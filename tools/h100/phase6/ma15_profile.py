"""Fail-closed CUDA Graph replay evidence for the fixed MA11 attention core.

Graph kernels are owned by the actual cudaGraphLaunch CUPTI correlation, not
by guessed temporal proximity to a CPU marker. Traces without that association
cannot validate MA15. CPU/GPU profile intervals never imply step-time savings.
"""
from collections import Counter
from itertools import groupby

import ma11_profile
from ma9_profile import Index, contains, end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-agraph')
FORWARD_MARKER = 'ma15.attention.graph.forward'
BACKWARD_MARKER = 'ma15.attention.graph.backward'
FORWARD_KERNEL = ma11_profile.FORWARD_KERNEL
BACKWARD_KERNEL = ma11_profile.BACKWARD_KERNEL
DEVICE_CATEGORIES = ('kernel', 'gpu_memcpy', 'gpu_memset')
# CUPTI end-timestamp collection can show sub-microsecond apparent overlap
# even for ordered kernels in one stream. NVIDIA describes overlaps normally
# below 500 ns: https://forums.developer.nvidia.com/t/73333/2
# Keep this exception local to nodes owned by the SAME graph launch, with
# matching device/context/stream. Staging and replay boundaries stay strict.
GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US = .5
GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION = .05


def interval_union_ms(events):
    """Union timestamps within one trace/clock domain; never across ranks."""
    intervals = sorted((e['ts'], end(e)) for e in events)
    total = 0.0
    right = None
    for left, stop in intervals:
        if right is None or left > right:
            total += stop - left
        elif stop > right:
            total += stop - right
        right = stop if right is None else max(stop, right)
    return total / 1000


def _is_graph_launch(event):
    return (event.get('cat') in ('cuda_runtime', 'cuda_driver') and
            event.get('name', '').split('_')[0] in ('cudaGraphLaunch', 'cuGraphLaunch'))


def _family(event):
    if event['cat'] == 'gpu_memset':
        require(event.get('args', {}).get('bytes') == 4,
                'Only the original cuBLAS 4-byte memset is allowed inside attention graphs')
        return 'blas_memset'
    if event['cat'] == 'gpu_memcpy':
        return 'copy'
    name = event['name']
    if FORWARD_KERNEL in name:
        return 'softmax_forward'
    if BACKWARD_KERNEL in name:
        return 'softmax_backward'
    if 'sm90_xmma_gemm_bf16bf16_bf16f32_f32_' in name and 'cublas' in name:
        return 'gemm'
    if 'direct_copy_kernel_cuda' in name:
        return 'copy'
    if 'AUnaryFunctor<c10::BFloat16' in name and 'MulFunctor<float>' in name:
        return 'scale'
    return 'unexpected'


def _same_stream_order(work, graph):
    work = sorted(work, key=lambda e: e['ts'])
    message = 'MA15 attention graph and staging copies'
    require(bool(work), message + ': no GPU work')
    stream = work[0]['args'].get('stream')
    require(stream is not None and all(e['args'].get('stream') == stream for e in work),
            message + ': compute stream changed')
    graph_ids = {id(e) for e in graph}
    tolerated = []
    for left, right in zip(work, work[1:]):
        overlap = end(left) - right['ts']
        if overlap <= .01:
            continue
        a, b = left.get('args', {}), right.get('args', {})
        limit = min(GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US,
                    GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION * min(left['dur'], right['dur']))
        same_launch = (id(left) in graph_ids and id(right) in graph_ids and
                       left['cat'] == right['cat'] == 'kernel' and
                       a.get('correlation') is not None and a.get('correlation') == b.get('correlation') and
                       all(a.get(key) is not None and a.get(key) == b.get(key)
                           for key in ('device', 'context')))
        require(same_launch and left['ts'] < right['ts'] and overlap <= limit,
                message + ': GPU dependency order changed')
        tolerated.append(dict(correlation=a['correlation'], device=a['device'], context=a['context'],
                              stream=stream, predecessor=left['name'], successor=right['name'],
                              predecessor_ts_us=left['ts'], successor_ts_us=right['ts'],
                              observed_overlap_us=overlap, allowed_overlap_us=limit))
    return stream, tolerated


def _rng_replay_prologue(index, scope, launch):
    """Account for the two unconditional PyTorch 2.6 replay_prologue fills.

    CUDAGraph::capture_begin always registers the default CUDA generator;
    replay calls its prologue even when wholegraph_increment is zero. The
    two tensors are int64[1] in the pinned CUDAGeneratorImpl.cpp. Ordinary
    record_shapes=False traces prove dtype/launch geometry and ownership,
    not tensor numel; do not report the source-derived [1] as measured shape.
    """
    fills = [e for e in index.cpu if e['name'] == 'aten::fill_' and contains(scope, e)]
    require(len(fills) == 2, 'Expected exactly two pinned PyTorch RNG replay prologue fills')
    work = []
    for fill in fills:
        require(end(fill) <= launch['ts'] + .01,
                'RNG replay prologue fill was submitted after graph launch')
        # Shape metadata is absent in the frozen record_shapes=False loop.
        # If supplied by an exporter, reject contrary evidence as well.
        shapes = fill.get('args', {}).get('Input Dims')
        require(shapes is None or isinstance(shapes, list) and shapes and shapes[0] == [1],
                'RNG replay prologue tensor shape differs from pinned int64[1]')
        owned, _ = index.work(fill)
        require(len(owned) == 1 and owned[0]['cat'] == 'kernel' and
                'FillFunctor<long>' in owned[0]['name'] and
                owned[0].get('args', {}).get('grid') == [1, 1, 1] and
                owned[0].get('args', {}).get('block') == [128, 1, 1],
                'RNG replay prologue must own one original scalar int64 fill kernel')
        work.extend(owned)
    require(len({id(e) for e in work}) == 2, 'RNG replay prologue fill kernel was attributed twice')
    return fills, work


def trace_attention_graph_records(events, arm):
    require(arm in ARMS, 'MA15 accepts only original MA11 and attention-graph arms')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and
               e.get('name') in (FORWARD_MARKER, BACKWARD_MARKER)]
    launches = [e for e in events if _is_graph_launch(e)]
    if arm == ARMS[0]:
        require(not markers and not launches, 'MA11 control acquired MA15 graph replay')
        return dict(installed=False, graph_replay=False)

    forward = [e for e in markers if e['name'] == FORWARD_MARKER]
    backward = [e for e in markers if e['name'] == BACKWARD_MARKER]
    require(len(forward) == len(backward) == 128 and len(launches) == 256,
            'MA15 needs 128 forward and 128 backward markers and graph launches')
    require(len({e.get('args', {}).get('correlation') for e in launches}) == 256 and
            all(e.get('args', {}).get('correlation') is not None for e in launches),
            'Graph launch correlation IDs are missing or reused; cannot validate graph node ownership')
    require(not any(e.get('cat') in ('cuda_runtime', 'cuda_driver') and
                    any(word in e['name'] for word in
                        ('BeginCapture', 'EndCapture', 'GraphInstantiate', 'GraphCreate', 'GraphAdd'))
                    for e in events), 'Graph capture or instantiation leaked into the profile window')
    counts = Counter(e['name'] for e in events if e.get('cat') == 'cpu_op')
    require(not counts['aten::baddbmm'] and not counts['aten::bmm'] and
            not counts['BaddbmmBackward0'] and not counts['BmmBackward0'] and
            not any(e.get('cat') == 'user_annotation' and e.get('name') in
                    (ma11_profile.FORWARD_MARKER, ma11_profile.BACKWARD_MARKER) for e in events),
            'Candidate retained eager attention submissions or original MA11 markers')
    require(not counts['aten::_softmax'] and not counts['aten::_softmax_backward_data'] and
            not counts['MaskedFillBackward0'] and not counts['AsStridedBackward0'],
            'Candidate fell back to original attention softmax/mask/view replay')
    require(counts['torch::autograd::CopySlices'] == 4,
            'Original LM-head CopySlices coverage changed')

    phases = sorted([(e, 'forward') for e in forward] + [(e, 'backward') for e in backward],
                    key=lambda pair: pair[0]['ts'])
    require([(phase, len(list(rows))) for phase, rows in groupby(phases, key=lambda p: p[1])] ==
            [('forward', 32), ('backward', 32)] * 4,
            'Attention four-microbatch forward/backward order changed')
    index = Index(events)
    seen_launch, seen_work, graph_work, staging_work, streams = set(), set(), [], [], set()
    rng_cpu, rng_work, timestamp_overlaps = [], [], []
    phase_counts, signatures = {}, {}
    for scope, phase in phases:
        owned_launches = [e for e in launches if contains(scope, e)]
        require(len(owned_launches) == 1, 'Each MA15 marker must own exactly one graph launch')
        launch = owned_launches[0]
        require(id(launch) not in seen_launch, 'Graph launch is attributed more than once')
        seen_launch.add(id(launch))
        correlation = launch['args']['correlation']
        graph = sorted(index.device.get(correlation, []), key=lambda e: e['ts'])
        require(graph, 'No graph nodes match cudaGraphLaunch correlation; CUPTI association is unavailable')
        work, _ = index.work(scope)
        require({id(e) for e in graph} <= {id(e) for e in work},
                'Graph nodes are not uniquely associated with their replay marker')
        require(not any('nccl' in e['name'].lower() or 'flux' in e['name'].lower() or
                        'taco_' in e['name'] for e in graph),
                'Communication or Flux projection entered an attention graph')
        families = Counter(_family(e) for e in graph)
        expected = (Counter(gemm=2, softmax_forward=1, copy=1, blas_memset=1)
                    if phase == 'forward' else
                    Counter(gemm=4, softmax_backward=1, scale=2, blas_memset=2))
        require(families == expected, 'Unexpected graph node coverage: ' + phase + ' ' + str(families))
        # Discard only cuBLAS scratch clears when checking the original compute
        # dependency chain. All clears remain in the work accounting above.
        sequence = [_family(e) for e in graph if e['cat'] == 'kernel']
        expected_sequence = (['gemm', 'softmax_forward', 'gemm', 'copy'] if phase == 'forward' else
                             ['gemm', 'gemm', 'softmax_backward', 'gemm', 'scale', 'gemm', 'scale'])
        require(sequence == expected_sequence, 'Original attention graph dependency order changed')
        native = next(e for e in graph if _family(e).startswith('softmax_'))
        block = 1024 if phase == 'forward' else 128
        require(native['args'].get('grid') == [8192, 1, 1] and
                native['args'].get('block') == [block, 1, 1],
                'Original MA11 softmax launch geometry changed')
        # Replays must keep the same concrete GEMM and native launch identity.
        signature = [(e['name'], e.get('args', {}).get('grid'), e.get('args', {}).get('block'))
                     for e in graph if e['cat'] == 'kernel']
        require(phase not in signatures or signatures[phase] == signature,
                'Attention graph replay changed its CUDA kernel or launch identity')
        signatures[phase] = signature
        fill_scopes, fill_work = _rng_replay_prologue(index, scope, launch)
        fill_ids = {id(e) for e in fill_work}
        staging = [e for e in work if e.get('args', {}).get('correlation') != correlation and
                   id(e) not in fill_ids]
        require(staging and all(_family(e) == 'copy' for e in staging),
                'Graph replay staging contains non-copy work or lacks ownership copies')
        cpu_copy = [e for e in index.cpu if e['name'] == 'aten::copy_' and contains(scope, e)]
        clones = [e for e in index.cpu if e['name'] == 'aten::clone' and contains(scope, e)]
        require(len(cpu_copy) == (5 if phase == 'forward' else 4) and
                len(clones) == (1 if phase == 'forward' else 3),
                'QKV/mask/gradient staging or independent output clone coverage changed')
        copy_work = {id(k) for copy_scope in cpu_copy for k in index.work(copy_scope)[0]}
        require({id(e) for e in staging} == copy_work,
                'Staging GPU work lacks a unique original copy operation')
        before = [e for e in cpu_copy if end(e) <= launch['ts'] + .01]
        after = [e for e in clones if e['ts'] >= end(launch) - .01]
        require(len(before) == (4 if phase == 'forward' else 1) and len(after) == len(clones),
                'Graph input copies/replay/output ownership dependency changed')
        input_work = [k for copy_scope in before for k in index.work(copy_scope)[0]]
        output_work = [k for clone_scope in after for k in index.work(clone_scope)[0]]
        require(input_work and output_work and
                max(end(e) for e in input_work) <= graph[0]['ts'] + .01 and
                end(graph[-1]) <= min(e['ts'] for e in output_work) + .01,
                'Graph device execution overtook its input copies or independent output clones')
        require(max(end(e) for e in input_work) <= min(e['ts'] for e in fill_work) + .01 and
                max(end(e) for e in fill_work) <= graph[0]['ts'] + .01,
                'RNG replay prologue device fills must follow input copies and precede graph execution')
        stream, overlaps = _same_stream_order(work, graph)
        streams.add(stream)
        timestamp_overlaps.extend(dict(item, phase=phase) for item in overlaps)
        require(not (seen_work & {id(e) for e in work}), 'Attention GPU work was attributed twice')
        seen_work.update(id(e) for e in work)
        graph_work.extend(graph)
        staging_work.extend(staging)
        rng_cpu.extend(fill_scopes)
        rng_work.extend(fill_work)
        phase_counts[phase] = dict(families)

    require(len(streams) == 1, 'Attention graph replays moved between compute streams')
    require(seen_launch == {id(e) for e in launches}, 'Unowned cudaGraphLaunch in profile')
    native = [e for e in events if e.get('cat') == 'kernel' and
              (FORWARD_KERNEL in e['name'] or BACKWARD_KERNEL in e['name'])]
    require(len(native) == 256 and {id(e) for e in native} <= seen_work,
            'Original MA11 softmax kernels are missing or lack graph ownership')
    # Any nonzero graph-node tag must be among the nodes associated above.
    # The tag alone is never used as evidence of a particular replay launch.
    tagged = [e for e in events if e.get('cat') in DEVICE_CATEGORIES and
              any(e.get('args', {}).get(key, 0) for key in ('graph id', 'graph node id'))]
    require({id(e) for e in tagged} <= {id(e) for e in graph_work},
            'Unowned CUDA graph node appears outside the attention replay')
    return dict(installed=True, graph_replay=True, forward_calls=128, backward_calls=128,
                forward_graph_launches=128, backward_graph_launches=128, microbatches=4,
                softmax_compute_kernel_calls=256, compute_stream=next(iter(streams)),
                original_qk_calls=128, original_bmm_calls=640, eager_attention_calls=0,
                original_softmax_calls=0, original_softmax_backward_calls=0,
                attention_view_replays=0, copy_slices_calls=4,
                graph_nodes_per_replay=phase_counts, kernel_signatures=signatures,
                timestamp_order_check=dict(
                    strict_boundary_tolerance_us=.01,
                    graph_kernel_tolerance_us=GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US,
                    max_fraction_of_shorter_kernel=GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION,
                    tolerated_overlap_count=len(timestamp_overlaps),
                    max_observed_overlap_us=max((e['observed_overlap_us'] for e in timestamp_overlaps), default=0),
                    tolerated_overlaps=timestamp_overlaps,
                    note='Bounded CUPTI timestamp overlap within one correlated graph launch; '
                         'not evidence of physical kernel concurrency. Staging/RNG/replay boundaries remain strict.'),
                graph_gpu_work_ms=sum(e['dur'] for e in graph_work) / 1000,
                staging_gpu_work_ms=sum(e['dur'] for e in staging_work) / 1000,
                graph_and_staging_gpu_union_ms=interval_union_ms(graph_work + staging_work),
                rng_replay_prologue=dict(
                    fills_per_replay=2, fill_kernel_calls=len(rng_work),
                    logical_write_bytes_from_pinned_source=len(rng_work) * 8,
                    gpu_work_ms=sum(e['dur'] for e in rng_work) / 1000,
                    cpu_union_ms=interval_union_ms(rng_cpu),
                    kernel_dtype_observed='int64 (FillFunctor<long>)',
                    tensor_shape_from_pinned_source=[1], tensor_shape_observed=False,
                    source='PyTorch v2.6.0 CUDAGeneratorImpl.cpp register_graph/replay_prologue; '
                        'CUDAGraph.cpp capture_begin/replay',
                    note='Both fills run even when wholegraph_increment=0. This does not imply RNG '
                        'advancement. Frozen record_shapes=False traces cannot measure numel; [1] '
                        'and 8-byte logical writes follow the pinned source, while exact ownership, '
                        'count, int64 kernel dtype and scalar launch geometry are trace-checked.'),
                total_attention_device_work_ms=sum(e['dur'] for e in graph_work + staging_work + rng_work) / 1000,
                total_attention_device_union_ms=interval_union_ms(graph_work + staging_work + rng_work),
                marker_cpu_union_ms=interval_union_ms(markers),
                graph_launch_cpu_union_ms=interval_union_ms(launches),
                graph_correlation_policy='Every graph node must share the unique owning cudaGraphLaunch '
                    'CUPTI correlation. Missing association fails validation; no temporal matching fallback.',
                limitation='Graph work includes original cuBLAS clears. Staging/clone work and all 512 '
                    'PyTorch RNG replay prologue fill kernels are separately reported and included in total. '
                    'Profile CPU and GPU intervals overlap and are perturbed; none is unprofiled step savings.')


def trace_composed_records(events, arm, rank=None):
    graph = trace_attention_graph_records(events, arm)
    if arm == ARMS[0]:
        result = ma11_profile.trace_composed_records(events, ma11_profile.ARMS[1], rank)
    else:
        result = dict(
            activation_cache=ma11_profile.trace_activation_cache_records(events, ma11_profile.ARMS[1]),
            zero_count=ma11_profile.trace_zero_count_records(events, ma11_profile.ARMS[1], rank),
            row_backward=ma11_profile.trace_row_backward_records(events, ma11_profile.ARMS[1]),
            attention_softmax=graph)
    result['attention_graph'] = graph
    return result
