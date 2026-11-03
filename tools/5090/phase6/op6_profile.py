"""OP6 local-attention graph audit for the pinned NGC PyTorch 2.8 build.

Reuse MA15's launch-correlated ownership/lifecycle checks, with explicit SM120
dispatch and PyTorch 5228986c39 replay adaptations. All OP3 projection, cache,
zero-count and communication audits remain unchanged.
"""
from pathlib import Path
from types import ModuleType

import op3_profile
from rtx5090_loader import H100
from ma9_profile import contains, require

PROTOCOL = 'rtx5090-op6-local-attention-graph-v1'
VARIANTS = ('op4', 'op6')
# Local empirical admission bounds for the pinned SM120/CUPTI workload. These
# are not a hardware clock-accuracy guarantee. The first eight-rank capture
# showed <=0.896 us end-timestamp overlap (<=23.91% of the shorter kernel).
# Retain a hard absolute AND relative cap, unique graph-launch correlation,
# device/context/stream equality, ordered starts, and strict staging boundaries.
GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US = 1.0
GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION = .25
RNG_SOURCE = ('https://github.com/pytorch/pytorch/blob/5228986c39/'
              'aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L198-L208')
GEMM_PREFIXES = (
    'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8',
    'cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_nn_align8',
    'cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_nt_align8',
)


def _replace(source, old, new):
    require(source.count(old) == 1, 'MA15 graph-audit source contract changed: ' + old[:100])
    return source.replace(old, new)


def _family(event):
    if event['cat'] == 'gpu_memcpy':
        name = event['name'].lower()
        return 'copy' if 'dtod' in name or 'device -> device' in name else 'unexpected'
    if event['cat'] != 'kernel':
        return 'unexpected'
    name = event['name']
    if 'rtx5090_ma11_softmax::ma11_softmax_forward_kernel' in name:
        return 'softmax_forward'
    if 'rtx5090_ma11_softmax::ma11_softmax_backward_kernel' in name:
        return 'softmax_backward'
    if 'cutlass::Kernel2<' in name and any(token in name for token in GEMM_PREFIXES):
        return 'gemm'
    if 'direct_copy_kernel_cuda' in name:
        return 'copy'
    if 'AUnaryFunctor<c10::BFloat16' in name and 'MulFunctor<float>' in name:
        return 'scale'
    return 'unexpected'


def _rng_replay_prologue(index, scope, launch):
    # Unlike PyTorch 2.6, the pinned 2.8 commit guards both generator-state
    # fills with wholegraph_increment. This attention graph contains no RNG.
    fills = [e for e in index.cpu if e['name'] == 'aten::fill_' and contains(scope, e)]
    require(not fills, 'RNG fills are unexpected for the pinned nonrandom OP6 graph')
    return [], []


def _graph_auditor():
    source_path = H100 / 'ma15_profile.py'
    source = source_path.read_text()
    source = _replace(source, 'Counter(gemm=2, softmax_forward=1, copy=1, blas_memset=1)',
                      'Counter(gemm=2, softmax_forward=1, copy=1)')
    source = _replace(source, 'Counter(gemm=4, softmax_backward=1, scale=2, blas_memset=2)',
                      'Counter(gemm=4, softmax_backward=1, scale=2)')
    source = _replace(source,
        "        block = 1024 if phase == 'forward' else 128\n"
        "        require(native['args'].get('grid') == [8192, 1, 1] and\n"
        "                native['args'].get('block') == [block, 1, 1],",
        "        grid, block = (([2048, 1, 1], [32, 4, 1]) if phase == 'forward'\n"
        "                       else ([8192, 1, 1], [128, 1, 1]))\n"
        "        require(native['args'].get('grid') == grid and\n"
        "                native['args'].get('block') == block,")
    source = _replace(source,
        "        require(max(end(e) for e in input_work) <= min(e['ts'] for e in fill_work) + .01 and\n"
        "                max(end(e) for e in fill_work) <= graph[0]['ts'] + .01,\n"
        "                'RNG replay prologue device fills must follow input copies and precede graph execution')",
        "        require(not fill_work, 'Nonrandom PyTorch 2.8 graph acquired RNG device work')")
    module = ModuleType('rtx5090_op6_ma15_graph_audit')
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    module._family = _family
    module._rng_replay_prologue = _rng_replay_prologue
    module.GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US = GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US
    module.GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION = GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION
    return module


graph_auditor = _graph_auditor()
FORWARD_MARKER = graph_auditor.FORWARD_MARKER
BACKWARD_MARKER = graph_auditor.BACKWARD_MARKER
_original_analyze = op3_profile.analyze


def attention_graph_records(events, variant, rank=None):
    require(variant in VARIANTS, 'Unknown OP6 profile variant')
    result = graph_auditor.trace_attention_graph_records(
        events, graph_auditor.ARMS[variant == 'op6'])
    if variant == 'op4':
        return result
    # Reject accidental broadening of the GEMM implementation allowed by the
    # family predicate. These are the actual OP4 attention launch signatures.
    expected = {
        'forward': [(GEMM_PREFIXES[0], [512, 8, 4]), (GEMM_PREFIXES[1], [16, 1, 4])],
        'backward': [(GEMM_PREFIXES[2], [16, 1, 4]), (GEMM_PREFIXES[0], [512, 8, 4]),
                     (GEMM_PREFIXES[1], [16, 1, 4]), (GEMM_PREFIXES[2], [8, 4, 4])],
    }
    for phase, signature in result['kernel_signatures'].items():
        gemms = [item for item in signature if any(token in item[0] for token in GEMM_PREFIXES)]
        require(len(gemms) == len(expected[phase]) and
                all(token in actual[0] and actual[1] == grid and actual[2] == [128, 1, 1]
                    for actual, (token, grid) in zip(gemms, expected[phase])),
                'OP6 changed the original OP4 attention GEMM dispatch')
    # OP3 replay compares the reanalysis directly with JSON-loaded evidence.
    # MA15 originally returned tuple signatures, which cannot survive that
    # comparison even when every kernel is identical.
    result['kernel_signatures'] = {
        phase: [list(item) for item in signature]
        for phase, signature in result['kernel_signatures'].items()}
    result['timestamp_order_check'].update(
        policy='op6-sm120-correlated-graph-kernel-timestamps-v1',
        note='Empirical OP6 trace-admission bounds, not guaranteed timestamp accuracy. '
             'Only kernels within the same uniquely correlated graph launch and '
             'device/context/stream may have bounded endpoint overlap. Starts, '
             'staging copies, replay boundaries and compute dependency sequence '
             'remain checked; raw timestamps are never rewritten.')
    if rank is not None:
        require(type(rank) is int and 0 <= rank < 8, 'Wrong OP6 graph rank')
        launches = {e['args']['correlation'] for e in events if graph_auditor._is_graph_launch(e)}
        require(all(e.get('args', {}).get('device') == rank for e in events
                    if e.get('cat') in graph_auditor.DEVICE_CATEGORIES and
                    e.get('args', {}).get('correlation') in launches),
                'OP6 graph node is on the wrong CUDA device')
    result.update(protocol=PROTOCOL, pytorch_commit='5228986c39',
        rng_replay_prologue=dict(fills_per_replay=0, fill_kernel_calls=0,
            logical_write_bytes_from_pinned_source=0, gpu_work_ms=0, cpu_union_ms=0,
            source=RNG_SOURCE, note='The pinned replay_prologue skips state fills when '
            'wholegraph_increment is zero; attention dropout is zero and capture RNG is checked.'),
        limitation='Graph work, staging and owned-output copies are included. Profile '
        'intervals overlap and are perturbed; none is an unprofiled step saving.')
    return result


def analyze(events, arm, rank, variant='op6'):
    require(variant in VARIANTS, 'Unknown OP6 profile variant')
    graph = attention_graph_records(events, variant, rank)
    if variant == 'op4':
        result = _original_analyze(events, arm, rank)
    else:
        # Call the original composition with only its attention checker changed.
        # Function cloning avoids modifying process-global MA11 helpers.
        from types import SimpleNamespace, FunctionType
        original_base = op3_profile.base
        local_base = SimpleNamespace(**vars(original_base))
        local_base.ma11_profile = SimpleNamespace(
            trace_attention_softmax_records=lambda *_args: graph)
        original = _original_analyze
        composed = FunctionType(original.__code__, dict(original.__globals__, base=local_base),
                                original.__name__, original.__defaults__, original.__closure__)
        result = composed(events, arm, rank)
    return dict(result, op6_variant=variant, attention_graph=graph)


def install(variant):
    """Install one child audit; return an idempotent restoration callback."""
    require(variant in VARIANTS, 'Unknown OP6 profile variant')
    previous = op3_profile.analyze
    replacement = lambda events, arm, rank: analyze(events, arm, rank, variant)
    op3_profile.analyze = replacement

    def restore():
        if op3_profile.analyze is replacement:
            op3_profile.analyze = previous
    return restore
