"""Strict launch evidence for complete MA11 versus resident-exponential MA13.

The two arms retain the same attention graph and backward implementation.
Durations are profiled work, never a claim of unprofiled critical-path savings.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import groupby

import ma8_profile
import ma9_profile
from ma9_profile import Index, contains, end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-resident')
FORWARD_MARKER = 'ma13.attention.softmax.forward'
BACKWARD_MARKER = 'ma13.attention.softmax.backward'
FORWARD_KERNEL = 'ma13_softmax_forward_kernel'
CONTROL_FORWARD_KERNEL = 'ma11_softmax_forward_kernel'
BACKWARD_KERNEL = 'ma11_softmax_backward_kernel'


def _arm(arm):
    require(arm in ARMS, 'MA13 accepts only complete MA11 and resident softmax arms')


def _one(values, message):
    require(len(values) == 1, message)
    return values[0]


def _kernels(index, scope):
    work, _ = index.work(scope)
    return [event for event in work if event['cat'] == 'kernel']


def _stream_path(events, message):
    require(bool(events), message + ': no GPU work')
    stream = events[0]['args'].get('stream')
    require(stream is not None and all(e['args'].get('stream') == stream for e in events),
            message + ': compute stream changed')
    require(all(end(a) <= b['ts'] + .01 for a, b in zip(events, events[1:])),
            message + ': GPU dependency order changed')
    return stream


class CPUIndex:
    def __init__(self, cpu):
        self.rows = defaultdict(list)
        for event in cpu:
            self.rows[event['tid'], event['name']].append(event)
        for rows in self.rows.values():
            rows.sort(key=lambda e: e['ts'])
        self.starts = {key: [e['ts'] for e in rows] for key, rows in self.rows.items()}

    def neighbor(self, scope, name, before):
        key = scope['tid'], name
        rows = self.rows.get(key, [])
        starts = self.starts.get(key, [])
        position = (bisect_left(starts, scope['ts']) - 1 if before
                    else bisect_right(starts, end(scope)))
        require(0 <= position < len(rows), 'Missing attention neighbor ' + name)
        value = rows[position]
        require((end(value) <= scope['ts'] + .01 if before else end(scope) <= value['ts'] + .01),
                'Attention CPU dependency order changed')
        return value


def trace_attention_softmax_records(events, arm):
    _arm(arm)
    candidate = arm == ARMS[1]
    prefix = 'ma13' if candidate else 'ma11'
    marker_names = {phase: prefix + '.attention.softmax.' + phase
                    for phase in ('forward', 'backward')}
    kernel_names = dict(forward=FORWARD_KERNEL if candidate else CONTROL_FORWARD_KERNEL,
                        backward=BACKWARD_KERNEL)
    index = Index(events)
    cpu = CPUIndex(index.cpu)
    counts = Counter(e['name'] for e in index.cpu)
    require(counts['aten::baddbmm'] == 128 and counts['aten::bmm'] == 640,
            'Original attention QK/PV and backward GEMM coverage changed')
    scopes = {phase: [e for e in events if e.get('cat') == 'user_annotation' and
                     e['name'] == marker] for phase, marker in marker_names.items()}
    all_markers = [e for e in events if e.get('cat') == 'user_annotation' and
                   e['name'].startswith(('ma11.attention.softmax.', 'ma13.attention.softmax.'))]
    native = [e for e in events if e.get('cat') == 'kernel' and
              any(token in e['name'] for token in
                  (FORWARD_KERNEL, CONTROL_FORWARD_KERNEL, BACKWARD_KERNEL))]
    require(len(scopes['forward']) == len(scopes['backward']) == 128 and
            len(native) == len(all_markers) == 256,
            'Softmax marker/kernel coverage or arm identity changed')
    require(not counts['aten::_softmax'] and not counts['aten::_softmax_backward_data'] and
            not counts['MaskedFillBackward0'] and not counts['AsStridedBackward0'],
            'Softmax fell back to original ATen/mask/view replay')
    require(counts['torch::autograd::CopySlices'] == 4,
            'Attention CopySlices survived or original LM-head copies changed')
    require(counts['aten::masked_fill_'] == 20 and counts['aten::mul'] == 532,
            'Dense attention mask/product pass or unrelated operations changed')
    require(not any(e.get('cat') == 'kernel' and
                    ('cunn_SoftMaxForward' in e['name'] or 'cunn_SoftMaxBackward' in e['name'])
                    for e in events), 'Unowned original softmax kernel')
    phases = sorted([(e, phase) for phase, rows in scopes.items() for e in rows],
                    key=lambda pair: pair[0]['ts'])
    require([(name, len(list(rows))) for name, rows in groupby(phases, key=lambda p: p[1])] ==
            [('forward', 32), ('backward', 32)] * 4,
            'Attention four-microbatch forward/backward order changed')
    selected, streams = [], set()
    work_ms, cpu_ms = Counter(), Counter()
    for scope, phase in phases:
        kernel = _one(_kernels(index, scope), 'Marker must own exactly one softmax kernel')
        require(kernel_names[phase] in kernel['name'], 'Marker owns the wrong softmax kernel')
        work, _ = index.work(scope)
        require(len(work) == 1, 'Softmax has extra copies, clears or fallback work')
        block = 256 if candidate else 1024
        if phase == 'backward':
            block = 128
        require(kernel['args'].get('grid') == [8192, 1, 1] and
                kernel['args'].get('block') == [block, 1, 1],
                'Softmax launch geometry changed')
        selected.append(kernel)
        work_ms[phase + '_fused'] += kernel['dur'] / 1000
        cpu_ms[phase] += scope['dur'] / 1000
        if phase == 'forward':
            qk = cpu.neighbor(scope, 'aten::baddbmm', True)
            pv = cpu.neighbor(scope, 'aten::bmm', False)
            qk_kernel = _one(_kernels(index, qk), 'Original QK GEMM kernel missing')
            pv_kernel = _one(_kernels(index, pv), 'Original PV GEMM kernel missing')
            streams.add(_stream_path([qk_kernel, kernel, pv_kernel], 'QK-softmax-PV'))
        else:
            upstream = cpu.neighbor(scope, 'aten::bmm', True)
            downstream = cpu.neighbor(scope, 'BaddbmmBackward0', False)
            preceding = _one(_kernels(index, upstream), 'Attention gradient producer missing')
            following = sorted(_kernels(index, downstream), key=lambda e: e['ts'])
            require(following, 'Attention score-gradient consumer missing')
            streams.add(_stream_path([preceding, kernel] + following, 'Attention backward'))
    require(len(streams) == 1, 'Attention moved between compute streams')
    require(len({id(e) for e in selected}) == len(selected), 'Attention work attributed more than once')
    require({id(e) for e in native} == {id(e) for e in selected},
            'Softmax kernel lacks a unique launch in its owning marker')
    return dict(arm=arm, candidate=candidate, forward_calls=128, backward_calls=128,
                original_qk_calls=128, original_bmm_calls=640, microbatches=4,
                softmax_compute_kernel_calls=len(selected), compute_stream=next(iter(streams)),
                forward_kernel=kernel_names['forward'], backward_kernel=BACKWARD_KERNEL,
                forward_block=256 if candidate else 1024, backward_block=128,
                original_softmax_calls=counts['aten::_softmax'],
                original_softmax_backward_calls=counts['aten::_softmax_backward_data'],
                attention_view_replays=counts['AsStridedBackward0'],
                copy_slices_calls=counts['torch::autograd::CopySlices'],
                global_masked_fill_calls=counts['aten::masked_fill_'],
                global_mul_calls=counts['aten::mul'], gpu_work_ms=dict(work_ms),
                softmax_cpu_ms=dict(cpu_ms), original_ma11_backward=True,
                limitation='Profile work and CPU intervals overlap and are perturbed; '
                           'they are not additive critical-path time or unprofiled step savings.')


def trace_activation_cache_records(events, arm):
    _arm(arm)
    return ma8_profile.trace_activation_cache_records(events, 'F4-cache')


def trace_row_backward_records(events, arm):
    _arm(arm)
    return ma8_profile.trace_row_backward_records(events, 'F4-cache')


def trace_zero_count_records(events, arm, rank=None):
    _arm(arm)
    return dict(ma9_profile.trace_zero_count_records(events, 'F4-zcount', rank),
                arm=arm, optimizer_source_arm='F4-zcount')


def trace_composed_records(events, arm, rank=None):
    _arm(arm)
    return dict(activation_cache=trace_activation_cache_records(events, arm),
                zero_count=trace_zero_count_records(events, arm, rank),
                row_backward=trace_row_backward_records(events, arm),
                attention_softmax=trace_attention_softmax_records(events, arm))
