"""Launch-correlated MA11 causal-softmax and unchanged MA89 path evidence.

Profile durations describe GPU work and perturbed spans, never unprofiled
critical-path savings. The reference uses its original ATen scopes, without
adding a marker or a per-call wrapper to the measured baseline.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import groupby

import ma8_profile
import ma9_profile
from ma9_profile import Index, contains, end, require

ARMS = ('F4-cache-zcount', 'F4-cache-zcount-csoftmax')
FORWARD_MARKER = 'ma11.attention.softmax.forward'
BACKWARD_MARKER = 'ma11.attention.softmax.backward'
FORWARD_KERNEL = 'ma11_softmax_forward_kernel'
BACKWARD_KERNEL = 'ma11_softmax_backward_kernel'


def _arm(arm):
    require(arm in ARMS, 'MA11 accepts only MA89 and MA89 causal-softmax arms')


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
    index = Index(events)
    cpu = CPUIndex(index.cpu)
    counts = Counter(e['name'] for e in index.cpu)
    require(counts['aten::baddbmm'] == 128 and counts['aten::bmm'] == 640,
            'Original attention QK/PV and backward GEMM coverage changed')
    forward = [e for e in events if e.get('cat') == 'user_annotation' and e['name'] == FORWARD_MARKER]
    backward = [e for e in events if e.get('cat') == 'user_annotation' and e['name'] == BACKWARD_MARKER]
    native = [e for e in events if e.get('cat') == 'kernel' and
              (FORWARD_KERNEL in e['name'] or BACKWARD_KERNEL in e['name'])]
    if candidate:
        require(len(forward) == len(backward) == 128 and len(native) == 256,
                'Candidate softmax marker/kernel coverage changed')
        require(not counts['aten::_softmax'] and not counts['aten::_softmax_backward_data'] and
                not counts['MaskedFillBackward0'] and not counts['AsStridedBackward0'],
                'Candidate fell back to original softmax/mask/view replay')
        require(counts['torch::autograd::CopySlices'] == 4,
                'Candidate retained attention CopySlices or changed original LM-head copies')
        require(counts['aten::masked_fill_'] == 20 and counts['aten::mul'] == 532,
                'Candidate retained a dense attention mask/product pass or changed unrelated operations')
        require(not any(e.get('cat') == 'kernel' and
                        ('cunn_SoftMaxForward' in e['name'] or 'cunn_SoftMaxBackward' in e['name'])
                        for e in events), 'Candidate has an unowned original softmax kernel')
    else:
        require(not forward and not backward and not native, 'MA89 control has MA11 wrappers or kernels')
        forward = [e for e in index.cpu if e['name'] == 'aten::_softmax']
        backward = [e for e in index.cpu if e['name'] == 'aten::_softmax_backward_data']
        require(len(forward) == len(backward) == counts['MaskedFillBackward0'] ==
                counts['AsStridedBackward0'] == 128 and counts['torch::autograd::CopySlices'] == 132,
                'Original softmax/mask/view replay coverage changed')
        require(counts['aten::masked_fill_'] == 276 and counts['aten::mul'] == 660,
                'Original dense mask/product or unrelated-operation coverage changed')

    phases = sorted([(e, 'forward') for e in forward] + [(e, 'backward') for e in backward],
                    key=lambda pair: pair[0]['ts'])
    require([(name, len(list(rows))) for name, rows in groupby(phases, key=lambda p: p[1])] ==
            [('forward', 32), ('backward', 32)] * 4,
            'Attention four-microbatch forward/backward order changed')
    selected, streams = [], set()
    work_ms = Counter()
    view_cpu_ms = Counter()
    view_device = {label: Counter() for label in ('as_strided', 'copy_slices')}
    for scope, phase in phases:
        local = _kernels(index, scope)
        if candidate:
            token = FORWARD_KERNEL if phase == 'forward' else BACKWARD_KERNEL
            kernel = _one(local, 'Candidate marker must own exactly one softmax kernel')
            require(token in kernel['name'], 'Candidate marker owns the wrong softmax kernel')
            device_work, _ = index.work(scope)
            require(len(device_work) == 1, 'Candidate softmax has extra copies, clears or fallback work')
            selected.append(kernel)
            chain = [kernel]
            work_ms[phase + '_fused'] += kernel['dur'] / 1000
        elif phase == 'forward':
            kernel = _one(local, 'Original forward softmax kernel coverage changed')
            require('cunn_SoftMaxForwardSmem' in kernel['name'] and
                    'BFloat16, float' in kernel['name'], 'Original BF16 softmax implementation changed')
            mask = cpu.neighbor(scope, 'aten::masked_fill_', True)
            mask_kernel = _one(_kernels(index, mask), 'Original forward mask kernel coverage changed')
            require('masked_fill_kernel' in mask_kernel['name'], 'Original forward mask changed')
            chain = [mask_kernel, kernel]
            selected.extend(chain)
            work_ms['forward_mask'] += mask_kernel['dur'] / 1000
            work_ms['forward_softmax'] += kernel['dur'] / 1000
        else:
            require(len(local) == 2, 'Original softmax backward must retain BF16 product and reduction')
            local.sort(key=lambda e: e['ts'])
            require('MulFunctor' in local[0]['name'] and 'BFloat16' in local[0]['name'] and
                    'cunn_SoftMaxBackward' in local[1]['name'], 'Original BF16 backward implementation changed')
            mask = cpu.neighbor(scope, 'MaskedFillBackward0', False)
            mask_kernel = _one([e for e in _kernels(index, mask) if 'masked_fill_kernel' in e['name']],
                               'Original backward mask kernel coverage changed')
            view = cpu.neighbor(scope, 'AsStridedBackward0', False)
            copies = cpu.neighbor(scope, 'torch::autograd::CopySlices', False)
            require(end(view) <= copies['ts'] + .01 and contains(copies, mask),
                    'Original in-place view replay no longer owns its mask backward')
            for label, owner in [('as_strided', view), ('copy_slices', copies)]:
                view_cpu_ms[label] += owner['dur'] / 1000
                owner_work, _ = index.work(owner)
                # CopySlices contains real DtoD memcpy operations, not just
                # compute kernels. Exclude its child mask kernel, which is
                # already attributed to backward_mask below.
                owner_work = [e for e in owner_work if 'masked_fill_kernel' not in e['name']]
                work_ms[label + '_device'] += sum(e['dur'] for e in owner_work) / 1000
                for event in owner_work:
                    category = event['cat']
                    view_device[label][category + '_calls'] += 1
                    view_device[label][category + '_ms'] += event['dur'] / 1000
                    if category in ('gpu_memcpy', 'gpu_memset'):
                        view_device[label][category + '_bytes'] += event['args'].get('bytes', 0)
            chain = local + [mask_kernel]
            selected.extend(chain)
            work_ms['backward_product'] += local[0]['dur'] / 1000
            work_ms['backward_softmax'] += local[1]['dur'] / 1000
            work_ms['backward_mask'] += mask_kernel['dur'] / 1000
        streams.add(_stream_path(chain, 'Attention softmax'))
        if phase == 'forward':
            qk = cpu.neighbor(scope, 'aten::baddbmm', True)
            pv = cpu.neighbor(scope, 'aten::bmm', False)
            qk_kernel = _one(_kernels(index, qk), 'Original QK GEMM kernel missing')
            pv_kernel = _one(_kernels(index, pv), 'Original PV GEMM kernel missing')
            streams.add(_stream_path([qk_kernel] + chain + [pv_kernel], 'QK-softmax-PV'))
        else:
            upstream = cpu.neighbor(scope, 'aten::bmm', True)
            downstream = cpu.neighbor(scope, 'BaddbmmBackward0', False)
            preceding = _one(_kernels(index, upstream), 'Attention softmax gradient producer missing')
            following = sorted(_kernels(index, downstream), key=lambda e: e['ts'])
            require(following, 'Attention score-gradient consumer missing')
            streams.add(_stream_path([preceding] + chain + following, 'Attention backward'))
    require(len(streams) == 1, 'Attention moved between compute streams')
    require(len({id(e) for e in selected}) == len(selected), 'Attention GPU work attributed more than once')
    if candidate:
        require({id(e) for e in native} == {id(e) for e in selected},
                'Candidate kernel lacks a unique launch in its owning marker')
    return dict(arm=arm, candidate=candidate, forward_calls=128, backward_calls=128,
                original_qk_calls=128, original_bmm_calls=640, microbatches=4,
                softmax_compute_kernel_calls=len(selected), compute_stream=next(iter(streams)),
                original_softmax_calls=counts['aten::_softmax'],
                original_softmax_backward_calls=counts['aten::_softmax_backward_data'],
                attention_view_replays=counts['AsStridedBackward0'],
                copy_slices_calls=counts['torch::autograd::CopySlices'],
                global_masked_fill_calls=counts['aten::masked_fill_'],
                global_mul_calls=counts['aten::mul'],
                gpu_work_ms=dict(work_ms), view_replay_cpu_ms=dict(view_cpu_ms),
                view_replay_device={label: dict(value) for label, value in view_device.items()},
                gpu_work_accounting='Named GPU families have disjoint launch ownership; view replay includes '
                    'kernel/memcpy/memset and excludes its separately counted backward mask kernel. '
                    'Their work sum is not a wall-clock critical-path interval.',
                limitation='Profile GPU work and CPU intervals overlap and are perturbed; '
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
