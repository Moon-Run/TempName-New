"""Launch-correlated native linear evidence, retaining all MA11 path checks.

Native profile scopes carry the actual saved (plan_id, call_id) pair. An
explicit, validated in-memory event view lets the existing kernel/stream/order
checkers consume these scopes. It never changes input traces or invents a
forward/backward pairing from list order. Device events and correlation ids
remain unchanged, and the original Python path is forbidden in the candidate.
"""
from collections import Counter, defaultdict
import re

import ma11_profile as baseline
from ma8_profile import end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-native-linear')
PREFIX = 'mav1.'
PATTERN = re.compile(r'^mav1\.(column|row)\.(forward|backward)\.(\d+)\.(\d+)$')
ALIASES = {('column', 'forward'): '_MA8CachedRawLinear',
           ('column', 'backward'): '_MA8CachedRawLinearBackward',
           ('row', 'forward'): 'ConsumerGemmRSFunction',
           ('row', 'backward'): 'ConsumerGemmRSFunctionBackward'}


def native_scopes(events, arm):
    require(arm in ARMS, 'Unknown MAV1 arm')
    markers = [e for e in events if e.get('name', '').startswith(PREFIX)]
    if arm == ARMS[0]:
        require(not markers, 'MA11 control unexpectedly contains native linear scopes')
        return {}, {}
    require(len(markers) == 1024, 'MAV1 requires 1024 native linear scopes')
    old = set(ALIASES.values())
    require(not any(e.get('cat') == 'cpu_op' and
                    (e.get('name') in old or
                     e.get('name', '').removeprefix('autograd::engine::evaluate_function: ') in old)
                    for e in events), 'MAV1 fell back to Python linear autograd')
    pairs = defaultdict(dict)
    per_plan = defaultdict(list)
    lookup = {}
    for scope in markers:
        match = PATTERN.fullmatch(scope['name'])
        require(match is not None and scope.get('ph') == 'X' and
                scope.get('cat') in ('cpu_op', 'user_annotation'),
                'Malformed MAV1 native scope')
        kind, phase, plan, call = match.groups()
        plan, call = int(plan), int(call)
        require((0 <= plan < 64 if kind == 'column' else 64 <= plan < 128)
                and call >= 1, 'MAV1 plan or call identity changed')
        require(scope.get('args', {}).get('External id') is not None,
                'MAV1 scope lacks actual profiler External id')
        key = (kind, plan, call)
        require(phase not in pairs[key], 'Duplicate MAV1 native call phase')
        pairs[key][phase] = scope
        lookup[id(scope)] = (key, phase)
        if phase == 'forward':
            per_plan[kind, plan].append(call)
    require(len(pairs) == 512 and len(per_plan) == 128,
            'MAV1 native module/microbatch coverage changed')
    for key, phases in pairs.items():
        require(set(phases) == {'forward', 'backward'}, 'Unpaired MAV1 native invocation')
        require(end(phases['forward']) <= phases['backward']['ts'] + .01,
                'MAV1 native backward precedes forward')
    for calls in per_plan.values():
        calls.sort()
        require(len(calls) == 4 and calls == list(range(calls[0], calls[0] + 4)),
                'MAV1 requires four consecutive microbatch invocations per plan')
    require(len({tuple(sorted(v)) for v in per_plan.values()}) == 1,
            'MAV1 plans have different warmup/call progress')
    return dict(pairs), lookup


def validated_event_view(events, arm):
    pairs, lookup = native_scopes(events, arm)
    if not pairs:
        return events
    # These are checker-local identifiers derived from validated native ctx
    # tokens, not claimed to be automatic autograd Sequence number records.
    sequence = {key: -(i + 1) for i, key in enumerate(sorted(pairs))}
    output = []
    for event in events:
        if id(event) not in lookup:
            output.append(event)
            continue
        key, phase = lookup[id(event)]
        alias = dict(event)
        alias['cat'] = 'cpu_op'
        alias['name'] = ALIASES[key[0], phase]
        alias['args'] = dict(event['args'], **{'Sequence number': sequence[key]})
        output.append(alias)
    return output


def trace_native_linear_records(events, arm):
    pairs, _ = native_scopes(events, arm)
    if arm == ARMS[0]:
        return dict(installed=False, original_python_linear_calls=1024)
    counts, work = Counter(), Counter()
    for (kind, plan, call), phases in pairs.items():
        for phase, scope in phases.items():
            name = kind + '_' + phase
            counts[name] += 1
            work[name] += scope['dur'] / 1000
    require(all(counts[k] == 256 for k in
                ('column_forward', 'column_backward', 'row_forward', 'row_backward')),
            'MAV1 per-phase native coverage changed')
    return dict(installed=True, native_plans=128, native_invocations=512,
                native_scopes=1024, calls=dict(counts), cpu_scope_ms=dict(work),
                original_python_linear_calls=0,
                pairing='explicit native plan_id/call_id saved in autograd context',
                checker_view='validated tokens only; original device events and CUDA correlations retained',
                limitation='Profile scope times include instrumentation and overlap; they do not '
                           'predict unprofiled complete-step savings.')


def fullstep_collective_inventory(events):
    """Frozen MA11 one-step totals include the untouched vocabulary/loss path."""
    kernels = [e for e in events if e.get('cat') == 'kernel']
    counts = {name: sum('nccl' in e['name'].lower() and name in e['name'] for e in kernels)
              for name in ('AllGather', 'ReduceScatter')}
    counts['publication'] = sum('CudaIpcBarrierAllKernel' in e['name'] for e in kernels)
    require(counts == {'AllGather': 524, 'ReduceScatter': 260, 'publication': 256},
            'MAV1 changed the complete MA11 collective/publication inventory')
    return counts


def trace_composed_records(events, arm, rank=None):
    checked = validated_event_view(events, arm)
    result = baseline.trace_composed_records(checked, baseline.ARMS[1], rank)
    result['activation_cache']['attribution'] = (
        'native_saved_context_token_and_fixed_topology' if arm == ARMS[1]
        else result['activation_cache']['attribution'])
    result['native_linear'] = trace_native_linear_records(events, arm)
    return result
