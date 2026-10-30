"""Launch-correlated checks for original CR3 and fused zero-count statistics.

The reference needs no Python wrapper or new marker. Its unchanged ATen calls
identify the block; the candidate has a profile-only marker around the same
statistic, including its original collective and scalar readback.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

ARMS = ('F4', 'F4-zcount')
MARKER = 'ma9.optimizer.zero_count'
CHUNKS = 'ma9_zero_count_chunks_kernel'
TOTAL = 'ma9_zero_count_ordered_total_kernel'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def end(event):
    return event['ts'] + event.get('dur', 0)


def contains(parent, child):
    return (parent['tid'] == child['tid'] and parent['ts'] <= child['ts'] + .01
            and end(child) <= end(parent) + .01)


class Index:
    def __init__(self, events):
        self.events = events
        self.cpu = [e for e in events if e.get('cat') == 'cpu_op']
        self.runtime, self.device = defaultdict(list), defaultdict(list)
        for event in events:
            if event.get('cat') in ('cuda_runtime', 'cuda_driver'):
                self.runtime[event['tid']].append(event)
            elif event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                self.device[event.get('args', {}).get('correlation')].append(event)
        for rows in self.runtime.values():
            rows.sort(key=lambda event: event['ts'])
        self.starts = {tid: [e['ts'] for e in rows] for tid, rows in self.runtime.items()}

    def work(self, scope):
        rows = self.runtime[scope['tid']]
        starts = self.starts.get(scope['tid'], [])
        lo, hi = bisect_left(starts, scope['ts'] - .01), bisect_right(starts, end(scope) + .01)
        launches = {e['args']['correlation']: e for e in rows[lo:hi]
                    if contains(scope, e) and e.get('args', {}).get('correlation') is not None}
        work = [event for correlation in launches for event in self.device[correlation]]
        require(len({id(e) for e in work}) == len(work), 'Device work was attributed twice')
        return work, launches


def _outer_counts(cpu):
    nested = sorted((e for e in cpu if e['name'] == 'aten::count_nonzero'),
                    key=lambda e: (e['ts'], -e['dur']))
    outer = []
    for event in nested:
        if not outer or not contains(outer[-1], event):
            outer.append(event)
    require(len(nested) == len(outer) * 2, 'Original count_nonzero overload coverage changed')
    return outer


def _family(name):
    if 'nccl' in name.lower():
        return 'allreduce' if 'AllReduce' in name else 'unexpected_collective'
    if 'CompareEqFunctor' in name or 'CompareNeFunctor' in name:
        return 'nonzero_predicate'
    if 'direct_copy_kernel_cuda' in name:
        return 'predicate_to_int64'
    if 'ReduceOp<long' in name:
        return 'int64_sum'
    if 'CUDAFunctorOnOther_add<long>' in name:
        return 'numel_minus_nonzero'
    if 'CUDAFunctor_add<float>' in name:
        return 'ordered_fp32_accumulation'
    if CHUNKS in name:
        return 'fused_chunks'
    if TOTAL in name:
        return 'fused_ordered_total'
    return 'unexpected_kernel'


def _one(events, message):
    require(len(events) == 1, message)
    return events[0]


def trace_zero_count_records(events, arm, rank=None):
    require(arm in ARMS, 'Unknown MA9 arm')
    require(rank is None or type(rank) is int and 0 <= rank < 8, 'Invalid TP rank')
    require(not any(e.get('cat') in ('cpu_op', 'user_annotation') and
                    (e['name'].startswith('MA8') or e['name'].startswith('ma8.column.'))
                    for e in events), 'MA8 column wrappers contaminated CR3')
    index = Index(events)
    candidate = arm != 'F4'
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e['name'] == MARKER]
    native = [e for e in events if e.get('cat') == 'kernel' and
              (CHUNKS in e['name'] or TOTAL in e['name'])]
    counts = _outer_counts(index.cpu)
    if candidate:
        require(not counts, 'Candidate fell back to count_nonzero')
        scope = _one(markers, 'Expected one candidate-only zero-count marker')
        require(len(native) == 2, 'Expected exactly two fused zero-count kernels')
        parameter_count = None
    else:
        require(not markers and not native, 'Original CR3 has MA9 wrappers or kernels')
        require(len(counts) in (194, 389), 'Original eligible parameter count changed')
        parameter_count = len(counts)
        tid = counts[0]['tid']
        require(all(e['tid'] == tid for e in counts), 'Original zero count moved CPU threads')
        last_count_end = max(map(end, counts))
        following = sorted((e for e in index.cpu if e['name'] == 'c10d::allreduce_' and
                            e['tid'] == tid and e['ts'] >= last_count_end), key=lambda e: e['ts'])
        require(following, 'Missing original zero-count AllReduce')
        collective = following[0]
        # The source has exactly final rsub/add, AllReduce, wait and item here.
        require(collective['ts'] - last_count_end < 1000, 'Original zero-count block is not contiguous')
        scalar_candidates = sorted((e for e in index.cpu if e['name'] == 'aten::item' and
                                    e['tid'] == tid and e['ts'] >= end(collective)), key=lambda e: e['ts'])
        require(scalar_candidates, 'Missing original zero-count scalar read')
        scalar = scalar_candidates[0]
        scope = dict(tid=tid, ts=counts[0]['ts'], dur=end(scalar) - counts[0]['ts'])

    cpu = [e for e in index.cpu if contains(scope, e)]
    collective = _one([e for e in cpu if e['name'] == 'c10d::allreduce_'],
                      'Expected one unchanged zero-count AllReduce')
    comm = _one([e for e in cpu if e['name'] == 'record_param_comms' and
                 contains(collective, e) and e.get('args', {}).get('Collective name') == 'allreduce'],
                'Missing original scalar collective metadata')
    comm_args = comm.get('args', {})
    require(comm_args.get('In msg nelems') == comm_args.get('Out msg nelems') == 1 and
            comm_args.get('dtype') == 'Float' and comm_args.get('Group size') == 8 and
            comm_args.get('Process Group Description') == 'MODEL_PARALLEL_GROUP',
            'Zero-count collective shape, dtype or TP group changed')
    scalar = _one([e for e in cpu if e['name'] == 'aten::item'],
                  'Expected one unchanged zero-count scalar read')
    require(end(collective) <= scalar['ts'] + .01, 'Scalar read precedes AllReduce submission')
    work, launches = index.work(scope)
    kernels = [e for e in work if e['cat'] == 'kernel']
    families = Counter(_family(e['name']) for e in kernels)
    if candidate:
        require(families == Counter(fused_chunks=1, fused_ordered_total=1, allreduce=1),
                'Candidate fused kernel or collective coverage changed')
        require({id(e) for e in native} <= {id(e) for e in kernels},
                'Fused kernel lacks a launch inside its owning marker')
    else:
        expected = Counter({name: parameter_count for name in (
            'nonzero_predicate', 'predicate_to_int64', 'int64_sum',
            'numel_minus_nonzero', 'ordered_fp32_accumulation')})
        expected['allreduce'] = 1
        require(families == expected, 'Original zero-count kernel coverage changed')
        # Associate each old reduction separately, not merely global name totals.
        for call in counts:
            local, _ = index.work(call)
            require(Counter(_family(e['name']) for e in local if e['cat'] == 'kernel') ==
                    Counter(nonzero_predicate=1, predicate_to_int64=1, int64_sum=1),
                    'Original count_nonzero lost its three-kernel implementation')

    compute = sorted([e for e in kernels if _family(e['name']) != 'allreduce'], key=lambda e: e['ts'])
    stream = compute[0]['args']['stream']
    require(all(e['args']['stream'] == stream for e in compute), 'Zero-count compute stream changed')
    require(all(end(a) <= b['ts'] + .01 for a, b in zip(compute, compute[1:])),
            'Zero-count kernels overlap or violate their stream order')
    if candidate:
        require(_family(compute[0]['name']) == 'fused_chunks' and
                _family(compute[1]['name']) == 'fused_ordered_total', 'Fused zero-count order changed')
    device = compute[0]['args'].get('device')
    if rank is None and type(device) is int and 0 <= device < 8:
        rank = device
    if rank is not None:
        expected_parameters = 389 if rank == 0 else 194
        if not candidate:
            require(parameter_count == expected_parameters, 'Zero count does not match TP rank filtering')
    else:
        expected_parameters = None

    allreduce = _one([e for e in kernels if _family(e['name']) == 'allreduce'],
                     'Missing zero-count collective kernel')
    comm_work, _ = index.work(collective)
    require(allreduce in comm_work, 'AllReduce kernel lacks its original collective owner')
    require(allreduce['args']['stream'] != stream and end(compute[-1]) <= allreduce['ts'] + .01,
            'AllReduce consumed unfinished zero statistics or changed stream')

    copies = [e for e in work if e['cat'] == 'gpu_memcpy']
    scalar_work, _ = index.work(scalar)
    copy = _one(copies, 'Only the original scalar readback may copy GPU data')
    require(copy in scalar_work and copy['args'].get('bytes') == 4 and
            ('DtoH' in copy['name'] or 'Device ->' in copy['name']) and
            copy['args']['stream'] == stream and end(allreduce) <= copy['ts'] + .01,
            'Unknown copy or premature zero-count scalar readback')

    clears = [e for e in work if e['cat'] == 'gpu_memset']
    if candidate:
        require(not clears, 'Candidate unexpectedly clears workspace in the measured path')
    else:
        used_owners = set()
        for clear in clears:
            launch = launches[clear['args']['correlation']]
            owners = [e for e in cpu if e['name'] == 'aten::sum' and contains(e, launch)]
            owner = _one(owners, 'Workspace clear is outside its original count reduction')
            require(clear['args'].get('bytes') == 4 and clear['args']['stream'] == stream and
                    id(owner) not in used_owners, 'Unknown or duplicate zero-count workspace clear')
            used_owners.add(id(owner))
            local, _ = index.work(owner)
            reduction = _one([e for e in local if e['cat'] == 'kernel' and _family(e['name']) == 'int64_sum'],
                             'Workspace clear has no original sum kernel')
            require(end(clear) <= reduction['ts'] + .01, 'Reduction workspace was cleared too late')

    adam = [e for e in events if e.get('cat') == 'kernel' and 'AdamFunctor' in e['name']]
    require(adam and min(e['ts'] for e in adam) >= end(copy) - .01,
            'Optimizer update began before zero-count completion')
    return dict(arm=arm, candidate=candidate, rank=rank,
                eligible_parameter_count=parameter_count if not candidate else expected_parameters,
                original_count_nonzero_calls=len(counts), kernel_families=dict(families),
                compute_kernel_calls=len(compute), original_allreduce_calls=1, scalar_readbacks=1,
                allowed_original_reduction_memsets=len(clears),
                cpu_statistic_span_ms=scope['dur'] / 1000,
                gpu_compute_work_ms=sum(e['dur'] for e in compute) / 1000,
                gpu_allreduce_ms=allreduce['dur'] / 1000,
                gpu_statistic_span_ms=(end(copy) - compute[0]['ts']) / 1000,
                optimizer_after_statistic=True,
                limitation='Profile proves execution and dependencies; timings are not unprofiled step savings.')
