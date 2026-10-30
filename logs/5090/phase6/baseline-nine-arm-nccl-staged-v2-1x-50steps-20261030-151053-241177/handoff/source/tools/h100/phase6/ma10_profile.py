"""Correlate bounded gradient commits with the original AccumulateGrad work.

Markers exist only in candidate profile runs. Original F4 has no MA10 hooks,
markers or kernels. Durations diagnose work, never estimate unprofiled savings.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict

ARMS = ('F4', 'F4-gacc')
ACCUMULATE = 'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad'
FLUSH = 'ma10.gradient_batch.flush:'
BACKWARD = 'ma10.microbatch.backward'
KERNEL = 'ma10_gradient_batch_kernel'


def require(value, message):
    if not value:
        raise ValueError(message)


def end(event):
    return event['ts'] + event.get('dur', 0)


def within(parent, child, same_thread=True):
    return ((not same_thread or parent['tid'] == child['tid']) and
            parent['ts'] <= child['ts'] + .01 and end(child) <= end(parent) + .01)


class Index:
    def __init__(self, events):
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
        lo, hi = bisect_left(starts, scope['ts']-.01), bisect_right(starts, end(scope)+.01)
        ids = {e.get('args', {}).get('correlation') for e in rows[lo:hi] if within(scope, e)}
        ids.discard(None)
        return [e for correlation in ids for e in self.device[correlation]]


def _owners(scopes, events):
    """Associate child CPU ops to nonoverlapping scopes on the same thread."""
    by_thread = defaultdict(list)
    for scope in scopes:
        by_thread[scope['tid']].append(scope)
    for rows in by_thread.values():
        rows.sort(key=lambda e: e['ts'])
    starts = {tid: [e['ts'] for e in rows] for tid, rows in by_thread.items()}
    result = defaultdict(list)
    for event in events:
        rows = by_thread[event['tid']]
        position = bisect_right(starts.get(event['tid'], []), event['ts']+.01)-1
        if position >= 0 and within(rows[position], event):
            result[id(rows[position])].append(event)
    return result


def trace_gradient_batching_records(events, arm):
    require(arm in ARMS, 'Unknown MA10 arm')
    require(not any(e.get('cat') in ('cpu_op', 'user_annotation') and
                    (e['name'].startswith(('MA8', 'ma7.', 'ma8.', 'ma9.')))
                    for e in events), 'An earlier mechanism contaminated CR3')
    index = Index(events)
    cpu = [e for e in events if e.get('cat') == 'cpu_op']
    leaves = [e for e in cpu if e['name'] == ACCUMULATE]
    require(len(leaves) == 4*389, 'Expected all 389 parameter hooks in four microbatches')
    additions = _owners(leaves, [e for e in cpu if e['name'] == 'aten::add_'])
    annotations = [e for e in events if e.get('cat') == 'user_annotation']
    flushes = sorted([e for e in annotations if e['name'].startswith(FLUSH)], key=lambda e:e['ts'])
    boundaries = sorted([e for e in annotations if e['name'] == BACKWARD], key=lambda e:e['ts'])
    native = [e for e in events if e.get('cat') == 'kernel' and KERNEL in e['name']]
    work, records = [], []
    if arm == 'F4':
        require(not flushes and not boundaries and not native,
                'Original CR3 must have no MA10 wrappers or kernels')
        for leaf in leaves:
            calls = additions[id(leaf)]
            require(len(calls) == 1, 'Original hook lost its single main_grad.add_')
            kernels = index.work(calls[0])
            require(len(kernels) == 1 and kernels[0]['cat'] == 'kernel' and
                    'CUDAFunctor_add<float>' in kernels[0]['name'],
                    'Original mixed-type gradient addition changed')
            work += kernels
        tensor_commits = len(work)
    else:
        require(not any(additions.values()), 'Candidate fell back to per-parameter add_')
        require(len(boundaries) == 4 and all(end(a) <= b['ts']+.01
                for a,b in zip(boundaries,boundaries[1:])), 'Missing or overlapping microbatch boundaries')
        require(4 <= len(flushes) < len(leaves), 'Gradient batching did not reduce submissions')
        tensor_commits = 0
        for flush in flushes:
            suffix = flush['name'][len(FLUSH):]
            require(suffix.isdigit() and 1 <= int(suffix) <= 32, 'Invalid flush tensor count')
            count = int(suffix)
            local = index.work(flush)
            require(len(local) == 1 and local[0]['cat'] == 'kernel' and KERNEL in local[0]['name'],
                    'Flush must own exactly one native kernel, without descriptor copies or memset')
            work += local
            tensor_commits += count
            records.append(dict(scope=flush, kernel=local[0], tensors=count))
        require(len(work) == len(native) and {id(e) for e in work} == {id(e) for e in native},
                'Native gradient kernel lacks a unique owning flush')
        require(all(end(a['kernel']) <= b['kernel']['ts']+.01
                    for a,b in zip(records,records[1:])),
                'GPU commit order differs from owning flush order')
        require(tensor_commits == len(leaves), 'Missing or duplicate committed gradients')
        owned = set()
        for boundary in boundaries:
            local_leaves = [e for e in leaves if within(boundary,e,False)]
            local_records = [r for r in records if within(boundary,r['scope'],False)]
            require(len(local_leaves) == 389 and sum(r['tensors'] for r in local_records) == 389,
                    'Gradient commit crossed a microbatch boundary')
            require(local_records, 'Microbatch did not flush gradients')
            owned.update(id(r['scope']) for r in local_records)
        require(len(owned) == len(flushes), 'Flush occurred outside microbatch backward')

    require(len({id(e) for e in work}) == len(work), 'Gradient kernel attributed twice')
    work.sort(key=lambda e:e['ts'])
    streams = {e.get('args',{}).get('stream') for e in work}
    require(len(streams) == 1 and None not in streams, 'Gradient commit uses an unexpected auxiliary stream')
    require(all(end(a) <= b['ts']+.01 for a,b in zip(work,work[1:])), 'Gradient writes overlap')
    # The optimizer is unchanged: its count_nonzero work and Adam must follow
    # all gradient writes. This is a GPU dependency check, not CPU clock mixing.
    counts = [e for e in cpu if e['name'] == 'aten::count_nonzero']
    require(len(counts) in (2*194,2*389), 'Original optimizer zero statistics changed')
    count_work = [k for e in counts for k in index.work(e) if k['cat']=='kernel']
    adam = [e for e in events if e.get('cat')=='kernel' and 'AdamFunctor' in e['name']]
    require(count_work and adam and min(e['ts'] for e in count_work+adam) >= end(work[-1])-.01,
            'Optimizer consumed unfinished gradient commits')
    require(all(e.get('args',{}).get('stream') in streams for e in count_work+adam),
            'Gradient commits must use the original optimizer compute stream')
    return dict(arm=arm, parameter_count=389, microbatches=4,
                accumulate_grad_calls=len(leaves), tensor_commits=tensor_commits,
                original_add_calls=sum(map(len,additions.values())),
                gradient_kernel_calls=len(work), flush_calls=len(flushes),
                flush_tensor_counts=[r['tensors'] for r in records],
                compute_stream=next(iter(streams)), original_zero_statistic_preserved=True,
                optimizer_after_commits=True,
                cpu_hook_work_ms=sum(e['dur'] for e in leaves)/1000,
                gpu_gradient_commit_work_ms=sum(e['dur'] for e in work)/1000,
                limitation='Profile verifies execution; summed work is not unprofiled step savings.')
