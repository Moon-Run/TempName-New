"""Launch-correlated path checks for MA8 activation retention.

No profiler time is interpreted as an unprofiled speedup. MA8-v2 uses automatic
autograd events and launch correlation, without per-call Python instrumentation.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def end(event):
    return event['ts'] + event.get('dur', 0)


def contains(parent, child):
    return (parent['tid'] == child['tid'] and parent['ts'] <= child['ts'] + .01
            and end(child) <= end(parent) + .01)


class TraceIndex:
    def __init__(self, events):
        self.events = events
        self.runtime = defaultdict(list)
        self.device = defaultdict(list)
        self.external = defaultdict(list)
        self.cpu = defaultdict(list)
        for event in events:
            cat, args = event.get('cat'), event.get('args', {})
            if cat in ('cuda_runtime', 'cuda_driver'):
                self.runtime[event['tid']].append(event)
            elif cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                if args.get('correlation') is not None:
                    self.device[args['correlation']].append(event)
                if args.get('External id') is not None:
                    self.external[args['External id']].append(event)
            elif cat == 'cpu_op':
                self.cpu[event['tid']].append(event)
        for rows in self.runtime.values():
            rows.sort(key=lambda e: e['ts'])
        self.starts = {tid: [e['ts'] for e in rows] for tid, rows in self.runtime.items()}

    def work(self, call):
        rows, starts = self.runtime[call['tid']], self.starts.get(call['tid'], [])
        lo = bisect_left(starts, call['ts'] - .01)
        hi = bisect_right(starts, end(call) + .01)
        launches = {e['args']['correlation']: e for e in rows[lo:hi]
                    if contains(call, e) and e.get('args', {}).get('correlation') is not None}
        associated = {id(e): e for e in self.external.get(call.get('args', {}).get('External id'), [])}
        for correlation in launches:
            associated.update({id(e): e for e in self.device[correlation]})
        for event in associated.values():
            require(event.get('args', {}).get('correlation') in launches,
                    'Device event lacks a launch inside its owning scope')
        return list(associated.values()), launches

    def clears(self, call, work, launches, qkv_bias=False):
        """Permit only observed original library workspace initialization."""
        require(not any(e['cat'] == 'gpu_memcpy' for e in work), 'Unexpected GPU copy')
        counts = Counter()
        owners_seen = set()
        for event in work:
            if event['cat'] != 'gpu_memset':
                continue
            launch = launches[event['args']['correlation']]
            size = event['args'].get('bytes')
            if size == 4:
                names, kind, token = ('aten::mm', 'aten::matmul'), 'cublas_4byte', 'gemm'
            elif qkv_bias and size == 48:
                names, kind, token = ('aten::sum',), 'qkv_bias_48byte', 'reduce_kernel'
            else:
                raise ValueError('Unexpected GPU memset size or owner')
            owners = [e for e in self.cpu[call['tid']]
                      if e['name'] in names and contains(call, e) and contains(e, launch)]
            require(owners, 'Allowed-size memset is outside the original library operation')
            owner = min(owners, key=lambda e: e['dur'])
            require(id(owner) not in owners_seen, 'Repeated workspace clear inside one library operation')
            owners_seen.add(id(owner))
            consumers = [e for e in work if e['cat'] == 'kernel' and token in e['name'].lower()
                         and contains(owner, launches[e['args']['correlation']])]
            require(len(consumers) == 1 and
                    consumers[0]['args']['stream'] == event['args']['stream'] and
                    end(event) <= consumers[0]['ts'] + .01,
                    'Workspace clear is not followed by its original same-stream kernel')
            counts[kind] += 1
        return dict(counts)


def _arm(arm):
    require(arm in ('F4', 'F4-cache'), 'Unknown MA8 arm')
    return arm != 'F4'


ORIGINAL_COLUMN = 'LinearWithGradAccumulationAndAsyncCommunication'
CACHED_COLUMN = '_MA8CachedRawLinear'


def _raw_column_pairs(index, arm, mode):
    """Identify the fixed 32-layer, four-microbatch topology from raw Functions.

    Each microbatch has 32 alternating QKV/FC1 pairs and one original vocabulary
    head. Forward bias kernels independently verify the alternating roles. The
    autograd sequence pairs each forward with its actual backward; list order
    never pairs different calls. Historical analysis is an explicit mode, not
    an alternate path used when a candidate fails its identity checks.
    """
    require(mode in ('ma8-v2', 'historical-f4'), 'Unknown column trace mode')
    candidate = _arm(arm)
    require(mode != 'historical-f4' or not candidate, 'Historical mode only accepts original F4')
    forbidden = ('MA8ReferenceLinear', 'MA8RetainedActivationLinear', 'ma8.column.')
    require(not any(any(token in e.get('name', '') for token in forbidden) for e in index.events),
            'MA8-v2 rejects old per-call wrappers or annotations')
    automatic_cached_names = {CACHED_COLUMN, CACHED_COLUMN + 'Backward',
                              'autograd::engine::evaluate_function: ' + CACHED_COLUMN + 'Backward'}
    require(not any(e.get('cat') == 'cpu_op' and 'MA8' in e.get('name', '')
                    and e['name'] not in automatic_cached_names for e in index.events),
            'Unexpected MA8 wrapper in the raw column path')
    names = (ORIGINAL_COLUMN, CACHED_COLUMN)
    forwards = sorted([e for e in index.events if e.get('cat') == 'cpu_op' and e['name'] in names],
                      key=lambda e: e['ts'])
    backwards = [e for e in index.events if e.get('cat') == 'cpu_op'
                 and e['name'] in tuple(n + 'Backward' for n in names)]
    require(len(forwards) == len(backwards) == 260, 'Expected 256 raw columns and four original heads')
    expected = {ORIGINAL_COLUMN: 4, CACHED_COLUMN: 256} if candidate else {ORIGINAL_COLUMN: 260}
    require(dict(Counter(e['name'] for e in forwards)) == expected,
            'Wrong raw column arm or original head coverage')
    by_sequence = {}
    for call in backwards:
        seq = call.get('args', {}).get('Sequence number')
        require(isinstance(seq, int) and seq not in by_sequence, 'Missing or duplicate raw backward sequence')
        by_sequence[seq] = call
    pairs, heads, seen = [], [], set()
    for i, forward in enumerate(forwards):
        seq = forward.get('args', {}).get('Sequence number')
        require(isinstance(seq, int) and seq not in seen and seq in by_sequence,
                'Missing or duplicate raw forward/backward sequence')
        seen.add(seq)
        backward = by_sequence[seq]
        require(backward['name'] == forward['name'] + 'Backward', 'Raw forward/backward class differs')
        require(end(forward) <= backward['ts'] + .01, 'Raw backward precedes its forward')
        head = i % 65 == 64
        classname = CACHED_COLUMN if candidate and not head else ORIGINAL_COLUMN
        require(forward['name'] == classname, 'Raw column/head topology changed')
        if head:
            heads.append((forward, backward))
        else:
            pairs.append(('qkv' if i % 65 % 2 == 0 else 'fc1', forward, backward))
    return pairs, heads


def _head_records(index, heads):
    """Keep all four unmodified vocabulary heads outside the 256-call target."""
    require(len(heads) == 4, 'Expected four original vocabulary heads')
    for forward, backward in heads:
        fw, fl = index.work(forward)
        bw, bl = index.work(backward)
        index.clears(forward, fw, fl)
        index.clears(backward, bw, bl)
        fk = [e for e in fw if e['cat'] == 'kernel']
        bk = [e for e in bw if e['cat'] == 'kernel']
        fag = [e for e in fk if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        fm = [e for e in fk if 'gemm' in e['name'].lower()]
        bag = [e for e in bk if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        rs = [e for e in bk if 'nccl' in e['name'].lower() and 'ReduceScatter' in e['name']]
        mm = sorted([e for e in bk if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        require(len(fk) == 2 and len(fag) == len(fm) == 1 and
                len(bk) == 4 and len(bag) == len(rs) == 1 and len(mm) == 2,
                'Original vocabulary head path changed')
        require(end(fag[0]) <= fm[0]['ts'] + .01 and
                end(bag[0]) <= rs[0]['ts'] + .01 and end(bag[0]) <= mm[1]['ts'] + .01 and
                end(mm[0]) <= rs[0]['ts'] + .01 and end(mm[0]) <= mm[1]['ts'] + .01,
                'Original vocabulary head dependencies changed')
        require(fm[0]['args']['stream'] == mm[0]['args']['stream'] == mm[1]['args']['stream'] and
                fag[0]['args']['stream'] != fm[0]['args']['stream'] and
                bag[0]['args']['stream'] == rs[0]['args']['stream'] != mm[0]['args']['stream'],
                'Original vocabulary head streams changed')
    return dict(calls=4, original_autograd_class=ORIGINAL_COLUMN, forward_allgather_kernels=4,
                backward_allgather_kernels=4, reduce_scatter_kernels=4,
                excluded_from_target_column_counts=True)


def _column_records(index, pairs, candidate):
    records = defaultdict(list)
    used = set()
    for role, forward, backward in pairs:
        fw, fl = index.work(forward)
        bw, bl = index.work(backward)
        require(not used.intersection(map(id, fw + bw)), 'Column device work attributed twice')
        used.update(map(id, fw + bw))
        fclears = index.clears(forward, fw, fl)
        bclears = index.clears(backward, bw, bl, qkv_bias=role == 'qkv')
        fk = [e for e in fw if e['cat'] == 'kernel']
        bk = [e for e in bw if e['cat'] == 'kernel']
        fag = [e for e in fk if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        fm = [e for e in fk if 'gemm' in e['name'].lower()]
        adds = [e for e in fk if 'elementwise' in e['name'].lower() and 'add' in e['name'].lower()]
        ag = [e for e in bk if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        rs = [e for e in bk if 'nccl' in e['name'].lower() and 'ReduceScatter' in e['name']]
        mm = sorted([e for e in bk if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        bias = [e for e in bk if 'reduce_kernel' in e['name']]
        require(len(fag) == len(fm) == 1 and len(adds) == int(role == 'qkv') and
                len(fk) == 2 + int(role == 'qkv'), 'Column forward path changed')
        require(end(fag[0]) <= fm[0]['ts'] + .01, 'Forward GEMM consumed an unfinished gather')
        require(fag[0]['args']['stream'] != fm[0]['args']['stream'], 'Forward communication stream changed')
        if adds:
            require(adds[0]['args']['stream'] == fm[0]['args']['stream'] and
                    end(fm[0]) <= adds[0]['ts'] + .01, 'Forward bias add order or stream changed')
        require(len(ag) == int(not candidate) and len(rs) == 1 and len(mm) == 2 and
                len(bias) == int(role == 'qkv') and len(bk) == 3 + int(not candidate) + int(role == 'qkv'),
                'Column backward AllGather/ReduceScatter/GEMM/bias coverage changed')
        dgrad, wgrad = mm
        reduce_scatter = rs[0]
        require(dgrad['args']['stream'] == wgrad['args']['stream'] and
                dgrad['args']['stream'] == fm[0]['args']['stream'] and
                reduce_scatter['args']['stream'] != dgrad['args']['stream'], 'Column streams changed')
        require(end(dgrad) <= wgrad['ts'] + .01 and end(dgrad) <= reduce_scatter['ts'] + .01,
                'Column GEMM order or dgrad-to-ReduceScatter dependency changed')
        require(end(bl[dgrad['args']['correlation']]) <= bl[reduce_scatter['args']['correlation']]['ts'] + .01 and
                end(bl[reduce_scatter['args']['correlation']]) <= bl[wgrad['args']['correlation']]['ts'] + .01,
                'Original dgrad, ReduceScatter, wgrad submission order changed')
        if ag:
            require(end(ag[0]) <= wgrad['ts'] + .01 and end(ag[0]) <= reduce_scatter['ts'] + .01,
                    'Baseline consumed unfinished gathered activation')
        if bias:
            require(bias[0]['args']['stream'] == wgrad['args']['stream'] and
                    end(wgrad) <= bias[0]['ts'] + .01, 'Column bias reduction moved')
        metrics = dict(forward_cpu_us=forward['dur'], backward_cpu_us=backward['dur'],
                       forward_allgather_us=fag[0]['dur'], dgrad_us=dgrad['dur'], wgrad_us=wgrad['dur'],
                       backward_allgather_us=ag[0]['dur'] if ag else 0.,
                       allgather_tail_after_dgrad_us=max(0., end(ag[0]) - end(dgrad)) if ag else 0.,
                       allgather_dgrad_overlap_us=max(0., min(end(ag[0]), end(dgrad)) - max(ag[0]['ts'], dgrad['ts'])) if ag else 0.,
                       dgrad_end_to_reduce_scatter_us=reduce_scatter['ts'] - end(dgrad),
                       allgather_end_to_reduce_scatter_us=reduce_scatter['ts'] - end(ag[0]) if ag else 0.,
                       gpu_chain_us=max(end(reduce_scatter), end(wgrad)) - min(ag[0]['ts'] if ag else dgrad['ts'], dgrad['ts']))
        records[role].append(dict(metrics=metrics, clears=dict(Counter(fclears) + Counter(bclears))))
    require(set(records) == {'qkv', 'fc1'} and all(len(v) == 128 for v in records.values()),
            'Expected 128 QKV and 128 FC1 calls per rank')
    result = {}
    for role, rows in records.items():
        clears = Counter()
        for row in rows:
            clears.update(row['clears'])
        result[role] = dict(calls=len(rows), forward_allgather_kernels=128,
                            backward_allgather_kernels=0 if candidate else 128,
                            dgrad_reduce_scatter_kernels=128, dgrad_kernels=128, wgrad_kernels=128,
                            bias_gradient_kernels=128 if role == 'qkv' else 0,
                            allowed_workspace_memsets=dict(clears),
                            median_us={key: statistics.median(r['metrics'][key] for r in rows) for key in rows[0]['metrics']},
                            sum_ms={key: sum(r['metrics'][key] for r in rows) / 1000 for key in rows[0]['metrics']})
    return dict(roles=result, column_calls=256, retained_forward_allgathers=256,
                backward_allgathers=0 if candidate else 256, retained_reduce_scatters=256,
                no_gpu_copies=True, original_gemm_order_and_streams=True)


def trace_activation_cache_records(events, arm):
    """Validate MA8-v2 raw automatic profiler events without per-call markers."""
    index = TraceIndex(events)
    pairs, heads = _raw_column_pairs(index, arm, 'ma8-v2')
    result = _column_records(index, pairs, _arm(arm))
    result.update(attribution='raw_autograd_sequence_and_fixed_topology_v2',
                  original_vocabulary_heads=_head_records(index, heads),
                  per_call_wrappers_and_annotations=False)
    return result


def historical_column_records(events):
    """Diagnostic only: frozen 32-layer, four-microbatch F4 trace without MA8 markers.

    This explicit diagnostic mode shares raw topology validation with MA8-v2;
    it is never used to accept a candidate with missing raw Function events.
    """
    index = TraceIndex(events)
    pairs, heads = _raw_column_pairs(index, 'F4', 'historical-f4')
    _head_records(index, heads)
    return _column_records(index, pairs, False)


def trace_row_backward_records(events, arm):
    """Confirm unchanged CR3 row backward and classify its cuBLAS clears narrowly."""
    _arm(arm)
    index = TraceIndex(events)
    forwards = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] == 'ConsumerGemmRSFunction']
    backwards = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] == 'ConsumerGemmRSFunctionBackward']
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e['name'] in ('scope.attn.candidate', 'scope.mlp.candidate')]
    require(len(forwards) == len(backwards) == 256, 'Missing row projection autograd coverage')
    roles = {}
    for call in forwards:
        matching = [e for e in scopes if contains(e, call)]
        require(len(matching) == 1, 'Missing or duplicate row forward scope')
        seq = call['args']['Sequence number']
        require(seq not in roles, 'Duplicate row forward sequence')
        roles[seq] = matching[0]['name'].split('.')[1]
    result = defaultdict(list)
    used, seen = set(), set()
    for call in backwards:
        seq = call['args']['Sequence number']
        require(seq in roles and seq not in seen, 'Unmatched row backward')
        seen.add(seq)
        work, launches = index.work(call)
        require(not used.intersection(map(id, work)), 'Row GPU work attributed twice')
        used.update(map(id, work))
        clears = index.clears(call, work, launches)
        kernels = [e for e in work if e['cat'] == 'kernel']
        bias = [e for e in kernels if 'reduce_kernel' in e['name']]
        ag = [e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        mm = sorted([e for e in kernels if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        require(len(kernels) == 4 and len(bias) == len(ag) == 1 and len(mm) == 2,
                'Unchanged row backward kernel coverage differs')
        bias, ag = bias[0], ag[0]
        dgrad, wgrad = mm
        require(bias['args']['stream'] == dgrad['args']['stream'] == wgrad['args']['stream'] and
                ag['args']['stream'] != dgrad['args']['stream'], 'Row backward stream changed')
        require(end(bias) <= ag['ts'] + .01 and end(ag) <= dgrad['ts'] + .01 and
                end(dgrad) <= wgrad['ts'] + .01, 'Row bias, gather, dgrad, wgrad order changed')
        require(end(launches[bias['args']['correlation']]) <= launches[ag['args']['correlation']]['ts'] + .01,
                'Row gather moved ahead of bias')
        result[roles[seq]].append(dict(backward_cpu_us=call['dur'], bias_us=bias['dur'],
            allgather_us=ag['dur'], dgrad_us=dgrad['dur'], wgrad_us=wgrad['dur'],
            cublas_4byte_memsets=clears.get('cublas_4byte', 0)))
    require(set(result) == {'attn', 'mlp'} and all(len(rows) == 128 for rows in result.values()),
            'Expected 128 Attention and 128 MLP row backwards')
    return {role: dict(backward_calls=128, bias_kernels=128, allgather_kernels=128,
                      dgrad_kernels=128, wgrad_kernels=128, gather_schedule='baseline',
                      allowed_cublas_4byte_memsets=sum(r['cublas_4byte_memsets'] for r in rows),
                      no_gpu_copies=True, original_order_and_streams=True,
                      median_us={key: statistics.median(r[key] for r in rows)
                                 for key in rows[0] if key != 'cublas_4byte_memsets'})
            for role, rows in result.items()}
