"""Private H100 audit copies for target PyTorch 2.8 NCCL and cuBLAS behavior.

At the frozen target commit 5228986c39, ProcessGroupNCCL.cpp:3186 launches
asyncOp=false collectives on the current CUDA stream. Async collectives
still use a separate NCCL stream. The frozen Megatron paths use synchronous
forward/head/row AllGather and statistic AllReduce, but asynchronous backward
activation AllGather and dgrad ReduceScatter. Keep those distinctions strict.

Change those four synchronous stream predicates and explicitly check that the
control column's asynchronous backward gather shares the ReduceScatter stream.
Also account for the observed row-dgrad cuBLAS split-K reduction within its
original aten::mm owner, including its duration and completion dependency.
Launch ownership, autograd pairing, GPU/CPU dependency order, scalar readback,
and asynchronous stream checks remain strict. H100 modules and input trace
events are never modified.
"""
from pathlib import Path
from types import ModuleType


def _is_splitk(event):
    return (event['cat'] == 'kernel'
            and event['name'].startswith('void cublasLt::splitKreduce_kernel<')
            and '__nv_bfloat16' in event['name'])


def _row_splitk(index, call, launches, gemms, splits):
    """Account for the observed cuBLAS BF16 dgrad reduction, never bias work."""
    from ma8_profile import contains, end, require

    tails = []
    durations = []
    claimed = set()
    for gemm in gemms:
        launch = launches[gemm['args']['correlation']]
        owners = [event for event in index.cpu[call['tid']]
                  if event['name'] == 'aten::mm' and contains(call, event) and contains(event, launch)]
        require(len(owners) == 1, 'Row GEMM lacks its original aten::mm launch owner')
        owner = owners[0]
        work, _ = index.work(owner)
        local = [event for event in work if event['cat'] == 'kernel']
        reductions = [event for event in local if _is_splitk(event)]
        require(len(reductions) <= 1 and len(local) == 1 + len(reductions)
                and any(event is gemm for event in local), 'Row GEMM has unregistered cuBLAS work')
        tail = gemm
        if reductions:
            tail = reductions[0]
            require(not tails, 'Unexpected row wgrad split-K implementation')
            require(id(tail) not in claimed and any(tail is event for event in splits),
                    'Row split-K reduction attributed twice or outside backward')
            require(tail['args']['stream'] == gemm['args']['stream']
                    and tail['args'].get('device') == gemm['args'].get('device')
                    and end(gemm) <= tail['ts'] + .01
                    and end(launch) <= launches[tail['args']['correlation']]['ts'] + .01,
                    'Row split-K reduction stream or dependency changed')
            claimed.add(id(tail))
        tails.append(tail)
        durations.append(sum(event['dur'] for event in local))
    require(claimed == {id(event) for event in splits}, 'Unowned row split-K reduction')
    return dict(dgrad_end=end(tails[0]), dgrad_us=durations[0], wgrad_us=durations[1],
                splitk_count=len(splits))


def _adapt(module, replacements):
    source = Path(module.__file__).read_text()
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('PyTorch 2.8 NCCL profile source contract changed: ' + old)
        source = source.replace(old, new)
    adapted = ModuleType('rtx5090_pytorch28_' + module.__name__)
    adapted.__file__ = module.__file__
    exec(compile(source, adapted.__file__, 'exec'), adapted.__dict__)
    return adapted


def pytorch28_profiles(ma8, ma9):
    column = _adapt(ma8, [
        ("fag[0]['args']['stream'] != fm[0]['args']['stream'] and",
         "fag[0]['args']['stream'] == fm[0]['args']['stream'] and"),
        ("require(fag[0]['args']['stream'] != fm[0]['args']['stream'],",
         "require(fag[0]['args']['stream'] == fm[0]['args']['stream'],"),
        ("ag['args']['stream'] != dgrad['args']['stream'], 'Row backward stream changed'",
         "ag['args']['stream'] == dgrad['args']['stream'], 'Row backward stream changed'"),
        ("        if ag:\n            require(end(ag[0])",
         "        if ag:\n"
         "            require(ag[0]['args']['stream'] == reduce_scatter['args']['stream'] "
         "!= dgrad['args']['stream'], 'Asynchronous backward gather stream changed')\n"
         "            require(end(ag[0])"),
        ('original_gemm_order_and_streams=True',
         "original_gemm_order=True, collective_stream_policy='pytorch28-sync-current-async-nccl'"),
        ('original_order_and_streams=True',
         "original_order=True, collective_stream_policy='pytorch28-sync-current-async-nccl'"),
        ("        kernels = [e for e in work if e['cat'] == 'kernel']\n"
         "        bias = [e for e in kernels if 'reduce_kernel' in e['name']]",
         "        kernels = [e for e in work if e['cat'] == 'kernel']\n"
         "        splits = [e for e in kernels if _is_splitk(e)]\n"
         "        bias = [e for e in kernels if 'reduce_kernel' in e['name'] and not _is_splitk(e)]"),
        ('require(len(kernels) == 4 and len(bias) == len(ag) == 1 and len(mm) == 2,',
         'require(len(kernels) == 4 + len(splits) and len(bias) == len(ag) == 1 and len(mm) == 2,'),
        ('        bias, ag = bias[0], ag[0]\n        dgrad, wgrad = mm',
         '        bias, ag = bias[0], ag[0]\n        dgrad, wgrad = mm\n'
         '        gemm_work = _row_splitk(index, call, launches, mm, splits)'),
        ("end(dgrad) <= wgrad['ts'] + .01, 'Row bias, gather, dgrad, wgrad order changed'",
         "gemm_work['dgrad_end'] <= wgrad['ts'] + .01, 'Row bias, gather, dgrad, wgrad order changed'"),
        ("allgather_us=ag['dur'], dgrad_us=dgrad['dur'], wgrad_us=wgrad['dur'],",
         "allgather_us=ag['dur'], dgrad_us=gemm_work['dgrad_us'], wgrad_us=gemm_work['wgrad_us'],\n"
         "            splitk_reduction_kernels=gemm_work['splitk_count'],"),
        ("dgrad_kernels=128, wgrad_kernels=128, gather_schedule='baseline',",
         "dgrad_kernels=128, wgrad_kernels=128, gather_schedule='baseline',\n"
         "                      splitk_reduction_kernels=sum(r['splitk_reduction_kernels'] for r in rows),"),
        ("if key != 'cublas_4byte_memsets'", "if key not in ('cublas_4byte_memsets', 'splitk_reduction_kernels')"),
    ])
    column._is_splitk = _is_splitk
    column._row_splitk = _row_splitk
    zero = _adapt(ma9, [
        ("allreduce['args']['stream'] != stream and end(compute[-1])",
         "allreduce['args']['stream'] == stream and end(compute[-1])"),
    ])
    return column, zero
