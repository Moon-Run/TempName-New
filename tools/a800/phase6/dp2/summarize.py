"""Validate all eight ranks and report balanced, paired full optimizer steps."""
import gzip
import json
import math
from pathlib import Path
import random
import statistics

from protocol import POLICIES, orders, replica_checks, sha, write_json
from routing import module_routes, codec_placement, selection_entry, logical_bytes
from decoder_protocol import expected_decoder_counts


def check_window(path, cfg, spec, policy, mode):
    assert (path/'exit-code.txt').read_text().strip() == '0'
    for node in range(2):
        assert (path/f'node{node}-exit-code.txt').read_text().strip() == '0'
        meta = json.loads((path/f'node{node}.json').read_text())
        assert meta['job_id'] == spec['jobs'][node] and meta['host'] == spec['nodes'][node]
    model = dict(cfg['common'], **cfg['cases'][0])
    m, n = model['sequence']*model['micro_batch'], model['hidden']
    microsteps = model['global_batch']//(model['micro_batch']*2)
    calls = 2*model['layers']*microsteps
    plan = json.loads((Path(__file__).parent/'selection-plan.json').read_text())
    rows = [json.loads((path/f'rank{r}.json').read_text()) for r in range(8)]
    replica_checks(rows)
    for rank, r in enumerate(rows):
        assert r['model'] == model and r['world_size'] == 8
        assert r['case'] == model['name'] and r['policy'] == policy and r['mode'] == mode
        assert r['completed'] and r['passed'] and r['parameters_updated']
        assert r['job_id'] == str(spec['jobs'][rank//4])
        assert r['host'] == spec['nodes'][rank//4]
        assert r['fallback_calls'] == r['skips'] == 0
        assert not r['sampling'] and r['profiling'] == (mode == 'profile')
        assert r['timing_protocol'] == 'full-step-one-clear-v2'
        warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else cfg['warmup_steps']
        count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else cfg['timed_steps']
        assert r['warmup_steps'] == warmup
        if mode != 'profile':
            assert r['timed_steps'] == count and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0
        assert len(r['losses']) == len(r['grad_norms']) == count
        assert all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']])
        assert r['tokens_per_optimizer_step'] == model['sequence']*model['global_batch']
        assert len(r['target_modules']) == 2*model['layers']
        assert len(r['warmup_calls']) == 2*model['layers']
        assert all(v == warmup*microsteps for v in r['warmup_calls'].values())
        assert r['target_calls'] == r['warmup_calls']
        assert r['replaced_modules'] == (0 if policy == 'native' else model['layers'] if policy == 'native_taco' else model['layers']*2)
        assert {(v['M'],v['N'],v['K_global']) for v in r['observed_shapes'].values()} == {(m,n,n),(m,n,model['ffn'])}
        assert r['module_policies'] == module_routes(r['observed_shapes'],policy,model['layers'],model)
        assert r['codec_placements'] == ({} if policy == 'native' else {
            name: (codec_placement(policy) if name.endswith('.mlp.linear_fc2') else
                   'native' if policy == 'native_taco' else 'bf16') for name in r['target_modules']})
        if policy in ('native','original'):
            assert not r['selection_checks']
        else:
            mask = selection_entry(plan,policy,m,n,model['ffn'])['mask'] if policy.endswith('_selective') else None
            import hashlib
            digest = hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest() if mask is not None else None
            wire,bf16 = logical_bytes(mask,rank%4,m,n,4)
            assert len(r['selection_checks']) == model['layers']
            for name,c in r['selection_checks'].items():
                assert name.endswith('.mlp.linear_fc2')
                assert (c['mask_sha256'],c['wire_bytes'],c['bf16_wire_bytes']) == (digest,wire,bf16)
        if policy not in ('native','native_taco'):
            for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                expected = (Path(spec['build'])/policy/lib).resolve()
                assert r[lib]['path'] == str(expected) and r[lib]['sha256'] == sha(expected)
        elif policy == 'native_taco':
            assert not r['flux_library_loaded'] and r['native_linear_forward_preserved']
        if mode == 'smoke':
            assert len(r['forward_checks']) == (0 if policy == 'native' else model['layers'] if policy == 'native_taco' else 2*model['layers'])
            assert all(c.get('passed', True) for c in r['forward_checks'].values())
        if mode == 'profile':
            with gzip.open(path/f'rank{rank}-profile.json.gz','rt') as f:
                events = json.load(f)['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel']
            gemms = [e for e in kernels if 'flux_bf16' in e['name']]
            assert len(gemms) == (0 if policy in ('native','native_taco') else calls)
            assert all('64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic' in e['name'] for e in gemms)
            expected = expected_decoder_counts(cfg,plan,model,policy,rank%4,calls)
            actual = dict(legacy=sum(any(s in e['name'] for s in ('taco_decode_ring_kernel','taco_decode_tile4_kernel')) for e in kernels),
                          bf16=sum('taco_reduce_bf16_compact4_kernel' in e['name'] for e in kernels),
                          compact=sum('taco_decode_compact4_kernel' in e['name'] for e in kernels))
            assert actual == expected, (policy,rank,actual,expected)
            assert not any(any(s in e['name'] for s in ('taco_decode_pair4_kernel','taco_decode_flat4_kernel','taco_encode_scatter_kernel')) for e in kernels)
            if policy.endswith('_selective'):
                assert sum('CudaIpcBarrierAllKernel' in e['name'] for e in kernels) == calls
            if policy == 'native_taco':
                for name in ('encode','exchange','decode_reduce'):
                    assert sum(e.get('cat') == 'user_annotation' and e.get('ph') == 'X' and e.get('name') == 'megatron_taco.'+name for e in events) == calls//2
            # DP gradient collectives must be present in a full step, for every policy.
            assert any('nccl' in e['name'].lower() and 'allreduce' in e['name'].lower().replace('_','') for e in kernels)
    return rows


def quantile(values, p):
    values = sorted(values)
    index = (len(values)-1)*p
    low = int(index)
    return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(index-low)


def paired(candidate, baseline):
    logs = [math.log(a/b) for a,b in zip(candidate,baseline)]
    rng = random.Random(20261004)
    samples = [100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000)]
    return dict(reduction_percent=100*(1-math.exp(statistics.mean(logs))),
                ci95=[quantile(samples,.025),quantile(samples,.975)])


def analyze(directory, spec, repetition):
    cfg = json.loads((directory/'config.json').read_text())
    rows, signatures = [], {}
    window = 0
    for block, order in enumerate(orders(repetition)):
        for policy in order:
            path = directory/f'round-{repetition}'/f'window-{window:02d}-{policy}'
            ranks = check_window(path,cfg,spec,policy,'timing')
            for rank,r in enumerate(ranks):
                assert r['block'] == block and r['window'] == window
                sig = tuple(r[k] for k in ('initial_parameters_sha256','initial_rng_sha256',
                                           'tokens_sha256','global_tokens_sha256','gpu_uuid','host'))
                if rank not in signatures:
                    signatures[rank] = sig
                assert signatures[rank] == sig, (rank,policy,'initialization changed')
            latency = max(r['ms_per_step'] for r in ranks)
            rows.append(dict(block=block,window=window,policy=policy,ms_per_step=latency,
                             tokens_per_second=ranks[0]['tokens_per_optimizer_step']*1000/latency,
                             initial_loss=ranks[0]['initial_loss'],last_loss=ranks[0]['losses'][-1],
                             grad_norm=ranks[0]['grad_norms'][-1]))
            window += 1
    assert len(list((directory/f'round-{repetition}').glob('window-*'))) == window == 36
    summary = []
    points = {p:[r['ms_per_step'] for r in rows if r['policy'] == p] for p in POLICIES}
    for policy in POLICIES:
        values = points[policy]
        summary.append(dict(policy=policy, median_ms_per_step=statistics.median(values),
                            cv_percent=statistics.stdev(values)/statistics.mean(values)*100,
                            comparisons={b:paired(values,points[b]) for b in POLICIES[:4]}))
    result = dict(completed=True,round=repetition,windows=rows,summary=summary,
                  initialization_signatures=signatures,rank_records=36*8,
                  interval_unit='6 paired blocks; bootstrap 10000; maximum elapsed time over all eight ranks',
                  config=cfg)
    write_json(directory/f'analysis-{repetition}.json', result)
    lines = [f'# TP4/DP2 {cfg["cases"][0]["name"]}: round {repetition}', '',
             '| Policy | ms/step | vs native_taco % [95% CI] | vs taco_fused % [95% CI] | gap vs native % | gap vs original % |',
             '| --- | ---: | --- | --- | ---: | ---: |']
    for r in summary:
        parts = [r['policy'], f"{r['median_ms_per_step']:.3f}"]
        for b in ('native_taco','taco_fused'):
            v = r['comparisons'][b]
            parts.append(f"{v['reduction_percent']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]")
        parts += [f"{-r['comparisons'][b]['reduction_percent']:+.3f}" for b in ('native','original')]
        lines.append('| '+' | '.join(parts)+' |')
    (directory/f'report-{repetition}.md').write_text('\n'.join(lines)+'\n')
    if repetition == 2:
        first = json.loads((directory/'analysis-1.json').read_text())
        assert first['initialization_signatures'] == json.loads(json.dumps(signatures))
        candidates = {}
        for p in POLICIES[4:]:
            rounds = [next(r for r in a['summary'] if r['policy'] == p) for a in (first,result)]
            passed = all(r['comparisons'][b]['reduction_percent'] >= 4 and r['comparisons'][b]['ci95'][0] > 0
                         for r in rounds for b in ('native_taco','taco_fused'))
            candidates[p] = dict(rounds=rounds,meets_target_both_rounds=passed)
        write_json(directory/'acceptance.json',dict(completed=True,target_percent=4,
            candidates=candidates,target_achieved=any(c['meets_target_both_rounds'] for c in candidates.values()),
            formal_windows=72,rank_records=576,timed_optimizer_steps=1440))
    return result
