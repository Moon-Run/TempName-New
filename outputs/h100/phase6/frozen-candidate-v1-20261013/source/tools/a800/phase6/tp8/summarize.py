"""Eight-rank BF16 TP8 verification and paired full-step measurements."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random
import re
import statistics

from protocol import POLICIES,orders,sha,validate_topology,write_json


def check_window(path,c,spec,policy,mode):
    assert (path/'exit-code.txt').read_text().strip()=='0'
    for node in range(2):assert (path/f'node{node}-exit-code.txt').read_text().strip()=='0'
    rows=[json.loads((path/f'rank{rank}.json').read_text()) for rank in range(8)]
    validate_topology(rows)
    expected_shapes={(8192,2048,2048),(8192,2048,8192)}
    root=Path(__file__).resolve().parent.parent
    for rank,r in enumerate(rows):
        assert r['completed'] and r['passed'] and r['parameters_updated']
        assert r['world_size']==8 and r['model']==c and r['policy']==policy and r['mode']==mode
        assert r['job_id']==str(spec['jobs'][rank//4]) and r['host']==spec['nodes'][rank//4]
        assert r['skips']==r['fallback_calls']==0 and not r['sampling']
        assert r['profiling']==(mode=='profile') and r['timing_protocol']=='full-step-one-clear-v2'
        warmup=2 if mode=='smoke' else 10;count=3 if mode=='smoke' else 1 if mode=='profile' else 20
        assert r['warmup_steps']==warmup and len(r['losses'])==len(r['grad_norms'])==count
        if mode!='profile':
            assert r['timed_steps']==count and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0
        assert r['tokens_per_optimizer_step']==8192
        assert r['padded_vocab_size']==8704
        assert len(r['target_modules'])==len(r['warmup_calls'])==24
        assert all(v==warmup for v in r['warmup_calls'].values())
        assert r['replaced_modules']==(0 if policy=='native' else 24)
        assert {(x['M'],x['N'],x['K_global']) for x in r['observed_shapes'].values()}==expected_shapes
        assert all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']])
        if policy!='native':
            for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                expected=(Path(spec['build'])/policy/lib).resolve()
                assert r[lib]['path']==str(expected) and r[lib]['sha256']==sha(expected)
            if mode=='smoke':
                assert len(r['numerical_checks'])==24 and all(x['passed'] for x in r['numerical_checks'].values())
        if mode=='profile':
            with gzip.open(path/f'rank{rank}-profile.json.gz','rt') as f:events=json.load(f)['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel']
            assert not any('taco_' in e['name'] for e in kernels)
            gemms=[e for e in kernels if 'flux_bf16' in e['name']]
            assert len(gemms)==(0 if policy=='native' else 48),(policy,rank,len(gemms))
            if gemms:
                expected=Counter()
                for k in (1024,4096):
                    text=(root/'mapping'/f'mapping-{policy}-k{k}-node{rank//4}.log').read_text()
                    grid=int(re.search(rf'PASS rank={rank%4} .*grid=(\d+)',text).group(1));expected[grid]+=24
                assert Counter(e['args']['grid'][0] for e in gemms)==expected
                assert all('64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic' in e['name'] for e in gemms)
    return rows


def paired(a,b):
    logs=[math.log(x/y) for x,y in zip(a,b)];rng=random.Random(20261004)
    boot=sorted(100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000))
    def q(p):
        x=9999*p;i=int(x);return boot[i]+(boot[min(i+1,9999)]-boot[i])*(x-i)
    return dict(reduction_percent=100*(1-math.exp(statistics.mean(logs))),ci95=[q(.025),q(.975)])


def analyze(directory,spec,repetition):
    c=json.loads((directory/'config.json').read_text());windows=[];signatures={};w=0
    for block,order in enumerate(orders(repetition)):
        for policy in order:
            path=directory/f'round-{repetition}'/f'window-{w:02d}-{policy}'
            rows=check_window(path,c,spec,policy,'timing')
            for rank,r in enumerate(rows):
                assert r['block']==block and r['window']==w
                sig=tuple(r[k] for k in ('initial_parameters_sha256','initial_rng_sha256','tokens_sha256','global_tokens_sha256','gpu_uuid','host'))
                assert signatures.setdefault(rank,sig)==sig,(rank,policy)
            latency=max(r['ms_per_step'] for r in rows)
            windows.append(dict(block=block,window=w,policy=policy,ms_per_step=latency,tokens_per_second=8192000/latency))
            w+=1
    assert w==16==len(list((directory/f'round-{repetition}').glob('window-*')))
    points={p:[r['ms_per_step'] for r in windows if r['policy']==p] for p in POLICIES}
    summary=[]
    for p in POLICIES:
        times=points[p]
        summary.append(dict(policy=p,median_ms_per_step=statistics.median(times),
            cv_percent=100*statistics.stdev(times)/statistics.mean(times),
            comparisons={b:paired(times,points[b]) for b in ('native','original')}))
    result=dict(completed=True,round=repetition,config=c,summary=summary,windows=windows,
                initialization_signatures=signatures,rank_records=128,
                scope='BF16 hierarchical TP8 only; no arrival/quantization acceptance',
                interval_unit='4 paired blocks within each round, bootstrap 10000; eight-rank maximum latency')
    write_json(directory/f'analysis-{repetition}.json',result)
    if repetition==2:
        prior=json.loads((directory/'analysis-1.json').read_text())
        assert prior['initialization_signatures']==json.loads(json.dumps(signatures))
        write_json(directory/'verification.json',dict(completed=True,formal_windows=32,rank_records=256,
            timed_optimizer_steps=640,initialization_matched=True,reverse_order_checked=True,
            quantization_tested=False))
    lines=[f'# {c["name"]}: round {repetition}','','| Policy | ms/step | vs native % [95% CI] | vs original hierarchy % [95% CI] |',
           '| --- | ---: | --- | --- |']
    for r in summary:
        vals=[r['policy'],f"{r['median_ms_per_step']:.3f}"]
        for b in ('native','original'):
            x=r['comparisons'][b];vals.append(f"{x['reduction_percent']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]")
        lines.append('| '+' | '.join(vals)+' |')
    (directory/f'report-{repetition}.md').write_text('\n'.join(lines)+'\n')
    return result
