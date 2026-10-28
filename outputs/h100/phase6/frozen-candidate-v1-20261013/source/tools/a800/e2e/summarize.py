"""Validate complete all-rank model results and summarize independent paired windows."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def quantile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def load_process(path, policy, mode, require_pass=True):
    records=[json.loads((path/f'rank{rank}.json').read_text()) for rank in range(4)]
    if require_pass:
        assert (path/'exit-code.txt').read_text().strip()=='0'
    for rank,r in enumerate(records):
        assert r['completed'] and (r['passed'] or not require_pass) and r['rank']==rank
        assert r['policy']==policy and r['mode']==mode and not r['sampling']
        assert r['fallback_calls']==0 and len(r['target_modules'])==8
        assert r['replaced_modules']==(0 if policy=='native' else 8)
        assert {(s['M'],s['N'],s['K_global']) for s in r['observed_shapes'].values()}=={(1024,1024,1024),(1024,1024,4096)}
    assert len({r['gpu_uuid'] for r in records})==4
    assert len({r['tokens_sha256'] for r in records})==1
    return records


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text())
    spec=protocol['config'];policies=spec['policies']
    exploratory=protocol.get('exploratory',False)
    if protocol['stage']=='validate':
        diagnostics=[];checks=[];native=None;profile_rows=[];pilot_rows=[]
        for policy in ['native','native-repeat',*policies[1:]]:
            actual='native' if policy=='native-repeat' else policy
            rs=load_process(out/f'validation-{policy}',actual,'validate',require_pass=not exploratory)
            if policy=='native':native=rs
            for rank,r in enumerate(rs):
                for key in ['initial_parameters_sha256','initial_rng_sha256','tokens_sha256']:
                    assert r[key]==native[rank][key]
                assert r['finite_training'] and r['skips']==0
                assert len(r['losses'])==spec['validation_steps']
                assert all(c==spec['validation_steps']*4 for c in r['target_calls'].values())
                assert max(r['loss_differences'])<=spec['budgets']['loss_absolute']
                for name,c in r['checks'].items():
                    if not exploratory:
                        assert c['passed'],(policy,rank,name,c)
                    checks.append(dict(policy=policy,rank=rank,tensor=name,**c))
                if not exploratory:
                    for c in r['shape_fp32_checks']:assert c['passed']
                diagnostics.append(dict(policy=policy,rank=rank,numerical_passed=r['passed'],checks=len(r['checks']),max_loss_abs=max(r['loss_differences']),
                    max_layer_output_rel_l2=max((c.get('relative_l2',0) for n,c in r['checks'].items() if n.startswith('layer_output/')),default=0),
                    max_parameter_gradient_rel_l2=max((c.get('relative_l2',0) for n,c in r['checks'].items() if n.startswith('parameter_gradient/')),default=0)))
        for policy in policies:
            rs=load_process(out/f'pilot-{policy}',policy,'pilot')
            elapsed=max(r['elapsed_seconds'] for r in rs)
            pilot_rows.append(dict(policy=policy,ms_per_step=elapsed*1000/spec['pilot_timed_steps'],
                                   tokens_per_second=spec['pilot_timed_steps']*4096/elapsed))
            prs=load_process(out/f'profile-{policy}',policy,'profile')
            for rank,r in enumerate(prs):
                with gzip.open(out/f'profile-{policy}/rank{rank}-profile.json.gz','rt') as f:
                    events=json.load(f)['traceEvents']
                kernels=[e for e in events if e.get('cat')=='kernel']
                flux=[e for e in kernels if 'flux_bf16' in e['name']]
                assert len(flux)==(0 if policy=='native' else 32),(policy,rank,len(flux))
                hparams=set()
                for e in flux:
                    assert 'rcr_gemmv2_false_false_intranode_' in e['name']
                    h=re.search(r'64x64x32_16x8x16_streamksk_nil_\d+x\d+x\d+_gemmstreamk_\d+_rasterheuristic',e['name'])
                    assert h;hparams.add(h.group())
                    assert e['args']['registers per thread']==254 and e['args']['block']==[128,1,1]
                if flux:assert hparams=={'64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic'}
                profile_rows.append(dict(policy=policy,rank=rank,flux_gemm_rs_calls=len(flux),
                    fused_kernel_sum_us=sum(e['dur'] for e in flux),hparams=';'.join(hparams),
                    note='Independent diagnostic; kernel sums are not model critical-path fractions'))
        for policy in policies[1:]:
            for m,n,k in spec['shapes']:
                path=out/f'mapping-{policy}-m{m}-n{n}-k{k}.csv'
                with path.open() as f:rows=list(csv.DictReader(f))
                assert len(rows)==4*(m//128)*((n+127)//128)
                for rank in range(4):
                    rs=[r for r in rows if int(r['rank'])==rank]
                    assert {(int(r['m']),int(r['n'])) for r in rs}=={(i,j) for i in range(m//128) for j in range((n+127)//128)}
        write_csv(out/'validation_summary.csv',diagnostics)
        # Native-reference records do not contain errors; keep a union schema.
        fields=sorted(set().union(*(r.keys() for r in checks)))
        write_csv(out/'tensor_errors.csv',[{key:r.get(key,'') for key in fields} for r in checks])
        write_csv(out/'pilot.csv',pilot_rows);write_csv(out/'profile_summary.csv',profile_rows)
        report=dict(passed=all(r['numerical_passed'] for r in diagnostics),execution_passed=True,exploratory=exploratory,script_sha256=protocol['script_sha256'],
            flux_manifest_sha256=protocol['flux_manifest_sha256'],budgets=spec['budgets'],
            diagnostics=diagnostics,pilot=pilot_rows,profile=profile_rows,
            scope=spec['scope'],limitations=['Small fixed synthetic GPT only; not a convergence guarantee.',
            'Pilot is not formal performance evidence; no paired confidence interval yet.'])
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    else:
        windows=[];blocks={};by_rank={};all_rs=[]
        expected=sum(len(o) for o in spec['orders'])
        assert len(list(out.glob('window-*')))==expected
        window=0
        for block,order in enumerate(spec['orders']):
            blocks[block]={}
            for policy in order:
                rs=load_process(out/f'window-{window:02d}-{policy}',policy,'timing')
                for rank,r in enumerate(rs):
                    assert r['block']==block and r['window']==window
                    assert r['timed_steps']==spec['timed_steps'] and r['warmup_steps']==spec['warmup_steps']
                    assert not r['profiling'] and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0
                    assert len(r['losses'])==spec['timed_steps'] and all(math.isfinite(x) for x in r['losses']+r['grad_norms'])
                    assert r['skips']==0 and all(c==spec['warmup_steps']*4 for c in r['warmup_calls'].values())
                    signature=tuple(r[key] for key in ['initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'])
                    if rank not in by_rank:by_rank[rank]=signature
                    assert by_rank[rank]==signature
                    all_rs.append(r)
                elapsed=max(r['elapsed_seconds'] for r in rs)
                row=dict(window=window,block=block,policy=policy,ms_per_step=1000*elapsed/spec['timed_steps'],
                    tokens_per_second=4096*spec['timed_steps']/elapsed,min_rank_seconds=min(r['elapsed_seconds'] for r in rs),max_rank_seconds=elapsed)
                windows.append(row);blocks[block][policy]=row;window+=1
        paired=[];summary=[]
        for policy in policies:
            times=[b[policy]['ms_per_step'] for b in blocks.values()]
            row=dict(policy=policy,median_ms_per_step=statistics.median(times),
                median_tokens_per_second=statistics.median(b[policy]['tokens_per_second'] for b in blocks.values()),
                min_ms=min(times),max_ms=max(times),cv_percent=100*statistics.stdev(times)/statistics.mean(times))
            for baseline in ['native','original']:
                logs=[math.log(b[policy]['ms_per_step']/b[baseline]['ms_per_step']) for b in blocks.values()]
                ratio=math.exp(statistics.mean(logs));rng=random.Random(20261002)
                boot=[100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000)]
                reduction=100*(1-ratio)
                row.update({f'latency_reduction_vs_{baseline}_percent':reduction,
                    f'throughput_gain_vs_{baseline}_percent':100*(1/ratio-1),
                    f'ci95_low_vs_{baseline}':quantile(boot,.025),f'ci95_high_vs_{baseline}':quantile(boot,.975),
                    f'faster_blocks_vs_{baseline}':sum(x<0 for x in logs)})
                for block,b in blocks.items():
                    paired.append(dict(block=block,policy=policy,baseline=baseline,baseline_ms=b[baseline]['ms_per_step'],
                        candidate_ms=b[policy]['ms_per_step'],latency_reduction_percent=100*(1-b[policy]['ms_per_step']/b[baseline]['ms_per_step'])))
            summary.append(row)
        write_csv(out/'windows.csv',windows);write_csv(out/'paired_blocks.csv',paired);write_csv(out/'summary.csv',summary)
        report=dict(passed=True,execution_passed=True,status='EXPLORATORY_ONLY' if exploratory else 'FORMAL',
            numerical_admission_passed=json.loads((out/'validation-gate.json').read_text())['passed'] if (out/'validation-gate.json').exists() else True,
            summary=summary,windows=len(windows),scope=spec['scope'],
            practical_threshold_percent=spec['practical_latency_reduction_percent'],
            interval_unit='8 paired blocks, not individual optimizer steps',
            limitations=['One small GPU-resident synthetic model configuration.',
                         'Each job interval is descriptive within that GPU allocation; compare both independent jobs.'])
        (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':analyze(Path(sys.argv[1]))
