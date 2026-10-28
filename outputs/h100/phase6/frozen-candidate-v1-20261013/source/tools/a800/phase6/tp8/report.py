"""Report BF16 TP8 measurements and separately measured TP4 before/after controls."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from protocol import POLICIES,sha,write_json


def change(before,after):
    logs=[math.log(a/b) for b,a in zip(before,after)]
    rng=random.Random(20261004)
    samples=sorted(100*(math.exp(statistics.mean(rng.choices(logs,k=len(logs))))-1) for _ in range(10000))
    def q(p):
        x=(len(samples)-1)*p;i=int(x)
        return samples[i]+(samples[min(i+1,len(samples)-1)]-samples[i])*(x-i)
    return dict(increase_percent=100*(math.exp(statistics.mean(logs))-1),ci95=[q(.025),q(.975)])


def regression(root):
    verified=json.loads((root/'verification.json').read_text());assert verified['completed'] and verified['windows']==72
    config=json.loads((root/'scripts/config.json').read_text());policies=config['policies']
    records=[];times=[]
    for phase in (1,2):
        table={};timings={}
        for path in sorted((root/f'results/model-{phase}').glob('window-*')):
            rr=[json.loads((path/f'rank{r}.json').read_text()) for r in range(4)]
            r=rr[0];block=r['block'] if phase==1 else 5-r['block']
            key=(block,r['policy']);table[key]=rr;timings[key]=max(x['ms_per_step'] for x in rr)
        assert len(table)==36
        records.append(table);times.append(timings)
    result=[]
    for policy in policies:
        before=[times[0][b,policy] for b in range(6)];after=[times[1][b,policy] for b in range(6)]
        controlled_before=[times[0][b,policy]/times[0][b,'original'] for b in range(6)]
        controlled_after=[times[1][b,policy]/times[1][b,'original'] for b in range(6)]
        trajectory={key:0. for key in ('losses','grad_norms')}
        for b in range(6):
            for r0,r1 in zip(records[0][b,policy],records[1][b,policy]):
                for key in ('initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'):
                    assert r0[key]==r1[key],(policy,key)
                for key in trajectory:
                    trajectory[key]=max(trajectory[key],max(abs(a-b) for a,b in zip(r0[key],r1[key])))
        result.append(dict(policy=policy,before_median_ms=statistics.median(before),after_median_ms=statistics.median(after),
            raw_change=change(before,after),relative_to_original_change=change(controlled_before,controlled_after),
            max_trajectory_difference=trajectory))
    return dict(root=str(root),job_id=json.loads((root/'submission.json').read_text())['job_id'],
                config=config,summary=result,completed=True,windows=72,rank_records=288,
                caution='Time-separated before/after; shared drift is not causal evidence of a TP8 code effect. Relative-to-original comparison controls common drift but is not an equivalence proof.')


def main(root,regression_roots):
    root=root.resolve();spec=json.loads((root/'submission.json').read_text());state=json.loads((root/'state.json').read_text())
    assert state['status']=='completed' and not state['active_steps']
    for name,h in spec['protected_sources'].items():assert sha(name)==h,name
    for name,h in spec['files'].items():assert sha(root/name)==h,name
    manifest=json.loads((Path(spec['build'])/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(Path(spec['build'])/name)==h,name
    ib=0
    for entry in state['windows']:
        assert entry['passed']
        if entry['mode']=='mapping':continue
        logs=list(Path(entry['out']).glob('nccl-node*-*.log'));assert len(logs)==8
        for p in logs:
            text=p.read_text();assert 'Using network IB' in text and 'Using network Socket' not in text
        ib+=1
    cases=[]
    lines=['# 双节点TP8 BF16对照与单节点TP4回归','','本次按用户选择测量已有BF16分层TP8路径，不扩展到达优先/选择性量化，不进行性能优化。TP8四组不构成两个量化必须baseline的完整4%验收。','',
           'TP8为两节点各4卡、DP1；每step8192 token，实际补齐词表8704，保持与单节点TP4相同的全局模型/batch。Flux在节点内执行两次M4096的GEMM-RS，再在对应节点rank间BF16交换并相加；MLP排序只作用于节点内计算，attention保留原始顺序。','',
           '| 配置 | 轮次 | 策略 | ms/step | 对Megatron下降% [95% CI] | 对原始顺序Flux分层适配下降% [95% CI] |',
           '| --- | --- | --- | ---: | --- | --- |']
    for case in spec['cases']:
        path=Path(case['config']).parent
        verification=json.loads((path/'verification.json').read_text());assert verification['completed']
        rounds=[]
        for i in (1,2):
            r=json.loads((path/f'analysis-{i}.json').read_text());rounds.append(r)
            c=r['config'];label=f"S{c['sequence']}/mb{c['micro_batch']}/gb{c['global_batch']}"
            for row in r['summary']:
                parts=[label,str(i),row['policy'],f"{row['median_ms_per_step']:.3f}"]
                for b in ('native','original'):
                    x=row['comparisons'][b];parts.append(f"{x['reduction_percent']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]")
                lines.append('| '+' | '.join(parts)+' |')
        cases.append(dict(name=case['name'],rounds=rounds,verification=verification))
    regressions=[regression(p.resolve()) for p in regression_roots]
    lines+=['','## 单节点TP4前后回归','','每节点使用相同配置和原冻结库；before在TP8之前，after在TP8之后，after反序。变化为正表示变慢，负值表示变快；配对按原始平衡块匹配，10000次bootstrap。时间分离的变化不能直接归因为TP8，另报告相对原生Flux的变化以观察共同漂移。','',
            '| 配置 / 作业 | 策略 | before ms | after ms | 耗时变化% [95% CI] | 相对原生Flux变化% [95% CI] | 最大loss / 梯度范数差 |',
            '| --- | --- | ---: | ---: | --- | --- | --- |']
    for reg in regressions:
        c=reg['config'];label=f"S{c['cases'][0]['sequence']}/mb{c['common']['micro_batch']} / {reg['job_id']}"
        for row in reg['summary']:
            parts=[label,row['policy'],f"{row['before_median_ms']:.3f}",f"{row['after_median_ms']:.3f}"]
            for key in ('raw_change','relative_to_original_change'):
                x=row[key];parts.append(f"{x['increase_percent']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]")
            parts.append(' / '.join(f'{v:.6g}' for v in row['max_trajectory_difference'].values()))
            lines.append('| '+' | '.join(parts)+' |')
    lines+=['', 'TP8共64正式窗口/512份rank记录/1280个计时optimizer step；另有16个模型预检窗口、网络与映射检查。单节点TP4回归共144正式窗口/576份rank记录/2880个计时optimizer step，另含各自原协议预检。所有计时窗口10步预热＋20步完整训练，固定合成token，不证明真实数据收敛。','',
            '原单节点及DP2源码保持不变，旧冻结库未覆盖；TP8所需的BF16形状路由在独立构建中完成。单节点前后记录的初始化、数据、RNG与GPU身份一致，所有窗口无跳步或回退。']
    result=dict(completed=True,tp8_cases=cases,tp4_regressions=regressions,ib_windows_verified=ib,
                tp8_formal_windows=64,tp4_formal_windows=144,performance_tuning=False,quantized_tp8_tested=False,
                protected_sources_unchanged=True,report_sha256=sha(Path(__file__)))
    write_json(root/'summary.json',result);(root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(root/'summary.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path)
    p.add_argument('--regressions',type=Path,nargs=2,required=True)
    a=p.parse_args();main(a.root,a.regressions)
