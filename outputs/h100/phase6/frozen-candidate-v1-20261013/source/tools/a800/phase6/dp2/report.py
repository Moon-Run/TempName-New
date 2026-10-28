"""Collect completed frozen DP2 cases, with no cross-layout latency ratios."""
import argparse
import gzip
import json
from pathlib import Path

from protocol import sha, write_json


def main(root):
    root = root.resolve()
    spec = json.loads((root/'submission.json').read_text())
    state = json.loads((root/'state.json').read_text())
    assert state['status'] == 'completed'
    assert not state['active_steps']
    assert json.loads((root/'network-check.json').read_text())['passed']
    assert json.loads((root/'source-consistency.json').read_text())['passed']
    network_windows = 0
    for entry in state['windows']:
        path = Path(entry['out'])
        assert entry['passed']
        logs = list(path.glob('nccl-node*-*.log'))
        assert len(logs) == 8, (path, len(logs))
        for logfile in logs:
            text = logfile.read_text()
            assert 'Using network IB' in text, logfile
            assert 'Using network Socket' not in text, logfile
        network_windows += 1
    for path,digest in spec['protected_sources'].items():
        assert sha(path) == digest, path
    for path,digest in spec['files'].items():
        assert sha(root/path) == digest, path
    manifest = json.loads((Path(spec['build'])/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha(Path(spec['build'])/name) == digest, name
    cases = []
    profile_rows = []
    model_descriptions = []
    for item in spec['cases']:
        cfg = json.loads(Path(item['config']).read_text())
        model = dict(cfg['common'], **cfg['cases'][0])
        model_descriptions.append(f"{model['layers']}层/H{model['hidden']}/FFN{model['ffn']}，S{model['sequence']}/mb{model['micro_batch']}/global{model['global_batch']}，{model['sequence']*model['global_batch']}全局token/step")
    lines = ['# 双节点 TP=4、DP=2 测量结果', '',
             '两个节点分别运行一个TP4组，跨节点同步DP梯度；复用原冻结实现，仅适配测试入口，没有性能调优。', '',
             '；'.join(model_descriptions)+'。DP2保持来源TP4的每副本工作量，全局batch翻倍；不直接计算跨布局加速比。', '',
             '| 配置 | 轮次 | 策略 | ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |',
             '| --- | --- | --- | ---: | --- | --- | ---: | ---: |']
    for case in spec['cases']:
        path = Path(case['config']).parent
        for policy in spec['policies']:
            for rank in range(8):
                with gzip.open(path/f'profile-{policy}'/f'rank{rank}-profile.json.gz','rt') as f:
                    events = json.load(f)['traceEvents']
                kernels = [e for e in events if e.get('cat') == 'kernel']
                reductions = [e for e in kernels if 'nccl' in e['name'].lower() and 'allreduce' in e['name'].lower().replace('_','')]
                profile_rows.append(dict(case=case['name'],policy=policy,rank=rank,
                    allreduce_calls=len(reductions),allreduce_kernel_sum_ms=sum(e['dur'] for e in reductions)/1000,
                    flux_gemm_kernel_sum_ms=sum(e['dur'] for e in kernels if 'flux_bf16' in e['name'])/1000))
        acceptance = json.loads((path/'acceptance.json').read_text())
        assert acceptance['completed']
        rounds = []
        for repetition in (1,2):
            r = json.loads((path/f'analysis-{repetition}.json').read_text())
            assert r['completed'] and len(r['windows']) == 36
            rounds.append(r)
            c = r['config']
            label = f"S{c['cases'][0]['sequence']}/mb{c['common']['micro_batch']}/gb{c['common']['global_batch']}"
            for entry in r['summary']:
                vals = [label, str(repetition), entry['policy'], f"{entry['median_ms_per_step']:.3f}"]
                for baseline in ('native_taco','taco_fused'):
                    x = entry['comparisons'][baseline]
                    vals.append(f"{x['reduction_percent']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]")
                vals += [f"{-entry['comparisons'][b]['reduction_percent']:+.3f}" for b in ('native','original')]
                lines.append('| '+' | '.join(vals)+' |')
        cases.append(dict(name=case['name'], acceptance=acceptance, rounds=rounds))
    lines += ['', '耗时取八rank中最慢窗口时间；每轮6个位置平衡块，第二轮反序，每窗口10步预热＋20步计时。下降百分比为配对耗时比的几何平均，95%区间来自10000次配对块bootstrap，不能由ms中位数相除替代。', '',
              '## 两轮4%门槛', '', '| 配置 | 候选 | 通过 |', '| --- | --- | --- |']
    for case in cases:
        for policy,value in case['acceptance']['candidates'].items():
            lines.append(f"| {case['name']} | {policy} | {'是' if value['meets_target_both_rounds'] else '否'} |")
    formal = 72*len(cases)
    lines += ['', f'正式{formal}窗口 / {formal*8}份rank记录 / {formal*20}个全局计时optimizer step；另有每场景6个smoke和6个profile窗口。', '',
              '检查覆盖实际IB传输、TP/DP分组、DP数据分片、同步梯度及更新后参数分片一致性、初始化/RNG/GPU身份、MLP-only范围、mask/通信字节和精确解码kernel计数。源代码和冻结库SHA按本轮准备时的清单核验。', '',
              '合成固定token，反向BF16 STE；不含数据加载、checkpoint或评估，不证明真实数据收敛。未启用梯度通信重叠，未执行TP8，也未为达标调整任何性能参数。']
    result = dict(completed=True, started_at=state['started_at'], finished_at=state['finished_at'],
                  jobs=spec['jobs'], nodes=spec['nodes'], cases=cases,
                  formal_windows=formal, rank_records=formal*8, timed_optimizer_steps=formal*20,
                  source_consistency_passed=True, network_ib_passed=True,
                  all_windows_ib_verified=network_windows,
                  reporting_code_sha256={'report.py':sha(__file__),
                                         'protocol.py':sha(Path(__file__).with_name('protocol.py'))},
                  performance_tuning=False)
    write_json(root/'summary.json', result)
    write_json(root/'profile-components.json',dict(rows=profile_rows,
        scope='Diagnostic profiled step, not formal timing. AllReduce sums include TP/DP collectives and waiting; not isolated DP transfer or critical-path attribution.'))
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(root/'summary.md')


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    main(p.parse_args().root)
