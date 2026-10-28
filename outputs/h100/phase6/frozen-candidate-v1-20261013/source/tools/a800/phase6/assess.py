"""Require same-campaign baselines and reversed rounds at the frozen target."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REQUIRED = ('native_taco', 'taco_fused')
REFERENCES = ('native', 'original')
CANDIDATES = ('remote_arrival_selective', 'interleaved_arrival_selective')


def assess(root):
    info = json.loads((root/'submission.json').read_text())
    config = json.loads((root/'scripts/config.json').read_text())
    # Historical campaigns without this field retain their original 5% target.
    acceptance = config.get('acceptance', dict(version='legacy-5pct', target_percent=5.))
    target = float(acceptance['target_percent'])
    assert 0 < target < 100
    candidates=tuple(p for p in info['policies'] if p.endswith('_arrival_selective'))
    assert set(CANDIDATES)<=set(candidates)
    assert set(info['policies']) == set(REQUIRED + REFERENCES + candidates)
    assert info['blocks'] == len(info['policies']) and info['repetitions'] == 2
    sys.path.insert(0, str(root/'scripts'))
    sys.path.insert(1, str(root))
    from verify import verify
    verify(root)
    result = dict(completed=True, target_percent=target, acceptance_version=acceptance['version'], metric='1 - geometric mean of paired block latency ratios',
                  interval_unit=f"{info['blocks']} paired blocks per round; bootstrap 10000 resamples",
                  scope='Two fresh-process repetitions in one Slurm allocation; fixed synthetic tokens',
                  candidates={}, baseline_manifests={}, graph_forward=config.get('graph_forward',False))
    build = Path(info['artifact_root'])/'build'
    result['double_buffered']=config.get('double_buffered',False)
    result['scenario_extension']=config.get('scenario_extension',False)
    manifest = json.loads((build/'manifest.json').read_text())
    result['baseline_manifests'] = manifest['source_manifests'].get('taco_fused')
    # The required Flux baseline must be the independently frozen phase4 warp-v2
    # original-order build, not a rebuilt candidate or a reordered all-remote path.
    frozen = Path(info['repo'])/'outputs/a800/phase4/taco-fused-warp-20261003/taco/libflux_cuda.so'
    assert manifest['policies']['taco_fused']['library_sha256'] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    lines = ['# Phase6 同轮完整 step 验收', '',
             f'正数表示配对耗时下降。{target:g}%目标按每轮点估计判断，并分别给出95%区间；参考baseline差距为候选相对参考的配对耗时增加百分比。', '',
             '| 轮次 | 候选 | ms/step中位数 | vs Megatron＋量化下降% [95% CI] | vs Flux＋融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |',
             '| --- | --- | ---: | --- | --- | ---: | ---: |']
    for candidate in candidates:
        rounds = []
        for i in (1,2):
            data = json.loads((root/f'results/model-{i}/analysis.json').read_text())
            row = next(r for r in data['summary'] if r['policy']==candidate)
            r = dict(round=i, median_ms_per_step=row['median_ms_per_step'], required={}, reference_gap_percent={})
            for b in REQUIRED:
                r['required'][b] = dict(reduction_percent=row[f'latency_reduction_vs_{b}_percent'],
                                       ci95=[row[f'ci95_low_vs_{b}'],row[f'ci95_high_vs_{b}']])
            for b in REFERENCES:
                r['reference_gap_percent'][b] = -row[f'latency_reduction_vs_{b}_percent']
            r['meets_target'] = all(v['reduction_percent']>=target for v in r['required'].values())
            r['positive_intervals'] = all(v['ci95'][0]>0 for v in r['required'].values())
            def fmt(b):
                v=r['required'][b]
                return f"{v['reduction_percent']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
            lines.append(f"| {i} | {candidate} | {r['median_ms_per_step']:.3f} | {fmt(REQUIRED[0])} | {fmt(REQUIRED[1])} | {r['reference_gap_percent']['native']:+.3f} | {r['reference_gap_percent']['original']:+.3f} |")
            rounds.append(r)
        result['candidates'][candidate] = dict(rounds=rounds, meets_target_both_rounds=all(r['meets_target'] and r['positive_intervals'] for r in rounds))
    result['target_achieved'] = any(c['meets_target_both_rounds'] for c in result['candidates'].values())
    lines += ['', '达到目标。' if result['target_achieved'] else f'尚未达到相对两个必须baseline均快{target:g}%的目标。', '',
              '完整step包含运行时查表、Q/DQ、通信同步、反向和optimizer。图重放启用时，输入复制、图启动和独立输出复制均计时；一次性建图/校准在预热阶段，不计入稳态step。仅优化MLP，attention保持原有后端。有限loss和数值预算通过不等于真实数据收敛。',
              ('各基础使用冻结计划中的BF16校准拟合到达表及选择mask；0/1遍拟合、2遍留出诊断。校准可由既有同形状实验复用，不代表本次重新采样；未在到达置换或量化后再次采样。组合差异同时包含顺序与mask差异，不能全部归因于首槽位。'
               if result['scenario_extension'] else
               '量化mask与到达表仍源于旧BF16校准；本轮隔离执行开销优化，未宣称完成重排后的联合自适应重校准。')]
    (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'acceptance.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    assess(p.parse_args().root.resolve())
