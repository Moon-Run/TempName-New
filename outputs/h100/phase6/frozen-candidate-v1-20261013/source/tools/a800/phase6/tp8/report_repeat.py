"""Combine two immutable TP8 BF16 campaigns as rounds 1/2 and 3/4."""
import argparse
import json
from pathlib import Path

from protocol import POLICIES, sha, write_json


def read(path):
    return json.loads(path.read_text())


def collect(root):
    spec, state = read(root/'submission.json'), read(root/'state.json')
    assert state['status'] == 'completed' and not state['active_steps']
    assert len(state['windows']) == 82 and all(w['passed'] for w in state['windows'])
    for name, digest in spec['files'].items():
        assert sha(root/name) == digest, name
    for name, digest in spec['protected_sources'].items():
        assert sha(name) == digest, name
    build = Path(spec['build'])
    assert sha(build/'manifest.json') == spec['original_build_manifest_sha256']
    for name, digest in read(build/'manifest.json')['files'].items():
        assert sha(build/name) == digest, name
    ib = 0
    for window in state['windows']:
        if window['mode'] == 'mapping':
            continue
        logs = list(Path(window['out']).glob('nccl-node*-*.log'))
        assert len(logs) == 8
        for p in logs:
            text = p.read_text()
            assert 'Using network IB' in text and 'Using network Socket' not in text
        ib += 1
    assert ib == 81
    cases = []
    for case in spec['cases']:
        directory = Path(case['config']).parent
        verification = read(directory/'verification.json')
        assert verification['completed'] and verification['formal_windows'] == 32
        assert verification['rank_records'] == 256
        rounds = [read(directory/f'analysis-{i}.json') for i in (1, 2)]
        assert rounds[0]['initialization_signatures'] == rounds[1]['initialization_signatures']
        cases.append(dict(name=case['name'], rounds=rounds))
    return spec, cases


def ci(value):
    return (f"{value['reduction_percent']:+.3f} "
            f"[{value['ci95'][0]:+.3f},{value['ci95'][1]:+.3f}]")


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def main(previous, current):
    previous, current = previous.resolve(), current.resolve()
    old_spec, old_cases = collect(previous)
    new_spec, new_cases = collect(current)
    for key in ('build', 'jobs', 'nodes', 'policies'):
        assert old_spec[key] == new_spec[key], key
    mapping = read(current/'round-mapping.json')
    assert mapping['local_to_display_round'] == {'1': 3, '2': 4}
    assert Path(mapping['previous_root']).resolve() == previous
    for name in ('adapter.py', 'worker.py', 'node.py', 'run.py', 'protocol.py', 'summarize.py', 'probe.py'):
        assert sha(previous/'scripts'/name) == sha(current/'scripts'/name), name
    merged, absolute, paired = [], [], []
    for old, new in zip(old_cases, new_cases):
        assert old['name'] == new['name']
        rounds = []
        for source, case, offset in ((previous, old, 0), (current, new, 2)):
            for analysis in case['rounds']:
                assert analysis['config'] == old['rounds'][0]['config']
                assert analysis['initialization_signatures'] == old['rounds'][0]['initialization_signatures']
                rounds.append(dict(display_round=analysis['round']+offset,
                                   source_root=str(source), local_round=analysis['round'], analysis=analysis))
        config = rounds[0]['analysis']['config']
        label = f"S{config['sequence']}/mb{config['micro_batch']}/gb{config['global_batch']}"
        for row in rounds:
            stats = {x['policy']: x for x in row['analysis']['summary']}
            absolute.append([label, row['display_round'],
                             *[f"{stats[p]['median_ms_per_step']:.3f}" for p in POLICIES]])
        for policy, name in (('remote_first', 'MLP远端BF16'), ('interleaved', 'MLP交替BF16')):
            for row in rounds:
                stats = next(x for x in row['analysis']['summary'] if x['policy'] == policy)
                paired.append([label, row['display_round'], name,
                               ci(stats['comparisons']['native']), ci(stats['comparisons']['original'])])
        merged.append(dict(name=old['name'], rounds=rounds))
    absolute_table = table(['配置', '轮次', '原生Megatron', '原始顺序Flux分层适配', 'MLP远端BF16', 'MLP交替BF16'], absolute)
    paired_table = table(['配置', '轮次', '排序', '对Megatron下降% [95% CI]', '对原始顺序Flux分层适配下降% [95% CI]'], paired)
    result = dict(completed=True, cases=merged, previous_root=str(previous), current_root=str(current),
                  new_formal_windows=64, new_rank_records=512, new_timed_optimizer_steps=1280,
                  total_formal_windows=128, total_rank_records=1024, total_timed_optimizer_steps=2560,
                  same_build_and_gpu_scripts=True, cross_campaign_initialization_match=True,
                  ib_windows_per_campaign=81, optimization=False, quantized_tp8_tested=False,
                  tp4_regression_repeated=False, report_sha256=sha(Path(__file__)),
                  absolute_table=absolute_table, paired_table=paired_table)
    write_json(current/'summary.json', result)
    (current/'summary.md').write_text('\n\n'.join([
        '# TP8 BF16第1–4轮对照（第3、4轮复测，无优化）',
        '第1、2轮保留原目录，第3、4轮来自本目录内部round-1/2，映射见round-mapping.json。两次配置、冻结库和GPU测量脚本一致，初始化、输入、RNG及GPU身份跨四轮核验一致。',
        '以下ms/step为八rank最大窗口耗时的中位数；每窗口10步预热、20步计时。每轮四个平衡块，轮1/2、轮3/4各为正序/反序。',
        absolute_table,
        '配对耗时下降为正表示更快；每轮单独计算配对比值的几何平均及10000次块bootstrap的95%区间，不用中位数相除，不混合四轮重新挑选统计口径。按形状、排序、轮次排列。',
        paired_table,
        '本次新增64正式窗口/512份rank记录/1280计时step；四轮合计128窗口/1024份rank记录/2560计时step。两次所有训练及网络预检窗口均使用IB。原TP4六组回归记录保留在第1、2轮会话，本次没有重跑TP4。四个BF16策略不构成完整TP8量化验收。',
    ])+'\n')
    print(current/'summary.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    main(args.previous, args.current)
