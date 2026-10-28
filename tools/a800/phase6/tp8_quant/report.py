"""Report full-step TP8 compatibility measurements, with no speed admission gate."""
import json
from pathlib import Path


def write_report(root):
    info = json.loads((root/'submission.json').read_text())
    plan = json.loads((root/'scripts/selection-plan.json').read_text())
    config = json.loads((root/'scripts/config.json').read_text())
    model = dict(config['common'], **config['cases'][0])
    padding = config.get('expected_padded_vocab_size', 9216)
    lines = ['# 单节点 TP8 完整量化组合首测', '',
             f'{model["layers"]}层GPT，H{model["hidden"]}/FFN{model["ffn"]}，S{model["sequence"]}/mb{model["micro_batch"]}/global{model["global_batch"]}，TP8/DP1，{model["sequence"]*model["global_batch"]} token/optimizer step，padded vocab{padding}。', '',
             '仅MLP linear_fc2前向远端贡献采用FP8 E4M3/H128；GEMM、本地贡献和反向保持BF16，attention保持各自原后端。候选沿用两套完整工作区和通用解码；必须Flux基线为冻结的原顺序融合v2。', '',
             '到达表与mask从本次分配已有的TP8 BF16校准生成，0/1遍拟合、2遍留出诊断；未在重排或量化后重校准。window64及1/64预算沿用既有规则，未调参。', '',
             '| 基础顺序 | 每来源选中tile | 远端贡献选中比例 | 留出近临界命中率 |',
             '| --- | --- | ---: | ---: |']
    for base, entries in plan['policies'].items():
        e = entries[0]
        hit = 'N/A' if e['heldout_near_critical_fraction'] is None else f"{100*e['heldout_near_critical_fraction']:.2f}%"
        lines.append(f"| {base} | {e['selected_per_source']} | {100*e['selected_remote_fraction']:.4f}% | {hit} |")
    lines += ['', '每轮6个位置平衡块，第二轮反转块及组顺序；每窗口10步预热＋20步完整optimizer step。配对下降=100×(1−exp(mean(log(T候选/T基线))))，95%区间按块bootstrap 10000次；不能用表内中位数相除替代。', '']
    for i in (1, 2):
        path = root/f'results/model-{i}/analysis.json'
        if not path.exists():
            continue
        lines += [f'## 第{i}轮', '', '| 策略 | ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |',
                  '| --- | ---: | --- | --- | ---: | ---: |']
        for r in json.loads(path.read_text())['summary']:
            def pct(b):
                return f"{r[f'latency_reduction_vs_{b}_percent']:+.3f} [{r[f'ci95_low_vs_{b}']:+.3f}, {r[f'ci95_high_vs_{b}']:+.3f}]"
            lines.append(f"| {r['policy']} | {r['median_ms_per_step']:.3f} | {pct('native_taco')} | {pct('taco_fused')} | {-r['latency_reduction_vs_native_percent']:+.3f} | {-r['latency_reduction_vs_original_percent']:+.3f} |")
        lines.append('')
    lines += [f"作业{info['job_id']}同一分配内两轮新进程；数值校验和有限loss不证明真实数据收敛。逻辑字节统计不等于实测链路流量。跨节点TP8量化仍未实现。"]
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import sys
    write_report(Path(sys.argv[1]))
