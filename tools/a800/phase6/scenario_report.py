"""Same-model results for the communication-intensity scenario extension."""
import json


def write_report(root):
    info=json.loads((root/'submission.json').read_text())
    cfg=json.loads((root/'scripts/config.json').read_text());case=cfg['cases'][0]
    m=case['sequence']*cfg['common']['micro_batch'];n=case['hidden'];k=case['ffn']//case['tp']
    lines=['# Phase6 通信场景完整 step 对照','',
           f"{case['name']}；12层、FFN={case['ffn']}、global batch{cfg['common']['global_batch']}、{case['sequence']*cfg['common']['global_batch']} token/step。MLP [M,N,K_local]=[{m},{n},{k}]，每rank远端BF16逻辑字节为 {2*m*n*3//4/2**20:g} MiB/调用。",'',
           '支持的基础顺序为原有rank+1分区偏移、L,R1,R2,R3、R1,L,R2,R3、R1,R2,R3,L；本次实际候选以下表及冻结配置为准。后两种是远端开头的交替，不等于严格全部远端tile先执行。每种基础独立校准到达优先及选择mask，故完整组合差异不能全部归因于首槽位；attention不参与优化。',
           '同轮保留两个必须baseline和两个原生参考。固定原始顺序Flux＋融合量化v2版本。候选使用两套完整BF16/FP8工作区保护复用，仍保留每次GEMM的全rank发布barrier。',
           f"每轮{info['blocks']}块，每窗口10步预热＋20步完整optimizer step，第二轮反序；正值为配对耗时下降，95%区间按块bootstrap。",'']
    for i in (1,2):
        path=root/f'results/model-{i}/analysis.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())['summary']
        lines += [f'## 第{i}轮','','| 策略 | ms/step | vs Megatron＋量化下降% [95% CI] | vs Flux＋融合v2下降% [95% CI] | 比原生Flux耗时增加% |','| --- | ---: | --- | --- | ---: |']
        for r in rows:
            def pct(b):return f"{r[f'latency_reduction_vs_{b}_percent']:+.3f} [{r[f'ci95_low_vs_{b}']:+.3f}, {r[f'ci95_high_vs_{b}']:+.3f}]"
            lines.append(f"| {r['policy']} | {r['median_ms_per_step']:.3f} | {pct('native_taco')} | {pct('taco_fused')} | {-r['latency_reduction_vs_original_percent']:+.3f} |")
        if any(r['policy']=='interleaved_arrival_selective' for r in rows):
            lines += ['','| 远端开头候选 / 当前交替组合 | 配对耗时下降% [95% CI] |','| --- | --- |']
            for r in rows:
                if r['policy'].startswith('interleaved_remote'):
                    b='interleaved_arrival_selective'
                    lines.append(f"| {r['policy']} / {b} | {r[f'latency_reduction_vs_{b}_percent']:+.3f} [{r[f'ci95_low_vs_{b}']:+.3f}, {r[f'ci95_high_vs_{b}']:+.3f}] |")
    lines += ['', '不同模型场景之间不直接相除宣布收益。有限loss及codec预算通过不代表真实数据收敛；GPU校准扰动、同一分配内重复和合成token的限制仍适用。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
