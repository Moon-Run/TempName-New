"""Write this allocation's results without changing historical reports."""
import csv
import json
from pathlib import Path
import sys


def main(root):
    state = json.loads((root/'campaign-state.json').read_text())
    lines = [f'# TP8 v2：分配 {state["job_id"]}', '',
             '同一单节点8×A800分配内运行；两轮是独立进程重复，不是独立Slurm作业。',
             'BF16，ring_reduction=True，tail分数，五组算子对照；正值为相对各自基础耗时下降。', '',
             '| 轮次 | shape M/N/K | 策略 | 下降% | 95%区间 |', '| --- | --- | --- | ---: | --- |']
    for r in (1, 2):
        data = json.loads((root/f'results/operator-{r}/analysis.json').read_text())
        for row in data['comparisons']:
            if (row['policy'], row['baseline']) not in [('remote_arrival','remote_first'), ('interleaved_arrival','interleaved')]:
                continue
            lines.append(f'| {r} | {row["M"]}/{row["N"]}/{row["K_global"]} | {row["policy"]} | {row["latency_reduction_percent"]:+.2f} | [{row["ci95_low"]:+.2f}, {row["ci95_high"]:+.2f}] |')
    lines += ['', '模型计时准入：同一候选在两个模型实际shape、两轮中的配对区间下界均大于0。',
              '通过候选：'+(', '.join(state['model_gate']['accepted_policies']) or '无；跳过正式模型计时，保留预检结果。'), '',
              '独立重采样的来源命中、尾部排序、尾10%召回率、准入集合大小和采样开销见 [label-validation.json](results/label-validation.json)。',
              '这批样本仍不代表低扰动或重排后机制验收。gap仅生成了离线消融plan；恒等/随机查表控制尚未进行GPU测试。']
    for r in (1, 2):
        p = root/f'results/model-{r}/summary.csv'
        if not p.exists():
            continue
        lines += ['', f'## 模型第{r}轮', '', '| 策略 | ms/step |', '| --- | ---: |']
        with p.open() as f:
            for row in csv.DictReader(f):
                lines.append(f'| {row["policy"]} | {float(row["median_ms_per_step"]):.3f} |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
