"""Summarize completed jobs without selecting the best repetition or policy."""
import csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
state=json.loads(Path(sys.argv[1]).read_text());LOG=ROOT/'logs/a800/arrival8'
def read(path):
 with path.open() as f:return list(csv.DictReader(f))
lines=['## 单节点TP8到达优先实测（自动汇总）','',
 '两种候选在每目标分区16个槽位窗口内稳定排序。评分方式以各作业冻结plan与构建manifest为准；历史v1使用滞后分数，审计后v2默认使用接收端相对join时刻，另有一致来源gap消融，三者需分别解释。全部通信保持BF16，ring_reduction=True；采样标签仅用于离线启发式，不用于证明精确网络时长或低扰动机制。',
 '', 'FP32误差按用户要求仅记录、不阻断；BF16组合容差、映射及运行健康仍须通过。正式性能使用冻结映射与无插桩库。以下正值表示耗时下降，区间为单作业内配对块bootstrap 95%区间，各作业分别列出。校准/构建成本不在算子稳态计时内，运行时查表成本已计入。','',
 '### 算子：两种叠加策略相对各自基础','',
 '| 作业 | M/N/K_global | 候选 | 基础 | 候选 μs | 下降% [95%区间] |',
 '| --- | --- | --- | --- | ---: | --- |']
all_ops=[];all_models=[]
for job in state['jobs']:
 out=LOG/str(job['job_id'])
 if job['stage']=='operator':
  assert json.loads((out/'analysis.json').read_text())['completed']
  summary=read(out/'summary.csv');comparisons=read(out/'comparisons.csv')
  all_ops.extend(dict(job=job['job_id'],**r) for r in comparisons)
  for r in comparisons:
   if (r['policy'],r['baseline']) not in [('remote_arrival','remote_first'),('interleaved_arrival','interleaved')]:continue
   t=next(t for t in summary if all(t[k]==r[k] for k in ['M','N','K_global','policy']))
   lines.append(f"| {job['job_id']} | {r['M']}/{r['N']}/{r['K_global']} | {r['policy']} | {r['baseline']} | {float(t['median_us']):.2f} | {float(r['latency_reduction_percent']):+.2f} [{float(r['ci95_low']):+.2f}, {float(r['ci95_high']):+.2f}] |")
lines += ['', '### 12层GPT：六组完整optimizer step','',
 '| 作业 | 策略 | ms/step | tokens/s | 相对原生下降% | 相对原始Flux下降% | 相对对应基础下降% [95%区间] |',
 '| --- | --- | ---: | ---: | ---: | ---: | --- |']
for job in state['jobs']:
 if job['stage']!='timing':continue
 out=LOG/str(job['job_id']);assert json.loads((out/'analysis.json').read_text())['completed']
 for r in read(out/'summary.csv'):
  all_models.append(dict(job=job['job_id'],**r))
  base={'remote_arrival':'remote_first','interleaved_arrival':'interleaved'}.get(r['policy'])
  delta=(f"{float(r['latency_reduction_vs_'+base+'_percent']):+.2f} [{float(r['ci95_low_vs_'+base]):+.2f}, {float(r['ci95_high_vs_'+base]):+.2f}]" if base else '—')
  lines.append(f"| {job['job_id']} | {r['policy']} | {float(r['median_ms_per_step']):.3f} | {float(r['median_tokens_per_second']):,.0f} | {float(r['latency_reduction_vs_native_percent']):+.2f} | {float(r['latency_reduction_vs_original_percent']):+.2f} | {delta} |")
lines += ['', '完整数据见各作业目录的summary.csv、comparisons.csv或paired_blocks.csv；全部rank记录、映射、实际kernel、loss、梯度范数、显存和退出状态随作业保存。有限loss/梯度和首次前向容差通过不代表真实语料收敛或完整训练轨迹等价。', '',
 '两种叠加候选均须相对各自基础判断，不将原始Flux相对Megatron的收益归给到达优先。若重复间方向不一致或区间跨零，保留“未显示稳定增量”的结论。该实验为单节点TP8，不推广到双节点调度。','']
for name,rows in [('operator-comparison.csv',all_ops),('model-comparison.csv',all_models)]:
 with (LOG/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
text='\n'.join(lines);(LOG/'report.md').write_text(text)
doc=ROOT/'docs/design/design-a800/only-tile.md';s=doc.read_text()
start='<!-- arrival8-results:start -->';end='<!-- arrival8-results:end -->'
if start in s:
 a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+start+'\n'+text+end+s[b:]
else:s+='\n'+start+'\n'+text+end+'\n'
doc.write_text(s)
print(LOG/'report.md')
