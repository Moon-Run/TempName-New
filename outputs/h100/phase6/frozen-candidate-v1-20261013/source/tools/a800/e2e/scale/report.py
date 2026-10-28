"""Refresh only the 12-layer expansion section; retain the four-layer historical tables."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'logs/a800/e2e/scale'
plan=json.loads(Path(__file__).with_name('config.json').read_text())
submissions=json.loads((OUT/'submissions.json').read_text())
labels={'native':'原生 Megatron-LM','original':'原始 Flux','remote_first':'远端重排','interleaved':'交替重排'}
start='<!-- scale-results:start -->';end='<!-- scale-results:end -->'
lines=[start,'## 当前实验：12层 GPT，扩大 hidden/sequence','',
 '本轮统一采用**12层GPT**。主对照为原生Megatron-LM、原始Flux、远端重排、交替重排；记录loss、完整optimizer step时间、tokens/s、梯度范数及显存峰值。', '',
 '| 层数 | hidden | sequence | FFN | heads | TP/GPU | 当前状态 |',
 '| ---: | ---: | ---: | ---: | ---: | ---: | --- |']
for entry in submissions['cases']:
 case=next(c for c in plan['cases'] if c['name']==entry['case'])
 complete=[j for j in entry['repetitions'] if (OUT/str(j)/'analysis.json').exists() and (OUT/str(j)/'exit-code.txt').exists() and (OUT/str(j)/'exit-code.txt').read_text().strip()=='0']
 state='已按用户要求取消全部8卡作业' if entry.get('status')=='CANCELLED_BY_USER' else f'预检{entry["preflight"]}；计时'+ '/'.join(map(str,entry['repetitions']))+f'，已完成{len(complete)}轮'
 lines.append(f"| 12 | {case['hidden']} | {case['sequence']} | {case['ffn']} | {case['heads']} | {case['tp']} | {state} |")
lines += ['', '两节点各4卡需要跨节点通信。Megatron可使用这种部署，但本次三种Flux接入设置`nnodes=1`、CUDA IPC/NVLink和单节点启动；跨节点需另行接入与验证，不能直接沿用当前对照。按用户要求，本轮8卡任务全部取消，继续12层4卡实验。8卡配置仅保留为未运行模板。', '',
 '### 12层4卡的实测结果','']
all_rows=[]
for entry in submissions['cases']:
 if entry.get('status')=='CANCELLED_BY_USER':continue
 case=next(c for c in plan['cases'] if c['name']==entry['case'])
 jobs=[]
 for job in entry['repetitions']:
  path=OUT/str(job)
  if not (path/'analysis.json').exists() or not (path/'exit-code.txt').exists() or (path/'exit-code.txt').read_text().strip()!='0':continue
  d=json.loads((path/'analysis.json').read_text());assert d['completed'] and d['model']['layers']==12 and d['case']==case
  rows={r['policy']:r for r in d['summary']};jobs.append((job,rows))
  all_rows.extend(dict(job=job,**r) for r in d['summary'])
 if not jobs:
  lines += ['计时作业仍在运行，完成后填入完整窗口统计；预检的短窗口时间不充当主表结果。','']
  continue
 lines += ['| 实现 | '+' | '.join(f'{j} ms/step | {j} tokens/s' for j,_ in jobs)+' |',
           '| --- | '+' | '.join('---: | ---:' for _ in jobs)+' |']
 for policy,label in labels.items():
  cells=[]
  for job,rows in jobs:cells += [f"{rows[policy]['median_ms_per_step']:.3f}",f"{rows[policy]['median_tokens_per_second']:,.0f}"]
  lines.append('| '+label+' | '+' | '.join(cells)+' |')
 lines += ['', '| 实现 | '+' | '.join(f'{j} 初始loss | {j} step 11 loss | {j} step 30 loss | {j} 峰值显存GiB' for j,_ in jobs)+' |',
           '| --- | '+' | '.join('---: | ---: | ---: | ---:' for _ in jobs)+' |']
 for policy,label in labels.items():
  cells=[]
  for job,rows in jobs:
   r=rows[policy];cells += [f"{r['initial_loss']:.6f}",f"{r['first_timed_loss']:.6f}",f"{r['last_timed_loss']:.6f}",f"{r['max_peak_allocated_gib']:.2f}"]
  lines.append('| '+label+' | '+' | '.join(cells)+' |')
 lines += ['', '表中所有完成窗口的loss和梯度范数均有限，跳步为0；显存取各rank/窗口中最大的PyTorch allocated峰值。', '',
           '| 实现 | '+' | '.join(f'{j} 相对原始Flux耗时下降% | {j} 末步梯度范数' for j,_ in jobs)+' |',
           '| --- | '+' | '.join('---: | ---:' for _ in jobs)+' |']
 for policy,label in labels.items():
  cells=[]
  for job,rows in jobs:
   r=rows[policy];cells += [f"{r['latency_reduction_vs_original_percent']:+.2f}",f"{r['final_grad_norm']:.6f}"]
  lines.append('| '+label+' | '+' | '.join(cells)+' |')
 lines += ['', '正值表示配对耗时下降，负值表示增加。绝对ms/step取窗口中位数，配对百分比按相同块内比值汇总。','']
 for job,_ in jobs:lines.append(f'- [{job} 完整汇总](../../../logs/a800/e2e/scale/{job}/summary.csv)、[窗口](../../../logs/a800/e2e/scale/{job}/windows.csv)、[配对块](../../../logs/a800/e2e/scale/{job}/paired_blocks.csv)。')
 lines.append('')
lines += ['### 扩展测试口径','',
 '保持BF16、sequence parallel、DP=PP=CP=1、micro batch=1/global batch=4、Apex Adam和原有融合开关。hidden=2048、sequence=2048时每步8192有效token；attention与MLP目标shape分别为(2048,2048,2048)、(2048,2048,8192)，局部K为512和2048。12层共24个目标模块，每step实际目标调用数为96。', '',
 '每轮4个位置平衡块、四种实现各4个独立窗口；每窗口10步预热后计时20步，计划两个独立作业重复。ms/step取全部rank中最慢墙钟，包含完整前向/反向、梯度同步、Adam和清梯度；数据预置GPU，排除初始化、预热、profile与checkpoint。与历史4层测试的窗口长度不同，结果分表报告。', '',
 '每个窗口重建相同模型/optimizer/RNG和输入；本轮初始loss直接取该窗口第一步预热，不再借用另一作业。固定合成token重复使用，loss下降不代表真实数据泛化。', '',
 '入口为`tools/a800/e2e/scale/`，脚本支持TP=4/8的全rank统计和映射检查。当前只验证4卡运行；被取消的8卡作业不计为已验证。', '',
 '```bash', '# 12层4卡预检（普通gpu_a800队列）',
 'sbatch --gpus=4 --cpus-per-task=8 --export=ALL,E2E_CASE=l12-h2048-s2048-tp4,E2E_SCALE_STAGE=preflight tools/a800/e2e/scale/run.sbatch', '',
 '# 预检完成后指定结果目录，运行两轮独立计时',
 'sbatch --gpus=4 --cpus-per-task=8 --export=ALL,E2E_CASE=l12-h2048-s2048-tp4,E2E_SCALE_STAGE=timing,E2E_SCALE_PREFLIGHT=/绝对路径/预检目录 tools/a800/e2e/scale/run.sbatch',
 'python3 tools/a800/e2e/scale/report.py', '```',end]
section='\n'.join(lines)+'\n'
p=ROOT/'docs/design/design-a800/base-e2e.md';text=p.read_text()
if start in text:
 a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+section+text[b:]
else:
 title,rest=text.split('\n',1)
 text='# Megatron-LM 与 Flux 端到端对比：12层扩展\n\n'+section+'\n## 历史基线：4层 GPT\n'+rest
if '## 历史基线：4层 GPT' in text:
 prefix,history=text.split('## 历史基线：4层 GPT',1)
 for old,new in [('## 1. 当前结果','### 4层历史结果'),('## 2. 固定测试配置','### 4层历史配置'),('## 3. 端到端时间包含什么','### 4层历史计时口径'),('## 4. 作业与原始结果','### 4层历史作业'),('## 5. 复跑入口','### 4层历史复跑入口')]:
  history=history.replace(old,new)
 text=prefix+'## 历史基线：4层 GPT'+history
p.write_text(text)
(OUT/'report.txt').write_text(section)
if all_rows:
 with (OUT/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
print(section)
