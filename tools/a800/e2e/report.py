"""Generate the requested loss/time/throughput report from completed four-way jobs."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
validation=Path(sys.argv[1]).resolve()
jobs=[Path(p).resolve() for p in sys.argv[2:]]
assert 1 <= len(jobs) <= 2
names={'native':'原生 Megatron-LM','original':'原始 Flux','remote_first':'远端重排','interleaved':'交替重排'}
policies=list(names)
gate=json.loads((validation/'validation.json').read_text())
assert gate['execution_passed']
job_rows={};raw={};metadata=[];flat=[];initial_states=None
for job in jobs:
    assert (job/'exit-code.txt').read_text().strip()=='0'
    analysis=json.loads((job/'analysis.json').read_text())
    assert analysis['execution_passed'] and analysis['windows']==32
    job_rows[job.name]={r['policy']:r for r in analysis['summary']}
    assert set(job_rows[job.name])==set(policies)
    raw[job.name]={p:[] for p in policies}
    for directory in sorted(job.glob('window-*')):
        assert (directory/'exit-code.txt').read_text().strip()=='0'
        rs=[json.loads((directory/f'rank{rank}.json').read_text()) for rank in range(4)]
        for rank,r in enumerate(rs):
            assert r['rank']==rank and r['completed'] and r['passed'] and r['skips']==0
            assert len(r['losses'])==len(r['grad_norms'])==30
            assert all(math.isfinite(x) for x in r['losses']+r['grad_norms'])
            assert r['warmup_steps']==20 and r['timed_steps']==30
            assert r['tokens_per_optimizer_step']==4096
            assert max(abs(a-b) for a,b in zip(r['losses'],rs[0]['losses'])) <= 1e-6
        r=rs[0];raw[job.name][r['policy']].append(r)
        flat.append(dict(job=job.name,window=r['window'],block=r['block'],policy=r['policy'],
            ms_per_step=max(x['ms_per_step'] for x in rs),loss_step21=r['losses'][0],loss_step50=r['losses'][-1],
            grad_norm_step21=r['grad_norms'][0],grad_norm_step50=r['grad_norms'][-1],skips=0,finite=True))
    assert all(len(rs)==8 for rs in raw[job.name].values())
    rs=[json.loads((job/f'window-00-native/rank{rank}.json').read_text()) for rank in range(4)]
    states=[tuple(r[key] for key in ['initial_parameters_sha256','initial_rng_sha256','tokens_sha256']) for r in rs]
    if initial_states is None:
        initial_states=states
    assert states==initial_states,'Model/RNG/data initial states differ across jobs'
    metadata.append(dict(job=job.name,node=(job/'hostname.txt').read_text().strip(),uuids=[r['gpu_uuid'] for r in rs]))

ids=[p.name for p in jobs]
lines=['# Megatron-LM 与三种 Flux 实现的端到端对比', '',
 '本页按同一小型GPT配置，对比**原生Megatron-LM、原始Flux、远端重排、交替重排**的loss、完整optimizer step时间、有效tokens/s和梯度范数。本页汇总实测运行指标。', '',
 '## 1. 当前结果', '',
 f"已完成{'、'.join(ids)}共{len(jobs)}轮独立Slurm作业，均正常退出。每轮32个计时窗口，四种实现各8个窗口；各窗口20步预热后计时30个完整optimizer steps。"]
if len(jobs)==1:
    lines += ['', '第二轮178712正在运行；本页先列完整首轮结果，复测结束后补齐第二轮列。']
lines += ['', '| 实现 | '+' | '.join(f'{j}：ms/step | {j}：tokens/s' for j in ids)+' |',
          '| --- | '+' | '.join('---: | ---:' for _ in ids)+' |']
for policy in policies:
    cells=[]
    for job in ids:
        r=job_rows[job][policy];cells += [f"{r['median_ms_per_step']:.3f}",f"{r['median_tokens_per_second']:,.0f}"]
    lines.append('| '+names[policy]+' | '+' | '.join(cells)+' |')
lines += ['', '上表为各实现8个窗口的中位数。每个窗口先取四个rank中最慢的耗时，再除以30步；`tokens/s = 4096 / 每步秒数`。TP=4处理同一批token，不再额外乘4。', '',
 '### 同一训练步位置的 loss', '',
 '每个窗口都从相同模型初始化和空optimizer状态重新开始，使用相同token、RNG和训练设置。初始step 1 loss取独立短训练作业178710；step 21和step 50分别取各计时窗口预热后的首步与最后一步。三列的采样位置分别标明。', '',
 '| 实现 | 初始loss（独立短训练step 1） | '+' | '.join(f'{j}：step 21 | {j}：step 50' for j in ids)+' |',
 '| --- | ---: | '+' | '.join('---: | ---:' for _ in ids)+' |']
for policy in policies:
    start=json.loads((validation/f'validation-{policy}/rank0.json').read_text())['losses'][0]
    cells=[f'{start:.6f}']
    for job in ids:
        rs=raw[job][policy]
        cells += [f"{statistics.median(r['losses'][0] for r in rs):.6f}",f"{statistics.median(r['losses'][-1] for r in rs):.8f}"]
    lines.append('| '+names[policy]+' | '+' | '.join(cells)+' |')
lines += ['', 'loss表中的计时段数值同样取8个窗口的中位数。四组都在重复使用的固定合成数据上训练，loss下降只说明本次短训练行为，不能代表真实语料的收敛或泛化能力。', '',
 '### 梯度与运行状态', '',
 '| 实现 | '+' | '.join(f'{j}：末步梯度范数' for j in ids)+' | 跳过更新 | loss/梯度NaN或Inf |',
 '| --- | '+' | '.join('---:' for _ in ids)+' | ---: | --- |']
for policy in policies:
    values=[f"{statistics.median(r['grad_norms'][-1] for r in raw[job][policy]):.6f}" for job in ids]
    lines.append('| '+names[policy]+' | '+' | '.join(values)+' | 0 | 无 |')
lines += ['', '梯度范数来自Megatron原生optimizer返回的全局范数；保留全部rank、全部step的原始值。', '',
 '### 配对耗时变化', '',
 '同一块内比较后取几何均值，正值表示耗时下降，负值表示耗时增加；因此下面的百分比不一定等于主表两个中位数直接相除。框架对比包含Flux接入与通信实现差异；判断额外重排的作用，应看相对原始Flux的列。', '',
 '| 实现 | '+' | '.join(f'{j}：相对原生Megatron下降% | {j}：相对原始Flux下降%' for j in ids)+' |',
 '| --- | '+' | '.join('---: | ---:' for _ in ids)+' |']
for policy in policies:
    cells=[]
    for job in ids:
        r=job_rows[job][policy]
        cells += [f"{r['latency_reduction_vs_native_percent']:+.2f}",f"{r['latency_reduction_vs_original_percent']:+.2f}"]
    lines.append('| '+names[policy]+' | '+' | '.join(cells)+' |')
lines += ['', '这些是固定配置下的测量差异；配对块、窗口波动和描述性区间保留在各作业CSV中，不用单个最快窗口代表整体。', '',
 '## 2. 固定测试配置', '',
 '| 项目 | 设置 |', '| --- | --- |',
 '| GPU/并行 | 单节点4×A800，NV8；TP=4，DP=PP=CP=1 |',
 '| 模型 | 4层GPT；hidden=1024，FFN=4096，16个attention heads |',
 '| 序列/batch | sequence=1024；micro batch=1，global batch=4；每步累积4个microsteps |',
 '| 词表 | token范围0–8191；实际padded vocabulary=8704 |',
 '| 精度/后端 | BF16；local Transformer、Apex LayerNorm；sequence parallel开启 |',
 '| 优化器 | 原生Megatron FP32主权重 + Apex Adam；lr=1e-4，weight decay=0.01，clip-grad=1.0 |',
 '| 数据 | 固定GPU驻留synthetic token bank；token seed=4321，模型seed=1234；4096有效tokens/step |',
 '| 其他 | dropout=0；不使用activation checkpointing或CUDA Graph；关闭梯度累积融合及所列bias/softmax融合 |',
 '| 软件 | Megatron-LM core_v0.12.3（3ea68ad6042cc1204386ae9364358f7c4de1bc37）；PyTorch 2.6.0+cu124；独立Conda环境flux-megatron-a800 |', '',
 '原生组保留Megatron的RowParallelLinear GEMM+NCCL路径。Flux三组共用已有IPC修复、BF16精度、归约选项及hparams，仅tile映射不同；本次模型接入采用`ring_reduction=True`。替换4层中的attention输出投影和MLP第二线性层，共8个模块。反向保持sequence AllGather和本地dgrad/wgrad计算，optimizer及其余模型结构相同。', '',
 '真实目标shape为`(M,N,K_global)=(1024,1024,1024)`及`(1024,1024,4096)`，局部K分别为256和1024。独立诊断确认三种Flux策略每rank每step实际执行32次目标GEMM-RS，无回退；tile=128×128×32、StreamK-SK、3 stages。', '',
 '## 3. 端到端时间包含什么', '',
 '一个optimizer step包括清梯度、4个梯度累积microsteps的forward/loss/backward、最终梯度同步、Adam和参数更新。窗口前后做GPU同步，使用单调墙钟计时；窗口内不新增逐算子同步。', '',
 '数据预先放在GPU上。数据读取/H2D、初始化、预热、checkpoint I/O、评估和profiler不计入这张表。Flux输出为保证autograd生命周期而进行的复制，包含在时间内。因此这里测的是完整训练计算步，而不是单个GEMM-RS，也不是包含数据加载和checkpoint的应用总墙钟。', '',
 '四组采用8个平衡块，每个实现在四种运行位置各出现2次；每个窗口重新启动并重放相同训练状态。profiler在独立进程运行，计时窗口关闭profiler和调用计数。', '',
 '## 4. 作业与原始结果', '',
 '| 作业 | 用途 | 节点 | 状态 |', '| --- | --- | --- | --- |',
 f'| {validation.name} | 四组准备、短训练、独立kernel诊断 | '+(validation/'hostname.txt').read_text().strip()+' | 脚本完成 |']
for m in metadata:lines.append(f"| {m['job']} | 四组整步计时 | {m['node']} | COMPLETED，退出码0:0 |")
if len(jobs)==2:
    lines += ['', '两轮使用相同四个GPU UUID。' if metadata[0]['uuids']==metadata[1]['uuids'] else '两轮GPU UUID不同，数据按作业分别报告。']
lines += ['', f'准备记录目录：`logs/a800/e2e/comparison/{validation.name}/`。此前的接入与逐张量诊断保留在原始日志中，本页按本轮要求汇总运行指标。', '']
for job in ids:
    lines += [f'- [{job} 时间汇总](../../../logs/a800/e2e/comparison/{job}/summary.csv)、[配对块](../../../logs/a800/e2e/comparison/{job}/paired_blocks.csv)、[所有窗口](../../../logs/a800/e2e/comparison/{job}/windows.csv)。']
lines += ['- [loss/梯度与时间联合汇总](../../../logs/a800/e2e/comparison/training_metrics.csv)。逐rank JSON还保留全部30步loss/梯度范数、GPU UUID、参数/RNG/token指纹和库哈希。', '',
 '## 5. 复跑入口', '',
 '脚本位于`tools/a800/e2e/`；需要现有独立Conda训练环境及`outputs/a800/phase3/three-way-build`。所有GPU作业通过普通`gpu_a800`队列提交。', '',
 '```bash', '# 从TempName根目录运行四组准备与诊断',
 'sbatch --export=ALL,E2E_STAGE=validate,E2E_ALLOW_EXPLORATORY=1 tools/a800/e2e/run.sbatch', '',
 '# 准备作业完成后，指定其结果目录，再执行下面的命令两次进行独立复测',
 'sbatch --export=ALL,E2E_STAGE=exploratory-timing,E2E_ALLOW_EXPLORATORY=1,E2E_VALIDATION_DIR=/绝对路径/准备作业目录 tools/a800/e2e/run.sbatch', '',
 '# 用已完成的两轮结果生成本页及CSV',
 f'python3 tools/a800/e2e/report.py logs/a800/e2e/comparison/{validation.name} \\',
 '  '+' \\\n  '.join(f'logs/a800/e2e/comparison/{job}' for job in ids), '```', '',
 '后续扩展优先增加更大hidden/sequence及真实数据，同时保持四组配置一致；当前表只代表上述固定小型GPT。']
text='\n'.join(lines)+'\n'
(ROOT/'docs/design/design-a800/base-e2e.md').write_text(text)
parent=jobs[0].parent
with (parent/'training_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
(parent/'report.txt').write_text(text)
print(text)
