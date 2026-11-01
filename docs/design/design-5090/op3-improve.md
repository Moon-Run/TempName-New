# OP3：A＋MA89＋MA11 上的 R＋CR3 融合与编码器改进

更新：2026-11-02。目标为单机 8×RTX 5090，TP8/DP1。**已实现两组代码和测试入口；新 OP3 的 GPU 正确性与性能结果尚待目标机运行。** 默认仍为 **2 个 block**，对应 `block00`、`block01`，每窗口 10 个 warmup＋50 个正式 optimizer step。

## 1. 本次比较内容

这里的 A 是 [OP1](op-1-A&B&C.md) 的 `flux-basic-interleaved`，即 Flux GemmOnly＋基础全远端量化＋交替 CTA 排序；不是 [baseline.md](baseline.md) 中使用另一套 native producer 的 `a`。MA89 取 MA8 激活缓存和 MA9 零梯度统计，MA11 取精确 causal softmax，三项沿用 [OP2](op2-ma11&ma89.md) 的安装方式，每个 worker 各安装一次。

| 实验组 | 脚本 ID | 编码 | 解码与后继消费者 |
| --- | --- | --- | --- |
| **A＋MA89＋MA11＋R＋CR3 消费者融合** | `op3-r` | 原 A 的独立 H128 编码器 | R packed-load/source-prefetch 解码；写回前执行 bias 和 residual 的两次 BF16 加法 |
| **A＋MA89＋MA11＋OP3 编码器改进** | `op3-improved` | 一个 warp 负责一个 H128 组，四个组组成一个 CTA | 保留 A 的基础解码器及原两个独立 BF16 add |

两组共用原 A 的同一份交替排序 GemmOnly 二进制、全远端量化、单次 NCCL All-to-All、MA8/MA9/MA11、模型和训练输入。第二组对应用户要求的“A＋MA89＋MA11＋你的改进”，其改动是编码器；本轮直接比较两个方案。没有把历史 OP2 步时加入新批次配对统计，因此本轮只证明两组之间的差异，不能单独证明任一组相对原 OP2 A 的净增量。

## 2. 回传数据与改进依据

### OP2 A 的正式结果和原始 trace

使用批次 [`op2-abc-ma11-v1-1x-50steps-20261102-013244-945721`](../../../../Load-5090/logs/5090/phase6/op2-abc-ma11-v1-1x-50steps-20261102-013244-945721/report.md)。已核对 96 份 rank 记录和 24 份 trace 的哈希；用正式 rank 记录重新计算全部 6 个窗口和比较统计，与原报告一致。A 的八 rank profile 也重新解析通过，覆盖 **2048 次投影**。跨 Python 版本重算的累加浮点字段存在小于 `1e-10` 的差异，离散计数一致。

| OP2 组 | 两个正式窗口（ms/step） | 平均（ms/step） |
| --- | --- | ---: |
| A＋MA89＋MA11 | 861.025 / 861.425 | **861.225** |
| B＋MA89＋MA11 | 990.580 / 998.847 | 994.713 |
| C＋MA89＋MA11 | 997.457 / 1006.530 | 1001.993 |

B/C 的分块路径未观察到通信与解码重叠。本次继续使用 A 的一次完整交换。

以下 profile 指标先求每 rank、每角色的调用中位数，再求八 rank 中位数；单位 μs。它们来自独立 profile，不能直接当作正式步时节省。

| A 的实际路径 | Attention | MLP |
| --- | ---: | ---: |
| 编码器 | **65.128** | **64.808** |
| 基础解码器 | 26.456 | 26.392 |
| 两个消费者 add 合计 | 7.616 | 7.560 |
| decoder 开始→LayerNorm 结束 | 52.536 | 52.320 |
| NCCL kernel | 660.783 | 282.671 |

按每 rank 整步 kernel 时长求和，再取八 rank 中位数，编码器约 **16.688 ms**、解码器约 **6.784 ms**、两个 add 约 **1.942 ms**、NCCL 约 **152.577 ms**。这些时间不是关键路径分解，不能相加推导整步收益。通信仍是更大的开销；当前数据支持优先减少独立编码器的 CTA 数和共享内存同步成本。

原编码器每个远端目的 rank 的 grid 为 8192，每 CTA 128 threads，总计 `8192×7=57344` 个 CTA；trace 中每 CTA shared memory 为 656 bytes。OP3 改进编码器仍使用 128 threads，但每 warp 完成一个组，每 CTA 完成四个组，grid 变成 `2048×7=14336`，消除该 kernel 的 CTA 内共享内存 butterfly/barrier。

### 为什么纳入 R 和 CR3 消费者融合

[5090 baseline 的同期数据](baseline.md#同期配对结果)中：

- R/A：977.081 / 986.816 ms，配对降幅 **0.987%［0.954%, 1.019%］**，18/18 块更快。
- CR3/C：1017.192 / 1022.978 ms，配对降幅 **0.566%［0.522%, 0.612%］**，18/18 块更快。

**CR3/C 不是单独融合的消融。** 其代码链还包含 R 的 mixed decoder 和 CR2 提交缓存，且使用选择性量化。不能把 0.566% 全归给 bias/residual 融合，也不能直接作为当前 A 的预期收益。

更直接的融合证据来自 [H100 MA4-F 的 F4/CR2 对照](../design-h100/New-Mechanism/ma-4.md#6-本轮-ma4-f-结果相对-cr2-有端到端增量)：保持相同 producer、选择策略和提交缓存，F4 为 501.433 ms、CR2 为 509.852 ms，融合新增量 **1.652%［1.297%, 2.034%］**，8/8 块更快；profile 每 rank 每步减少 512 个消费者 add kernel，并保留原 LayerNorm。它说明融合值得迁移测试，但属于 H100 历史证据。

因此，本次 **纳入 R＋CR3 的消费者融合部分**。是否对当前 5090 A 路径有效，由新的 `op3-r` 结果判断；不迁入 C 的选择性量化和分块到达路径。

## 3. 实现与算术约束

### R＋CR3 组

[R 的来源](../../../tools/h100/phase6/ma2_decode.cuh)使用 packed FP8 load、下一 source 预取、末尾本地 BF16 贡献预取，以及每 warp 一个 H128 组。本次将其地址计算改为已完成 NCCL 交换后的本地 `H128-AoS-136` packet，每个 packet 包含 128 bytes FP8＋两个 FP32 scale。

解码按 `rank+1 → … → rank+7 → local` 顺序，每个解码值及每次累加均保留 BF16 舍入。R 的收益机制作用于本地 decoder；它不恢复 H100 的跨卡直接访存或 peer 到达重叠。原 A 已有实例配置和独立输出，本轮不把这些已有机制再记为新增优化。

写回执行严格的 `BF16(residual + BF16(projection + bias))`，而不是三项 FP32 求和后只舍入一次。保留原 LayerNorm。融合 wrapper 绑定当前层实际 residual 和原 bias 参数，autograd 继续返回 input、weight、bias、residual 梯度；bias 归约先于原线性 backward 的序列 AllGather、dgrad 和 wgrad。相关生成器复用已验证的 [MA4-F adapter](../../../tools/h100/phase6/ma4_f_adapter.py)，由 `E2E_POLICY=op3-r` 显式启用。

### 编码器改进组

每个 lane 持有 `lane+32*j` 的四个原始系数。保留原来的 FP32 平方和归约树、adaptive scale、Hadamard butterfly 的距离顺序、max 归约、quant scale、E4M3 转换及 packet 布局。减少线程组数和共享同步，不改变量化算法、预算或远端字节数。

正式执行前必须证明新编码器与冻结 upstream 编码器生成完全相同的 packet，以及解码结果逐位一致。若不一致，直接停止在 probe 阶段，不允许通过放宽量化误差阈值进入计时。

### 两组共同运行规则

每个模块持久保存自己的收发 packet，使用固定 caller stream；每次投影只有一次异步 All-to-All，并调用 `Work.wait()` 建立通信完成依赖后解码。输出每次独立分配，不能覆盖仍被 autograd 使用的旧输出。提交异常会使相应进程组/GPU 的后续提交失效；不使用 clone 假装输出所有权。

每 rank 每投影发送 `7×8192×136=7798784` bytes，本地贡献不发送、不量化；这是逻辑贡献字节数，不等于 NCCL 实测链路流量。两组保持相同 `CUDA_DEVICE_MAX_CONNECTIONS=1` 和 baseline NCCL 设置。本次没有启用 CUDA Graph，也没有通过变更线程流或通信设置制造组间差异。

代码入口：

| 文件 | 职责 |
| --- | --- |
| [op3_codec.cu](../../../tools/5090/phase6/op3_codec.cu)、[op3_codec.py](../../../tools/5090/phase6/op3_codec.py) | 两条 CUDA 路径、固定流与输出生命周期 |
| [op3_build.py](../../../tools/5090/phase6/op3_build.py) | 新建小型 SM120 codec 库；只读复用原 A 的 GemmOnly 构建 |
| [op3_sources.py](../../../tools/5090/phase6/op3_sources.py) | 两组完整训练 worker、融合 adapter 和三项通用优化 |
| [op3_probe.py](../../../tools/5090/phase6/op3_probe.py)、[op3_validation.py](../../../tools/5090/phase6/op3_validation.py) | upstream 位串、融合舍入、完整模型及最终运行计数校验 |
| [op3_profile.py](../../../tools/5090/phase6/op3_profile.py) | 实际 kernel、NCCL、消费者、反向和 MA8/MA9/MA11 路径审计 |
| [op3.py](../../../tools/5090/phase6/op3.py)、[op3_protocol.py](../../../tools/5090/phase6/op3_protocol.py) | 分阶段执行、平衡配对、报告和只读重放 |

## 4. 测试与对比脚本

在**目标 5090 机器的 `TempName-New` 仓库根目录**运行，继续使用已配置的 5090 环境。本轮复用 OP2 已使用过的 ABC build；以下 `/workspace` 是该批次记录中的目标机路径，需要它及其依赖目录仍完整存在。更新整套 `tools/5090/phase6/` 新代码后再启动；正在运行的批次会校验冻结 controller 哈希。

### 仅原生正确性测试

可选的独立检查：编译 OP3 小库并检查两组完整形状、56 条有向 packet、原始 upstream 码流/解码 oracle、真实 GemmOnly 映射、独立输出复用及 R 融合边界；不运行完整模型：

```bash
bash tools/5090/phase6/run_op3.sh \
  --abc-build /workspace/Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/build \
  --transport-smoke-only --execute
```

完整两组对比会重复这些前置检查；无需先单独运行本节。若已启动独立检查，等待 `batch-state.json` 的状态为 `transport_smoke_completed` 后再启动完整对比，避免同时启动两个后台实验。

零值、随机、离群输入都需通过；融合另测 half-ULP、signed zero、subnormal、cancellation 等边界，half-ULP 必须能区分一次和两次 BF16 舍入。六个仍存活的输出在后续复用后必须保持不变。目标机任一探针失败就停止，不能把仅编译成功写成 GPU 验证成功。

### 完整两组对比：默认两个 block

```bash
bash tools/5090/phase6/run_op3_compare.sh \
  --abc-build /workspace/Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/build \
  --execute
```

入口默认 `--trials 1 --blocks 2`，正式顺序是：

```text
block00: op3-r → op3-improved
block01: op3-improved → op3-r
```

共 **4 个正式窗口、200 个正式 optimizer step、32 份正式 rank 记录**；两组各自先完成全模型 smoke 和八 rank profile，再开始计时。保留 TP8、32 层、hidden=4096、sequence=2048、micro batch=1、global batch=4。每窗口独立启动，八 rank 最大 elapsed / 50 作为步时。

不加 `--execute` 只打印计划。默认后台启动，查看打印的 `handoff/supervisor.log` 和 `batch-state.json`；可加 `--foreground` 前台执行。结果写入新目录 `Load-5090/logs/5090/phase6/op3-two-arm-v1-1x-50steps-<时间>/`，不会向历史 OP1/OP2 批次追加结果。

可将上述命令的 `--abc-build ...` 替换为 `--op3-build /path/to/completed-op3-batch/build`，只读复用本轮已编译的小库及原 ABC 依赖，避免重复编译该 codec；MA9/MA11 兼容小库及其数值检查仍由每个新批次准备。构建的 source、binary、SM120、Python/CUDA ABI 任一身份不符都会拒绝复用。

### 完成后的报告复核

```bash
bash tools/5090/phase6/run_op3_compare.sh \
  --out /path/to/completed-op3-batch --report
```

`--report` 只接受完整批次，重新验证冻结源码、构建/库身份、native probe、所有窗口和原始 profile，打印两组结果；不是失败批次的续跑入口。完整重放需要在仍持有构建及绝对路径依赖的目标环境执行。

主报告为 `report.md` / `report.json`，并保留 `results/compare1/` 下的每窗口 rank 记录、`native-probe/report.json`、`preflight.json` 和 `profile-summary.json`。正降幅表示**编码器改进组比 R＋融合组更快**；保留负收益及全部窗口。两块仅给描述统计与配对点估计，不计算 95% CI。需要更稳定统计时使用 `--blocks 6` 或更大的偶数；默认值仍为 2。

## 5. 验证范围和判读

- 源码生成检查在硬件预检前执行。GPU 正式计时必须先通过独立码流/映射/输出所有权、两组全模型 smoke、全部 16 份实际 profile。
- 全模型结束后、同步完成后才读取 64 个惰性创建投影实例的状态；timing 每模块必须恰好 `(10+50)×4=240` 次调用。R 的每次调用都必须经过融合消费；改进组保持裸投影。安装 MA8/MA9/MA11 各一次，并核对实际调用计数。
- profile 要求 R 的实际预取 decoder 后直接进入原 LayerNorm；改进组要求新编码器、原 decoder、两个独立 add 和原 LayerNorm。反向序列通信与两个 GEMM、允许的 split-K 归约、三项通用优化均单独审计。
- native probe 强制两组相同输入的裸投影 packet/output 逐位相同。两组全模型 smoke 必须具有相同 loss、grad_norm、最终参数及全部 389 个参数的逐步梯度/更新轨迹，任何差异都会阻止进入 profile 和计时；后续 profile/timing 的跨组轨迹继续完整记录。R 的 smoke 输出包含 bias/residual，另一组采样裸投影，两者输出指纹所处语义位置不同，不能直接用该指纹判定训练失配。
- 当前开发环境没有可用 GPU。新 CUDA 源码已经通过 CUDA 12.8 的 `sm_120` 编译及 SASS 检查：R 两个模板各 40 registers，新编码器 26 registers，均无 spill、无 CTA barrier；已检查 R 下一 source 和末尾本地贡献的提前加载。目标机仍会重新编译并归档自己的 `codec/compiled.sass`，CPU 检查不代替目标 CUDA 12.9 的数值或性能结果。

全套 5090 CPU 回归 **280 项通过**；在补强 smoke 训练等价门禁后，最终 **47 项 OP3 专项 CPU 测试重新通过**，覆盖算术映射、构建溯源、探针证据、实际 worker 生命周期、完整控制流程、profile 审计和 smoke 训练等价拒绝。两个 shell 入口的语法与只预览模式通过；新增文件的空白和文档链接检查通过。

本地回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/5090/phase6 -p 'test_*.py' -v
bash -n tools/5090/phase6/run_op3.sh tools/5090/phase6/run_op3_compare.sh
git diff --check
```

本次 OP3 的正式步时、显存峰值和胜出组均保持待测，不从 OP2 profile 的 kernel 时长外推出收益百分比。
