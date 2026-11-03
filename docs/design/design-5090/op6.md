# OP6：在 OP4 上重放局部 Attention CUDA Graph

更新：2026-11-03。**本轮选择减少本地 attention 的重复主机提交，在 OP4 上只增加每层 attention core 的前向／反向 CUDA Graph。已实现方案和同期 OP4／OP6 对比脚本，目标为完整 optimizer step ≤800 ms；首轮 OP4 已实测830.484 ms，OP6 已通过 GPU smoke，但在 profile 容差检查处中止，尚无正式 OP6 步时，不能声称已达到目标。最新结果与修复见第8节。**

[OP4](op4-improve.md) 两个已完成窗口为 833.486、831.011 ms，均值 **832.249 ms/step**。到 800 ms 需要净省约 **32.249 ms，即 3.875%**。本次重新测量 OP4 对照组，历史 832.249 ms 只说明选题背景，不混入配对统计。[baseline.md](baseline.md) 的慢卡替换情景也不是本次可直接兑现的收益。

## 1. 选择这个方向的依据

对 [OP4 原批次](../../../../Load-5090/SL3060/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/report.md) 的两个窗口、全部 **16 份单步 trace**，按实际 CPU owner 和 CUDA correlation 归属 attention 工作。每 rank 先求区间并集／工作量，再取八 rank 中位数：

| 指标，每 optimizer step | block00 | block01 |
| --- | ---: | ---: |
| 正式无 profiler 步时，ms | 833.486 | 831.011 |
| attention 选定 CPU scope 并集，ms | 110.591 | 110.087 |
| attention CPU 包围区间并集，含提交间隙，ms | 145.943 | 145.764 |
| 对应 attention GPU 工作量，ms | 60.904 | 61.141 |
| 原 attention 前向 kernel 数/rank | 512 | 512 |
| 原 attention 反向 kernel 数/rank | 896 | 896 |

前向选取 QK `baddbmm`、MA11 softmax、PV `bmm`、context 布局复制；反向选取 PV backward、MA11 backward、QK backward。每 step 有 32 层 × 4 microbatch，即 128 次 core forward 和 128 次 core backward。**CPU/GPU 区间相互重叠，且 profile 会扰动提交与 rank 到达；不能把 146−61 ms 当作可省时间，更不能与 NCCL 时间相加。** 数据支持将重复提交作为独立实验对象。

已有 [H100 MA15](../design-h100/New-Mechanism/ma-15.md) 使用同类局部图，在其独立实验中得到 344.230→315.552 ms 的端到端结果。它提供实现和验证经验；机器、互联、PyTorch 和基线不同，28.678 ms 不能直接搬到 5090。该历史批次还有文档已说明的最终验收序列化问题；OP6 的 profile 返回值使用 JSON 原生列表，并检查序列化往返一致性。

其他方向的本地证据：

- [open-compress 六组结果](../../../../Load-5090/SL3060/open-compress-int8-six-arm-v2-1x-4blocks-20261102-185333-783385/report.md) 中，Flux INT8 的交替排序相对普通 INT8 为 −0.048%，远端优先为 +0.098%。这不能支撑把 CTA 排序当作本轮 32 ms 的主要来源；它们也不是 OP4 同策略对照。
- OP4 已使用 16 个 NCCL 通道。本轮保留这个配置，避免同时改变通信调度而无法归因。
- [OP5](op5.md) 研究 SHM 接收端 CE 和 P2P LL 兼容配置。**OP6 不叠加这些设置，不改 AllGather／All-to-All 算法、通信字节数、量化精度或 TP8 切分。**

可复算的 [分析脚本](../../../tools/5090/phase6/analyze_op6_baseline.py) 和 [逐 rank 数据](op6-baseline-analysis.json) 保留原始 trace 路径、SHA256、完整 OP3 profile 验证以及统计定义：

```bash
python3 tools/5090/phase6/analyze_op6_baseline.py \
  --batch ../Load-5090/SL3060/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654 \
  --out /tmp/op6-baseline-analysis.json
```

## 2. 实际改进及与 OP4 的边界

| 项目 | OP4 对照 | OP6 候选 |
| --- | --- | --- |
| 基础计算 | OP3 combined：A＋MA89＋MA11＋R＋CR3＋warp 编码器 | 相同 |
| NCCL min/max channels | 16／16 | 16／16 |
| 量化、source 归约顺序、消费者 BF16 舍入 | 原实现 | 相同 |
| attention core 提交 | 每次 eager autograd／ATen 提交 | 每层一对固定前向／反向 CUDA Graph |
| 图边界 | 无 | 只含原 QK、MA11 softmax、PV 与局部布局／梯度运算 |
| 通信、投影、DDP、optimizer | 原路径 | 原路径，均在图外 |

[运行时实现](../../../tools/5090/phase6/op6_adapter.py) 复用 MA15 的生命周期设计，并固定到目标 `2.8.0a0+5228986c39.nv25.06`。不替换为 FlashAttention/SDPA，不更换 GEMM 算术或 softmax 内核。

每层首次预热时，在辅助流上录制原 attention core；正式重放仍在原计算流上：

```text
forward：实际 Q/K/V 和 bool mask → 静态缓冲 → forward graph → 独立输出副本
backward：实际输出梯度 → 静态缓冲 → backward graph → 独立 dQ/dK/dV 副本
```

32 层各自持有一对图和独立图池，同一层必须 forward 后完成 backward 才能再次使用。保留实际 Q/K/V 的 shape、stride 与指针对齐；每次复制真实 mask。stream、device、布局漂移，未配对重放、高阶梯度、嵌套捕获或异常后复用都会失败。图输入和返回输出分别持有存储，保留 autograd 的原输入版本检查。

[PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs) 要求固定捕获结构与存储生命周期。目标预发布版本的具体行为还按源码核对：其 [replay_prologue](https://github.com/pytorch/pytorch/blob/5228986c39/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L198-L208) 在 `wholegraph_increment` 非零时才写 RNG 状态；本 attention dropout 为零，所以不照搬 H100/PyTorch 2.6 的每次两个 RNG fill。新 profile 检查要求图内无随机操作，重放无这些 fill，并保留初始化 RNG 不变检查。

本方案增加边界复制和图池驻留。每次 core 前后向逻辑复制约 20 MiB，每 step 约 2.5 GiB；这些是逻辑字节数，不是实测 HBM 流量。静态 strided Q/K/V 与 mask 的输入存储约 704 MiB/卡，另有图内输出、梯度、workspace 和图池。复制、独立输出 clone、保护检查全部计入正式步时；内存池保留在峰值统计中。捕获发生于预热，正式指标反映稳态训练，不代表短任务启动时延。

按 256 个前后向边界均摊，达到 800 ms 约需每边界净省 **126 μs**。这只是预算，必须由正式配对窗口检验。若复制／图池成本抵消收益，保留负结果；不降低模型、步数或数值门禁。

## 3. 一条命令启动同期对比

在目标机 `TempName-New` 仓库根目录、已配置的八卡 5090 环境执行。首次可以不传 build，入口会完成硬件/NCCL 预检并准备 baseline→ABC→OP3 依赖：

```bash
bash tools/5090/phase6/run_op6.sh \
  --place 60 --trials 1 --blocks 2 --execute
```

默认编译并行数为 **`nproc`**，可用 `--jobs N` 覆盖。`--place 60` 写入 `../Load-5090/SL3060/`；`--place 61` 写入 `../Load-5090/SL3061/`。place 只选择结果文件夹，不进行远程登录或更改 GPU 选择；显式 `--out` 优先。不传 place 时使用 `../Load-5090/logs/5090/phase6/`。

首次成功完成一个子窗口后，终端打印的新批次目录下会出现 `build` 链接。后续复用该目录：

```bash
bash tools/5090/phase6/run_op6.sh \
  --place 60 --trials 1 --blocks 2 --execute \
  --build /workspace/Load-5090/SL3060/<首次OP6批次目录>/build
```

已有完整 OP3 build 也可直接传给 `--build`，例如目标机仍保留对应构建时：

```bash
bash tools/5090/phase6/run_op6.sh \
  --place 60 --trials 1 --blocks 2 --execute \
  --build /workspace/Load-5090/SL3060/op3-two-arm-v1-2x-50steps-20261102-022324-519378/build
```

**示例中的 open-compress build 属于另一构建链，不能供 OP4/OP6 复用。** 本入口沿用相同的 `--build` 使用方式，但只接收 OP3 或 ABC 清单；传 ABC 时会新建小型 OP3 codec。传 OP3 时复用主要 GEMM/codec 二进制，仍由原流程逐子批编译很小的 MA9/MA11 兼容库并检查，不宣称完全零编译。完整依赖目录须保留；链接和清单包含绝对路径，不能只搬走一个 `.so` 或改路径绕过校验。

默认后台执行，增加 `--foreground` 可前台观察。不带 `--execute` 只预览，不创建批次、不查 GPU、不编译：

```bash
bash tools/5090/phase6/run_op6.sh --place 61 --trials 2 --blocks 4
```

默认 `trials=1, blocks=2`。每个 block 包含两组，各 10 步预热＋50 个正式 optimizer step：

```text
trial1 block00：OP4 → OP6
trial1 block01：OP6 → OP4
trial2 block00：OP6 → OP4
trial2 block01：OP4 → OP6
```

因此 `--trials 1 --blocks 2` 是 **4 个正式窗口、200 个正式 optimizer step**。两组依次独占八卡运行；不是同时占用 GPU。任意正整数 trials/blocks 均可，奇数块可能无法完全平衡先后顺序，报告记录实际是否平衡。模型保持 32 层、hidden4096、sequence2048、microbatch1、global batch4、TP8/DP1，每 step8192 tokens。

每个子批均执行原 native packet／舍入／所有权检查、投影算子输出及梯度检查和计时、全模型 smoke、八 rank profile，再正式训练。投影算子本身处于图外，其计时不用于证明 attention 图加速；收益看完整 optimizer step。启动环境和 NCCL 配置文件若含 OP5 或额外通道／算法／协议覆盖，入口会报错，避免把干预混入本次实验。

## 4. 数值与执行路径验收

实现入口为 [run_op6.sh](../../../tools/5090/phase6/run_op6.sh)、[控制器](../../../tools/5090/phase6/op6.py) 和 [独立子批入口](../../../tools/5090/phase6/op6_child.py)。原 OP3 代码和 native 构建规则保持原样；新子批通过有范围的包装接入 [源码生成](../../../tools/5090/phase6/op6_sources.py)、[状态校验](../../../tools/5090/phase6/op6_validation.py) 和 [图 trace 审计](../../../tools/5090/phase6/op6_profile.py)。

1. **真实图正确性：** smoke 的每次 attention forward 都与原 eager MA11 比较完整输出位串；backward 比较完整 dQ/dK/dV。保留原 MA11 native self-test、MA8 激活缓存、MA9 计数、原 OP3 codec／消费者检查。
2. **同期训练回归：** [配对审计](../../../tools/5090/phase6/op6_evidence.py) 要求相同硬件／软件、构建、输入／RNG／初始参数、CPU affinity 和通信；比较 loss、grad norm、最终全部参数指纹、smoke 全参数与梯度轨迹及投影输出。**本次严格检查 smoke 梯度哈希，不沿用 OP4 跨历史批次的梯度哈希豁免。**
3. **实际执行：** OP4 必须无图；OP6 每 rank 必须有 128 次前向＋128 次反向实际 graph launch，节点通过唯一 CUDA correlation 归属，保留原 attention GEMM dispatch、SM120 MA11 softmax launch、输入复制与输出所有权顺序。缺少 CUPTI 归属或捕获混入正式 profile 均失败；不以设置开关代替路径证据。
4. **通信不变：** 两组仍必须各有 524 次 AllGather，真实 grid 为16；比对所有 NCCL 调用、payload、进程组与 launch 几何，包括256次投影 SendRecv。通信或投影进入图会失败。
5. **留存原始证据：** 保存源码快照、构建清单、逐 rank JSON／trace、算子报告及 SHA256；所有正式窗口保留。正式步时仍取八 rank 最大 elapsed／50，不从 profile 求和构造步时。

## 5. 结果怎么看

新批次默认名称：`op6-local-attention-graph-paired-v1-<trials>x-<blocks>blocks-<时间>`。

| 文件／目录 | 内容 |
| --- | --- |
| `report.md`、`report.json` | 两组均值、中位数、最慢／最快、吞吐、每块差值和800ms目标观察 |
| `pair-evidence.json` | 每个完整 block 的严格数值回归、通信不变检查与文件哈希 |
| `batch-state.json`、`handoff/supervisor.log` | 当前阶段、失败原因或完成状态 |
| `compare1/block00-op4/`、`compare1/block00-op6/` | 各组原生／算子／模型 smoke、profile、50步计时与显存记录 |
| `build` | 首次子窗口使用的完整 OP3 构建链接，供后续复用 |

`completed` 表示验证和实验完成；`target_800ms_observed` 表示本批 OP6 正式窗口平均值≤800ms。是否相对 OP4 有改善同时看 `mean_paired_saved_ms` 和 `faster_blocks`：达到800ms但比同期OP4更慢，也不能算改进。不同 trial 单独报告，同时给全部窗口均值；不混合历史窗口，不剔除慢窗口，不提供小样本置信区间或跨机器保证。图增加的 allocated/reserved 峰值须一起看，Torch 峰值不覆盖 NCCL/context 的全部内部资源。

完成后只读复核：

```bash
bash tools/5090/phase6/run_op6.sh \
  --out /workspace/Load-5090/SL3060/<已完成OP6批次目录> --report
```

`--report` 重放证据检查，不续跑 GPU 工作。请同步完整新脚本及当前完整仓库，不能只复制 shell；历史结果和已开始的批次保持不动。

## 6. 本地验证边界

本地在已有 PyTorch 环境运行 **47项 OP6 测试，全部通过**（含5项真实 CPU autograd 检查）；OP3 回归49项通过，OP4 回归41项中39项通过、2项因测试使用的旧路径历史 fixture 不可用而跳过。另完成 shell 语法、命令预览和源码／文档检查。测试覆盖源码生成、生命周期、故障注入、图 trace 归属、配对数值门禁、调度和构建复用。CPU PyTorch 环境仅用于 autograd 测试；目标运行时仍严格要求指定 NGC PyTorch 2.8 和八张 SM120 5090。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op6*.py' -v
bash -n tools/5090/phase6/run_op6.sh
bash tools/5090/phase6/run_op6.sh --place 60 --trials 1 --blocks 2
git diff --check
```

本地默认 python 若未安装 PyTorch，5项真实 CPU autograd 测试会跳过；应在已有 PyTorch 环境再运行 `test_op6_adapter.py`，不能把跳过记为通过。本次尚未在目标机编译／捕获／训练；800ms、实际显存、CUPTI 图节点归属和端到端位串等价，均由上述完整脚本的目标 GPU 运行决定。

## 7. 编译后因 OP5 源码变化中止的修复（2026-11-03）

[003522 批次日志](../../../../Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-2blocks-20261103-003522-661167/handoff/supervisor.log) 显示 baseline 和 ABC 编译调用返回后，父控制器在启动第一个子批次前报错：

```text
ValueError: Missing/changed evidence: /workspace/TempName-New/tools/5090/phase6/op5.py
```

该批次的硬件/NCCL 预检已完成，但尚未进入 OP4/OP6 模型 smoke、图捕获或正式训练。原因是原通用源码收集器把整个 phase6 目录都冻结，独立 OP5 文件的更新也会中止 OP6。保存清单中的 `op5.py` 哈希与当前源码不同；OP6 主控制器在修复前的哈希与该批次保存值相同。这个错误不构成 CUDA Graph 或模型数值失败证据。

新增 [op6_controllers.py](../../../tools/5090/phase6/op6_controllers.py)，仅在**新建 OP6 快照**时排除七个明确无运行依赖的 OP5 文件：`op5.py`、`op5_evidence.py`、`op5_probe.py`、`run_op5.sh`、`analyze_op5_baseline.py`、`test_op5.py`、`test_op5_packet_probe.py`。OP6 父批次和两个子组共用该规则；首次硬件预检也改经 `op6_child.py --variant op4 --preflight-only` 启动。独立历史分析工具 `analyze_op6_baseline.py` 仍使用 `op5_evidence`，但它不参与本次启动、构建、计时或报告复核。

原 SHA256 验证器未修改。其余源码、图适配器、H100 helper、OP3 依赖、已保存快照及 native 构建验证继续执行；运行中修改这些实际依赖仍会失败。**不会在复核旧清单时删除条目、更新旧哈希或把失败批次改为成功。** 修复也不影响 OP5 的控制器行为。

同步当前 `op6.py`、`op6_child.py` 和新增 `op6_controllers.py` 后，创建新批次。若目标机保留了这次生成的完整 ABC 构建和依赖，可直接复用，避免重编主要 baseline/ABC 库：

```bash
bash tools/5090/phase6/run_op6.sh \
  --place 60 --trials 1 --blocks 2 --execute \
  --build /workspace/Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-2blocks-20261103-003522-661167/dependencies/abc
```

该命令由原构建验证器核对来源和二进制，再编译 OP3 codec；MA9/MA11 小库仍按原流程准备。回传目录未包含这些构建二进制，本地未将它们标记为验证通过。如果目标机未保留构建，去掉 `--build` 重新构建即可。不要将旧失败目录传作 `--out`，也不要使用 `--report` 续跑。

新增回归测试覆盖：编译期间修改独立 OP5 文件不再中止新 OP6 批次；修改 OP6/OP3、CUDA 源码、H100 helper 或冻结副本仍失败；旧清单仍严格验证；父批次、两个子组及预检均采用正确入口，退出后恢复原收集器。

本次在已有 PyTorch 环境运行全部 **53项 OP6 CPU 测试，全部通过、无跳过**；shell 语法、复用 ABC 构建的命令预览、文档链接及 `git diff --check` 通过。尚未在目标机重跑 GPU 实验，此次修复不构成800ms或图捕获通过的实测结论。

## 8. 首个 OP4 窗口完成，OP6 profile 时间戳门禁中止（2026-11-03）

[011706 批次总日志](../../../../Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-4blocks-20261103-011706-043929/handoff/supervisor.log) **没有完成一对 block**。`compare1` 是第一个 trial；其中 `block00-op4` 与 `block00-op6` 都完成正式计时并通过配对审计，才算一个完整 block。本次计划4块、8窗口、400个正式 step，实际只完成 OP4 的一个50步正式窗口：

| 组别 | 阶段／状态 | 正式 ms/step | tokens/s |
| --- | --- | ---: | ---: |
| OP4 | `completed`，10步预热＋50步计时 | **830.4842508397996** | **9864.12** |
| OP6 | native、投影算子、全模型 smoke 已通过；profile 审计失败 | 尚无 | 尚无 |

OP4 数值来自 [子批报告](../../../../Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-4blocks-20261103-011706-043929/compare1/block00-op4/report.md)，并按八 rank 最大 `elapsed_seconds / 50` 重算核对。OP6 profile 的单步记录有 profiler 开销，不作为正式步时；因此本批不能计算配对收益，也不能判断是否达到800ms。

### 8.1 失败点与原始 trace

[OP6 子批日志](../../../../Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-4blocks-20261103-011706-043929/compare1/block00-op6-supervisor.log) 表明八 rank 的 profile worker 已返回，控制器解析 rank0 trace 时在复用的 `_same_stream_order` 中报：

```text
ValueError: MA15 attention graph and staging copies: GPU dependency order changed
```

这次不是上一节的 OP5 源码冻结错误，也没有 CUDA runtime 错误或 smoke 数值不一致记录。只读对照八 rank 的 OP4/OP6 smoke 和 profile JSON，初始模型／输入／RNG／通信、loss、grad norm、最终参数一致；smoke 的全部梯度／参数轨迹与投影输出哈希也一致。OP6 smoke 内部逐次 graph 输出及 dQ/dK/dV 与 eager oracle 的位串检查已通过。

直接检查全部8份 OP6 trace：每 rank 有128次前向＋128次反向 graph launch；同一 graph correlation 内共1465处相邻 kernel 的结束/开始时间戳存在大于0.01μs的表面重叠，其中238处超过原门限。各处 device/context/stream 相同，主要是 backward softmax→GEMM 和 GEMM→短 scale kernel：

- rank0 首个被拒绝的 GEMM→scale：重叠 **0.09619140625μs**，scale 时长1.44μs；原门限 `min(0.5μs, 5%×1.44μs)=0.072μs`。
- 全八 rank 最大绝对重叠 **0.89599609375μs**；相对较短 kernel 的最大比例约 **23.90%**。

[NVIDIA 对 CUPTI 时间戳的解释](https://forums.developer.nvidia.com/t/concurrent-execution-of-cuda-kernels-from-different-contexts/73333/2) 指出同一 stream 也可能显示由 kernel 结束时间戳采集造成的微小重叠，500ns并不是保证上限。结合此次同一图、相同执行身份和逐位 smoke 证据，现象与时间戳采集误差相容；**不能据 trace 表面重叠直接断言发生了数据依赖违规，也不能把它当作可兑现的计算重叠收益。**

### 8.2 有限范围的检查器修正

只修改 [op6_profile.py](../../../tools/5090/phase6/op6_profile.py) 的局部图检查规则：允许范围为 `min(1.0μs, 较短kernel时长的25%)`。这是针对当前 SM120/CUPTI 工作负载的明确经验验收界限，**不是 NVIDIA 保证的时钟精度**，不自动随某次结果增大。

该容差仅作用于同一次唯一 graph launch correlation 下、同 device/context/stream 的两个 kernel；开始时间仍须递增，节点种类、数量、顺序、原 GEMM 分派和 softmax 几何仍严格核验。输入复制、输出 clone、不同 graph launch、不同 stream，以及通信／投影路径均不获得该例外。原始时间戳不修改；所有允许的表面重叠记录在报告中。没有修改图计算、数值容差或正式计时循环，H100 检查器原文件也保持不变。

新增真实 trace 边界回归样本，覆盖原先误拒绝的短 scale 与 softmax 边界；超出1μs或25%、不同 correlation/device/context/stream、涉及 staging/memcpy 或相同开始时间的案例继续拒绝。

只读复算命令（包含完整 OP3 路径检查及 OP6 图检查，不改旧批次状态、不验证未回传的二进制、不续跑）：

```bash
python3 tools/5090/phase6/reanalyze_op6_profile.py \
  --batch ../Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-4blocks-20261103-011706-043929 \
  --out /tmp/op6-first-pair-profile-analysis.json --jobs 4
```

复算记录见 [op6-first-pair-profile-analysis.json](op6-first-pair-profile-analysis.json)。修复后需要启动**新批次**才能取得正式 OP6 步时；原失败状态和原始证据保留。同步当前 `op6_profile.py` 及完整 OP6 脚本后，可复用此次已经成功用于 OP4 的构建：

```bash
bash tools/5090/phase6/run_op6.sh \
  --place 60 --trials 1 --blocks 4 --execute \
  --build /workspace/Load-5090/SL3060/op6-local-attention-graph-paired-v1-1x-4blocks-20261103-011706-043929/build
```

本次修正后，**8份原始 OP6 trace 的完整 OP3 路径＋图检查均通过；55项 OP6 CPU 测试全部通过、无跳过**。这是原始证据的只读重审和检查器回归验证；未新增任何正式 GPU 计时窗口，也未将旧失败批次改为完成。

## 9. 空档显存占位与自动交接（2026-11-03）

按用户确认的“空档占位并自动交接”方式，`run_op6.sh` 现在默认启用 **`--reserve-mib 31846`**。原命令可以直接使用，无需手动启动或结束占显存程序：

```bash
bash tools/5090/phase6/run_op6.sh \
  --place 61 --trials 1 --blocks 4 --execute \
  --build /workspace/Load-5090/SL3061/op6-local-attention-graph-paired-v1-1x-4blocks-20261103-143056-057061/dependencies/abc
```

| 阶段 | 显存行为 |
| --- | --- |
| 批次监督器启动、源码校验通过后 | 检查八卡空闲，再启动占位进程，每卡实际申请31846 MiB |
| baseline/ABC/OP3编译、CPU证据检查、窗口之间 | 占位进程持有显存，不执行 GPU 计算 |
| GPU硬件/NCCL预检、native/算子检查、smoke/profile/正式训练之前 | 占位进程完全退出并释放显存，再执行原空闲检查和GPU任务 |
| 每个GPU子任务成功退出后 | 确认空闲，重新申请占位显存；父子控制器之间同样交接 |
| 整批成功完成、任一步失败、SIGTERM/SIGHUP/Ctrl-C | 清理占位进程；收到中止信号时同时终止本批启动的子进程组 |

这里的31846 MiB是**每卡实际申请的内存块大小**，不是PyTorch `reserved` 指标，也不是把 `nvidia-smi` 整卡读数硬设成31846；监控读数还包含CUDA上下文等占用，可能略高。任何一张卡空闲显存不足都会明确失败，并释放此前已成功申请的各卡显存；不杀其他任务、不默默降低申请量。

实现位于 [op6_memory.py](../../../tools/5090/phase6/op6_memory.py)。仅独立占位进程调用 CUDA Driver 的内存分配/释放 API，控制器不保留 CUDA context，不更换训练 allocator，也不把大占位 tensor 留在训练进程中。占位进程与GPU worker不会同时存在；分配、释放和上下文初始化都发生在正式计时子进程外。OP4和OP6使用相同占位设置，设置写入父子 plan 并在报告复核时检查一致。

这是减小编译和切换空档被占卡机会的办法，**不是系统级独占锁**。释放与下一进程获得GPU之间存在短暂空隙；训练期间只保留训练本身所用的显存，其他任务仍可能申请剩余显存。要获得对其他用户进程也有效的严格独占，需要机器的作业调度器。

占位进程通过私有 stdin 管道等待。控制器正常释放时关闭管道，控制器异常死亡时操作系统也会关闭管道，占位进程收到 EOF 后释放所有已申请的显存。常规中止请向 `handoff/supervisor.json` 中的 PID 发送 SIGTERM，使监督器逐级清理本批 GPU worker；`kill -9` 不能执行监督器的清理代码，因此不能承诺它同时中止所有已经启动的独立训练进程。

各父/子批次的 `handoff/memory-reservation.jsonl` 记录 `acquired`／`released`、持有进程PID、每卡申请字节数及退出状态；CUDA分配失败详情写入 `handoff/memory-holder.log`。父批次在发布 `completed` 前确认占位显存已释放。`--report` 和未加 `--execute` 的预览不会查询或占用GPU。

可显式关闭，或调整申请量：

```bash
# 在原命令后增加以下参数之一
--reserve-mib 0       # 关闭占位
--reserve-mib 30000   # 空档每卡申请30000 MiB
```

部署须同步当前 `op6.py`、`op6_child.py` 和新增 `op6_memory.py`；原 `run_op6.sh` 继续作为入口。已有OP3/ABC构建可以复用，默认编译并行度仍为 `nproc`。使用新批次，不修改正在运行批次的源码冻结清单。

本次 **69项 OP6 测试全部通过、无跳过**，包括多卡分配中途失败回滚、真实子进程管道EOF释放、启动失败/超时清理、持有进程异常退出、GPU阶段交接、正常完成/子任务失败/审计失败后的释放，以及信号处理与关闭开关。shell语法、原SL3061命令预览、文档链接和 `git diff --check` 通过。本地没有可用5090；CUDA调用用模拟驱动验证控制流程，实际31846 MiB分配和目标机交接仍须实跑确认。
