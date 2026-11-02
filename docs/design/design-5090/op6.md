# OP6：在 OP4 上重放局部 Attention CUDA Graph

更新：2026-11-03。**本轮选择减少本地 attention 的重复主机提交，在 OP4 上只增加每层 attention core 的前向／反向 CUDA Graph。已实现方案和同期 OP4／OP6 对比脚本，目标为完整 optimizer step ≤800 ms；尚无 OP6 的 5090 GPU 正确性或性能实测，不能声称已达到目标。**

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
