# OP5：保留 OP4 的 16 通道，改用 SHM 接收端 copy engine

更新：2026-11-02。**方案与启动脚本已实现，尚未在目标 8×RTX 5090 上实测 OP5，不能声称已获得 50ms 提升。** 本轮沿用 OP4 的完整计算组合和 16 通道，把下一步优化放在共享内存传输的接收端搬运。目标是将完整 optimizer step 从指定批次的 **832.248790ms 降至不高于 782.248790ms**，即至少减少 50ms。只测一个固定候选，不自动扫描参数或挑选最快窗口。

## 1. 从指定 OP4 批次出发

参考：[OP4 设计与结果](op4-improve.md)、[5090 baseline](baseline.md)，以及用户指定的 [OP4 批次报告](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/report.md)。从原始八 rank elapsed 重新计算，两个完整窗口为：

| 正式结果 | block 0 | block 1 | 参考均值 |
| --- | ---: | ---: | ---: |
| ms/step | 833.486118 | 831.011461 | **832.248790** |
| 正式 optimizer step | 50 | 50 | 共 100 步 |

两窗口均保留；每窗口为 10 步预热＋50 步正式计时，步时取八 rank 最大 elapsed 除以 50。这里使用 OP4 父批次及其两个子批次，不再把 OP3 的 850.195ms 当作 OP5 的性能基准，也不使用先前失败的 OP4 批次。

重新解析两个窗口的 **16 份整步 trace**，按实际 CPU/autograd owner 归属通信，每 rank 先求和，再取八 rank 中位数：

| 每 optimizer step 的 profile 工作 | 次数/rank | block 0，ms | block 1，ms |
| --- | ---: | ---: | ---: |
| 全部 AllGather | 524 | 402.005 | 408.304 |
| MA8 QKV/FC1 前向 AllGather | 256 | 222.228 | 219.860 |
| row projection 反向 AllGather | 256 | 157.118 | 164.418 |
| 上述两个主要类别合计，先逐 rank 相加 | 512 | **381.030** | **385.919** |
| ReduceScatter | 260 | 157.736 | 159.960 |
| 投影 SendRecv | 256 | 164.787 | 174.231 |

512 次主要 AllGather 均未与同 rank 的非 NCCL 计算 kernel 重叠。全部 AllGather 中少量重叠来自其他 owner；不能将“全部 524 次”都描述为零重叠。两个主要类别的中位数之和，与先逐 rank 相加再取中位数不同，预算采用后者。

所有 AllGather 实际 grid 都已经是 `[16,1,1]`。这说明 OP4 的通道设置确实生效，但现有 profile 没显示它稳定缩短 AllGather；继续只从 16 调到 32 缺少依据。两个窗口、全部八 rank 的 AllGather 区间内均没有记录到 HtoD 拷贝。旧批次使用 WARN 日志，没有 SHM 连接的 INFO 证据，所以**这些旧记录不足以直接证明当时具体采用哪一种 host transport**。

这些时长含 rank 等待和 profiler 扰动，不能相加还原正式步时，也不是可直接兑现的收益。编码与解码已较短，单靠进一步消除它们没有足够的 50ms 空间；本轮保留既有 OP3 codec。

[只读分析工具](../../../tools/5090/phase6/analyze_op5_baseline.py)与[逐 rank 分析结果](op5-baseline-analysis.json)包含 trace 路径、SHA256、消息尺寸、调用数、launch 几何、计算重叠和 HtoD 诊断，可复算：

```bash
python3 tools/5090/phase6/analyze_op5_baseline.py \
  --batch ../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654 \
  --out /tmp/op5-baseline-analysis.json
```

## 2. 唯一候选：接收端 CE staging

候选继承 **A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 warp 编码器＋OP4 16 通道**，额外固定下列三个 SHM 参数：

| 配置 | OP4 历史记录 | OP5 |
| --- | --- | --- |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `16` / `16` | `16` / `16` |
| `NCCL_SHM_USE_CUDA_MEMCPY` | 未设置 | **`1`** |
| `NCCL_SHM_MEMCPY_MODE` | 未设置 | **`2`**，接收端 |
| `NCCL_SHM_LOCALITY` | 未设置 | **`2`**，显式固定默认接收端 locality |
| `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` / `NCCL_NVLS_ENABLE` | `1` / `0` / `0` | `1` / `0` / `0` |
| 正式训练 `NCCL_DEBUG` | `WARN` | `WARN` |

[NCCL 2.27.3 的 SHM 源码](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/transport/shm.cc#L58-L66)定义了这些参数；[接收端 progress 实现](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/transport/shm.cc#L378-L410)在 Simple 路径上提交 `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, ...)`，完成后发布就绪状态。上述参数属于本次固定版本的实验控制，探针要求实际 NCCL 为 **2.27.3**，不把它们当作跨版本通用配置。

候选的数据依赖为：

```mermaid
flowchart LR
    A[发送 GPU 的 NCCL kernel] --> B[主机共享内存 FIFO]
    B --> C[接收端 copy engine：HtoD]
    C --> D[接收 GPU 的本地 FIFO]
    D --> E[NCCL kernel 消费本地数据]
    E --> F[原 GEMM / 反向计算]
```

**性能假设：** 让接收 GPU 从本地显存消费数据，有机会减轻 NCCL SM 直接读取主机内存的开销，改善仍在依赖链上的大消息 AllGather。这里只是基于实现和现有 trace 提出的候选，旧数据没有证明直接 host read 已经是瓶颈。通信字节数、ring 转发次数和 PCIe/NUMA 物理带宽并未减少。

选择接收端模式 `2` 是为了只改变接收搬运；发送端仍保留 direct 路径。`SHM_LOCALITY=2` 显式锁定原默认值，不叠加 sender-locality 或 CPU affinity 调整。CE staging 也会增加 proxy 调度、memcpy/event 提交、本地 FIFO 和同步开销；16 通道分出的较小块可能使这些开销抵消带宽收益。这里没有预先把任何开销算成已经省下的时间。

设置作用于所有采用相关 SHM/Simple 路径的通信，可能同时影响 ReduceScatter、AllReduce、SendRecv。不能将最终整步变化全部归给 AllGather。也不从 kernel 名字里的 `_RING_LL` 推断协议；这个入口在当前 NCCL 中可以承载不同分派，实际 HtoD 活动才是本轮路径证据。

模型、TP8/DP1、32 层、hidden 4096、sequence 2048、micro batch 1、global batch 4、8192 tokens/step、FP8 packet 内容、原 source 归约顺序、BF16 舍入、消费者融合、MA8 缓存、梯度统计、Adam 和 stream 依赖均沿用 OP4。不缩短正式窗口、不关闭正确性检查、不启用 CUDA Graph、不引入新的近似量化。

## 3. 50ms 预算与结果解释

```text
历史均值 = (833.4861183352768 + 831.0114610381424) / 2
         = 832.2487896867096 ms/step
目标线   = 历史均值 − 50
         = 782.2487896867096 ms/step

如果仅由 512 次主要 AllGather 提供收益：
每次所需平均净减少 = 50,000 / 512 = 97.65625 μs
```

50ms 相当于约 **6.01% 的正式整步耗时**，或上述 381–386ms profile 量级的约 **13%**。后一个比例仅用于筛选目标难度：profile 与正式计时不同，既不能预测 CE 会节省多少，也不能按通信改善比例外推整步收益。其他通信的变化可能增加或抵消收益。

报告同时保留：

- 所有候选窗口的均值、中位数、最快/最慢值、tokens/s 和原始显存记录。
- `historical_delta_ms`：相对全部 OP4 历史窗口均值的观察差值。
- `historical_target_observed`：全部候选窗口平均步时是否不高于 782.248790ms；算子模式始终为 false。
- `target_validated=false`：当前入口只测候选，没有同期 OP4 对照，不能据此声称已经验证因果性的 50ms 增量。

若完成但未达线，仍报告全部结果；不会挑最快窗口、自动换设置重跑或把未达标写成达标。若探针或数值/路径审计失败，则父批次标记 `failed`，保留失败记录，不生成成功报告。后续要确认因果收益，需要另做同期 OP4/OP5 对照；本入口不偷偷增加旧组训练。

## 4. 测试启动命令

在目标机的 `TempName-New` 仓库根目录执行：

```bash
bash tools/5090/phase6/run_op5.sh \
  --op3-build /workspace/Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/build \
  --trials 1 --blocks 1 --execute
```

**一条命令自动完成：同尺寸 AllGather 探针 → 原有算子检查/计时 → 全模型 smoke/profile → 10 步预热＋50 步正式训练 → OP5 审计和报告。** 默认后台运行；加 `--foreground` 可前台观察。不带 `--execute` 只预览，不访问 GPU、不创建结果目录、不编译。

默认参考即用户指定的 OP4 父批次：

```text
../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/
```

若它存放在其他位置，加 `--reference-batch /path/to/completed-op4-parent-batch`，应传父目录，不能传其中 `block00-candidate`。入口校验该批次所有正式窗口和文件哈希，从原始 rank elapsed 重算均值；数值轨迹及 trace 对照固定使用第一个历史子窗口。没有重新运行历史 GPU 工作，也不声称重新验证了归档中缺失的历史二进制。

第一次完整测试建议保持上面的 `--trials 1 --blocks 1`；需要观察重复性时可使用 `--blocks 2`，它表示两个候选正式窗口，共 100 步。`--operator-only --foreground --execute` 会运行同尺寸通信探针和投影算子后结束，不运行完整模型，也不产生整步达标结论。

需要同步下列新增文件，以及已更新的 `op4_evidence.py`；其余依赖使用当前完整仓库，不能只复制 shell：

```text
tools/5090/phase6/run_op5.sh
tools/5090/phase6/op5.py
tools/5090/phase6/op5_evidence.py
tools/5090/phase6/op5_probe.py
tools/5090/phase6/op4_evidence.py
```

OP4 审计 helper 只新增可选的通信检查器、参考通道数和干预说明参数；不传参数时沿用原行为。OP3 worker、codec、GEMM 构建和训练入口未修改，可以复用命令中已有的 OP3 build。MA9/MA11 兼容库继续按原流程准备。

启动环境、`/etc/nccl.conf`、`~/.nccl.conf` 和 `NCCL_CONF_FILE` 中若有本轮通道、SHM CE、算法、协议或 tuner 覆盖，入口会明确报错，避免隐藏变量污染实验；脚本不会编辑配置文件。只在八卡空闲时运行，不与其他 GPU 实验并行。

## 5. 新增验证与证据位置

[控制器](../../../tools/5090/phase6/op5.py)和[启动脚本](../../../tools/5090/phase6/run_op5.sh)复用 OP3 的完整门禁，再通过 [OP5 审计](../../../tools/5090/phase6/op5_evidence.py)与历史 OP4 核对。

### 5.1 同尺寸通信探针

[op5_probe.py](../../../tools/5090/phase6/op5_probe.py)先执行原有硬件/Apex/NCCL 预检，再启动独立的八 rank 进程：

1. 使用与训练相同的 GPU、NCCL 2.27.3、通信配置和 `CUDA_DEVICE_MAX_CONNECTIONS=1`。每 rank 输入 `1,048,576` 个 BF16（2MiB），输出 `8,388,608` 个 BF16（16MiB）。
2. 三轮不同位模式输入，检查全体 rank 的完整输出位串；在非默认 stream 上复用缓冲、调用异步 Work 并等待，其中一轮延迟 rank7，覆盖数据就绪与复用依赖。
3. 20 次预热＋100 次正式 AllGather，记录 CUDA event 和同步墙钟平均延迟，报告八 rank 最大值；另做 10 次独立 profile。它是通信微基准，不是 optimizer step，不能直接将其改善乘以 512 作为正式收益。
4. 仅探针使用 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,SHM`。要求实际连接日志为 `SHM/direct/CE`，八 rank 均覆盖 16 通道；其他 transport 或不完整日志不能通过。正式训练仍使用原来的 WARN。
5. 每次 profile AllGather 必须实际启动 16 CTA，并包含同 device、完整落在该 kernel 时间区间内的 HtoD 拷贝，总接收量为七份远端贡献，即 **14MiB/次**。只出现环境变量或无关 memcpy 不能通过。

探针通过后才启动完整候选。它与训练使用独立 communicator，因此完整候选结束后的审计还会对八 rank 的 **524 次模型 AllGather 逐次检查同样的 HtoD 字节覆盖**。若当前 profiler 无法提供这些证据，批次失败并要求检查原 trace，不降级成“设置变量即成功”。HtoD 时长包含在等待它的 NCCL kernel 区间中，不能再加一次。

### 5.2 原有数值与路径门禁

继续执行 native packet/舍入/输出所有权、投影算子输出与四类梯度及三步更新逐位检查、模型 smoke、八 rank 全模型 profile、完整正式训练与原始文件哈希核验。

与 OP4 历史记录相比，只允许三个 CE 配置差异；保留硬件、软件、初始参数、输入、RNG、CPU affinity 和模型身份检查。loss、gradient norm、全部最终参数指纹、smoke 参数更新轨迹及投影输出仍要求一致。沿用 OP4 的既有规则：历史 smoke 梯度哈希差异记录但不阻断，哈希不能说明误差大小；没有把这一项改成新的放宽条件。

AllGather 调用数、BF16 消息尺寸、TP group、真实 launch、16-CTA grid 继续严格审查；其余 NCCL 入口、payload、进程组和调用数与 OP4 核对。完整候选必须同时通过原 OP3 replay 和 OP5 的 CE 审计。

### 5.3 输出与只读复核

新批次位于：

```text
../Load-5090/logs/5090/phase6/op5-shm-recv-ce-single-v1-<trials>x-50steps-<时间>/
```

| 位置 | 内容 |
| --- | --- |
| `report.md` / `report.json` | 正式步时、历史差值、50ms 目标线与达线观察 |
| `batch-state.json` / `handoff/supervisor.log` | 总体状态、失败原因与当前阶段 |
| `transport-probe/report.json` | 同尺寸 AllGather 延迟、实际 SHM/CE 连接和拷贝覆盖 |
| `transport-probe/process.log` / `rank*-profile.json.gz` | 探针原始 INFO 日志、八 rank trace；trace 也在 `transport-probe/` 下 |
| `candidate-evidence.json` | 历史数值回归、全部模型 AllGather CE 覆盖、原始文件哈希 |
| `compare1/block00-candidate/` | 原 OP3 子批：构建校验、算子、smoke、profile、50 步正式窗口 |

`completed` 表示检查完成，是否达到历史 50ms 目标另看 `historical_target_observed`。完整运行的每窗口显存记录位于子批 `report.json` 和逐 rank JSON；Torch allocated/reserved 不覆盖 NCCL 的新增 GPU FIFO、主机共享内存和 CUDA context，不能把它们当作整卡总占用。

对已完成结果只读复核：

```bash
bash tools/5090/phase6/run_op5.sh \
  --out /path/to/completed-op5-batch --report
```

`--report` 重新核对探针、子批、历史记录、trace 和源码快照，不续跑训练。结果及依赖 build 应保留在目标机；本地回传副本缺少二进制时，不能把只读历史分析当成完整 replay。

## 6. 本地验证与当前边界

本地已核对 OP4 两个正式窗口原始 elapsed、历史证据哈希，并解析全部 16 份整步 trace。**13 项 OP5 测试、41 项 OP4 回归、49 项 OP3 回归，共 103 项全部通过**；shell 语法、完整/算子模式预览、文档本地链接及文件空白检查通过。新增 CPU 测试覆盖候选独占配置、只预览无副作用、50ms 判据、历史双窗口均值、数值门禁、CE 完整字节覆盖、真实 transport 日志、失败中止、探针先于训练，以及报告与探针证据的绑定。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op5.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op4*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op3*.py' -v
bash -n tools/5090/phase6/run_op5.sh
bash tools/5090/phase6/run_op5.sh --trials 1 --blocks 1
git diff --check
```

**当前没有 OP5 GPU 正确性通过记录，也没有 OP5 算子或整步实测值。** 这份方案选择了仍有足够时间空间的通信路径，并提供可运行、可否证的实验；是否净省至少 50ms，以目标机完整运行的结果为准。
