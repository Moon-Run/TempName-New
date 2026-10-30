# RTX 5090 九组 NCCL staging v2 baseline：MA11 / MA89 / MA8 / MA9 / CR3 / R / C / A / Flux＋基础量化

更新：2026-10-30。目标机器为**单机 8×RTX 5090，TP8、DP1**。本页定义九组同场对比，并提供与 `bash tools/h100/phase6/run_ma19.sh --execute` 相同使用方式的 [5090 入口](../../../tools/5090/phase6/run_baseline.sh)。

**当前状态：代码改为不依赖 P2P 的 NCCL staging v2，尚无 RTX 5090 训练实测结果。** 2026-10-30 用户回传的官方 `nvcr.io/nvidia/pytorch:25.06-py3` 临时容器检查确认，8 张 RTX 5090 均为 SM120 / 170 SM，但全部 56 条跨卡有向 peer-access 链路不可用。新的投影路径为 **本卡 producer 写入本地 packet → NCCL All-to-All → 本卡 decoder**，不再把其他 GPU 的地址交给 CUDA kernel。P2P 矩阵仍记录在预检证据中，但不再要求链路可用。

本轮实验协议为 `rtx5090-phase6-nine-arm-nccl-staged-v2`，传输协议为 `rtx5090-nccl-staged-v2`；旧的直接访存 v1 构建与结果不能复用。2026-10-30 目标机回传日志已推进到 `prepare-workers`，此前硬件/NCCL 预检及七组原生构建已完成；本轮在 Python 3.12 的 AST 校验处中止。数值检查、路径检查及正式计时仍须在目标机执行，不把 H100 耗时填作 5090 耗时。

## 1. 对比目标与 H100 参考

[H100 baseline](../design-h100/New-Mechanism/baseline.md) 中 MA11 的完整 optimizer step 平均耗时为 **450.662 ms/step**，原始值为 `450.6624015075795 ms`。该页取 MA11 首轮已复核批次 `ma11-v1-causal-softmax-1x-50steps-20261027-121342-060643`；本页保留机制来源和量化算术，使用新的 staging 传输实现，不把 MA19、MA20 或 MAV1 算作 MA11。

| 指标 | MA11 | MA89 | MA8 | MA9 | CR3 | R | C | A | Flux＋基础量化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 历史平均耗时（ms/step） | **450.662** | 459.806 | 474.004 | 489.878 | 501.433 | 515.716 | 516.784 | 536.692 | 582.318 |
| 5090 同场平均耗时（ms/step） | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5090 同场中位数（ms/step） | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5090 吞吐（tokens/s） | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5090 MA11 相对各组均值耗时下降 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

H100 行来自历史多批次汇总，供背景参考；5090 待测行将填写本次九组同场实验。H100 原表的“选定窗口耗时”混合了均值、中位数和最慢值，**本实验九组一律用全部正式窗口的算术均值计算主表降幅**。跨机器比值同时包含 GPU、互联、CUDA/PyTorch、GEMM 和传输后端差异，不表示某一机制的净收益。结果显式记录 `original_mechanisms_not_identical=true` 和 `original_peer_overlap_preserved=false`。

## 2. 九组实现与量化范围

| 展示名称（均为 staged 迁移版） | 脚本组名 | 原机制 ID | 新后端中保留的比较内容 |
| --- | --- | --- | --- |
| MA11 | `ma11` | `F4-cache-zcount-csoftmax` | 完整 MA89，加注意力 mask/softmax 中间态消除；保留原归约树、BF16 舍入和 QK/PV 算术。 |
| MA89 | `ma89` | `F4-cache-zcount` | CR3，加 MA8-v2 前后向激活复用和 MA9 保序多张量零梯度统计。 |
| MA8 | `ma8` | `F4-cache` | CR3，加 MA8-v2 激活缓存；QKV/FC1 反向复用前向完整输入，减少重复 AllGather。 |
| MA9 | `ma9` | `F4-zcount` | CR3，加分块计数与保序汇总两个零统计 kernel；不改梯度范数、裁剪和 Adam。 |
| CR3 | `cr3` | `F4` | decoder 内融合 bias/residual，保留两次 BF16 舍入和原 LayerNorm；使用本地 producer 的提交缓存。 |
| R | `r` | `R`（MA2-R） | 保留原 decoder 与全远端量化；由新前端持有实例配置和独立输出。 |
| C | `c` | `C` | 保留冻结 swizzle、选择性量化 mask、双工作区和实例配置；不保留原跨卡到达时序。 |
| A | `a` | `A` | 全远端量化；复用持久 CUDA 工作区，但每次调用重新配置 native producer，作为新后端中的提交方式对照。 |
| Flux＋基础量化 | `flux-basic` | `basic` | Flux GemmOnly、独立 TACO CUDA 编码、NCCL All-to-All、接收端融合解码/归约 **v2**。 |

量化均覆盖 32 层 Attention 输出投影与 MLP 输出投影，每 rank 共 64 个目标模块；本地贡献保持 BF16。MA11/MA89/MA8/MA9/CR3/C 使用原 `1/64` 预算上限的选择性量化 mask，R/A/Flux＋基础量化使用全远端量化。MA11 的注意力 bool mask 与量化选择 mask 是两个不同对象。

选择性量化各组共用**冻结的 H100 到达表和 mask**，用于固定算法条件下的跨架构移植对照；这些表不代表 5090 校准结果。若后续在 5090 上重新采样到达顺序或选择 mask，须新建协议和结果批次，不能与本次混算。

原 H100 的 A/R/C/CR3 主机提交、IPC 屏障和工作区生命周期与新后端不同；上表是新后端中的对应消融，不能把 R/A 等差值解释为原 IPC wrapper 的逐调用成本。MA8 激活复用、MA9 保序零统计、MA11 softmax 及 decoder BF16 算术继续接受独立的数值和实际执行路径检查。

原 MA11 Python helper 锁定 PyTorch 2.6；5090 生成副本改为绑定预检记录中的精确 PyTorch 版本，包括目标 NGC 25.06 的 `2.8.0a0+5228986c39.nv25.06`。原 Megatron 类源码身份、native softmax self-test，以及完整模型 smoke 中每次前后向的逐位对照均保留。版本门禁调整不代表新 PyTorch 的 softmax 已证明等价；任何精确检查失败都会阻止正式计时。

## 3. 本地 packet、NCCL 传输与同步

H100 Phase6 的量化投影使用 CUTLASS V2，内部注册键为 SM80/A100。5090 构建将机器码编译为 **`sm_120`**，本地 producer 使用冻结 GEMM/epilogue 算术和 swizzle，将按目的 rank 打包的内容写入本 GPU 的发送缓冲，不构造 CUDA IPC peer 地址或跨卡 GroupBarrier。NCCL 交换完成后，decoder 只访问本 GPU 的接收缓冲和本地 BF16 贡献，保留原 source 归约顺序与舍入。

生产、交换、消费按 CUDA stream 依赖排序：先完成本地 producer 的 packet 写入，再提交字节 All-to-All，等待通信的 stream 依赖后启动本地 decoder；工作区再次写入前必须排在上次消费之后。失败的实例不继续复用。**这一版本不尝试恢复原 GEMM tile 尚在产生时跨卡发送和到达优先消费的重叠**，通信开始时间与源端排序的性能意义都已改变。

全部九组冻结 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`，保留并记录其他 NCCL/NIC 设置。这里允许 NCCL 使用主机中转，不把环境变量本身作为“实际采用 SHM”的证据；具体 transport 由 NCCL/目标系统选择。

Flux＋基础量化使用适合 5090 的 **GemmOnly V2** 后端；H100 该列的 Hopper V3/GMMA 后端不能直接用于 5090。因此这一列保留基础量化 v2 的算法定义，但不能解释为与 H100 使用了同一个 GEMM kernel。

执行前置检查要求：

- 恰好暴露 8 张不同的 RTX 5090，计算能力为 `(12, 0)`，每卡 170 SM，且没有占用 GPU 的其他计算任务。
- CUDA Toolkit 至少 12.8，PyTorch CUDA 构建能够实际执行 SM120 运算。Apex 等 Megatron 所需扩展须在相同 PyTorch/CUDA 环境中重新编译并可导入。**不要用 `env/setup_h100.sh` 把 5090 环境安装回 H100 的旧版 cu124 环境。**
- 不要求 CUDA peer access；预检记录完整矩阵，并启动真实八 rank NCCL 检查。BF16 AllGather、AllReduce、ReduceScatter，以及非等长 `uint8` All-to-All（self split 为 0）必须全部正确。探针在非默认 CUDA stream 上循环三次并等待异步 Work，记录每个 rank 的 GPU、软件环境和通信配置。
- 仓库中的冻结源码、选择表、相应依赖以及指定的 Megatron 源码完整可用。构建器按源文件身份校验，在新的输出目录生成 SM120 二进制，不复用 H100 的 `.so`。

新增本地 staging 需要显存。按当前选择性量化配置估算，每个 slot 的发送/接收等 packet 工作区约 77 MiB，64 个目标模块、双 slot 约为 9.6 GiB 量级；还需叠加模型、optimizer、激活缓存和临时张量。32 GiB 5090 上的实际峰值必须通过完整 smoke 确认，代码不会为绕过 OOM 自动缩小 batch、关闭 MA8 缓存或减少模型层数。

GEMM 私有 workspace 使用独立 `cudaMalloc`，不在 PyTorch allocator 峰值中。每个 rank 另记所有 TACO slot 的 `native_workspace_bytes`，报告同时展示 Torch allocated 峰值及逐 rank 加上 TACO workspace 后的最大值；该合计仍不包含 CUDA context、NCCL 等内部显存，不能视为整卡总占用。

环境/Apex 检查通过尚不能完成预检，必须取得八个成功的 NCCL rank 记录。随后依次执行 native mapping、八组 TACO 原生传输 smoke、九组完整模型 smoke、九组 profile，再开始正式计时。原生传输 smoke 不依赖模型执行，覆盖 Attention/MLP 两个真实形状、零值/随机/离群输入、56 条有向数据包 SHA256 互证、pack/unpack 位串检查、消费者融合与普通解码后两次 BF16 add 的逐位一致，以及 rank 延迟和七次复用后的六个输出存储保持。

### 目标机回传检查（2026-10-30）

证据来源为用户在目标机执行临时容器命令后粘贴的终端输出，不是本地执行记录或完整训练报告。

| 检查项 | 回传结果 |
| --- | --- |
| 镜像 | `nvcr.io/nvidia/pytorch:25.06-py3`，NVIDIA build `177567386` |
| Python | `3.12.3`，`/usr/bin/python` |
| PyTorch | `2.8.0a0+5228986c39.nv25.06` |
| CUDA | PyTorch linked CUDA `12.9`；nvcc `V12.9.86` |
| cuDNN | `91002`（9.10.2） |
| GPU 架构 | PyTorch 编译架构包含 `sm_120` 和 `compute_120`；8 张 RTX 5090 均为 `(12, 0)` / 170 SM |
| 跨卡 peer access | **0/56 可用**；矩阵仅对角线为 1，对角线不计作跨卡链路 |
| Apex 导入 | `apex_C`、`amp_C`、`fused_layer_norm_cuda` 均成功；尚未验证实际 GPU kernel 执行 |

NVIDIA 在 [RTX 5090 P2P 问题的公开答复](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8)中说明 GeForce RTX 50 系列不支持 GPU 间 P2P；本机回传结果与之相符。这份证据解释了旧版直接访存方案的阻塞，是本次新增 staging 路径的原因，不是新版 NCCL 探针或训练已通过的证据。

普通 NCCL 通信可在不能使用 P2P 时使用主机共享内存等传输方式；`NCCL_P2P_DISABLE=1` 仅禁用 NCCL 的 P2P transport，不会改写应用中自定义 CUDA kernel 的远端指针访问。[NCCL 环境变量说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2276/user-guide/docs/env.html#nccl-p2p-disable)

新的统一入口已把八卡 NCCL 集合通信作为正式前置检查。`Flux＋基础量化` 保留 GemmOnly＋独立编码＋NCCL All-to-All＋接收端融合 v2；其余八组使用本地 fused producer 和 staging 传输。后续回传的批次 `baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-151053-241177` 已通过前置检查并完成七组原生编译/链接，随后在 `prepare-workers` 报出 `Original training functions changed`。该次尚未进入原生传输 smoke，两类投影路径仍待目标机验证。

## 4. 一键执行

在 `TempName-New` 仓库根目录、已准备好的 5090 Python 环境内执行：

官方 25.06 镜像沿用自身的 CUDA 12.9、Python 和 PyTorch；为运行新协议，需要更新完整仓库中的脚本和原生源码，不能只复制入口 shell。

已有 `hjr/TempName-New` 和 `hjr/Megatron-LM` 时，在容器内的 `TempName-New` 根目录先运行环境配置脚本。它复用现有源码，创建继承镜像包的轻量 venv，补齐缺失依赖并执行八卡预检，不重装 PyTorch/CUDA/Apex，也不启动训练：

```bash
bash env/setup_5090.sh
source env/activate_5090.sh
```

测试入口会自动读取生成的 Python、CUDA 和 Megatron 路径；参数、日志、软件预配置模式见 [5090 环境说明](../../../env/README_5090.md)。已配置环境无需再手动重设 `RTX5090_PYTHON` 为镜像系统解释器。

```bash
# 仅打印执行计划；不查询 GPU、不构建、不启动训练。
bash tools/5090/phase6/run_baseline.sh

# 只做硬件、Apex 实际运算和八 rank NCCL 检查；不编译 Flux 或训练模型。
bash tools/5090/phase6/run_baseline.sh --execute --preflight-only

# 构建并检查新的原生传输；通过后停止，不运行完整模型和正式计时。
bash tools/5090/phase6/run_baseline.sh --execute --transport-smoke-only --foreground

# 九组完整批次：预检查、构建、原生/模型 smoke、profile、正式计时与报告。
bash tools/5090/phase6/run_baseline.sh --execute
```

`--preflight-only` 无需 Megatron 源码；`--transport-smoke-only` 使用统一构建器，首次构建仍需实验指定的 Megatron 源码及依赖，也可以通过 `--baseline-build` 复用已完成的新协议构建。每次使用新的输出目录。仅硬件/NCCL 预检成功并不会生成原生构建。

若旧版脚本在 Python 3.12 的 `prepare-workers` 阶段报 `Original training functions changed`，原因是 [Python 3.12 为函数和类 AST 新增了 `type_params` 字段](https://docs.python.org/3.12/library/ast.html#ast.FunctionDef)，普通函数也会带 `type_params=[]`，无法直接匹配 Python 3.10 生成的冻结 AST 哈希。5090 的 `baseline_sources.py` 已在局部解析树中统一这个空字段；原训练函数和计时循环哈希保持不变，非空类型参数及实际代码修改仍会被拒绝。无需更换镜像或重装环境。

同步修复后的 `tools/5090/phase6/baseline_sources.py` 后，可复用上述批次已完成的构建重试；入口会校验旧构建并创建新的结果目录：

```bash
bash tools/5090/phase6/run_baseline.sh \
  --execute --transport-smoke-only --foreground \
  --baseline-build /workspace/TempName-New/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-151053-241177/build
```

与 H100 的 `run_ma19.sh` 类似，默认执行模式交给后台控制进程；使用 `--foreground` 可在当前终端运行。后台任务持续运行要求容器保持运行；如果容器的主进程是交互式 bash，退出该 bash 会停止容器及其子进程。可在持续运行的容器中通过 `docker exec` 启动测试，并查看批次状态与阶段日志。

显式指定 Python、CUDA 和 Megatron 路径：

```bash
RTX5090_PYTHON=/path/to/5090-env/bin/python \
bash tools/5090/phase6/run_baseline.sh \
  --execute \
  --cuda-home /usr/local/cuda \
  --build-cuda-home /usr/local/cuda \
  --megatron /path/to/Megatron-LM \
  --out logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-01
```

`--python` 也可指定 worker 解释器；`RTX5090_PYTHON` 决定入口解释器，建议二者使用同一个环境。默认 `--trials 1 --blocks 18`；可使用 `--trials 2` 独立重复，或将 `--blocks` 设置为 18 的正整数倍。`--jobs` 控制编译并发，`--worker-timeout` 控制单个八 rank worker 窗口的超时。

只有新 native 协议 `rtx5090-nine-arm-native-nccl-staged-sm120-v2` 已完成且验证通过的构建才能复用；原 direct-peer v1、H100 和只准备源码的构建都会被拒绝：

```bash
bash tools/5090/phase6/run_baseline.sh \
  --execute --baseline-build /path/to/previous-batch/build \
  --out logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-02
```

离线复核原始结果、重新分析 profile 并打印报告（不改写原结果文件）：

```bash
bash tools/5090/phase6/run_baseline.sh \
  --report --out logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-01
```

## 5. 统一计时、正确性与结果文件

模型保持与 H100 对齐：32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1；固定模型和 token seed，每个 optimizer step 为 **8192 token**。计时包含前向、反向、通信、optimizer 更新与清梯度，不启用 CUDA Graph。

每组每窗口 **10 步预热＋50 步正式计时**。一个完整块包含九组；默认采用 9 次位置轮转加对应逆序的 **18 块平衡设计**，预先确定执行顺序。每组在每个执行位置出现两次，任意两组的前后关系平衡。每个 trial 共：

| 工作量 | 数量 |
| --- | ---: |
| 每组正式窗口 | 18 |
| 每组正式 optimizer step | 900 |
| 九组正式窗口 | 162 |
| 正式 rank 记录 | 1296 |
| 正式 optimizer step | 8100 |

以上数量不含原生传输 smoke、模型 smoke、预热和 profile；模型 smoke 每组 2 步预热＋3 步检查，profile 独立运行并在正式计时前通过，不混入正式步时。

- **窗口耗时：** `max(8 个 rank 的 elapsed_seconds) × 1000 / 50`。
- **主表耗时：** 每组全部 18 个窗口的算术均值；另报中位数、最慢窗口及显存峰值，保留所有有效慢窗口。
- **吞吐：** `8192000 / 平均 ms`。
- **MA11 主表降幅：** `100 × (1 − MA11 平均 ms / 对照组平均 ms)`；正数表示 MA11 更快，负数照常保留。
- **同期配对收益：** 按完整块配对，使用对数耗时比及 10,000 次 bootstrap 给出 95% CI；与主表算术均值比单独报告。区间描述该固定工作负载，36 个两两比较未作多重比较校正。

同一 trial 中各组保持相同初始参数、输入和运行环境。MA11/MA89/MA8/MA9/CR3/C 在组内检查 loss、梯度和参数更新一致性，R/A 在另一组内检查；不同量化策略不强行要求逐位相等。基础量化 v2 单独检查 codec 和输出所有权。各组还检查实际模块覆盖、无 fallback、完整梯度/参数轨迹及相应 profile 路径，防止只改组名而优化未启用。

每个批次保留 `environment-preflight.json`（硬件/Apex/NCCL 检查汇总）、`preflight-communication.json`、`nccl-preflight/rank*.json`（八 rank 集合通信原始记录）、`local-smoke/<arm>/rank*.json` 与 `local-smoke-summary.json`（八组 TACO 原生传输检查）、`preflight.json`（九组模型 smoke）、构建清单、worker 配置、逐 rank 训练记录、profile 证据及阶段日志；最终生成 **`report.json` 和 `report.md`**。后台状态看 `batch-state.json`，控制进程日志看 `handoff/supervisor.log`。`--transport-smoke-only` 成功状态为 `transport_smoke_completed`，不生成训练性能报告。离线复核检查 NCCL/原生传输探针证据、逐 rank 完成状态、软件/硬件身份及固定通信环境；失败或缺失的窗口不能计入已完成报告，收益为负不会提前停止。完成实机测试后，应以该批次报告及原始记录回填第 1 节的 5090 待测行，同时记录批次目录、驱动、PyTorch/CUDA、NCCL 和 PCIe/P2P 拓扑。

相关实现：[入口](../../../tools/5090/phase6/run_baseline.sh)、[九组协议与统计](../../../tools/5090/phase6/baseline_protocol.py)、[八 rank NCCL 探针](../../../tools/5090/phase6/baseline_collectives.py)、[隔离构建](../../../tools/5090/phase6/baseline_build.py)、[worker 生成](../../../tools/5090/phase6/baseline_sources.py)。
