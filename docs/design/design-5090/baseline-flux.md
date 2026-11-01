# RTX 5090 Flux 四组量化对比

更新：2026-11-01。目标机器为单机 **8×RTX 5090，TP8/DP1**。本实验按要求从**旧九组 baseline 的 C/A**拆出量化对照，另比较 Flux＋基础量化和 Flux 无量化。独立入口为 [run_baseline_flux.sh](../../../tools/5090/phase6/run_baseline_flux.sh)，协议为 **`rtx5090-phase6-flux-quant-four-arm-v1`**。

**当前状态：四组实验已有完成批次 `20261101-203719-996417`；后续批次 `20261101-231152-305302` 在硬件预检时遇到 CUDA OOM。** 本次失败原因和重跑方法见第 8 节，早期 native 检查器错误及修复见第 7 节。“改进量化”是本组的命名，不预设一定快于其他组。旧 [baseline](baseline.md) 和 [OP1-A](op-1-A.md) 的实验数据不混入四组报告。

## 1. 四组定义

| 展示名称 | 脚本组名 | 从哪里派生 | 实际路径与保留内容 |
| --- | --- | --- | --- |
| **Flux＋改进量化** | `flux-improved` | 旧 C，去掉交替排序、到达优先和选择性量化 | 原 Flux rank-offset 排序；全部远端贡献量化；保留 C 的融合 producer、warp decoder、实例配置、双工作区和独立输出。 |
| **Flux＋量化融合** | `flux-fused` | 旧 A，去掉交替排序 | 原 Flux rank-offset 排序；全部远端贡献量化；保留 A 的融合 producer、scalar decoder、单工作区、逐调用配置和 adapter 输出 clone。 |
| **Flux＋基础量化** | `flux-basic` | 现有 5090 基础量化 v2 | 原 Flux GemmOnly＋独立 TACO CUDA 编码＋NCCL 字节 All-to-All＋接收端融合解码/归约 v2。 |
| **Flux（无量化）** | `flux-bf16` | 同一个 5090 GemmOnly 构建 | 原 Flux GemmOnly＋NCCL BF16 Reduce-Scatter；无编码、解码、选择 mask、量化 packet 或 All-to-All。 |

三个量化组都覆盖 32 层 Attention 输出投影和 MLP 输出投影，每 rank 共 **64 个模块**。远端贡献全部采用原 H128/FP8 codec，本地贡献保持 BF16。无量化组的本地和远端贡献均为 BF16。

**“去掉选择性量化”指取消稀疏选择策略，改为全部远端量化，不是将 C 变成无量化组。** 新 C 的 mask 指针为 `nullptr`；原 `taco_selected` 在这个条件下对每个远端贡献返回 true，对本地贡献返回 false。worker 不读取历史 mask 或到达表。

本轮以旧 C/A 为消融来源，**不加入 OP1-A 的 direct-wire、删除 pack/unpack 或删除 A 配置/clone 等额外改动**。这样 C/A 之间仍可观察原解码方式、配置和工作区生命周期的组合差异。也不叠加 R、CR3、MA8、MA9、MA11 或 CUDA Graph。

## 2. 实现边界

### 2.1 旧 C/A 恢复原 Flux 排序

[flux_baseline_native.py](../../../tools/5090/phase6/flux_baseline_native.py) 分别从已验证九组构建的 `selective/c` 和 `selective/a` 复制登记源码，使用仓库原始的 `ThreadblockSwizzleStreamKRankOffset` 替换实验排序头文件。

保留原 Flux 基础 tile 映射以及 rank 的 M 维偏移：

```text
(base_m, base_n) = 原 Stream-K tile 映射
tile_m = (base_m + tiled_m / world * rank) % tiled_m
tile_n = base_n
```

不再使用 `dst=(i%8+rank)%8` 的目的 rank 交替映射，也不读取 `phase6_device_lookup` 到达表。原 Flux 的 rank-offset 是其默认分工方式，本实验没有把它替换成新的交替算法。

CUDA mapping 检查覆盖两个目标形状、八个 rank 和每个输出 tile，验证完整覆盖、无重复及上面的原始映射；实际 GEMM 注册增加 `FluxBaselineOriginalOrder` 名称标记，profile 检查每次目标 GEMM 确实来自这一构建。源码和二进制哈希同时固定，不能只改 Python 组名通过验证。

### 2.2 C 去掉到达与选择性策略，保留原 decoder

新 C 将 local transport 的 `RTX5090_LOCAL_SELECTIVE` 改为 0，禁止传入非空 mask。其 producer、量化公式和 decoder 源码保持旧 C：TP8 走 `taco_decode_ring_kernel<false>`，每 warp 负责一个 H128 group，每个 CTA 处理四组。

这与旧 A 的 scalar decoder 是不同实现。保留原 C 的双工作区轮转、实例配置和独立输出；不借用后续 R/CR 的预取 decoder。新 C 不再带选择 mask 的工作区字段，工作区验证只接受形状和 packet 指针。

### 2.3 使用 5090 已验证的 NCCL staging

本机跨卡 peer access 为 0/56，C/A 继续使用旧 5090 staged 实现：

```text
本卡 GEMM epilogue 融合量化
  → 本卡 pack
  → NCCL 字节 All-to-All
  → 本卡 unpack
  → 原 C 或 A decoder
```

双方均不恢复 H100 的跨卡 tile 到达及计算通信重叠。全远端量化时，每个目的 packet 含 `8192×136 = 1,114,112` 字节，每 rank 七个远端 packet 共 **7,798,784 字节**，self split 为 0。C/A 的 tile wire 与基础量化的 SoA wire 布局不同，但这三个量化组的逻辑发送字节预算相同。

共享 scratch、接收 packet 的私有存储、NCCL `Work.wait()` 的 stream 依赖及失败后禁止复用等检查沿用旧后端；不同工作区 slot 不共享 scratch。C 使用双 slot，A 使用单 slot；这些差异按原机制保留并单独计数。

### 2.4 无量化组的准确含义

无量化组与基础量化组使用**同一份 Sm89/RTX5090 GemmOnly V2 / StreamkSK 二进制**，实际机器码为 SM120。GEMM 产生完整 BF16 partial 后，无量化组调用 `torch.distributed.reduce_scatter_tensor`；每次分配独立的 `256×4096` 输出。

这是适用于当前无 P2P 机器的 Flux 无量化实现，不是 H100 上直接访问 peer 的原生 GemmRS 路径。代码见 [flux_baseline_bf16.py](../../../tools/5090/phase6/flux_baseline_bf16.py)。保留适配器要求的类名只是接口兼容；模块不加载 codec 动态库，不调用 encoder/decoder，不分配 packet，也不执行 All-to-All。

记录中的 BF16 逻辑远端贡献为 `7×256×4096×2 = 14,680,064` 字节；这个值便于描述数据量，**不是对 NCCL 实际链路字节数的测量**，不能直接据此推断通信耗时。

## 3. 允许归约差异，保留数值与训练门禁

沿用本次讨论的要求，允许排序、GEMM、量化及 NCCL 归约产生不同舍入，**不要求四组 loss/梯度/参数轨迹逐位相同**。每组仍必须通过独立数值检查：

- 三个量化组：与未量化 BF16 参考的相对 L2 误差不超过 **5%**，与相同全远端 codec 参考的相对 L2 误差不超过 **1%**；保留零值、随机、离群输入的独立 native 检查。
- 无量化组：以独立 `torch.matmul`＋NCCL BF16 Reduce-Scatter 为参考，实际相对 L2 上限为 **1%**；不与量化输出比较，也不执行“codec 自检”。
- 完整模型保留有限 loss/gradient norm、64 模块覆盖、零 fallback、全部可训练参数的 smoke 梯度/更新记录、初始权重/RNG/token 身份和输出所有权检查。
- 六次返回的输出必须保持独立存储，并在第七次工作区复用之后仍不被覆盖。C/A native 探针还验证两个实例共享 scratch、交替提交和关闭一个实例后另一实例继续正确执行。

报告为全部六组配对保留逐窗口 loss、gradient norm 最大绝对/相对差，以及输出/最终参数是否逐位相同的观测。数值不相同不会单独阻止计时；非有限值、超出误差预算、漏算或错误执行路径仍会停止。短程固定负载通过不等于长期收敛或模型质量等价。

## 4. 一键运行

在目标 5090 容器的 `TempName-New` 根目录、已配置的环境中运行。沿用 `env/setup_5090.sh` / `env/activate_5090.sh` 的 Python、CUDA 和 Megatron 配置，更新整个 `tools/5090/phase6/`，不能只复制入口 shell。

```bash
# 仅预览，不查询 GPU、不创建结果目录、不构建或训练。
bash tools/5090/phase6/run_baseline_flux.sh

# 四组完整批次，默认后台运行。
bash tools/5090/phase6/run_baseline_flux.sh --execute
```

已有完整九组构建时，建议显式复用，只新编译两份去掉排序的 A/C 变体；原 GemmOnly、基础 codec 和 Megatron 快照保持共用：

```bash
bash tools/5090/phase6/run_baseline_flux.sh --execute \
  --baseline-build /workspace/TempName-New/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-151053-241177/build
```

上面的路径与旧 baseline 文档的已用目标机构建相同；如果实际路径不同，请替换为目标机的完整 `build` 目录。只有日志而缺少二进制的回传归档不能当作构建复用。不指定时会先创建冻结的基准构建，再创建本轮 A/C 变体，不需要重新执行九组训练。

```bash
# 只做硬件、Apex 实际运算和八 rank NCCL 预检。
bash tools/5090/phase6/run_baseline_flux.sh --execute --preflight-only

# 构建后只做 A/C 原排序 mapping 和全远端传输检查。
bash tools/5090/phase6/run_baseline_flux.sh --execute --foreground \
  --transport-smoke-only --baseline-build /path/to/verified-baseline/build

# 完整实验，当前终端显示阶段。
bash tools/5090/phase6/run_baseline_flux.sh --execute --foreground \
  --baseline-build /path/to/verified-baseline/build

# 后续批次复用本轮已编译的四组构建，所有 GPU 门禁和计时重新运行。
bash tools/5090/phase6/run_baseline_flux.sh --execute \
  --flux-build /path/to/previous-flux-batch/build

# 只读重放已完成批次，重新验证原始记录和 profile。
bash tools/5090/phase6/run_baseline_flux.sh --report --out /path/to/completed-flux-batch
```

`--baseline-build` 和 `--flux-build` 二选一；后者仍依赖其清单记录的原基准构建。新 C/A 在新目录生成，原生源码与二进制不覆盖旧构建。针对已登记旧 Python 前端的迁移仍使用原入口的验证、备份和回滚流程。

每次运行使用新的输出目录，默认保存于：

```text
../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-<时间>/
```

后台状态看 `batch-state.json`，阶段日志看 `handoff/supervisor.log`，完成后生成 `report.json` 和 `report.md`。后台运行要求容器持续存活。参数 `--python`、`--cuda-home`、`--build-cuda-home`、`--megatron`、`--jobs`、`--worker-timeout` 和 `--out` 与原入口含义一致。

## 5. 公平计时与证据

模型保持 **32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1**；每 optimizer step 为 **8192 tokens**。计时包括前向、反向、通信、optimizer 和清梯度，不启用 CUDA Graph。

默认 `--trials 1 --blocks 8`。每八块使用四组 Williams 轮转加逆序；每组在每个位置出现两次，两两前后关系平衡。`--blocks` 支持 8 的正整数倍，`--trials` 支持 1 或 2，各 trial 独立汇总。

| 默认正式工作量 | 数量 |
| --- | ---: |
| 每组窗口 | 8 |
| 每窗口预热 / 正式 optimizer step | 10 / 50 |
| 每组正式 step | 400 |
| 四组正式窗口 / rank 记录 | 32 / 256 |
| 四组正式 optimizer step | 1600 |

执行顺序为：源码/冻结训练 AST 检查→硬件/Apex/八 rank NCCL→隔离构建→两组 mapping/native 传输检查→四组完整模型 smoke→四组各八 rank profile→正式计时→从原始证据重算报告。

四组都固定 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`，其余通信环境记录并保持一致。允许主机中转不等于声称实测使用了某种指定 transport。

profile 必须证明：C/A 各自执行融合 GEMM、pack、NCCL、unpack 和原 decoder；A 保留逐次配置与 clone，C 保留实例配置与双工作区；基础量化执行独立 encoder 和 receiver v2；无量化执行 GemmOnly＋BF16 Reduce-Scatter，量化/All-to-All kernel 均为零。缺少实际 CUDA launch 归属或依赖证据不能通过。

窗口步时为 `max(8 rank elapsed_seconds)×1000/50`。主表报告所有窗口算术均值、中位数、最慢窗口、吞吐和相对无量化组的均值耗时降幅；另报告六项完整 block 配对的对数比与 10,000 次 bootstrap 95% CI。负收益和慢窗口照常保留；六项比较未作多重比较校正。

显存报告 Torch allocated/reserved 和独立 TACO workspace，逐 rank 相加后取最大。allocated＋native 不包含 CUDA context/NCCL 内部占用，不能当作整卡总峰值。C/A 与 GemmOnly 组的 GEMM、codec 和通信路径不同，端到端差值不能归因于单一 kernel。

## 6. 本地验证状态

执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/5090/phase6 -p 'test_*.py' -v`，**182 项测试全部通过**，其中本次新增 19 项。覆盖四组的真实 worker 生成与库路由、全部记录校验、四组完整控制流程、原排序/量化范围检查、无量化 Reduce-Scatter 调用和输出所有权，以及错误路径拒绝。CPU fixture 不作为 GPU 测量数据保存。

初次交付时，本地 CUDA 12.8 已将两组 A/C 的 transport、decoder 和实际 GEMM 注册编译为 **SM120 对象文件**，并成功编译各自的 CUDA mapping 检查可执行文件；没有在本地运行这些 GPU 程序。入口预览、shell 语法、文档链接和 whitespace 检查通过。后续目标机已越过完整构建阶段，但八卡正确性门禁和正式性能仍未完成，见下节。

## 7. 2026-11-01 回传失败：双工作区空 mask 数量误判

批次 `flux-quant-four-arm-v1-1x-50steps-20261101-201716-772182` 在 `native-reference-wire-and-mapping-probes` 阶段失败，八 rank 均报 `RuntimeError: All-remote instances have unexpected masks`。较早的 `20261101-200546-250921` 批次也是同一错误。证据见最新批次的 [supervisor 日志](../../../../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-201716-772182/handoff/supervisor.log)和 [C 探针日志](../../../../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-201716-772182/native-probe/flux-improved/process.log)。

这是四组实验的 Python 检查器错误：原共享 scratch 检查只允许单工作区全远端组，要求两个实例的 mask 列表为 `[None, None]`。移植时将 C 也设为全远端，却漏掉了它仍有双工作区；C 的实际列表应为 `[None, None, None, None]`。旧检查在所有 mask 都为空时仍必然失败。修复在 [flux_baseline_probe.py](../../../tools/5090/phase6/flux_baseline_probe.py)，期望长度改为 `2 × 每实例 slot 数`，保留实例数、slot 数、非空 mask、私有 packet/native handle、共享 scratch、输出保留和同步计数等检查，不放宽数值预算。

最新批次的 `environment-preflight.json` 记录 `completed=true`；目标机已编译 A/C 两份变体并进入 native 检查。C 的 Attention/MLP mapping 日志各含八 rank 共 4096 条 tile 记录。失败 rank 文件仍为 `completed=false`、`roles={}`，且没有创建模型 `results` 目录，**不把此前未保存的中间检查记为完整 native smoke 通过，也没有正式训练数据可汇总**。

此次只修改探针检查器，构建器、native 源码及前端与目标机冻结快照一致。同步修复后的工具后，在目标机 `TempName-New` 根目录用下面命令启动新批次，可复用已完成构建，**不必重编译 CUDA**：

```bash
bash tools/5090/phase6/run_baseline_flux.sh --execute --foreground \
  --flux-build /workspace/Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-201716-772182/build
```

该命令重新执行全部门禁和四组训练，不向失败目录追加结果。目标机构建目录及其依赖的原九组构建须保留；本地回传归档不含 `build` 二进制。新增回归测试覆盖 A/C 两种 slot 数、每个 mask 位置的非空拒绝、缺失/额外 slot、错误别名/同步，以及完整八 rank 报告校验。执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/5090/phase6 -p 'test_flux_baseline*.py' -v`，**24 项针对性测试全部通过**，入口预览和 `git diff --check` 也通过；这些 CPU fixture 不作为 GPU 通过证据，修复后尚未重跑八卡。

## 8. 2026-11-01 回传失败：硬件预检 CUDA OOM

批次 `flux-quant-four-arm-v1-1x-50steps-20261101-231152-305302` 的 [batch-state.json](../../../../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-231152-305302/batch-state.json) 显示失败阶段为 `hardware-preflight`。原始 [environment-preflight.log](../../../../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-231152-305302/environment-preflight.log) 显示，预检在第一个逻辑 GPU 调用 `torch.cuda.mem_get_info` 时报告 `CUDA error: out of memory`，尚未运行 NCCL 探针、构建验证、模型或正式计时。日志中的上层 `Child exited 1` 是包装错误，CUDA OOM 才是子进程记录的原始错误。

当时 `devices=[]`，旧代码没有及时保存当前设备 UUID 或故障前的 `nvidia-smi` 设备/进程信息；仅凭本次记录无法确定当时各卡的剩余显存、占用来源，也不能认定一定是 `temp.sh`。预检前的空闲进程检查已经通过，但容器中的进程列表未必能看到全部宿主机占用；不能仅凭空进程列表认定显存足够。更早完成批次的显存峰值是历史参考，不是此次故障时的快照。

预检已增强：在 CUDA 上下文初始化前及发生异常后尽力保存 `nvidia-smi` 显存/计算进程快照，保存 `CUDA_VISIBLE_DEVICES`，并在可取得设备属性时记录失败逻辑 GPU 的 UUID；父控制器不再用 `Child exited 1` 覆盖子进程错误。快照查询失败仍保留原错误，显存或进程不符合条件仍会阻止测试。这是**诊断改进，不会自动释放显存或保证重跑成功**。物理索引和 CUDA 逻辑索引可能不同，应按 UUID 对照。

在目标机检查 GPU 显存和进程：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory --format=csv
```

如容器内无法解释显存占用，应在宿主机检查对应 UUID；确认并停止自己的占用任务后再跑。如果之前启动了 `temp.sh` 的 24 小时计算，它会持续占用 GPU，须先确认对应进程是否仍在运行。不要停止无法确认归属的进程。

同步更新整个 `tools/5090/phase6/` 后，可先执行八卡预检；通过后沿用本次 `handoff/supervisor.json` 记录的目标机构建重新开始完整实验：

```bash
bash tools/5090/phase6/run_baseline_flux.sh --execute --preflight-only

bash tools/5090/phase6/run_baseline_flux.sh --execute --foreground \
  --flux-build /workspace/Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-220324-139713/build
```

两条命令分别生成新批次；完整实验会重新执行所有门禁。构建路径必须在目标机仍存在且通过原有身份校验；本地归档不含该构建的二进制。若预检继续失败，请同时查看新批次的 `environment-preflight.json` 中 `driver_snapshot_before_cuda`、`driver_snapshot_on_failure`、`devices` 与原始 `error`，以及 `environment-preflight.log`。这些快照是采样时刻的观察，不是持续空闲或容量保证。

相关实现：[控制器](../../../tools/5090/phase6/flux_baseline.py)、[协议统计](../../../tools/5090/phase6/flux_baseline_protocol.py)、[隔离构建](../../../tools/5090/phase6/flux_baseline_build.py)、[native 消融](../../../tools/5090/phase6/flux_baseline_native.py)、[worker 生成](../../../tools/5090/phase6/flux_baseline_sources.py)、[记录检查](../../../tools/5090/phase6/flux_baseline_validation.py)、[profile 检查](../../../tools/5090/phase6/flux_baseline_profile.py)。
