# H100 单节点 Phase6

**MLP＋Attention 五组量化消融使用 [`ablation4.py`](ablation4.py)：[消融实验 4](../../../docs/design/design-h100/消融实验/消融实验4.md)。** Megatron＋量化、固定 Flux 融合 v2、同融合 v2 实现的 A/B，以及消融 3 双缓冲完整版 C；默认单轮 25 个正式窗口。75 的消融 3 五套结果见 [汇总与 74 历史封存](../../../docs/design/design-h100/消融实验/base-phase6-ablation3.md)。

**native＋G0–G6 消融实验使用独立入口：[README-ablation.md](README-ablation.md)。** 它固定 GPT 6.7B/TP8，分别测量 remote_first 与 interleaved，显式支持 BF16、定向、随机同数量及全远端模式。下文仍是原六策略协议，不将其 72 窗口或 4% 验收套用到单项消融。

本入口把 A800 Phase6 的完整 step 对比协议移植到单节点 H100，默认 TP8，支持 TP4，DP/PP/CP 均为 1。无需 Slurm，使用独立 `torchrun --standalone` 进程。H100构建只生成独立快照，A800冻结库、到达表和历史结果均不改写。

新服务器首次部署请按 [单机八卡 H100 部署指南](../../../docs/design/design-h100/deploy-tp8.md) 配置环境、重新编译并运行。

**默认模型为GPT 6.7B。** 执行端先读[测试与交付说明](../../../docs/design/design-h100/instruction-h100.md)，开发端先读[数据与源码索引](../../../docs/design/design-h100/instruction.md)。截至提交 `e7a0c49`，旧0.64B、6.7B前次续跑及最新完整复测均有报告，具体状态和证据见开发端索引。下文命令用于按任务执行新会话；`calibrate.py run` 和 `run.py` 没有 `--execute` 时只打印命令。

## 算法与 baseline

| 策略 | MLP `linear_fc2` 前向 | attention 前向 | 类别 |
| --- | --- | --- | --- |
| `native` | 原生 Megatron | 原生 Megatron | 参考 baseline |
| `original` | 原生 Hopper Flux GemmV3 | 原生 Hopper Flux GemmV3 | 参考 baseline |
| `native_taco` | 原生 Megatron GEMM＋朴素张量 TACO | 原生 Megatron | 必须 baseline |
| `taco_fused` | A800 冻结融合 v2 的 H100 适配：原始顺序、全远端量化、CTA 解码 | 原生 Hopper Flux GemmV3 | 必须 baseline |
| `remote_arrival_selective` | rank+1 基础顺序＋到达优先＋选择性量化 | 原生 Hopper Flux GemmV3 | 候选 |
| `interleaved_arrival_selective` | L,R1,… 基础顺序＋到达优先＋选择性量化 | 原生 Hopper Flux GemmV3 | 候选 |

`config.json` 的 `bases` 还可加入 `interleaved_remote`（R1,L,R2,…）及 `interleaved_remote_group`（R1,R2,…,L），形成八组对比。基础顺序描述分区调度，不保证所有远端 CTA 都比本地 CTA 更早完成。

量化 MLP 路径保留 Phase6 的 CUTLASS GemmV2 算法，**重新编译为 SM90 代码**，没有将 FP8 编解码直接移植到 Hopper WGMMA/TMA epilogue。其注册名称仍含 `sm80/a100`，代表 V2 算法键，不代表执行 A800 cubin；实际架构由编译命令和设备检查记录。原生 Flux 参考及 attention 使用 SM90 GemmV3。后续若开发 Hopper 原生融合量化，需另建版本和 baseline。

**2026-10-11 起新构建的融合基线标识为 `a800-fused-v2-cta-v1`。** [fused_v2](fused_v2/README.md) 固定了 A800 原基线的 codec、runtime、epilogue 和 C++/Python 封装：恢复 CTA H128 解码、`__nv_cvt_float_to_fp8` 编码转换、每次 forward configure/reset、单工作区及解码后复用 barrier；模型 adapter 也保留旧基线的输出 clone。候选继续使用当前 warp 解码、优化编码和双工作区。已有 H100 会话使用的是 `phase6-warp-v1` 基线，其结果不改写，也不能与新基线的收益合并。对齐实现不保证两种硬件的绝对耗时相同。

H100 wrapper 按构造时确定的实例身份，统一选择缓冲区布局、GEMM 注册项、padding、归约与发布 barrier。候选保存完整 TACO 配置；A800 对齐基线仅保存 V2/V3 身份，逐次配置仍由旧 Python 流程管理。量化路径保留 V2 的全 rank 发布 barrier；不会套用 Hopper V3 的单节点免 barrier 分支。BF16 GEMM、本地 BF16、FP8 E4M3/H128 远端编码及 BF16 STE 反向保持原协议。

H100 源码快照还移除了 PCIe swizzle 头文件中未被使用的全局 `cudaMalloc` 初始化，并让采样 worker 先绑定本地 rank 的设备再导入 Flux，避免导入库时意外分配到默认设备。A800 共享源码保持原样。

H100 baseline 从受版本控制的源码生成并冻结，其中 `taco_fused` 使用 `tools/h100/phase6/fused_v2/` 的固定源码及哈希清单，**构建时不读取 `outputs/a800`、不复制 A800 的 `.o/.so`**。到达表、mask 必须来自相同 H100 节点与设备顺序的新校准，不能把 A800 数据改个架构标签后复用。Hopper 编译原则见 [NVIDIA 兼容性指南](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html)。

## 环境与范围

- Python 3.10、PyTorch 2.6.0+cu124、带 C++/CUDA 扩展的 NVIDIA Apex、CMake、Ninja、GCC/G++；详细包版本见 [公共环境记录](../../../docs/design/env.md)，保持可比较的软件版本。
- 编译建议 CUDA Toolkit 12.8，运行及 Apex 环境沿用 CUDA 12.4。builder 显式关闭 NVSHMEM、protobuf、GPU 架构自动探测，并设置 `CUDA_VISIBLE_DEVICES=''`，可以在没有 GPU 的编译节点上执行。
- Megatron-LM 固定为 `core_v0.12.3` / `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，路径通过 `--megatron` 指定；不依赖 `/data/run01/...` 或固定 Conda 位置。
- `sm_count` 默认 132，对应完整 H100 SXM 配置；也可指定 114 生成相应注册配置。实际运行仍要求 H100 SM90、同质设备、完整单节点 TP 组、NVLink 路径和所有设备两两可进行 CUDA peer access，不支持 MIG、跨节点或任意 PCIe 拓扑。
- 当前只实例化 BF16/RCR/无 bias/单节点 RS。MLP 使用 128×128×32、3 stages 的 V2 配置；Hopper 保留上游两个 BF16 cluster 选择。未覆盖其他 dtype、转置或融合 reduction。
- 形状要求 `M % (128*TP) == 0`、`N % 128 == 0`，attention/MLP 的本地 K 都是 32 的倍数，物理 tile 总数不超过 2048，且每个目标分区的 tile 数是 4 的倍数。

默认模型为 **32层、H4096/FFN16384、32 heads、词表32768，约6.7B**，S2048、micro batch1/global batch4、TP8，8192 token/step。[config-tp4-s1024.json](config-tp4-s1024.json) 为6.7B TP4、S1024、micro1/global8，使用梯度累积维持8192 token/step。原12层配置另存为 [TP8旧配置](config-0.64b-tp8.json) 和 [TP4旧配置](config-0.64b-tp4-s1024.json)。参数量计算和新shape见[模型说明](../../../docs/design/model-gpt-6.7b.md)。新模型必须重新构建和校准，不能复用旧H2048表；两个构建阶段传入同一配置。不同TP的词表补齐/参数量可能不同，不做跨TP直接强扩展结论。

所有产物按会话放到 `outputs/h100/phase6/<会话>/` 和 `logs/h100/phase6/<会话>/`；完整分类、合成token生成方式与Git规则见[文件放置说明](../../../docs/design/design-h100/instruction-h100.md#生成数据与文件放置规范)。下面的短路径示例只用于说明阶段，实际新实验按部署指南使用 `gpt67-tp8-<时间>` 等全新会话名。

## 1. 准备并编译 baseline 和 sampler（不使用 GPU）

在已经激活的主 Python 环境、仓库根目录下执行：

```bash
git submodule update --init --recursive

python tools/h100/phase6/build.py prepare \
  outputs/h100/phase6/base \
  --config tools/h100/phase6/config.json

python tools/h100/phase6/build.py compile \
  outputs/h100/phase6/base \
  --cuda-home /usr/local/cuda-12.8 --jobs 2
```

`prepare` 只生成独立源码快照与 manifest，不编译、不检查 GPU。`compile` 才调用 CMake/nvcc 和 Python 扩展构建；日志为输出目录内的 `compile.log`。NCCL 默认在该输出目录内由固定子模块源码构建；已有兼容的 NCCL 开发安装也可通过 `--nccl-root /path/to/nccl` 指定，必须同时具有 `include/nccl.h` 和 `lib/libnccl_static.a`。

产物包括 `original`、`taco_fused`、每种基础顺序对应的 `sampler_*`。各库及 Python binding 均独立，后续每个策略使用全新进程，避免加载混用。只有 `manifest.json` 中 `status=built` 的产物可进入准备与运行阶段；编译成功仍会保持 `gpu_validated=false`。

## 2. 新的 H100 到达校准（之后才在 GPU 上执行）

准备校准目录只做 CPU 操作；第二条命令才会使用 GPU：

```bash
python tools/h100/phase6/calibrate.py prepare \
  logs/h100/phase6/calibration --build outputs/h100/phase6/base

# 仅在之后允许使用 H100 GPU 时执行。
python tools/h100/phase6/calibrate.py run \
  logs/h100/phase6/calibration --cuda-home /usr/local/cuda-12.4 --execute
```

校准先运行相应库的 GPU swizzle 检查器，再测量每个物理 tile 的三遍 BF16 贡献观察。检查器从该动态库导出的函数查询实际 V2 kernel occupancy，不沿用 A800 的固定 CTA/SM 数；候选实际坐标与计划不一致时拒绝运行。保留八个 fragment 的发布/acquire 协议、接收端本地时钟、首次扫描拒绝、超时和重复记录检查，并报告采样扰动。每个基础顺序独立校准。

```bash
python tools/h100/phase6/plan.py \
  logs/h100/phase6/calibration \
  outputs/h100/phase6/selection-plan.json
```

计划生成是 CPU 操作：只用 passes 0/1 拟合，pass 2 只做诊断。默认每个目标分区内 64 个合法槽位重排，每个来源的远端量化预算为 1/64；保持物理 mask、本地 BF16、目标分区配额与窗口边界。无可靠训练信号时保留基础顺序、零量化选择，并如实记录，不强行选满预算。该步骤不是重排/量化后的联合重新校准。

## 3. 编译组合候选并冻结完整对比（不使用 GPU）

```bash
python tools/h100/phase6/build.py prepare \
  outputs/h100/phase6/candidates \
  --config tools/h100/phase6/config.json \
  --plan outputs/h100/phase6/selection-plan.json

python tools/h100/phase6/build.py compile \
  outputs/h100/phase6/candidates \
  --cuda-home /usr/local/cuda-12.8 --jobs 2

python tools/h100/phase6/prepare.py logs/h100/phase6/compare \
  --base-build outputs/h100/phase6/base \
  --candidate-build outputs/h100/phase6/candidates \
  --plan outputs/h100/phase6/selection-plan.json \
  --megatron /workspace/Megatron-LM
```

候选编译需要第 2 步的真实 H100 计划；本次不运行 GPU，因此不会伪造该计划或生成可用于正式性能结论的候选产物。CPU 单元测试使用的合成输入仅验证协议。

`prepare.py` 冻结配置、worker、适配器、独立 codec reference、计划和动态库哈希。实验脚本来自受维护的 A800 模型入口，但 H100 快照移除了 Slurm/旧节点/固定 Conda 依赖，并使用 H100 设备检查。Megatron 的 HEAD 和工作区在准备、启动时都核验。

## 4. GPU 对比与 CPU 报告

```bash
# 只查看完整执行计划，不使用 GPU。
python tools/h100/phase6/run.py logs/h100/phase6/compare

# 以下命令仅在之后允许使用 H100 GPU 时执行。
python tools/h100/phase6/run.py logs/h100/phase6/compare \
  --stage all --cuda-home /usr/local/cuda-12.4 --execute

# 已完成 GPU 测量后的离线 CPU 核验和统计。
python tools/h100/phase6/report.py logs/h100/phase6/compare
```

`all` 顺序执行候选 swizzle 检查、所有策略的模型 smoke、所有策略的独立 profile，再进行两轮位置平衡完整 optimizer-step 测量。默认六策略：每轮六个配对块，第二轮反序，每窗口 10 步 warmup＋20 步计时。包含前向、编码/解码、通信、同步、反向、optimizer 和一次清梯度。profile 不计入主表。

也可分别使用 `--stage preflight`、`--stage timing`；timing 必须具备相同冻结输入的完整 preflight。失败/部分结果保留，禁止覆盖旧目录或从缺失窗口得出正式结论。锁防止同一实验目录同时启动两个控制器。

核验内容包括所有 rank、单节点分组、H100 UUID/设备顺序与校准一致、相同初始参数/RNG/token、无跳步/回退、有限 loss/梯度、实际加载库、每个 MLP 的 mask/逻辑字节数，以及 profile 中 Hopper/V2/TACO kernel 的实际出现。未来 smoke 会执行原有 BF16/codec 前向误差检查；这些检查不证明真实语料收敛。

输出 `report.json`、`report.md`。各轮分别用配对块计算 `100*(1-exp(mean(log(T_candidate/T_baseline))))` 与 10000 次 bootstrap 95% 区间，同时列出两个必须 baseline 和两个原生参考的差距。默认沿用 A800 的 4% 门槛，可在实验前修改 `target_percent`；没有 GPU 数据时不生成收益结论。

## CPU 检查

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
  python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v
```

这些检查不导入 Torch、不初始化 CUDA，覆盖 TP4/TP8、四种基础顺序、训练/留出分离、mask 预算、本地 BF16、不合法计划拒绝、两轮平衡顺序、worker 快照、架构/布局/barrier 分派及配对统计。编译检查与 GPU 正确性、吞吐测试应分别记录。

2026-10-09旧0.64B配置：15项CPU检查通过，132 SM的四套库及Python binding完成编译/CPU导入，合成候选及mapping检查器仅完成编译。记录见[历史编译验证](../../../docs/design/design-h100/instruction.md#本次无-gpu-验证2026-10-09)。上述CPU/编译记录仍是旧模型历史证据；后续gxn74的GPU实验与当前6.7B报告见开发端数据索引。
