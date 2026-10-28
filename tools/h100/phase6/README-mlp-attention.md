# H100 GPT6.7B TP8：Attention/MLP四组作用范围对比

独立入口 `mlp_attention.py` 实现 [mlp-attetion.md](../../../docs/design/design-h100/mlp-attetion.md)，协议为 `h100-gpt67-tp8-mlp-attention-v1`。旧 `native_taco` 仍为MLP-only，旧五组 `ablation2.py` 及历史冻结数据保持原样。

| 策略 | self_attention.linear_proj | mlp.linear_fc2 |
| --- | --- | --- |
| native_taco_attn_mlp | 原生GEMM＋朴素tensor TACO | 原生GEMM＋朴素tensor TACO |
| native_taco_attn | 原生GEMM＋朴素tensor TACO | 原生Megatron BF16 |
| interleaved_full_attn | interleaved＋到达优先＋定向量化 | 原生Megatron BF16 |
| interleaved_full_attn_mlp | 同一完整候选 | 同一完整候选 |

模型固定32层、hidden4096、FFN16384、32 heads、S2048、micro batch1、global batch4，TP8/DP1/PP1/CP1。seed1234、token_seed4321、词表32768、padded vocab33792，每完整step8192 token。H100为单节点8张132 SM设备。选中的候选投影均用SM90编译的GemmV2，保留冻结解码、双工作区和BF16 STE反向；不要求出现旧候选的V3 attention投影kernel。

两个候选链接**同一个双shape库**，共享同一份Attention计划。Attention `[2048,4096,4096]` / local K512，MLP `[2048,4096,16384]` / local K2048，分别记录到达表、物理mask及哈希，CUDA查表也按本地K精确分派。全程不修改模型FFN来模拟Attention。

适配器只对指定模块使用原生通信替换或候选。`stop_audit()` 后按scope绑定原生/候选forward，全部64个投影的正式调用数实际计数；每组使用相同计数包装。预检核对32/64个启用模块、原生forward代码对象、实际路由、两种shape的mask/逻辑字节和两个工作区的独立地址。profile用角色scope与CUDA launch correlation核对相应V2/解码kernel，避免仅凭全局kernel名称误认作用范围。

## 环境与会话

在已安装PyTorch、Apex及Megatron依赖的H100环境运行；沿用 [环境说明](../../../docs/design/env.md) 和 [H100部署说明](../../../docs/design/design-h100/deploy-tp8.md)。Megatron固定 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，tracked files须干净。候选实际来源为冻结提交 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`，不是维护HEAD。

```bash
cd /workspace/TempName
export H100_SCOPE_SESSION="mlp-attention-gpt67-tp8-$(date +%Y%m%d-%H%M%S)"
export H100_SCOPE_OUT="$PWD/outputs/h100/phase6/$H100_SCOPE_SESSION"
export H100_SCOPE_LOG="$PWD/logs/h100/phase6/$H100_SCOPE_SESSION"
export H100_MEGATRON=/workspace/Megatron-LM
export H100_CUDA_BUILD=/usr/local/cuda-12.8
export H100_CUDA_RUN=/usr/local/cuda-12.4
export PYTHONNOUSERSITE=1
```

路径与本机安装不同时调整变量。所有新输出目录必须不存在；失败时保留日志和状态，在新目录修复重做，不覆盖已关闭的证据。

## 1. 准备并编译BF16采样库（不使用GPU）

采样库从冻结计算源码派生，支持两个精确shape；默认仅采集Attention，既有MLP计划可在同host、GPU UUID/顺序、配置和来源一致时复用。

```bash
python tools/h100/phase6/mlp_attention.py calibration-build "$H100_SCOPE_OUT/calibration" \
  --roles attn
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/mlp_attention.py compile "$H100_SCOPE_OUT/calibration" \
  --cuda-home "$H100_CUDA_BUILD" --jobs 2
python tools/h100/phase6/mlp_attention.py calibration-prepare "$H100_SCOPE_LOG/calibration" \
  --build "$H100_SCOPE_OUT/calibration"
python tools/h100/phase6/mlp_attention.py calibration-run "$H100_SCOPE_LOG/calibration" \
  --cuda-home "$H100_CUDA_RUN"
```

最后一步默认dry-run，只打印命令。编译可选 `--nccl-root /path/to/nccl`，要求 `include/nccl.h` 和 `lib/libnccl_static.a`；否则在新构建目录编译NCCL。构建器隐藏GPU，不加载新扩展或运行mapping。

更换设备/顺序或host时，在第一条命令使用 `--roles attn mlp`，在同一新校准会话分别采集两种shape，后面显式传入新MLP计划。不能沿用旧设备计划、修改manifest或只换shape标签。

## 2. H100端真实校准与计划拟合（校准命令使用GPU）

以下GPU命令留给后续明确执行实验的H100会话，本次开发不运行。

```bash
python tools/h100/phase6/mlp_attention.py calibration-run "$H100_SCOPE_LOG/calibration" \
  --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/mlp_attention.py fit "$H100_SCOPE_LOG/calibration" \
  "$H100_SCOPE_OUT/plans/measured"
```

每种shape使用独立BF16采样：192个repeat覆盖3遍×64个本地tile offset，每个repeat包含off及receiver_sparse。pass0/1拟合，pass2只用于留出诊断；沿用原参考、合法目的分区/窗口64及每来源1/64预算。本地mask为零，不强行补满预算。拟合核对八rank、完整shape、全部pass/offset、BF16 V2 kernel、库哈希、设备身份和完成记录。

只采Attention时输出 `attn.json`；采两种shape时另有 `mlp.json`。原始校准rank JSON每rank包含本次采集的所有shape，mapping和profile按完整shape单列。Attention没有真实校准时，正式构建会失败；CPU合成计划不能升级为实测计划。

## 3. 构建共同候选、冻结四组比较

默认复用原冻结MLP计划，必须与新Attention计划设备身份完全一致。旧MLP-only动态库不能直接用于这项双shape实验，需新建共同候选库。

```bash
python tools/h100/phase6/mlp_attention.py build-prepare "$H100_SCOPE_OUT/candidates" \
  --attn-plan "$H100_SCOPE_OUT/plans/measured/attn.json"
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/mlp_attention.py compile "$H100_SCOPE_OUT/candidates" \
  --cuda-home "$H100_CUDA_BUILD" --jobs 2
python tools/h100/phase6/mlp_attention.py prepare "$H100_SCOPE_LOG/compare" \
  --build "$H100_SCOPE_OUT/candidates" --megatron "$H100_MEGATRON" --blocks 4
python tools/h100/phase6/mlp_attention.py run "$H100_SCOPE_LOG/compare"
```

如果重新校准了MLP，在 `build-prepare` 增加 `--mlp-plan "$H100_SCOPE_OUT/plans/measured/mlp.json"`。两种计划分别冻结到 `plans.json`，不按两个候选分别拟合。`prepare` 保存原生baseline来源、实际适配器/worker、计划、构建manifest、控制器快照和哈希；原生两组没有Flux库路径。

默认4组×4块×2轮＝32个正式窗口，另有8个预检窗口。第一轮固定seed20261013随机排列一次后循环移位；第二轮反转块及块内顺序。`--blocks` 仅接受4的正整数倍，运行前冻结，不自动追加复测。每窗口10步预热＋20步完整optimizer step，两轮都包含前向、反向、通信、更新和清梯度。

## 4. H100端四组预检、正式计时和报告

```bash
python tools/h100/phase6/mlp_attention.py run "$H100_SCOPE_LOG/compare" \
  --stage all --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/mlp_attention.py report "$H100_SCOPE_LOG/compare"
```

也可分别使用 `run ... --stage preflight --execute` 与 `run ... --stage timing --execute`。预检包括三个“候选×启用shape”映射、四组smoke、四组profile；任何缺rank、路由/shape/mask/库不符、工作区别名、数值超预算、非有限loss/梯度、skip或意外fallback都会停止。Attention-only的MLP原生路径属于登记路径，不算fallback。计时不加载采样库。

每套校准和每套比较仅开始前检查一次GPU，记录host、八卡UUID/顺序、计算进程、显存和利用率；比较还要求与两份校准计划一致。忙时直接退出，不创建Slurm、等待器或自动重试。预检及正式窗口之间不轮询GPU；[watcher.md](../../../docs/design/design-h100/watcher.md) 的队列规则只适用于另行明确安排的外部队列。

## 5. 报告与证据交付

每窗口取八rank最大elapsed/20，吞吐按8192 token/step计算。逐轮与同会话合并8块分别报告四组中位ms/step、tokens/s；两个主要加速表分别使用**同作用范围**的native_taco作分母，输出配对比值中位数及最好单块，附轮次/block和双方耗时。

四个配对比较为：候选Attention/native Attention、候选双模块/native双模块、候选双模块/候选Attention、native双模块/native Attention。耗时下降为 `100*(1-exp(mean(log(T_X/T_Y))))`；完整配对块bootstrap10000次、seed20261009，95%区间不把step或rank当独立重复。负收益和跨零区间保留，不套4%门槛。扩展到双模块的变化包含MLP从原生BF16切换到完整候选的整体影响。

全样本为主。单独慢窗口附表使用每轮各策略原始中位数×1.10，一组触发就剔除整块四组，所有比较使用同一保留集合，不迭代阈值。列明触发策略、剔除块及剩余数；零块显示“无可用配对块”，一块只给点估计、不估计区间。没有跨历史会话拼接功能。

默认交付：256份正式rank、64份预检rank、32份压缩profile trace、3份候选/shape映射，以及新增Attention校准原始rank/trace/mapping、两种角色计划、冻结脚本/调度/manifest/源码与库哈希、command/process日志和全样本/筛选报告。失败时保留部分数据和状态，不补写完成。原环境先严格核验；接收端缺库时可用 `report /path/to/compare --offline`，仅核对冻结文本及记录中的库身份，不声称重新读取实际动态库。

## CPU演练与无GPU检查

以下流程可以在开发端执行。`source_only` 固定在manifest、计划、preparation和worker中，直接调用worker也会在导入Torch前失败。编译通过不会让合成Attention计划变为可执行计划。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v

# 使用独立CPU目录，避免占用将来的正式输出名称。
export H100_CPU_OUT="$PWD/outputs/h100/phase6/mlp-attention-gpt67-tp8-cpu-$(date +%Y%m%d-%H%M%S)"
export H100_CPU_LOG="$PWD/logs/h100/phase6/$(basename "$H100_CPU_OUT")"
python tools/h100/phase6/mlp_attention.py calibration-build "$H100_CPU_OUT/calibration" --roles attn mlp
python tools/h100/phase6/mlp_attention.py calibration-prepare "$H100_CPU_LOG/calibration" \
  --build "$H100_CPU_OUT/calibration" --source-only
python tools/h100/phase6/mlp_attention.py calibration-run "$H100_CPU_LOG/calibration"
python tools/h100/phase6/mlp_attention.py build-prepare "$H100_CPU_OUT/candidates" --source-only
python tools/h100/phase6/mlp_attention.py prepare "$H100_CPU_LOG/compare" \
  --build "$H100_CPU_OUT/candidates" --source-only
python tools/h100/phase6/mlp_attention.py run "$H100_CPU_LOG/compare"
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/mlp_attention.py compile-check "$H100_CPU_OUT/calibration" \
  --out "$H100_CPU_OUT/cpu-compile/sampler" --cuda-home "$H100_CUDA_BUILD"
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/mlp_attention.py compile-check "$H100_CPU_OUT/candidates" \
  --out "$H100_CPU_OUT/cpu-compile/candidate" --cuda-home "$H100_CUDA_BUILD"
```

`compile-check` 分别运行两个shape的CPU表/mask检查、编译双K的SM90 mapping对象，以及sampler或codec runtime对象，不运行mapping或CUDA kernel。`compile` 可进一步在隐藏GPU的环境链接完整库和Python binding；CPU候选仍是diagnostic-only，必须在H100真实校准后新建正式构建。

CPU演练和编译检查产物仅本机保留，由 `.gitignore` 忽略；交付维护代码、测试与本说明中的验证摘要。正式H100源码/计划/manifest/原始证据仍按 [交付规范](../../../docs/design/design-h100/instruction-h100.md) 回传，`.so/.o/.a`、构建树、环境及缓存保持忽略。

## 本次开发验证（2026-10-14）

71项CPU回归通过（57项既有、14项scope新增）；最终scope修改后14项再次通过。覆盖校准完整shape/pass/库核验、K512/K2048分派、四组在timing/profile中 `stop_audit()` 前后的实际适配器调用、原生无Flux、独立工作区、40窗口完整合成证据（8预检＋32计时）、同范围配对统计、统一整块筛选及失败路径。合成记录只存在于测试临时目录，不作为GPU证据。

实际CPU演练完成两种shape校准dry-run、三项候选映射命令及四组32正式＋8预检窗口dry-run。采样和候选各完成两种shape的CPU表/mask检查、SM90 mapping对象、sampler/codec runtime对象编译；进一步完成两个完整 `libflux_cuda.so`、`libflux_cuda_ths_op.so`、Python binding及mapping程序的编译/链接，导出符号核验通过。构建使用CUDA Toolkit12.8.61、G++11.4.0、Python3.10环境，并设置 `CUDA_VISIBLE_DEVICES=''`；没有加载新Flux扩展或运行mapping/CUDA kernel。

本地会话为 `outputs/h100/phase6/mlp-attention-gpt67-tp8-cpu-20261014-dev/` 及同名 `logs/`，检查记录在 `logs/.../handoff/`，全部由Git忽略。旧冻结SHA256SUMS与REFERENCES核验通过。候选库包含明确标记的CPU Attention合成计划，保持diagnostic-only；它不能用于真实实验，后续必须先真实H100校准，再新建正式候选和compare。本次没有GPU正确性或性能结论，没有启动Slurm、占用监控、等待器或队列。

## H100 profile 计数修正（2026-10-15）

真实 PyTorch trace 会同时保存同名的 CPU `user_annotation` 和 `gpu_user_annotation`。角色调用计数与 CUDA launch correlation 只使用 CPU 注解；仍严格要求每个角色 32 层 × 4 微批＝128 次，不把 GPU 镜像重复计入，也不接受缺失或重复的 CPU 调用。新增四组镜像注解、缺失 CPU span 及重复 CPU span 回归后，72 项 CPU 测试通过；V2/解码、原生无 Flux、数值和工作区校验保持不变。

原失败会话 `TTention-MLP-gpt67-tp8-20261014-231236` 的状态、submission 和 trace 保留。修正后新建 `TTention-MLP-gpt67-tp8-20261015-003756`，先核验原真实双 shape 校准、计划与完整共享库，再冻结新的控制器哈希并重新执行全部预检和计时；不修改原清单绕过旧哈希。
