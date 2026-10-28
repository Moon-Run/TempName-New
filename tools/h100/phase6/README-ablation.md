# H100 GPT 6.7B TP8 消融入口

2026-10-14新增的 `native_taco＋G1–G4` 交替五组实验使用独立 [README-ablation2.md](README-ablation2.md) / `ablation2.py` 入口，默认50窗口。本页旧八组默认协议与历史产物保持兼容。

分组依据是 [消融实验.md](../../../docs/design/design-h100/消融实验/消融实验.md)：`native`＋G0–G6，共八组，仅比较完整 optimizer step。`remote_first`、`interleaved` 分别调度、分别报告。这里不运行旧六策略验收，不新增 E0、工程消融或算子性能对照。

实现起点是冻结登记 `candidate` 指向的 a–h 实测源码（`ff38f5b1ff1c80cb831f58c64b95f42b0379879a`），不是维护快照提交。所有派生源码写入新会话，原始源码、动态库、计划及历史报告不改。共享 `src/`、`python/`、A800 worker、旧 H100 入口与固定 `fused_v2/` 均保持原样。

## 分组及执行路径

| 组别 | 顺序 | 到达重排 | 显式量化模式 |
| --- | --- | --- | --- |
| native | Megatron 原生 | 无 | native_bf16 |
| G0 | 候选 V2 原始顺序 | 关闭 | bf16 |
| G1 | B | 关闭 | bf16 |
| G2 | B | 开启 | bf16 |
| G3 | B | 关闭 | targeted |
| G4 | B | 开启 | targeted |
| G5 | B | 开启 | random_matched |
| G6 | B | 开启 | all_remote |

G0–G6 的 GEMM、tile、epilogue、warp decoder、双工作区、输出生命周期和 BF16 STE 反向都来自同一实测实现。MLP 是编译为 SM90 的 GemmV2，attention 是原候选的 Hopper GemmV3。native 保留原生 Megatron 完整训练路径，不加载 Flux。

BF16 明确传入全零 `[source rank, physical tile]` mask；G6 明确枚举每个来源全部 448 个远端 tile，本地始终为零。两者不调用 fraction 选择器。候选实例的 `owned_taco.enabled` 保持为真，保证 BF16 MLP 不转为 V3；现有 `taco_selected` 分支旁路 BF16 贡献的编解码，原混合接收 kernel 仍执行 BF16 归约。因此 BF16 profile 中仍应出现 `taco_decode_ring_kernel`，这不代表发生 FP8 解码。没有另写 BF16 专用快路径或改变工作区生命周期。

G3/G4 的物理 mask 完全相同；G2/G4/G5/G6 的到达表相同。G0 从实测 remote_first 的 StreamK 基础坐标撤销一个分区偏移，恢复 rank 原始顺序，不假定 StreamK 为行优先。G5 默认预登记 seed `17 29 43`，逐来源、逐目标分区匹配 G4 实际选择数，记录每分区可选组合数。至少提供三个互不重复的 seed，全部保留并逐个报告。

## 入口

| 文件 | 功能 |
| --- | --- |
| `ablation_plan.py` | 八组定义、计划验证与派生、随机 mask、固定调度 |
| `candidate.py` | 核对实测源码，独立准备及编译各组 |
| `ablation_prepare.py` | 冻结 worker、模型路由、计划、库、调度与 submission |
| `ablation_routing.py` | G 组统一候选路径与 native 分流 |
| `ablation_run.py` | 默认 dry-run，显式 `--execute` 才启动 GPU |
| `ablation_report.py` | 预检、全部 rank 核验、配对报告及跨会话汇总 |
| `ablation_compile_check.py` | CPU 表/mask 测试与仅编译的 SM90 检查 |
| `test_ablation.py` | CPU 正向与失败路径回归 |

各个构建独立放在 `builds/<B>-<组或seed>/<同名变体>/`。G5 的三个 seed 是同一算法组的三个固定 mask，并非增加算法组。默认每种 B 有十个测量配置：native、六个非随机 G 组、三个 G5 seed。每轮十个完整配对块，round2 同时反转块与块内顺序：`2 bases × 2 rounds × 10 blocks × 10 configs = 400` 个正式窗口，另有 40 个 smoke/profile 窗口与 18 次 GPU 映射检查。`--blocks N` 在准备时固定块数，不能看完收益后选择保留数量；每轮至少两块。默认块数只是预登记起点，不保证区间精度，应在正式实验前结合预跑方差和预算确定。

## 1. CPU 检查与会话设置

以下命令在仓库根目录、已配置的 Python 环境运行。新机器的依赖版本和编译工具见 [公共环境](../../../docs/design/env.md)；Megatron 固定 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
  python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v

export H100_SESSION="ablation-gpt67-tp8-$(date +%Y%m%d-%H%M%S)"
export H100_OUT="$PWD/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$PWD/logs/h100/phase6/$H100_SESSION"
export H100_FREEZE="$PWD/outputs/h100/phase6/frozen-candidate-v1-20261013"
export H100_MEGATRON=/workspace/Megatron-LM
export H100_CUDA_BUILD=/usr/local/cuda-12.8
export H100_CUDA_RUN=/usr/local/cuda-12.4

(cd "$H100_FREEZE" && sha256sum --quiet -c SHA256SUMS.txt)
sha256sum --quiet -c "$H100_FREEZE/REFERENCES.sha256.txt"
```

默认可复用冻结的真实 H100 校准，但 GPU 执行仍严格要求原 host、八张 GPU UUID 及顺序、132 SM、模型和映射相符。回传证据中的 `/workspace/TempName/` 可以在准备阶段按仓库相对路径定位，必须逐文件匹配原 SHA256；此解析记录在新 manifest，不改历史路径或清单。其他绝对路径必须实际存在且匹配。完整构建和模型预检仍需要 H100 端的运行环境。

## 2. 必要时重新校准

同设备、同模型、同映射可直接跳到下一节。更换 host/GPU UUID/顺序，或映射检查不匹配时，先用新目录校准。当前入口固定 132 SM、GPT 6.7B、TP8；更换 shape/SM 等条件属于新的实验协议，不能只改 manifest 绕过检查。

```bash
# CPU 准备和编译采样构建。旧库和固定 fused_v2 源码均不修改。
python tools/h100/phase6/build.py prepare "$H100_OUT/calibration-base" \
  --config tools/h100/phase6/config.json
python tools/h100/phase6/build.py compile "$H100_OUT/calibration-base" \
  --cuda-home "$H100_CUDA_BUILD" --jobs 2
python tools/h100/phase6/calibrate.py prepare "$H100_LOG/calibration" \
  --build "$H100_OUT/calibration-base"

# 先只打印命令；只有后续明确允许 H100 执行时才加 --execute。
python tools/h100/phase6/calibrate.py run "$H100_LOG/calibration" \
  --cuda-home "$H100_CUDA_RUN"
python tools/h100/phase6/calibrate.py run "$H100_LOG/calibration" \
  --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/plan.py "$H100_LOG/calibration" "$H100_OUT/parent-plan.json"
```

两种 B 分别采集三遍，pass 0/1 拟合、pass 2 只作留出诊断。下一节 `candidate.py prepare` 加 `--plan "$H100_OUT/parent-plan.json"` 即可使用新计划；仍从 a–h 的相同计算源码派生，不做额外优化。原计划及新计划均保留。

## 3. 独立构建与冻结模型输入

```bash
# 默认父计划是冻结登记里的真实 H100 计划；输出目录必须不存在。
python tools/h100/phase6/candidate.py prepare "$H100_OUT/candidates" \
  --seeds 17 29 43 --bases remote_first interleaved

# 完整编译各组；不使用 GPU，CUDA_VISIBLE_DEVICES 由构建器置空。
# 可选 --nccl-root /path/to/nccl，须有 include/nccl.h 与 lib/libnccl_static.a。
python tools/h100/phase6/candidate.py compile "$H100_OUT/candidates" \
  --cuda-home "$H100_CUDA_BUILD" --jobs 2

# 要先逐项编译，可加 --variants remote_first-G0 remote_first-G1 等。
# 已成功且哈希通过的独立构建会跳过；失败的构建保留 compile.log。
python tools/h100/phase6/ablation_prepare.py "$H100_LOG/compare" \
  --builds "$H100_OUT/candidates" --megatron "$H100_MEGATRON" \
  --blocks 10 --schedule-seed 20261013
python tools/h100/phase6/ablation_run.py "$H100_LOG/compare"
```

只演练 CPU 流程时，可以不编译库，使用 `ablation_prepare.py ... --source-only`。该 submission 永久标记为不可执行，`--execute` 也会拒绝。正式执行必须完成所有构建后重新准备新的 compare，不编辑标记。dry-run 只核对文件、列出命令和调度，不导入 Torch、不初始化 CUDA、不调用映射可执行文件，也不启动 worker。worker 自身还要求执行控制器设置的显式启用标记。

无需 GPU 的表与 CUDA 编译检查：

```bash
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation_compile_check.py \
  --builds "$H100_OUT/candidates" --out "$H100_OUT/cpu-compile" \
  --cuda-home "$H100_CUDA_BUILD"
```

它运行 g++ 编译的 CPU 表及 `taco_selected` 测试，对全部变体只编译 GPU mapping 对象，并编译共同的 SM90 codec runtime；不运行 CUDA kernel，不产生可用于性能测试的完整动态库。

`ablation-manifest.json`、每组 `manifest.json`、派生计划、`submission.json` 记录组别、B、实际顺序、到达开关、量化模式、seed、parent_freeze、父/派生计划 SHA256、实测源码/维护提交、逐文件改动前后哈希及动态库来源。native 没有 tile 派生计划或 Flux 库，相应字段不适用。每个 rank 重复记录实际组别、计划身份与加载的库哈希。

## 4. H100 预检与完整 step 测试

以下命令会实际使用 GPU，只在 H100 端另行获准执行时运行。遵守 [watcher.md](../../../docs/design/design-h100/watcher.md) 的启动前检查约定，本入口不启动监控、等待器或 Slurm，不在运行中周期查询 GPU。

```bash
python tools/h100/phase6/ablation_run.py "$H100_LOG/compare" \
  --stage preflight --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/ablation_run.py "$H100_LOG/compare" \
  --stage timing --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/ablation_report.py "$H100_LOG/compare"
```

预检包含所有 G 组实际 GPU 映射、所有组完整模型 smoke 与独立 profile。核验双射、物理坐标、目的分区、实际 V2/V3/mixed-wire kernel、32 层量化 mask/逻辑字节、原误差预算（MLP 对 BF16 相对 L2 ≤0.05，对独立 codec 参考 ≤0.01）、相同初始化/RNG/token、有限 loss/梯度、零 skip、零 fallback。native 不得加载 Flux；任何 G 组缺 V2 MLP kernel 都不能计时。

每个正式窗口为 10 步预热＋20 步完整 optimizer step，包含前向、反向、通信、参数更新和清梯度；8192 tokens/step。每窗口取八 rank 最大 elapsed/20。全部 rank、命令、日志和 profile 原样保留。失败阶段不自动重试、不覆盖目录，修复后建立新会话并说明失败原因。

## 5. 统计、复测及回传

`report.json` 保存全部窗口的八 rank elapsed、ms/step、tokens/s、每轮绝对耗时和完整配对 log 比值；`report.md` 给出比较表。收益是 `100 × (1 − exp(mean(log(T_X/T_Y))))`，按完整配对块 bootstrap 10000 次给 95% 区间。报告 G0–G6 对 native、设计文档列出的内部对照，以及到达/量化交互项；G5 每 seed 单列，在块内对全部 seed 几何平均后另报汇总，不把 seed 或 rank 当独立重复。主结果不剔除慢窗口，不执行单项 4% 验收；区间跨零显示“尚未确认正增益”。

独立复测复用已核验构建时，另建 compare，重复 prepare/preflight/timing。两种 B 始终分开；两轮均是完整训练 step。多会话汇总须在实验前规定，并要求相同父计划、源码哈希、配置和随机seed。还须匹配 `measurement_identity`：预检中的初始化/输入/运行软件指纹、冻结worker及配置/计划文件哈希、实际运行库哈希和控制脚本哈希；目录位置可不同，内容必须一致。软件环境、编译结果或worker变化的会话分别报告，不能仅因CUDA源码相同就合并。以下汇总对独立会话等权，先抽会话、再在会话内抽完整配对块，单会话与单轮结果仍保留：

```bash
python tools/h100/phase6/ablation_report.py \
  "$H100_LOG/compare" /path/to/independent-session/compare \
  --aggregate "$H100_LOG/independent-sessions.json"
```

执行端先在原环境完成严格报告。接收端没有 `.so` 时可用下面的离线检查，仍核对冻结 worker/计划/调度、preflight 证据和全 rank 记录，但只核验 rank 所记录的库哈希，不能声称重新读取了原动态库：

```bash
python tools/h100/phase6/ablation_report.py "$H100_LOG/compare" --offline
```

回传源码、计划、manifest、完整校准/预检/窗口记录及报告，不回传编译库/对象/缓存。保留原 `.so` 在 H100 端。下面命令只准备本会话交付，不提交其他会话：

```bash
mkdir -p "$H100_LOG/handoff"
git rev-parse HEAD > "$H100_LOG/handoff/code-commit.txt"
git submodule status > "$H100_LOG/handoff/submodules.txt"
git diff -- tools/h100/phase6 docs/design/design-h100 \
  > "$H100_LOG/handoff/tracked-changes.patch"
python -m pip freeze > "$H100_LOG/handoff/python-packages.txt"
git status --short -- "outputs/h100/phase6/$H100_SESSION" "logs/h100/phase6/$H100_SESSION"
git add -- "outputs/h100/phase6/$H100_SESSION" "logs/h100/phase6/$H100_SESSION"
git diff --cached --stat
git diff --cached --name-only
# 补充 handoff/README.md：会话、代码与数据提交、复用关系、完成/失败状态、缺项。
# 核对暂存内容后回传（本轮开发任务不执行这些发布命令）。
git commit -m "Record H100 ablation $H100_SESSION"
git push
git rev-parse HEAD
```

新写且尚未跟踪的实现文件不包含在 `git diff` 中，须随代码提交交付；不要仅依赖补丁。源码中的链接脚本、`.clang-format`、Triton `.bc/.ll` 输入由消融目录的精确 Git 例外保留，编译产物仍忽略。`source-only` 演练与 CPU 合成测试记录不是 H100 正确性/吞吐证据。

## 本轮 CPU 交付（2026-10-13）

本轮完成43项CPU回归（19项新增、24项既有），真实冻结计划的两种B、18份派生源码与mask核验，400个正式窗口的dry-run，18份SM90映射对象及共同codec runtime编译，`remote_first-G0` 的完整Flux动态库、Python binding和mapping可执行文件构建。只运行了纯CPU表/mask检查，没有运行mapping可执行文件、加载新Flux扩展或执行任何GPU kernel。

本轮自检产物仅在本机保留：`outputs/h100/phase6/ablation-gpt67-tp8-cpu-20261013-{v1,final}/` 及同名 `logs/` 目录，包含生成的源码副本、计划、构建、dry-run与详细验证记录。上述CPU会话及消融目录中的 `cpu-compile/` 已加入 `.gitignore`，不随Git上传；提交维护代码、测试代码、使用说明和本节验证摘要即可。最终会话逐变体确认351个来源文件与已编译版本完全相同；旧冻结目录与REFERENCES清单均通过核验，原基线/A800实现未修改。正式H100实验的源码、计划和原始结果仍按前述回传规范保留。

开发端其余17份完整库尚未构建；开发端已编译的G0库也不随Git迁移。H100端必须确认18份候选库全部完成构建，仅可复用本端实际存在且哈希/环境符合要求的构建，随后完成所有组实际GPU映射/数值/profile/零fallback验证及完整step测量。当前没有性能收益、显存可用性或收敛结论，也没有提交Slurm、启动监控或等待GPU队列。

## 代码复核（2026-10-13）

复核重新运行原43项CPU测试，全部通过；随后发现跨会话汇总只检查父计划、配置和计算源码，可能合并运行软件、模型worker、控制脚本或实际动态库不同的会话。已增加 `measurement_identity` 的记录与一致性检查，新增改变上述身份时拒绝汇总、相同身份允许独立会话汇总的回归测试，最终45项CPU测试全部通过。分组、物理mask、构建源码与单会话计时流程未改。

本次没有重新执行CUDA编译或GPU验证，前节编译记录仍是初次实现时的验证范围。修复改变了 `ablation_report.py`，正式执行应在同步修复后新建compare并重新prepare，以冻结最新控制脚本哈希；不修改旧submission的校验清单来绕过检查。已有候选构建可先按原manifest核验，是否复用由本端实际库与环境决定。
