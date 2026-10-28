# H100 GPT 6.7B TP8：交替五组消融

Attention-only/Attention＋MLP四组范围对比使用独立 [README-mlp-attention.md](README-mlp-attention.md) / `mlp_attention.py`。本页 `native_taco` 和G1–G4仍保留旧MLP范围含义。

实现对应 [消融实验2.md](../../../docs/design/design-h100/消融实验/消融实验2.md)，协议为 `h100-gpt67-tp8-ablation2-v1`。独立入口是 `ablation2.py`，复用原消融的候选构建、物理计划、完整训练 worker、预检与 rank 核验。旧 [八组入口](README-ablation.md)、冻结源码、构建、计划和报告保留。

| 策略 | 路径 | 到达优先 | 物理量化 mask |
| --- | --- | --- | --- |
| native_taco | 原生 Megatron 线性层及 attention；仅 MLP 前向通信使用朴素 tensor TACO | 不适用 | 无候选 mask，全部远端贡献量化 |
| G1 | 冻结 SM90 编译的 V2 MLP / Hopper V3 attention | 关 | 显式全零 |
| G2 | 与 G1 相同 | 开 | 与 G1 相同 |
| G3 | 与 G1 相同 | 关 | 冻结定向 mask |
| G4 | 与 G1 相同 | 开 | 与 G3 完全相同 |

G1–G4 均为 `interleaved`，保留同一解码、双工作区及 BF16 STE 反向实现。G1/G2 的 `owned_taco.enabled` 仍打开以保持 V2 路径；混合接收 kernel 名称包含 `taco_decode_ring_kernel` 并不表示发生量化，预检按实际物理 mask 与 wire bytes 核对。冻结交替 mask 各来源选择数为 `[7,7,7,0,7,7,7,3]`，本地贡献为零。

`native_taco` 沿用 `prepare.py` 的 TP8 适配（`self.world in (4,8)`），复制原生 `RowParallelLinear.forward` 的代码对象，只替换该 MLP 模块的通信函数全局变量。无候选动态库、映射计划、Flux Python 搜索路径；smoke 检查32层通信数值及字节，profile 要求 tensor TACO 三个 scope 与 all-to-all 证据，并拒绝 Flux kernel。

固定模型：32层、hidden4096、FFN16384、32 heads、S2048、micro batch1、global batch4、TP8/DP1/PP1/CP1；每步8192 token。seed1234、token_seed4321、词表32768、padded vocab33792。设备要求同一节点8张132 SM的H100，UUID及顺序与父校准一致。

## 环境与会话

在 H100 容器内、仓库根目录执行。Python/PyTorch/Apex/CUDA 依赖沿用 [环境说明](../../../docs/design/env.md) 和 [H100部署说明](../../../docs/design/design-h100/deploy-tp8.md)。Megatron 固定 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，tracked files 必须干净。以下路径对应既有部署，安装位置不同时修改环境变量。不要把 CPU rehearsal 目录改标记作为正式会话。

```bash
cd /workspace/TempName
export H100_SESSION="ablation2-gpt67-tp8-interleaved-$(date +%Y%m%d-%H%M%S)"
export H100_OUT="$PWD/outputs/h100/phase6/$H100_SESSION"
export H100_LOG="$PWD/logs/h100/phase6/$H100_SESSION"
export H100_MEGATRON=/workspace/Megatron-LM
export H100_CUDA_BUILD=/usr/local/cuda-12.8
export H100_CUDA_RUN=/usr/local/cuda-12.4
export H100_FREEZE="$PWD/outputs/h100/phase6/frozen-candidate-v1-20261013"
export PYTHONNOUSERSITE=1
(cd "$H100_FREEZE" && sha256sum --quiet -c SHA256SUMS.txt)
sha256sum --quiet -c "$H100_FREEZE/REFERENCES.sha256.txt"
```

## 复用四个旧构建，或新编译四个构建

二选一。复用只核验参与的四个旧变体，不要求旧 G0/G5/G6/remote 构建可用；所有源码、计划及库必须在本端实际存在并匹配哈希。新 registry 保存来源绝对路径、原 manifest 哈希和四份计划；不复制旧构建树，不修改旧文件，不复用计时。实际测量源码身份仍是 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`。

```bash
python tools/h100/phase6/ablation2.py reuse "$H100_OUT/candidates" \
  --from-builds outputs/h100/phase6/ablation-gpt67-tp8-20261013-233551/candidates
```

没有完整本地库时，从冻结源码新派生并编译四个候选，不展开其余14个旧变体。编译器隐藏全部GPU，不加载新扩展。可以用 `--groups G1` 逐组编译；不传则编译四组。

```bash
python tools/h100/phase6/ablation2.py build-prepare "$H100_OUT/candidates"
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation2.py compile "$H100_OUT/candidates" \
  --cuda-home "$H100_CUDA_BUILD" --jobs 2
```

可选 `--nccl-root /path/to/nccl`，目录需包含 `include/nccl.h` 和 `lib/libnccl_static.a`。设备/顺序/映射身份改变时，按 [旧入口的重新校准命令](README-ablation.md#2-必要时重新校准) 在新目录校准一次，再给 `build-prepare` 传 `--plan "$H100_OUT/parent-plan.json"`；四组共享新计划，不能逐组挑选 mask。改变shape或SM数超出本协议。

## CPU 回归、无GPU编译与 dry-run

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation2.py compile-check "$H100_OUT/candidates" \
  --out "$H100_OUT/cpu-compile" --cuda-home "$H100_CUDA_BUILD"
```

`compile-check` 运行四个纯CPU物理表/`taco_selected`测试，编译四份SM90映射对象和共同codec runtime，不运行映射可执行文件或CUDA kernel，不产生可计时的完整库。

仅做CPU演练时，用独立目录运行下列命令。`source_only` 同时冻结在 preparation 记录和worker中，不能只改submission标记来升级。正式执行必须新建compare。

```bash
python tools/h100/phase6/ablation2.py prepare "$H100_LOG/compare-cpu" \
  --builds "$H100_OUT/candidates" --megatron "$H100_MEGATRON" --source-only
python tools/h100/phase6/ablation2.py run "$H100_LOG/compare-cpu"
```

编译或复用成功后，准备正式会话；prepare只做CPU操作。默认每轮5块、两轮50窗口；`--blocks` 仅接受5的正整数倍，在运行前冻结。调度seed默认20261013；第一轮随机排列一次后循环移位，第二轮同时反转块及块内顺序。每窗口10步预热＋20步完整训练（前向、反向、通信、参数更新、清梯度）。不自动追加复测。

```bash
python tools/h100/phase6/ablation2.py prepare "$H100_LOG/compare" \
  --builds "$H100_OUT/candidates" --megatron "$H100_MEGATRON" \
  --blocks 5 --schedule-seed 20261013
python tools/h100/phase6/ablation2.py run "$H100_LOG/compare"
```

dry-run会列出10个smoke/profile窗口和50个正式窗口，不导入Torch、不查询GPU、不启动任何worker。`submission.json` 冻结源码/库/控制器哈希、GPU顺序、模型配置及调度；`provenance/` 保存父计划、构建manifest、baseline来源与控制脚本快照。

## H100 上实际启动（会使用GPU）

以下命令留给后续明确执行实验时运行。本次实现任务不运行它们。推荐一次 `--stage all` 连续完成四组映射、五组smoke、五组profile及50窗口计时。

```bash
python tools/h100/phase6/ablation2.py run "$H100_LOG/compare" \
  --stage all --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/ablation2.py report "$H100_LOG/compare"
```

也可分阶段；同一套的timing阶段不会重新查询GPU占用。

```bash
python tools/h100/phase6/ablation2.py run "$H100_LOG/compare" \
  --stage preflight --cuda-home "$H100_CUDA_RUN" --execute
python tools/h100/phase6/ablation2.py run "$H100_LOG/compare" \
  --stage timing --cuda-home "$H100_CUDA_RUN" --execute
```

执行器只在整套预检开始前做一次GPU检查（host、八卡UUID/顺序、无计算进程、显存≤1024MiB、利用率≤5%），保存 `start-gpu-check.json`；忙时失败退出，没有等待/重试/Slurm队列。运行中及组间不查询占用。遵守 [watcher.md](../../../docs/design/design-h100/watcher.md)；如另行安排排队，外部队列才采用600秒/10秒/连续3次约定。本入口不创建队列。任一缺rank、数值失败、非有限loss/梯度、skip/fallback、错误路径/库/mask或输入身份不一致都会停止。失败目录与状态保留，修复后新建会话。

## 报告与交付

`report.json`/`report.md` 包含两轮分别和同一会话合并10块的全样本统计：

- 五组中位ms/step及tokens/s；每窗口以八rank最大elapsed/20计算。
- 相对同期 `native_taco` 的配对比值中位数和最好值；最好值附轮次、block、双方耗时。
- 四组对baseline，以及G2/G1、G4/G3、G3/G1、G4/G2、G4/G1的配对耗时下降及95%区间。下降为 `100*(1-exp(mean(log(T_group/T_reference))))`，完整配对块bootstrap10000次，seed20261009。rank与step不是独立重复。
- 慢窗口附表固定使用各轮各策略原始中位数的1.10倍，任一策略触发就删除整块五组，所有对比共用剩余块。记录剔除块及触发策略；不迭代重算阈值，不改原数据。零剩余块显示“无可用配对块”；一块只给点估计，区间为空。

G1对基线不代表纯排序收益；没有G0、原生BF16或固定Flux融合baseline，不套用旧4%验收。负收益和跨零区间照实保留。没有跨历史会话合并功能。

H100端先严格核验再交付。接收端没有动态库时可运行 `python tools/h100/phase6/ablation2.py report /path/to/compare --offline`，核对冻结文本和rank中的库哈希；这不代表重新读取了原动态库。

默认完整证据：400份正式rank、80份预检rank、40份压缩trace、4份候选映射；以及全部command/process日志、start-gpu-check、schedule、submission、preflight、源码/计划/构建/控制器哈希、完整和筛选报告。窗口数改动以冻结调度计算，失败记录不补写为完成。

按 [H100交付规范](../../../docs/design/design-h100/instruction-h100.md) 回传维护代码、文档、源码快照、计划、manifest和原始证据。复用时交付新登记与旧来源引用，不重复提交旧构建。`.so/.o/.a`、CMake/Ninja树、Python缓存、环境及checkpoint保持忽略。`logs/.../handoff/` 写明实际环境、代码提交/补丁、复用来源、已完成和缺失项。CPU合成回归不属于H100性能数据。

## 本次CPU交付（2026-10-14）

57项CPU回归通过（45项既有、12项新增）；真实冻结交替计划派生的四组CPU表/mask检查、四份SM90映射对象和共同codec runtime编译通过；独立source_only会话完成50正式＋10预检窗口dry-run。native_taco的adapter、TP8 codec和reference与历史H100实测快照一致，原冻结清单核验通过。没有启动GPU实验、Slurm、监控或等待队列，也未构建新的完整Flux动态库。后续仍需H100端完整库、映射、smoke/profile与正式计时验证。

按本次交付约定，CPU检查、dry-run和无GPU编译生成的结果只保留在本机，不纳入Git：`.gitignore`忽略 `outputs/h100/phase6/ablation2-*-cpu-*/`、`logs/h100/phase6/ablation2-*-cpu-*/` 及五组会话的 `cpu-compile/`。本次本地会话名为 `ablation2-gpt67-tp8-interleaved-cpu-20261014-implementation`，详细记录位于其 `logs/.../handoff/README.md`。维护代码、回归测试和本节验证摘要继续交付；正式H100实验的源码、计划、manifest和原始证据仍按上文保留。
