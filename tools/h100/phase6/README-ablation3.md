# H100 GPT 6.7B TP8：九组策略与模块范围消融

独立入口为 `tools/h100/phase6/ablation3.py`，实验协议为 `h100-gpt67-tp8-ablation3-v1`，对应 [消融实验3](../../../docs/design/design-h100/消融实验/消融实验3.md)。默认三种策略 × 三种作用范围，共九组、两轮162个正式窗口。2026-10-17状态：原H100服务器已有完整构建及预检，回传124个完整正式窗口，第二轮异常中断；[部分结果](../../../docs/design/design-h100/消融实验/base-phase6-ablation3.md)按13个完整配对块统计，不能算作完整两轮成功。

**本次新服务器只跑一轮：`prepare --blocks 9 --rounds 1`，81个正式窗口。** 新增`--rounds {1,2}`，省略仍为两轮以保持兼容；旧冻结会话没有该字段时始终按两轮解释。本轮模型、九组、校准、mask和10/20步设置均不变，预检仍完整执行18个窗口。

**新H100服务器请从 [deploy-tp8.md](../../../docs/design/design-h100/deploy-tp8.md) 开始。** 它包含固定软件环境、`DEBUG=0`及共用NCCL Release开发库、本机Attention/MLP重新校准、完整链接和两种运行范围。下文第1节的历史计划路径仅适合同机身份核验通过后的复用，不适用于新机器。只要求可运行性检查时，实际执行`run --stage preflight --execute`和`report --preflight`后停止；明确要求跑完一套才进入正式计时。

## 分组与执行实现

| 策略族 | 到达优先 | 远端量化 | ID前缀 |
| --- | --- | --- | --- |
| A | 关，保持基础交替 | 全部远端H128/E4M3 | `interleaved_all` |
| B | 开 | 与A相同的全远端mask | `interleaved_arrival_all` |
| C | 与B相同的到达表 | 定向选择性mask | `interleaved_arrival_selective` |

每个前缀分别加 `_mlp`、`_attn`、`_attn_mlp`。Attention仅为32个 `self_attention.linear_proj`，MLP仅为32个 `mlp.linear_fc2`；未选中模块均为原生Megatron BF16，双模块启用64个投影。本地贡献和GEMM保持BF16，反向沿用BF16/STE。

模型固定S2048、micro batch1/global batch4、TP8/DP1/PP1/CP1、32层/H4096/FFN16384、8192 token/step。Attention的全局shape为 `[2048,4096,4096]`，本地K512；MLP为 `[2048,4096,16384]`，本地K2048。两角色各用独立实测计划，同角色在单模块/双模块中保持相同计划。

构建目录下仅有 `base/`、`arrival/` 两套双shape库，计算/codec/封装源码相同，允许不同的源码仅为设备swizzle和主机映射头。A共用base库，B/C共用arrival库，通过显式 `all_remote`/`targeted` mask区分。A/B每来源恰好448个远端tile，C最多7个；mask绑定物理tile。旧固定 `fused_v2`、A800实现、既有五组和四组协议保持原定义。

## 1. 复用真实计划，准备新构建

以下命令在仓库根目录执行。环境沿用 [H100环境说明](../../../docs/design/env.md)；正式构建需使用有PyTorch及编译依赖的Python环境。所有输出目录必须不存在。

```bash
export H100_A3_SESSION="ablation3-gpt67-tp8-scope-$(date +%Y%m%d-%H%M%S)"
export H100_A3_OUT="$PWD/outputs/h100/phase6/$H100_A3_SESSION"
export H100_A3_LOG="$PWD/logs/h100/phase6/$H100_A3_SESSION"
export H100_A3_PLANS="$PWD/outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/plans/measured"
export H100_A3_MEGATRON=/workspace/Megatron-LM
export DEBUG=0
export MAKEFLAGS=
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX="$CXX"

python tools/h100/phase6/ablation3.py build-prepare "$H100_A3_OUT/candidates" \
  --attn-plan "$H100_A3_PLANS/attn.json" \
  --mlp-plan "$H100_A3_PLANS/mlp.json"

CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation3.py compile "$H100_A3_OUT/candidates" \
  --cuda-home /usr/local/cuda-12.8 --jobs 2
```

实际路径以执行端为准。两份计划须对应相同host、GPU UUID/顺序、模型和shape；构建器核对父校准文件、计划哈希和物理坐标。示例显式复用同一次双shape校准；若不传 `--mlp-plan`，沿用原冻结MLP计划，这与示例计划可能不同，最终来源会记录在manifest中。真实Attention计划为必需输入。

`compile`隐藏GPU，调用已有编译器分别生成两套动态库、Python binding和mapping程序，不运行mapping或GPU kernel；成功后仍保留 `gpu_validated=false`。可传 `--nccl-root /path/to/nccl`，否则沿用builder的固定NCCL构建流程。新服务器按部署指南构建一次含`include/nccl.h`和`lib/libnccl_static.a`的Release开发库，在采样/base/arrival编译时引用同一路径。固定NCCL Makefile只把`DEBUG=0`识别为优化构建，`DEBUG=release`也会启用`-O0 -G`；CMake Release不会覆盖它。两套构建使用完整的相同计算来源，修改A/B/C的模式不修改冻结codec。

## 2. 如需新校准

已有同机计划通过核验时不必重新校准。新服务器、设备/顺序、shape或基础映射发生变化时，在新会话执行独立的BF16双shape校准；沿用现有scope校准协议，输出角色计划，再进入第1节，显式传入新的`attn.json`和`mlp.json`。下列编译也须沿用第1节的`DEBUG=0`和编译器环境。

```bash
python tools/h100/phase6/ablation3.py calibration-build "$H100_A3_OUT/calibration" --roles attn mlp
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation3.py compile "$H100_A3_OUT/calibration" \
  --cuda-home /usr/local/cuda-12.8 --jobs 2
python tools/h100/phase6/ablation3.py calibration-prepare "$H100_A3_LOG/calibration" \
  --build "$H100_A3_OUT/calibration"

# 默认只打印计划；这里没有执行GPU校准。
python tools/h100/phase6/ablation3.py calibration-run "$H100_A3_LOG/calibration"
```

后续明确安排GPU校准时，在 `calibration-run` 增加 `--execute --cuda-home /usr/local/cuda-12.4`。校准完成后才可拟合：

```bash
python tools/h100/phase6/ablation3.py fit "$H100_A3_LOG/calibration" "$H100_A3_OUT/plans/measured"
```

每shape三遍、192个repeat，pass0/1训练、pass2诊断；基于同一份角色评分派生A/B/C，不针对不同组重新拟合。采样库不进入正式训练计时。

## 3. 冻结九组与dry-run

```bash
python tools/h100/phase6/ablation3.py prepare "$H100_A3_LOG/compare" \
  --build "$H100_A3_OUT/candidates" --megatron "$H100_A3_MEGATRON" --blocks 9 --rounds 1
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare"
```

默认每轮9个块，每块九策略各一次；固定seed20261013排列后循环移位，使每组在九个位置各出现一次。仅选择两轮时，第二轮反转块及块内顺序。`--blocks`可在prepare时设置为9的正整数倍。每窗口10步预热＋20步完整step，包含四个微批的前向/反向、通信、更新和清梯度。

| prepare设置 | 正式窗口 | 正式rank记录 | 正式step | 含预检的worker窗口 |
| --- | ---: | ---: | ---: | ---: |
| 本次`--blocks 9 --rounds 1` | 81 | 648 | 1620 | 99 |
| 兼容默认`--blocks 9 --rounds 2` | 162 | 1296 | 3240 | 180 |

两种模式都另有18个smoke/profile窗口、144份预检rank、72份profile trace，以及12个策略/角色mapping任务。dry-run只读取冻结文本并打印12项mapping与所选轮数对应的worker任务，不导入Torch、不查询GPU、不启动任何mapping或worker。

轮数在prepare时写入submission的`rounds`、准备记录的`rounds`和worker配置的`repetitions`；run/report交叉核验并按该调度执行，不接受临时修改轮数。旧两轮实验即使只完成第一轮也仍是不完整实验；本次须新建compare，不能删除第二轮调度或补写完成标记。

## 4. 后续H100执行与报告

以下命令在H100执行端明确安排完整一套时使用；单纯阅读本页不会启动实验：

```bash
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare" \
  --stage all --cuda-home /usr/local/cuda-12.4 --execute
python tools/h100/phase6/ablation3.py report "$H100_A3_LOG/compare"
```

可拆为 `--stage preflight` 和 `--stage timing`，timing必须已有相同submission的完整预检。入口在preflight/all开始前检查GPU占用；单独启动timing不会重复该检查，执行者须先确认八卡仍空闲且属于本任务。忙时停止；不创建等待器、重试、Slurm或队列，运行中不周期查询占用。失败保留阶段状态和部分数据，需新会话恢复，不覆盖已启动目录，也不通过删改状态文件实现续跑。具体命令、控制器日志保存及仅预检分支见部署指南。

预检按实际角色路由核验模块调用、完整shape、量化字节、数值容差、独立packet/mask地址和模块实例。profile只计CPU角色注解，通过CUDA launch correlation识别对应V2/解码kernel，避免把GPU镜像注解重复计数；未选中投影不得出现候选kernel。训练窗口核对全部八rank、初始化/RNG/token/环境一致、实际库哈希、零skip/意外fallback。

报告主表保留全样本，按冻结的实际轮数输出九组完整step中位耗时/吞吐及合并统计，并明确`round_count`和`blocks_per_round`。本次仅有第1轮9块，`combined`也是这9块，不重复计样本；没有第二轮反转验证。两轮模式则分别给出两轮及合并18块。列出九项同范围B/A、C/B、C/A比较及九项同策略scope比较。每项提供配对几何平均耗时下降、10000次完整块bootstrap区间、中位加速比和最好单块及双方耗时。没有4%验收门槛，也没有同期native_taco/固定融合v2，不将这些未测组用作分母。

敏感性附表使用每轮各组全样本中位数的1.10倍阈值，任一组触发就排除整块九组，所有18项比较使用同一集合，阈值不迭代；零块不估计，单块无区间。最好单块不是稳定收益，逐项区间未做多重比较校正。`report --offline`仅用于回传文本/记录核验，不声称重新读取原H100运行库。

## 5. CPU演练与编译检查

```bash
export H100_A3_CPU_OUT="$PWD/outputs/h100/phase6/ablation3-gpt67-tp8-cpu-$(date +%Y%m%d-%H%M%S)"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

python tools/h100/phase6/ablation3.py build-prepare "$H100_A3_CPU_OUT/candidates" --source-only
python tools/h100/phase6/ablation3.py prepare "$H100_A3_CPU_OUT/compare" \
  --build "$H100_A3_CPU_OUT/candidates" --source-only --rounds 1
python tools/h100/phase6/ablation3.py run "$H100_A3_CPU_OUT/compare"
python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v

# 可选，仅运行CPU映射/mask检查程序并编译CUDA对象，不执行GPU程序。
python tools/h100/phase6/ablation3.py compile-check "$H100_A3_CPU_OUT/candidates" \
  --out "$H100_A3_CPU_OUT/cpu-compile" --cuda-home /usr/local/cuda-12.8
```

`source_only`和诊断状态写入构建、父/派生计划、submission及准备记录，生成worker的首行永久拒绝执行。编译不会改变该属性，不能改标记把合成计划用于实测；真实运行必须使用实际校准另建构建和compare。CPU测试中的模拟rank/trace仅生成于临时目录，worker仍禁用，不作为GPU证据。

编译检查分别覆盖两种顺序×两种shape的主机映射及既有选择判断，编译两份SM90 mapping对象和codec runtime对象；不会加载动态库或运行CUDA mapping程序。全远端/选择性mask及其预算/字节语义另外由九组CPU回归覆盖。正式运行库的完整链接及GPU检查是不同阶段。

## 6. 交付

维护代码、使用说明、九组方案，以及正式会话的源码/计划/manifest/调度/原始rank/trace和日志按既有规则交付。新增`.gitignore`精确保留九组源码快照所需`.ld`、`.clang-format`和Triton输入；CPU演练目录、`.so/.o/.a`、编译树、环境及缓存继续忽略。

## 开发验证记录（2026-10-16）

最终完整CPU回归85项全部通过，包括72项既有检查和13项新增检查。新增检查覆盖九组/162窗口平衡调度、A/B同全远端mask、B/C同到达表、两角色独立计划、九组实际adapter在预热与正式阶段的路由、显式codec模式、profile角色关联、source-only拒绝执行，以及临时模拟的完整180窗口预检/报告和错误路径。

CPU演练完成两类顺序的源码准备、九组compare冻结、12项映射与180项worker任务dry-run。两类顺序×两种shape的CPU映射/mask检查通过，CUDA Toolkit12.8完成两份SM90 mapping对象及两份codec runtime对象编译；没有执行CUDA mapping程序、加载新动态库或运行任何GPU kernel。本次未进行九组完整动态库/Python扩展链接，后续正式环境按第1节编译。

已另用回传的真实Attention/MLP计划完成CPU源码准备，分别保留 `[7,7,7,2,7,7,7,1]` 与 `[2,7,7,2,7,7,7,0]` 的选择量；该准备不构成GPU正确性验证。冻结目录660项校验和与2147项历史引用哈希通过，既有实验记录未修改。CPU演练和编译产物位于本机临时目录，正式交付仅含维护代码、文档和验证摘要。

以上保留开发阶段的验证边界；后续原H100执行状态以本页开头及回传结果文档为准。原两轮162窗口实验仍未完成；新服务器本次另行登记一轮81窗口，并独立通过实际预检，不能从CPU演练、对象编译或已有四组/五组结果推断。

## 单轮支持验证（2026-10-17）

`CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/h100/phase6 -p 'test_ablation3.py' -v`：19项CPU测试通过。覆盖显式一轮/默认两轮CLI、第一轮原调度、99/180个worker窗口dry-run、轮数冻结及篡改拒绝、旧两轮元数据兼容，并用临时模拟记录分别完成81/162个正式窗口的预检和报告；缺rank、缺窗口及单轮中出现额外第二轮均拒绝。模拟记录只存在于临时目录，worker保持禁用，不作为GPU证据。

原`ablation3-gpt67-tp8-scope-20261016-221018-release1/compare`另通过只读离线输入核验，仍识别为两轮162窗口；未改写历史数据或报告。文档命令语法与CLI参数核对通过。本次只修改Python调度/报告及说明，没有编译CUDA或启动GPU、Slurm、等待队列。
