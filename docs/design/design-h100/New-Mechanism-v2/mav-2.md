# MAV2：在 Flux＋基础量化上单独验证选择性变换

更新：2026-10-30。实现 [idea.md 的 Idea 1](idea.md#3-idea-1选择性变换量化不必总经过-h128)，本次对照采用 [baseline.md](../New-Mechanism/baseline.md) 中的 **Flux＋基础量化 v2**，具体路径见 [a-baseline.md](../New-Mechanism/a-baseline.md)。选择性候选只在 D8（直接块缩放 FP8）和 H8（原 H128/E4M3）之间选择；**所有远端贡献始终量化，本地贡献始终 BF16**。另加入 `d8` 组，强制全部远端使用直接 FP8，不执行选择判断或 H8 回退。

当前完成代码与测试入口，尚无本次 H100 八卡数值或性能实测。历史基础组的 582.318 ms 只用于识别基线；MAV2 的收益必须由本次同期对照计算。

## 1. 一条命令启动对比

在 `TempName-New` 仓库根目录运行：

```bash
# 只打印计划，不创建批次、不构建、不查询 GPU
bash tools/h100/phase6/run_mav2.sh

# 默认一次完整三臂比较（basic、mav2、d8）；构建、预检、profile、正式训练、报告自动执行
bash tools/h100/phase6/run_mav2.sh --execute

# 前台观察首轮构建和数值预检
bash tools/h100/phase6/run_mav2.sh --foreground --execute

# 两次独立比较，分别报告结果
bash tools/h100/phase6/run_mav2.sh --trials 2 --execute
```

启动器沿用 `run_ma11.sh` 的环境加载方式，读取 `outputs/h100/container-recovery/env.sh`（若存在）。可通过 `MAV2_PYTHON=/absolute/path/to/python` 选择原训练环境；其余参数见 `bash tools/h100/phase6/run_mav2.sh --help`。`--flux-build` 可复用通过来源检查的 GemmOnly 构建；`--codec-build` 必须是本次三臂版本的 MAV2 codec 构建；旧基础库及上一版双臂库都会被来源检查拒绝。

默认后台运行，双重 fork 并确认 supervisor 归属容器 PID 1；启动成功不代表实验完成。容器须保持运行。批次目录为 `logs/h100/phase6/mav2-three-arm-1x-50steps-<时间>/`，查看：

- `batch-state.json`：当前阶段、完成或失败原因。
- `handoff/container-supervisor.log` 及各阶段日志：构建、worker 和预检输出。
- `compare1/preflight.json`：GPU codec 自测、模型 smoke 和执行路径检查。
- `compare1/report.json` / `report.md`：完整正式窗口、误差、同期配对收益及区间。

## 2. 对照与候选的唯一机制差异

| 项目 | `basic`：Flux＋基础量化 v2 | `mav2`：选择性变换 | `d8`：强制直接 FP8 |
| --- | --- | --- | --- |
| GEMM | 原 Flux Hopper GemmOnly，先得到完整 BF16 partial | 同一构建、同一动态库 | 同一构建、同一动态库 |
| 量化对象 | Attention output 与 MLP FC2 的全部远端贡献 | 相同，无选择性量化 mask | 相同，无选择性量化 mask |
| 编码 | 原 TACO adaptive scale → H128 → quant scale → E4M3 | 逐组尝试 D8，超限走原 H8 | amax → scale → E4M3；全部 D8 |
| 在线误差选择 | 无 | 实际 FP8 回转及误差归约 | 无，跳过回转与选择归约 |
| 通信 | NCCL All-to-All | 相同包长、splits 和调用方式 | 相同包长、splits 和调用方式 |
| 接收 | 单 kernel 七来源解码＋按序 BF16 求和＋本地贡献 | 同样融合、顺序与舍入，按 tag 解码 | 同一候选接收 kernel，所有组走 D8 分支 |
| 每 128 元素实际包长 | 128 B FP8＋两个 FP32 槽＝136 B | 同为 136 B，第二槽复用为 D8 tag | 同为 136 B，第二槽全部为 D8 tag |
| 本地贡献 | BF16 | BF16 | BF16 |

没有引入 MA11、MA8 激活缓存、MA9、MA4-F、生产端融合、直接 IPC、到达排序或新的量化覆盖策略。三组沿用基础 v2 的投影前后向和训练配置。选择性候选新增的选择归约、FP8 回转、分支和 tag 读取全部保留在正式计时中。

保留固定 136 B 布局是有意的实验控制：本次检验少做正逆 Hadamard 是否带来净收益，不把 D8 理想 132 B 的载荷变化混入结果。实际通信字节没有因 D8 比例上升而减少。

## 3. 本版的组级选择规则

本版采用 **在线、单 token 的 128 通道组误差门禁**。`idea.md` 建议先使用独立校准冻结模式表；本次没有现成真实层校准样本，因此不伪造离线表，也不从随机矩阵推断真实层模式。在线判断是该思想的一种可运行实现，其额外成本可能抵消跳过变换的收益。

对 BF16 组 \(x\)，尝试：

\[
s=\max_j|x_j|/448,\qquad q=\operatorname{E4M3}(x/s),\qquad
\hat x_D=\operatorname{BF16}(s\operatorname{FP32}(q)).
\]

使用实际 BF16 回转结果计算误差，固定要求：

\[
\|\hat x_D-x\|_2/\|x\|_2\le 0.025,\qquad
\max|\hat x_D-x|/\max|x|\le 0.0625.
\]

同时满足才用 D8，否则用原 H8；全零组直接编码为 D8 零。门禁只读取当前 token 的当前通道组，不混入其他 token。极小正 scale 需要数值保护；非有限输入报错，不能清零后算通过。

H8 的两个 scale 保持原含义；D8 用第一槽保存直接 scale，第二槽写 `-1.0f` 作为标记。D8 接收路径直接乘 scale 并舍入到 BF16，跳过逆 H128。每个来源重建到 BF16 后，仍按 `(rank+1)%8, …, (rank+7)%8` 顺序累加，每次加法都舍入 BF16，最后加入本地 BF16。输出保持独立存储。

这些门槛是预注册实验参数，不是训练质量保证。H8 是基础 codec 的回退表示，并不保证每个 H8 组都比 D8 误差小；最终仍需检查完整归约输出和训练。此处不加入 BF16 组级回退，因为它会改变量化覆盖率，混入用户明确暂不研究的选择性量化。

### 强制 D8 组

`d8` 使用与选择性候选相同的 D8 scale 和量化公式，但所有远端组无条件编码为 D8。生产端得到 FP8 字节后直接写包，跳过实际 FP8 回转、误差归约和 H8 编码；接收端仍保留原融合归约，每组都走直接解码路径。非有限输入仍报错。

这组检验“完全省去变换与选择判断”能达到怎样的速度和误差。独立 smoke/profile 必须确认所有投影的真实样本 **D8=100%、H8=0**，数值 oracle 也使用强制 D8，而不是选择性参考。原有完整模型数值门禁仍适用；“强制 D8”只规定表示，不把质量失败视为通过。

## 4. 测试协议与判定

每次比较固定 GPT-6.7B 配置：TP8、32 层、hidden 4096、FFN 16384、序列 2048、micro batch 1/global batch 4、BF16、无 recompute，覆盖 Attention＋MLP。

两轮各六个完整三组配对块：第一轮覆盖三组的六种排列各一次，第二轮反向镜像，平衡每组的执行位置与先后顺序。共 **36 个正式窗口**，每臂 12 个。每窗口重新初始化，10 步预热＋50 步正式，共 1800 个正式 optimizer step、288 份正式 rank 记录；不是一条连续 1800 步训练轨迹。独立 smoke/profile 共六个八卡窗口、48 份 rank 记录和 24 份 trace。完整步包含前后向、通信、梯度处理、参数更新与清梯度。

窗口步时为八 rank 最大墙钟除以 50。按完整配对块的对数耗时比做 10,000 次 bootstrap，报告 `mav2/basic`、`d8/basic`、`d8/mav2` 三对耗时下降及 95% CI，保留全部慢窗口和负收益。区间下界大于零且相关候选质量通过，才支持该批正增量；三对区间是未做多重比较校正的逐对区间。`completed` 只表示实验流程完成。

正式计时前执行独立数值与 profile 检查。D8 改变了表示误差，因此不能沿用基础 v2 对 CUDA 组的“跨臂逐位一致”要求；必须分别检查 codec 实现正确性与训练质量。

| 检查 | 固定要求 |
| --- | --- |
| codec 输入 | zeros、small、outliers、极小 BF16、D8/H8 混合、真实远端形状；覆盖八个来源排列 |
| 强制 H8 | 与原编码包逐位一致，并检查原接收解码语义 |
| 强制 D8／混合包 | 用独立 tensor 参考与实际包字节解码参考交叉检查 |
| codec 重建误差 | 相对 BF16 L2 ≤5%；选择性参考 ≤1%，强制 D8 参考 **≤0.8%**；实际包参考 ≤0.5% |
| 来源归约 | 与先逐来源解码到 BF16、再按原顺序逐次 BF16 相加逐位一致 |
| 真实模型投影 | 沿用 smoke 的相对 BF16 5%／相对本策略 codec 1% 门槛；mav2 参考支持 D8/H8，d8 使用纯 D8 参考 |
| 跨臂 loss | 每点绝对差 ≤`0.05 + 0.05 × abs(basic)`，含初始、预热末步和正式轨迹 |
| 跨臂 grad_norm | 每点绝对差 ≤`0.1 + 0.10 × abs(basic)` |
| 路径 | 相同 SM90 GemmOnly 与 NCCL All-to-All；各自编码与融合接收 kernel 确实执行 |
| 存储与异常 | 保留输出不被后续调用覆盖；非有限输入由独立 CUDA context 验证报错 |

mav2 和 d8 分别相对 basic 检查上述 loss／grad_norm 门槛，并报告各自质量状态；不要求两个候选的轨迹彼此接近。上述门槛是短程筛查条件，**不是全部参数梯度逐位一致，也不是长期收敛结论**。接近零的 loss 主要受绝对容差控制，因此 JSON 同时保留均值、末步比值和曲线归一化 RMSE，避免只报“通过”掩盖相对差异。预检失败阻止正式计时；正式轨迹超限仍保存全部窗口和失败报告，但不认定为合格加速。

每个 rank、每个投影模块在独立 smoke/profile 中保留首次真实调用的编码包，窗口结束后读取 D8/H8 数量；报告按 Attention／MLP 汇总。这个比例只代表首包样本，不是 50 步训练的完整比例。正式计时不保留诊断包，也不执行额外统计归约或主机同步。

本机没有 CUDA 设备，不能在此确认 GPU 自测、完整训练质量或加速结果。上机数值门禁失败时应保留日志并停止，不把未通过的候选报告为成功；正式运行出现负收益时照常出报告。

## 5. 实现与本地验证

实现入口：

| 文件 | 职责 |
| --- | --- |
| [run_mav2.sh](../../../../tools/h100/phase6/run_mav2.sh)、[detach_mav2.py](../../../../tools/h100/phase6/detach_mav2.py) | 只读预览、自动构建、后台运行、完整报告复核 |
| [mav2_codec.cuh](../../../../tools/h100/phase6/mav2_codec.cuh)、[mav2_build.py](../../../../tools/h100/phase6/mav2_build.py) | D8/H8 编码和融合解码、冻结原 H8 源码、构建三臂共用库 |
| [mav2_codec.py](../../../../tools/h100/phase6/mav2_codec.py) | 原 GemmOnly 后的 packet、NCCL 交换、分臂解码及诊断 |
| [mav2_reference.py](../../../../tools/h100/phase6/mav2_reference.py)、[mav2_validation.py](../../../../tools/h100/phase6/mav2_validation.py) | 独立参考与 GPU 数值自测 |
| [mav2.py](../../../../tools/h100/phase6/mav2.py)、[mav2_prepare.py](../../../../tools/h100/phase6/mav2_prepare.py)、[mav2_sources.py](../../../../tools/h100/phase6/mav2_sources.py) | 冻结 worker、准备并运行完整训练 |
| [mav2_protocol.py](../../../../tools/h100/phase6/mav2_protocol.py)、[mav2_report.py](../../../../tools/h100/phase6/mav2_report.py) | 固定排程与门槛、执行路径核对、数值和配对统计 |

构建记录校验原 H8 源码、候选源码和二进制哈希；提交时冻结控制器、生成 worker、模型配置、通信环境与所有阈值。比较过程中修改这些输入会被拒绝，不能边跑边换实现或放宽门槛。

本次三臂扩展的本地验证已完成：

| 项目 | 结果 |
| --- | --- |
| codec / build / harness / launcher | **42 项测试通过**（13 / 6 / 11 / 12） |
| 三臂排程与报告 | 36 窗口、1800 正式步、三对收益；D8 质量失败不影响 MAV2 的独立判定 |
| 强制 D8 路径检查 | 正式策略禁止模式覆盖；跳过选择回转和归约；拒绝观察到 H8 的 D8 样本 |
| 新 codec CUDA 12.8／SM90 编译、链接、库加载与导出 | 通过；三组共用同一库 |
| source-only 准备／验证、冻结 worker 依赖导入 | 通过 |
| 脚本语法与只读预览 | 通过；预览未创建训练批次 |
| 完整 Flux 构建、GPU 数值自测、八卡训练和实测 profile | **本机未执行** |

本机 PyTorch 为 **2.6.0+cu124**，CUDA 设备数 **0**，编译器为 `/data/apps/cuda/12.8/bin/nvcc`。本次汇总见 [validation.json](../../../../outputs/h100/phase6/mav2-d8-validation-20261030/validation.json)，完整训练控制器测试输出见 [harness-tests.log](../../../../outputs/h100/phase6/mav2-d8-validation-20261030/harness-tests.log)，codec 构建见 [manifest.json](../../../../outputs/h100/phase6/mav2-d8-validation-20261030/codec-build/manifest.json) 和 [compile.log](../../../../outputs/h100/phase6/mav2-d8-validation-20261030/codec-build/compile.log)。这些记录不能当成 GPU 正确性或性能结果。

上一版双臂的 33 项 MAV2 测试及 20 项基础回归记录保留在 [旧 validation.json](../../../../outputs/h100/phase6/mav2-validation-20261030/validation.json)，未覆盖或混入本次记录。

可复跑本地测试：

```bash
python -m unittest discover -s tools/h100/phase6 -p 'test_mav2_*.py' -v
python -m unittest discover -s tools/h100/phase6 -p 'test_basic_quant*.py' -v
```

完整实验结束后，可从原始 rank 数据重算报告：

```bash
python tools/h100/phase6/mav2.py report <批次目录>/compare1 --offline
```

三组协议为 `h100-mav2-selective-transform-three-arm-v2`，与先前双臂记录分开，不合并窗口或复用旧批次。

若 D8 占比不低但完整步没有变快，结果说明本版选择成本、编码或融合接收开销抵消了变换节省；应保留该负结果，不将 D8 占比当作性能收益。

## 6. 回传 smoke 报错修复与 D8 参考门槛更新

回传批次 `mav2-three-arm-1x-50steps-20261030-215237-552847` 的 Flux 和 codec 构建均完成，首个 `smoke-basic` 在模型初始化之前的 codec 自测失败，正式训练窗口为 **0**。[worker 原始日志](../../../../../Load/logs/mav2-three-arm-1x-50steps-20261030-215237-552847/compare1/results/smoke-basic/process.log)记录 rank 3 的 `small` 用例：D8 相对 BF16 误差为 **2.5446%**，相对参考误差为 **0.513869%**，实际包解码参考误差为 **0**，按来源顺序的 BF16 累加及强制 D8 入口一致性均通过。因此 `compare1-controller.log` 中的进程退出异常只是 worker 自测失败的上层报错，不能当作编译或完整训练失败。

参考原来使用 `amax / 448`。PyTorch 2.6 的 CUDA 标量除法会改为乘 FP32 倒数，而本实现用 `__fdiv_rn(amax, 448.f)`；这两者的 scale 可能差 1 ULP。[PyTorch 对应源码](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu#L32-L44)明确实现了该特化。本地构造的 BF16 小值反例会将 E4M3 中点 `368` 移到中点下方，使 payload 从 `384` 变为 `352`。该反例复现了误差放大机制；回传没有实际输入和 packet dump，不能声称已逐元素重演原 rank 3 样本。

本次修改：

- `mav2_reference.py` 改用同设备、同 dtype 的 Tensor 常量 `448` 作除数，使参考执行 FP32 除法。正式 CUDA codec 路径未改。
- 按用户要求，D8 **相对参考**误差门槛从 **0.5% 更新为 0.8%**，由 `D8_REFERENCE_L2_LIMIT = .008` 统一控制 GPU 自测与报告证据校验，并将门槛写入自测记录。回传的 `0.513869%` 在新门槛内。
- 实际包解码门槛仍为 0.5%，输入重建门槛仍为 5%，投影参考门槛仍为 1%，选择性变换的 2.5% 组级选择规则不变。
- 增加确定性 `rounding_midpoint` GPU 用例，独立要求 scale、FP8 payload 和参考重建精确一致；另增加 CPU 模拟 CUDA 标量除法的回归，以及 0.8% 边界和其他误差门槛未变化的检查。

本次本地回归共 **49 项通过、1 项因无 CUDA 设备跳过**，包含 build、完整三组 harness 和 launcher；没有重新编译未改动的 CUDA kernel。修复记录见 [validation.json](../../../../outputs/h100/phase6/mav2-d8-reference-fix-20261030/validation.json)，构建协议／harness／启动器的 29 项回归输出见 [tests.log](../../../../outputs/h100/phase6/mav2-d8-reference-fix-20261030/harness-build-launcher-tests.log)。GPU 中点检查及原 rank 3 失败是否消失，仍需上机确认。

同步修复后的源码到 H100 后，创建新批次重跑。可以复用本次已成功的 Flux 构建，避免重复编译 GEMM；codec 的来源身份含参考和验证脚本，因此让启动器重新构建 codec，不传旧 `--codec-build`：

```bash
# 在 H100 的 TempName-New 仓库根目录执行
bash tools/h100/phase6/run_mav2.sh \
  --flux-build logs/h100/phase6/mav2-three-arm-1x-50steps-20261030-215237-552847/flux-build \
  --execute
```

原回传日志及失败批次保留。本机仍无 GPU，修复后的 H100 smoke 和完整三组性能需要以上新批次确认。
