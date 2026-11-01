# OP2：A/B/C 统一叠加 MA89 与 MA11 优化

更新：2026-11-02。目标为单机 **8×RTX 5090，TP8/DP1**。独立协议 **`rtx5090-phase6-op2-abc-ma11-v1`**，入口 [run_op2_abc.sh](../../../tools/5090/phase6/run_op2_abc.sh)。本轮只比较三组，默认 **2 个 block、6 个正式窗口、300 个正式 optimizer step**。

**当前状态：代码和运行入口已实现，尚无 OP2 八卡训练结果。** 本地编译和 CPU 回归不代表三组已通过目标 GPU 检查，也不预设叠加后的加速比。

## 1. 三组叠加什么

[baseline.md](baseline.md) 中，MA89 在 CR3 上加入 **MA8 激活复用＋MA9 保序零梯度统计**；MA11 在 MA89 上进一步加入 **attention mask/softmax 中间态消除**。MA11 已包含 MA89，因此三项优化各安装一次，不重复安装 MA89，也不拆成六组。

| 本轮展示组 | 脚本 ID | 保留的 OP1 投影路径 | 共同叠加 |
| --- | --- | --- | --- |
| **A＋MA89＋MA11** | `flux-basic-interleaved` | 交替排序 GemmOnly；全部远端量化；直接 packet；一次 All-to-All；融合解码/归约 | MA8＋MA9＋MA11 softmax |
| **B＋MA89＋MA11** | `flux-basic-arrival` | 同一交替 GEMM/codec；四个 64 行分块；通信完成事件触发解码 | MA8＋MA9＋MA11 softmax |
| **C＋MA89＋MA11** | `flux-basic-selective` | 与 B 相同的分块；仅每目的 packet 最后 1/64 H128 group 量化，其余 BF16 | MA8＋MA9＋MA11 softmax |

这里迁移的是 **MA89 与 MA11 相对原基底新增的模型和 optimizer 优化**。不把旧 CR3 的融合 producer、bias/residual consumer 或 H100 选择表替换进来；A/B/C 继续使用各自已经测过的 Flux 投影路径。没有增加“原 A/B/C”“MA89-only”或“Flux＋改进量化”作为第四组。

三组共用同一个 ABC 构建、同样的优化小库和同样的安装代码。模型、反向算术、Adam、梯度范数、裁剪、输入和训练计时循环均保留原有协议。MA8 改变激活保存与反向 AllGather；MA9 只改变统计实现；MA11 只改变指定 softmax 实现。

## 2. 优化落地与隔离

### MA8：复用前向完整激活

使用已验证的 `ma8_adapter.install_activation_cache(raw, 'F4-cache')`，覆盖全部 **32 个 QKV＋32 个 FC1** 列并行模块。前向 AllGather 结果由 autograd 上下文独立持有，反向 wgrad 直接复用，消除这 64 个模块重复的反向输入 AllGather。词表输出头和 A/B/C 的行并行反向不替换。

安装前继续校验原 Megatron 类、前向方法和 wrapper 的源码哈希、模型形状、BF16、sequence parallel 等条件。保留激活会增加显存占用，仍由完整模型 smoke 和峰值显存记录判断能否运行。

### MA9：保序零梯度统计

复用原 `ma9_adapter.py` 和 `ma9_zero_count.cuh`，分块计数后按原参数顺序进行 FP32 汇总；保留参数过滤、TP 重复参数处理、AllReduce 和原浮点舍入顺序。不能用一个整数总和最后转 FP32 代替原逐张量累加。

[op2_abc_build.py](../../../tools/5090/phase6/op2_abc_build.py) 把原 header 原样编译为独立的 `libop2_ma9_zero_count.so`，实际机器码为 **SM120**。这样无需向 ABC 进程额外加载旧 MA11 的整套 `libflux_cuda.so`，也不会替换已验证的 ABC GEMM 二进制。

### MA11：5090/PyTorch 2.8 softmax

复用 [ma11_softmax_5090.cu](../../../tools/5090/phase6/ma11_softmax_5090.cu) 和现有 `baseline_ma11_compat.py`。当前绑定已验证的 NGC 25.06 PyTorch **`2.8.0a0+5228986c39.nv25.06`**。不回退到 H100/PyTorch 2.6 的归约实现。

在目标 GPU 上先做独立六组前向/反向逐位自检；三组模型 smoke 再校验每次实际 softmax 的前后向结果。安装覆盖全部 32 层 attention softmax，QK/PV GEMM、attention dropout 和其他 attention 配置不改变。

### 安装和状态采集

[op2_abc_sources.py](../../../tools/5090/phase6/op2_abc_sources.py) 在模型和 ABC Adapter 创建后、第一次训练前按 **MA8→MA9→MA11** 的顺序安装三项机制。源码检查要求安装在顶层恰好一次，原训练函数和完整 step 计时 AST 必须保持一致。

动态状态在全部预热和正式/profile step 完成后采集，随后写 rank JSON：包括 ABC 的 64 份通信状态、MA9 最终计数、MA11 前后向审计次数及两份优化库的实际路径/哈希。不会在实例尚未创建时保存空状态，也不会只保存预热计数。该时机继续使用已修复的 OP1 实现。

## 3. 本轮保留的边界

上一轮 [OP1 完成报告](../../../../Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-002446-998423/report.md) 中，A/B/C 分别为 **966.738 / 1077.683 / 1074.402 ms/step**；B/C 没有观察到解码与后续通信的重叠。这些数值仅作历史背景，不计入 OP2 的统计。

按本轮“在 ABC 上都加优化”的范围，A/B/C 的 packet、量化比例、分块数、stream 依赖、`CUDA_DEVICE_MAX_CONNECTIONS=1` 和 NCCL 配置均延续 OP1。此前发现的 B/C 分块与工作队列问题没有在本轮同时改动，因此新增 MA89/MA11 不代表这些成本会消失。

本轮可以比较 **共同启用 MA89/MA11 后，哪条 ABC 路径更快**。由于没有同期未优化对照或 MA89-only 对照，不能从本轮单独证明 MA8、MA9 或 MA11 各自的增量收益，也不能直接用跨批次差值做严格归因。

## 4. 运行命令

同步整个 `tools/5090/phase6/` 到目标环境，在 `TempName-New` 根目录运行。优先复用已经完成的 ABC 构建，只新增编译两个小库：

```bash
# 无副作用预览：不查询 GPU、不创建目录、不启动训练。
bash tools/5090/phase6/run_op2_abc.sh

# 推荐：只测三组，默认后台运行；默认 2 个 block。
bash tools/5090/phase6/run_op2_abc.sh --execute \
  --abc-build /workspace/Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/build

# 在当前终端查看阶段日志。
bash tools/5090/phase6/run_op2_abc.sh --execute --foreground --blocks 2 \
  --abc-build /workspace/Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/build

# 只执行硬件/Apex/八 rank NCCL 预检。
bash tools/5090/phase6/run_op2_abc.sh --execute --preflight-only

# 若需要完整的三组排序平衡周期，使用六块。
bash tools/5090/phase6/run_op2_abc.sh --execute --blocks 6 \
  --abc-build /path/to/verified-op1-abc/build

# 完整批次的只读证据与报告重放。
bash tools/5090/phase6/run_op2_abc.sh --report --out /path/to/completed-op2-batch
```

上面的具体 ABC 路径来自上一轮成功批次实际复用的构建记录；目标机需保留该目录及其父构建。只有日志而缺少二进制的本地回传目录不能用作构建。`--abc-build` 复用的是投影构建，MA9/MA11 小库在本轮新目录编译和校验，不修改父构建。

如果没有 ABC 构建，入口也支持原有 `--baseline-build` / `--flux-build`；这三个复用参数互斥。不提供时会创建完整依赖构建，耗时更多；附带构建中的其他变体不会进入本轮 native 探针、模型或性能比较。`--transport-smoke-only` 停在 softmax 独立自检及三组 ABC native 探针之后，MA9 与三项组合的完整验证仍须模型 smoke。

结果保存为：

```text
../Load-5090/logs/5090/phase6/op2-abc-ma11-v1-1x-50steps-<时间>/
```

查看 `batch-state.json`、`handoff/supervisor.log`；完成后生成 `report.md` / `report.json`。本轮的优化库放在 `ma9-compat/` 和 `ma11-compat/`。默认后台运行需要容器持续存活。

## 5. 计时与验证

模型仍为 **32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1**；每 optimizer step 处理 **8192 tokens**。

默认 **1 trial × 2 blocks × 3 组**：`block00`、`block01` 各三组；每窗口 **10 步预热＋50 步正式计时**，共 **6 个正式窗口、48 份正式 rank 记录、300 个正式 optimizer step**。另有三组完整模型 smoke 与三组 profile，不混入正式统计。

两块或三块短测只给描述统计和三项配对点估计，不计算置信区间；六块采用三组的六种排列，平衡组别位置和前后关系。`--blocks` 接受 2、3 或 6 的正整数倍，`--trials` 支持 1 或 2。

全部源码、硬件、NCCL、构建身份、softmax 自检、三组 ABC native 检查通过后，依次运行三组 smoke、三组八 rank profile，全部通过才开始正式计时。四项实际路径检查同时要求：

- **ABC forward/row backward**：各组原编码、通信、解码、输出所有权和反向 GEMM/AllGather；不能套用旧 CR3 的 `ConsumerGemmRSFunction` 与融合 bias 反向计数。
- **MA8**：256 个列投影的前向 AllGather 保留，反向对应重复 AllGather 为零，四个词表输出头仍使用原路径。
- **MA9**：两次真实统计 kernel、原参数顺序与统计 AllReduce；smoke 保留六项边界自检及每步精确审计。
- **MA11**：128 次前向和 128 次反向实际 native softmax launch、完整 attention 邻接依赖；旧 mask/softmax 中间态路径不得混入。

三组都记录有限 loss/梯度范数、全部训练参数的 smoke 梯度/更新、初始权重/RNG/token 身份、64 个目标模块、零 fallback 和实际优化库身份。保留全部窗口与负收益。报告取八 rank 最大墙钟除以 50，给均值、中位数、最慢窗口、吞吐、显存及三项组间配对；显存仍分开报告 Torch allocated/reserved 与独立 native workspace，不把它们当作整卡总峰值。

## 6. 本地检查

本次新增 **12 项 OP2 检查全部通过**；整个 `tools/5090/phase6` 测试集 **234 项全部通过**。默认两块及完整六块入口预览、shell/Python 语法、文档文件链接和 whitespace 检查也通过。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op2_abc*.py' -v
```

回归覆盖三组调度、生成 worker 的实际安装与最终报告生命周期、完整 worker 包、每项机制证据缺失/错误拒绝、MA9 独立库来源和二进制篡改拒绝、ABC 反向执行路径，以及完整 controller 阶段顺序。所有 fixture 只用于 CPU 检查，不作为 GPU 结果。

新增 MA9 小库已用本地 CUDA 12.8 编译为 SM120；MA11 复用原有已验证构建器和 5090 kernel。另用 OP1 回传的三组实际 trace 检查了 ABC 行并行反向规则，并核对原 ABC 与 MA89 的 attention 操作计数一致。当前环境没有可见 GPU，尚未运行 OP2 八卡模型或生成性能结论。
