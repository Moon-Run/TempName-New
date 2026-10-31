# MA14–MA20 串行运行

在 `TempName-New` 根目录执行：

```bash
# 仅预览，不创建实验目录、不编译、不启动 GPU。
bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh

# 正式运行一轮；--trials 2 可运行两轮。
bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh --execute
```

顺序为 MA14 → MA15 → MA16 → MA17 → MA18 → MA19 → MA20。每个子实验使用 `--foreground`，等待包括 profile 和报告在内的整批工作完成后再开始下一个。任一进程或验证失败就停止；性能没有收益不阻止后续实验。

| 实验 | 完整训练分组 | 每轮正式窗口 | 每轮正式 optimizer step |
| --- | --- | ---: | ---: |
| MA14 | MA11 对照＋MA14 | 16 | 800 |
| MA15 | MA11 对照＋MA15 | 16 | 800 |
| MA16 | 仅 MA16 | 8 | 400 |
| MA17 | 仅 MA17 | 8 | 400 |
| MA18 | 仅 MA18 | 8 | 400 |
| MA19 | 仅 MA19 | 8 | 400 |
| MA20 | 仅选出的 Q 候选 | 8 | 400 |

每个正式窗口仍为 10 步预热＋50 步计时。MA20 保留包括 MA11 在内的算子级校准和 heldout 选择；它不会再运行 MA11 完整训练。若原选型规则没有选出合格候选，仍以 `no_candidate` 结束，不擅自改阈值或强选一个 Q。表中 MA20 的步数以选出候选为前提。

MA16–MA19 使用新增的 `candidate_fullstep.py` 调度层，复用各实验原 worker、训练循环、逐 rank 验证器和 profile 分析器。只执行候选自身的完整模型 smoke、正式计时及独立 profile，不执行原来双组的独立 operator suite。候选的 codec/机制状态检查、全参数梯度覆盖和已有数值门禁均保留；缺少同期控制组时不宣称配对梯度/更新等价。

单组报告保留绝对 ms/step、loss、梯度信息和执行验证。它不会引用 MA14/MA15 的 MA11 数据或历史数据，不生成相对 MA11 的加速比、配对置信区间或 2% 验收结论；`target_evaluated=false` 表示没有做该比较。MA14/MA15 的原配对统计保持原样。

MA16–MA19 的 `fullstep/plan.json` 是实际单组调度；`fullstep/source/` 保存原 worker 的冻结模板与实际候选结果，模板本身的双组定义不代表两组都被运行。`fullstep/report.json`、`report.md`、`profile-summary.json` 位于外层。冻结的 `candidate-controller.py` 支持独立离线读报告：

```bash
python3 /实际批次/fullstep/candidate-controller.py report /实际批次/fullstep --offline
```

单独运行某个实验时也可以显式选择单组模式：

```bash
bash tools/h100/phase6/run_ma16.sh --candidate-only --execute
bash tools/h100/phase6/run_ma20.sh --candidate-only --execute
```

MA16–MA20 的独立入口未传 `--candidate-only` 时保留原双组行为，以便需要同期对照时继续使用。MA20 的 `--candidate-only` 与 `--reuse-ma11` 互斥，前者不借用历史 MA11 数据。

原 `run_ma14_ma15_ma16_ma17_ma18.sh` 保留为兼容转发入口，现在也执行上述七个实验。原 MAV2 串行入口的兼容文件名不变，但内部顺序和状态记录已扩展到 MA20，新日志阶段为 `ma14-ma20`，且同样按上述规则跳过重复 MA11 完整训练。历史日志和旧冻结会话不改写。

构建策略仍由原启动器决定：默认会准备并编译构建包，包中可能仍含控制组库；单组模式控制的是实际测量，不会自动复用旧编译产物。新入口创建新会话，继续继承启动环境的 CPU affinity，不恢复旧单核强绑。

开发端 72 项 CPU 回归测试通过，覆盖七实验顺序与失败停止、旧入口转发、默认双组兼容、单组调度、冻结源码校验、候选 smoke/profile 调用和无控制组时不宣称收益。七个真实启动器的预览和 shell 语法检查通过，预览没有创建实验目录。没有进行新 GPU 性能实验。
