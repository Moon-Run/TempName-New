# Phase7：原融合路径上的到达优先改进

主入口是 `fused.py`，完整设计、实现边界及运行命令见 [base-phase7.md](../../../docs/design/design-h100/base-phase7.md)，后续路线见 [update.md](../../../docs/design/design-h100/update.md)。

- `fused_plan.py`：真实receiver采样、稳定剩余延迟、协调排序、保守回退。
- `fused_build.py`：生成base/legacy/improved三套同源候选，仅CUDA/host映射表不同。
- `fused_protocol.py`、`fused_prepare.py`、`fused_run.py`、`fused_report.py`：原融合模型接入、九组配对及严格证据检查。
- `snapshot.py`：校验本地保留的Phase6原版快照。

三种顺序使用完全相同的原选择性量化mask；GemmV2、codec、IPC、屏障和双缓冲保持一致。默认dry-run，GPU执行须显式指定 `--execute`。静态模型筛选不等于性能提升。

`prepare --rounds 1 --blocks 9`准备单轮九组均衡实验：81个正式窗口加18个预检窗口，共99个worker窗口。省略 `--rounds` 时保持两轮，共162个正式窗口加18个预检窗口。两种模式均有12项策略/shape映射和72份profile trace。轮数在准备时冻结，`run`按冻结调度执行，报告检查全部预期窗口且拒绝额外窗口；不能通过提前停止默认两轮实验得到完整单轮报告。

```bash
python3 tools/h100/phase7/fused.py --help
python3 -m unittest discover -s tools/h100/phase7 -p 'test_fused_*.py' -v
```

此前独立分块原型及其CPU验证产物已删除。新生成的 `fused-*-dev/` 目录仅保留本地，不随Git提交。
