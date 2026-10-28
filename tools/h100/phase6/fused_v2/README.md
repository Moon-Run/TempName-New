# A800 冻结融合 v2 的 H100 构建来源

实现标识：`a800-fused-v2-cta-v1`。

此目录保存 A800 必须基线的六份原始源码，用于新 H100 `taco_fused` 构建。来源为 `outputs/a800/phase4/taco-fused-warp-20261003/taco`，本次逐文件与 TP8 正式首测的 `outputs/a800/phase6/tp8-quant-measure-20261006/build/manifest.json` 核对。原始 SHA256 保存在 [provenance.json](provenance.json)。构建只读取本目录，不依赖本机被忽略的 A800 实验目录，也不复制其二进制。

保留的基线行为：

- BF16 GEMM、原始 tile 顺序、本地 BF16、全部远端 H128/E4M3 编码。
- 融合编码为 `encode_warp()`，内部仍调用 `__nv_cvt_float_to_fp8`。
- 接收端每个 128-thread CTA 处理一个 H128 组，使用共享 scratch；不使用候选的 `decode_warp()`。
- 每次 forward 在 Python 侧 configure/reset，单工作区，发布 barrier 和解码后的复用 barrier 均保留。
- 不声明 `owns_configuration`，共享模型 adapter 保留与 A800 相同的 detach/输出 clone 路径。

[build.py](../build.py) 只对 `role=fused` 安装本目录，再做 H100 必需适配：

1. Python 设备检查改为 SM90，并在构造 MLP `GemmRS` 时临时设置 TACO 配置，构造正常或异常退出都 reset。
2. C++ 在构造时仅保存 `h100_taco` 布尔身份，使缓冲区、GEMM、归约和发布 barrier 一致选用 V2；attention 实例保持 Hopper V3。每次 forward 的 TACO 配置仍由旧 Python 流程管理。
3. 使用现有 H100 注册器及 SM90 编译；保留现有独立库、设备绑定和无全局 CUDA 初始化的构建适配。

候选、采样器及原生 Hopper 参考仍从维护源码生成。此目录不用于修改 A800 历史基线，已有 H100 构建、日志和报告也不被替换。新 manifest/submission/report 带实现标识；没有标识的已有 H100 融合基线按历史 `phase6-warp-v1` 报告。

对齐的是算法与执行流程，不保证 A800/H100 的绝对延迟或相对收益相同；相同 CUDA FP8 intrinsic 在不同架构上的机器指令仍可能不同。新版本须在新会话完成 H100 smoke/profile 与正式 timing 后才能报告性能。

2026-10-11 开发端验证：18 项 CPU 协议/前端测试通过，覆盖构造与 forward 异常后的配置清理、基线来源哈希及版本传递；真实源码准备通过，候选/采样 C++ 适配与旧 builder 逐字一致。默认 132 SM 配置下，CUDA 12.8 完成新融合基线的 CUDA/C++ 动态库、Python 扩展及 GPU mapping 检查器编译，`cuobjdump` 确认 runtime 含 `sm_90a.cubin`；使用 SDK 驱动桩的 CPU 导入与旧 ABI 符号检查通过，未初始化 CUDA 上下文。编译位于 `/tmp/h100-fused-v2-0ded51lb/base`，标记 `diagnostic_only=true`、`gpu_validated=false`，不可用于正式测量；未执行 mapping 检查器或任何 GPU kernel。

保留各源码的原始许可声明。TACO/COCCL 适配的 BSD-3-Clause 归属见 [TACO_LICENSE.txt](../../../a800/phase4/TACO_LICENSE.txt)；Flux 源码许可见仓库 [LICENSE](../../../../LICENSE)。后续改动本固定基线须显式更新实现标识和来源记录，不能无说明地吸收候选优化。
