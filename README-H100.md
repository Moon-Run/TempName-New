# 精简版 H100 测试仓库

2026-10-28 从 `TempName` 的当前源码复制，保留相同目录布局。用于在新的单节点八卡 H100 服务器上重新构建和测试，包含 MA0 完整子层对比、MA8／MA9／MA89／MA11 等入口及 TP8-DP2 工具源码。

## 保留的内容

```text
TempName-New/
├── src/、include/、cmake/、python/、proto/   Flux 原生源码和构建依赖
├── tools/h100/                            H100 测试入口、配置及回归测试
├── tools/a800/                            H100 仍会导入的公共代码和 TACO 实现
├── test/、examples/、docs/                 测试源码、示例和机制文档
├── 3rdparty/cutlass/                      固定版本 Git 子模块
├── 3rdparty/nccl/                         固定版本 Git 子模块
├── outputs/h100/phase6/                   仅原路径中的少量冻结源码及配置
├── logs/h100/phase6/                      仅基线所需的三个固定 JSON 输入
└── CMakeLists.txt、setup.py、build.sh…      原构建入口
```

没有复制原仓库的 `.git` 历史、历史训练结果、profiler trace、原始校准样本、数据集、checkpoint、Python 环境、编译产物或旧的服务器 `env.sh`。新产生的结果和编译目录默认被 `.gitignore` 排除。

`outputs/` 中保留的都是注册过哈希的源码快照；`logs/` 中仅保留 `selection-plan.json`、`candidate-manifest.json`、`fused-manifest.json`。这些是构建 CR3／MA11 原实现的输入，不是性能测试输出。共约 **18.45 MiB**；精确路径及 SHA256 见 [slim-inputs.json](docs/design/design-h100/slim-inputs.json)。保持原路径可沿用原构建脚本，无需放宽冻结源码或选择表校验。

根目录 `README.md` 和 `LICENSE` 与原仓库逐字节一致，因为历史基线把它们纳入源码哈希。请将新部署说明写在本文件中。新仓库初始化时的两份文件备份在 `docs/TempName-New-initial-README.md` 和 `docs/TempName-New-initial-LICENSE.txt`；迁入的 Flux 源码继续遵循原 `LICENSE`／`NOTICE`。

## 第三方源码

子模块固定为原实验使用的提交：

| 路径 | 提交 |
| --- | --- |
| `3rdparty/cutlass` | `df8a550d3917b0e97f416b2ed8c2d786f7f686a3` |
| `3rdparty/nccl` | `8c6c5951854a57ba90c4424fa040497f6defac46` |

本地依赖已独立初始化，不使用旧 `TempName/.git/modules`、共享 Git objects 或指向旧仓库的绝对符号链接。CUTLASS 工作树采用 sparse checkout，仅保留构建所需头文件目录和根文件。

将新仓库的源码、`.gitmodules` 和两个子模块条目提交并传到新服务器后，在新 checkout 中执行：

```bash
git submodule update --init --recursive --depth 1
git submodule status
```

不要只在文件管理器中复制两个子模块的 `.git` 指针文件到另一台服务器；它们是相对于当前仓库的元数据。通过新仓库 clone／submodule update 获取依赖即可，不需要原仓库的 Git 历史。

## 新服务器的环境

全新 Ubuntu22.04 CUDA12.8 容器可直接使用 [env 一键配置脚本](env/README.md)：

```bash
cd /workspace/TempName-New
bash env/setup_h100.sh
```

默认创建 `/workspace/conda-envs/tempname-h100`，复用并核验同级 `Megatron-LM`，补齐固定依赖和 Apex，生成测试入口自动读取的环境文件。脚本不启动正式训练。

代码复制不包含软件环境。按 [H100 TP8 部署文档](docs/design/design-h100/deploy-tp8.md) 准备 Python、PyTorch、Apex、CUDA 和编译器；历史测试使用 Python 3.10、PyTorch 2.6.0+cu124、CUDA 12.8 编译自定义 Flux，运行时／Apex 的要求见部署文档。不要复制旧服务器的 Conda 路径配置。

Megatron-LM 仍为仓库外的独立依赖，默认放在同级 `../Megatron-LM`，固定提交为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`（`core_v0.12.3`）；可以通过测试入口的 `--megatron` 指定其他路径。仅有本仓库还不能替代 Megatron 或已安装的软件环境。

当前测试要求同节点八张完整 H100 SM90／132 SM，且 GPU 之间能够访问 peer memory；MIG 或其他 SM 数配置不在原协议内。

## 启动 MA11 与基础量化的 Attention／MLP 对比

在新服务器激活实际准备好的 Python 环境后：

```bash
cd /workspace/TempName-New
export H100_RECOVERY_PYTHON="$(command -v python)"
export H100_RECOVERY_BUILD_CUDA=/usr/local/cuda-12.8

# 先预览，不编译、不启动 GPU。
bash tools/h100/phase6/run_ma0_layer.sh

# 环境和八卡资源准备好后执行。
bash tools/h100/phase6/run_ma0_layer.sh --execute
```

如果 CUDA 实际安装路径不同，应替换上述路径。环境变量可避免入口继续使用 75 的旧 Python 默认路径。MA0 会在新服务器重新构建，不依赖 `/workspace/tp8-dp2-bench` 或旧服务器的 `.so`。

测量范围、输出文件和限制见 [ma-0-layer.md](docs/design/design-h100/New-Mechanism/ma-0-layer.md)。其他机制入口仍位于原来的 `tools/h100/phase6/`。文档中的历史日志链接未整体迁移；重分析旧性能结果或重做依赖旧校准原始样本的早期实验，需要另取相关数据或重新校准。

## 本地验证

本次在 CPU 主机上验证 MA11、Flux GemmOnly 和基础量化 codec 的源码准备与哈希校验，并在验证进程中禁止读取旧仓库路径，确认所需源文件位于新仓库中。MA0 层级测试 **9 项通过**，TP8-DP2 CPU 回归 **55 项通过**；MA0 启动预览与保留输入的哈希检查通过。

这些检查不编译 CUDA、不运行 H100，也不代表新服务器已经具备可运行的软件环境或已经测得性能结果。
