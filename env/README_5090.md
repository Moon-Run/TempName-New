# RTX 5090：复用 NGC 25.06 镜像配置环境

适用于已确认的 `nvcr.io/nvidia/pytorch:25.06-py3`，Ubuntu 24.04 x86_64、Python 3.12.3、PyTorch `2.8.0a0+5228986c39.nv25.06`、CUDA 12.9、cuDNN 9.10.2，以及镜像自带的 Apex。默认使用现有目录：

```text
hjr/
├── TempName-New/
├── Megatron-LM/
├── venvs/tempname-5090/    # 脚本创建的轻量 venv
└── .5090-env/             # 管理记录、安装锁
```

在容器中进入已有的 `hjr/TempName-New` 后执行：

```bash
bash env/setup_5090.sh
source env/activate_5090.sh
```

无需 `--execute`。脚本不重新 clone 这两个仓库，也不安装新的 Python、PyTorch、CUDA 或 Apex，不启动训练。它使用 `venv --system-site-packages --without-pip` 继承镜像的包和 pip，只向 venv 补缺失依赖。基础镜像必须保留，venv 不是可移到其他镜像中独立运行的环境。[Python venv 说明](https://docs.python.org/3.12/library/venv.html)

如系统依赖缺失，需要在容器内以 root 运行；依赖已齐备时可用有目录写权限的用户。GPU 检查需要容器暴露八张 RTX 5090，建议启动容器时使用 `--gpus all --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864`。

## 执行内容

1. 检查镜像自带的 Python 3.12、精确 PyTorch 版本、CUDA 12.9、SM120 编译支持与三项 Apex 扩展导入；记录版本、C++ ABI 和关键模块文件摘要。
2. 复用 `../Megatron-LM`，要求提交 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。已有仓库干净但提交不同，会获取所需提交并以 detached HEAD 切换；有已跟踪文件修改时停止并保留现场。不会 `reset --hard`、`git clean` 或覆盖源码目录。CUTLASS/NCCL 按 `TempName-New` 的子模块条目初始化和检查。
3. 只补缺失的系统构建工具、Python 3.12 开发/venv 支持、NUMA 和 verbs 用户态库；APT 计划若涉及驱动、CUDA 元包或删除已有包，会停止。
4. 创建或复用脚本管理的 venv。仅安装缺失的 `packaging`、`pybind11`、`numpy`、`einops`、`regex`、`psutil`、`PyYAML`、`tqdm`、`six`、`setuptools`、`wheel`，使用 `pip --no-deps`，不批量升级镜像组件。沿用 NVIDIA 的 `/etc/pip/constraint.txt` 和已配置的 pip 约束，不修改约束文件。[NGC 25.06 说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-06.html)
5. 再次核对 PyTorch/CUDA/Apex 身份，编译一个 SM120 CUDA 对象，在固定 Megatron 源码上做训练模块导入检查，并在独立日志目录编译、导入 Python 3.12 dataset helper；不会覆盖源码目录中已有的 helper。
6. 默认复用新版 5090 预检，实际执行 Apex kernel 和八 rank NCCL 集合通信。记录 P2P 矩阵，但允许全部跨卡链路为 false。冻结 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`。
7. 检查成功后生成 `outputs/5090/env-setup/env.sh`，供激活脚本和测试入口使用。

如果镜像组件已经被替换、Apex 导入/实际 kernel 失败，或 Megatron 在当前环境导入失败，脚本保留日志并停止，不以降级 PyTorch 或跳过检查伪装成功。`env/setup_h100.sh` 安装的是另一套环境，不用于本镜像。

## 参数与重复执行

```bash
# 只看计划，无安装、文件写入和 GPU 查询。
bash env/setup_5090.sh --dry-run

# 暂时没有空闲 GPU 时只配置并验证软件；GPU/NCCL 状态明确标为待验证。
bash env/setup_5090.sh --skip-gpu-check

# 路径不采用上述兄弟目录布局时显式指定。
bash env/setup_5090.sh \
  --base-python /usr/bin/python \
  --cuda-home /usr/local/cuda \
  --megatron /path/to/hjr/Megatron-LM \
  --prefix /path/to/hjr/venvs/tempname-5090 \
  --state-dir /path/to/hjr/.5090-env
```

默认 CUDA 路径来自 `CUDA_HOME`，未设置时使用 `/usr/local/cuda`，并解析为实际 CUDA 12.9 目录。脚本入口解释器可用 `RTX5090_BASE_PYTHON` 指定。已存在的非本脚本管理 venv、其他镜像对应的 venv 或不完整 venv 均保留，需选择新的 `--prefix`；正常重复执行会复用管理记录并重新验证。

已有仓库不会被重新 clone；若 Git 目录不存在，需修正挂载或 `--megatron` 路径。安装可能需要访问 Ubuntu 软件源、PyPI，以及补齐 Git 提交/子模块时的 GitHub。这里仅配置容器用户态环境，不改变宿主机 GPU 驱动。

日志与状态：

```text
outputs/5090/env-setup/runs/<时间>/install.log
outputs/5090/env-setup/runs/<时间>/report.json
outputs/5090/env-setup/runs/<时间>/gpu-preflight/
outputs/5090/env-setup/last-report.json
outputs/5090/env-setup/env.sh
```

`status=completed`、`software_ready=true` 表示软件检查通过；默认模式还要求 `gpu_checked=true`。`--skip-gpu-check` 会明确写入 `gpu_checked=false` 和 `gpu_checks_pending=true`。这不表示原生 Flux/TACO 已通过 GPU 验证或已经有性能数据。

## 接着验证传输或开始实验

测试 shell 会自动读取成功生成的配置，自定义 Megatron 路径也会自动传入。优先级为显式命令行参数、`RTX5090_*` 环境变量、配置文件、原默认值，因此不必每次重新手填解释器路径。

```bash
# 可选：单独重新做环境与通信预检。
bash tools/5090/phase6/run_baseline.sh --execute --preflight-only

# 编译实验库并验证本地 producer→NCCL→decoder，随后停止。
bash tools/5090/phase6/run_baseline.sh \
  --execute --transport-smoke-only --foreground

# 另行启动九组完整实验。
bash tools/5090/phase6/run_baseline.sh --execute
```

新协议保留量化/归约数值定义，改变传输和同步，不能与原 H100 的 P2P 重叠机制直接混算。实验详情见 [5090 baseline](../docs/design/design-5090/baseline.md)。

开发环境验证使用 CPU 回归和无副作用预览；实际 APT/pip、NGC 镜像导入、CUDA 编译与八卡执行仍需在目标容器运行：

```bash
python3 -B -m unittest discover -s env -p 'test_setup_5090.py' -v
```
