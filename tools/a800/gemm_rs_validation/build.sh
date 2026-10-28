#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/run01/scyb672/hjr/TempName
OUT="$ROOT/outputs/a800/gemm_rs_validation"
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2 OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.0 FLUX_FORCE_BUILD=1 FLUX_SHM_USE_NVSHMEM=0
export NCCL_ROOT="$ROOT/3rdparty/nccl/build/local"
cd "$OUT/source"
cmake -S . -B build -DCUDAARCHS=80 -DGPU_SM_CORES=108 \
  -DENABLE_NVSHMEM=OFF -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$OUT/source/python/flux"
cmake --build build --parallel 2
cmake --install build
python setup.py build_ext --inplace
g++ -O2 -std=c++17 -I"$CUDA_HOME/include" -I"$NCCL_ROOT/include" \
  "$OUT/scripts/nccl_gpu_check.cc" "$NCCL_ROOT/lib/libnccl_static.a" \
  -L"$CUDA_HOME/lib64" -lcudart_static -ldl -lrt -pthread -o "$OUT/nccl_gpu_check"
