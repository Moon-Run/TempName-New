#!/usr/bin/env bash
# Configure the existing NGC PyTorch 25.06 image; never launch training.
set -euo pipefail
setup_5090_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
setup_5090_python=${RTX5090_BASE_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
exec "$setup_5090_python" -B "$setup_5090_dir/setup_5090.py" "$@"
