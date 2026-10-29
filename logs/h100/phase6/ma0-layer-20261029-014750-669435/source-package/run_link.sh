#!/usr/bin/env bash
set -euo pipefail
bench_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$bench_root/tools/h100/phase6/check_tp8_dp2_link.py" "$@"
