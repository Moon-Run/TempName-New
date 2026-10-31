#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry: the maintained sequence now includes MA19 and MA20.
ma_sequence_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ma_sequence_dir/run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh" "$@"
