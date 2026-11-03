#!/usr/bin/env bash
# Flux + basic receiver fusion only; preview unless --execute, detached by default.
set -euo pipefail
basic_only_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$basic_only_tools/run_basic_quant.sh" --basic-only "$@"
