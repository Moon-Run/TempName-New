#!/usr/bin/env bash
set -euo pipefail
# Preview by default; --execute checks/migrates a supplied build and runs smoke.
rtx5090_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$rtx5090_scripts/run_baseline.sh" --transport-smoke-only --foreground "$@"
