#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/h100/phase6/run_ma8_ma9_ma10.sh [--execute] [--trials 1|2]

Run MA8, then MA9, then MA10. Without --execute, preview each launcher only.
Each child uses --foreground so its entire batch, including profiles, finishes
before the next starts. Every completed experiment proceeds to the next,
whether or not it has a performance gain. A process failure stops the sequence.
Each launcher creates its own timestamped batch directory and logs.

Examples:
  bash tools/h100/phase6/run_ma8_ma9_ma10.sh --execute
  bash tools/h100/phase6/run_ma8_ma9_ma10.sh --trials 2 --execute
EOF
}

ma8910_execute=false
ma8910_trials=1
while (( $# )); do
    case "$1" in
        --execute)
            ma8910_execute=true
            shift
            ;;
        --trials)
            if [[ ${2-} != 1 && ${2-} != 2 ]]; then
                printf '%s\n' 'Error: --trials requires 1 or 2.' >&2
                exit 2
            fi
            ma8910_trials=$2
            shift 2
            ;;
        --trials=1|--trials=2)
            ma8910_trials=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Error: unsupported argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ma8910_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$ma8910_repo"
ma8910_args=(--foreground --trials "$ma8910_trials")
ma8910_mode=preview
if [[ $ma8910_execute == true ]]; then
    ma8910_args+=(--execute)
    ma8910_mode=execute
fi

for ma8910_number in 8 9 10; do
    printf '\n[MA%s] %s, trials=%s\n' "$ma8910_number" "$ma8910_mode" "$ma8910_trials"
    if bash "tools/h100/phase6/run_ma${ma8910_number}.sh" "${ma8910_args[@]}"; then
        printf '[MA%s] %s finished.\n' "$ma8910_number" "$ma8910_mode"
    else
        ma8910_status=$?
        printf '[MA%s] failed with exit code %s; sequence stopped.\n' \
            "$ma8910_number" "$ma8910_status" >&2
        exit "$ma8910_status"
    fi
done
