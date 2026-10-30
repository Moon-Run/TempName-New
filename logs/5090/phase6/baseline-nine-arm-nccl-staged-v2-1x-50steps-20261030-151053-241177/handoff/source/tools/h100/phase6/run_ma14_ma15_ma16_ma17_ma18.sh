#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18.sh [--execute] [--trials 1|2]

Run MA14, then MA15, then MA16, then MA17, then MA18. Without --execute, preview each launcher only.
Each child uses --foreground so its entire batch, including profiles, finishes
before the next starts. Every completed experiment proceeds to the next,
whether or not it has a performance gain. A process failure stops the sequence.
Each launcher creates its own timestamped batch directory and logs.

Examples:
  bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18.sh --execute
  bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18.sh --trials 2 --execute
EOF
}

ma1415161718_execute=false
ma1415161718_trials=1
while (( $# )); do
    case "$1" in
        --execute)
            ma1415161718_execute=true
            shift
            ;;
        --trials)
            if [[ ${2-} != 1 && ${2-} != 2 ]]; then
                printf '%s\n' 'Error: --trials requires 1 or 2.' >&2
                exit 2
            fi
            ma1415161718_trials=$2
            shift 2
            ;;
        --trials=1|--trials=2)
            ma1415161718_trials=${1#*=}
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

ma1415161718_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$ma1415161718_repo"
ma1415161718_args=(--foreground --trials "$ma1415161718_trials")
ma1415161718_mode=preview
if [[ $ma1415161718_execute == true ]]; then
    ma1415161718_args+=(--execute)
    ma1415161718_mode=execute
fi

for ma1415161718_number in 14 15 16 17 18; do
    printf '\n[MA%s] %s, trials=%s\n' "$ma1415161718_number" "$ma1415161718_mode" "$ma1415161718_trials"
    if bash "tools/h100/phase6/run_ma${ma1415161718_number}.sh" "${ma1415161718_args[@]}"; then
        printf '[MA%s] %s finished.\n' "$ma1415161718_number" "$ma1415161718_mode"
    else
        ma1415161718_status=$?
        printf '[MA%s] failed with exit code %s; sequence stopped.\n' \
            "$ma1415161718_number" "$ma1415161718_status" >&2
        exit "$ma1415161718_status"
    fi
done
