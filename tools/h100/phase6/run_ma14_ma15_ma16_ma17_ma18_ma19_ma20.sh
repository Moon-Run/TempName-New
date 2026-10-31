#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh [--execute] [--trials 1|2]

Run MA14 -> MA15 -> MA16 -> MA17 -> MA18 -> MA19 -> MA20.
MA14/MA15 retain their MA11 control; MA16--MA20 use candidate-only fullstep.
MA20 retains its required operator calibration and heldout selection. Without --execute, preview each launcher only.
Each child uses --foreground so its entire batch, including profiles, finishes
before the next starts. Every completed experiment proceeds to the next,
whether or not it has a performance gain. A process failure stops the sequence.
Each launcher creates its own timestamped batch directory and logs.

Examples:
  bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh --execute
  bash tools/h100/phase6/run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh --trials 2 --execute
EOF
}

ma_sequence_execute=false
ma_sequence_trials=1
while (( $# )); do
    case "$1" in
        --execute)
            ma_sequence_execute=true
            shift
            ;;
        --trials)
            if [[ ${2-} != 1 && ${2-} != 2 ]]; then
                printf '%s\n' 'Error: --trials requires 1 or 2.' >&2
                exit 2
            fi
            ma_sequence_trials=$2
            shift 2
            ;;
        --trials=1|--trials=2)
            ma_sequence_trials=${1#*=}
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

ma_sequence_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$ma_sequence_repo"
ma_sequence_args=(--foreground --trials "$ma_sequence_trials")
ma_sequence_mode=preview
if [[ $ma_sequence_execute == true ]]; then
    ma_sequence_args+=(--execute)
    ma_sequence_mode=execute
fi

for ma_sequence_number in 14 15 16 17 18 19 20; do
    ma_sequence_child_args=("${ma_sequence_args[@]}")
    if (( ma_sequence_number >= 16 )); then
        ma_sequence_child_args+=(--candidate-only)
    fi
    printf '\n[MA%s] %s, trials=%s\n' "$ma_sequence_number" "$ma_sequence_mode" "$ma_sequence_trials"
    if bash "tools/h100/phase6/run_ma${ma_sequence_number}.sh" "${ma_sequence_child_args[@]}"; then
        printf '[MA%s] %s finished.\n' "$ma_sequence_number" "$ma_sequence_mode"
    else
        ma_sequence_status=$?
        printf '[MA%s] failed with exit code %s; sequence stopped.\n' \
            "$ma_sequence_number" "$ma_sequence_status" >&2
        exit "$ma_sequence_status"
    fi
done
