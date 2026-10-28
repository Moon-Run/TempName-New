# Phase5: MLP ordering plus all/selective TACO

Nine full-step groups: native Megatron, stock Flux, original Flux plus standalone
TACO, remote-only BF16, interleaved-only BF16, and each reordered build with all
remote contributions or calibrated selective contributions quantized. Only
`mlp.linear_fc2` changes; attention keeps original BF16 mapping. No arrival-priority
permutation is added. Mixed precision uses the phase4 warp-fused E4M3/H128 codec.

The immutable `[source][physical M tile][N tile]` mask controls both sender and
receiver. Selected remote tiles use FP8 packets plus two FP32 scales; unselected
tiles use the original BF16 source slots. Local contributions stay BF16. Decode
uses the same BF16 source order and workspace reuse barriers as phase4.

From the repository root, choose fresh output directories:

```bash
python3 -m unittest discover -s tools/a800/phase5 -p 'test_*.py'
python3 tools/a800/phase5/plan.py \
  logs/a800/arrival/arrival-v2-tp4-20261003/results/calibration-fit \
  logs/a800/phase5/NEW/selection-plan.json
python3 tools/a800/phase5/build.py --out outputs/a800/phase5/joint-build-NEW
python3 tools/a800/phase5/e2e/prepare.py "$PWD" \
  "$PWD/logs/a800/phase5/joint-e2e-NEW" \
  "$PWD/outputs/a800/phase5/joint-build-NEW" \
  "$PWD/logs/a800/phase5/NEW/selection-plan.json" --job-id 179147
```

Confirm the allocation is still alive and no other GPU experiment is running.
Run the frozen `campaign.py` using an overlapping internal step of the existing
179147 allocation; preserve its resident step `179147.1` and do not create
`RELEASE`. The campaign verifies attention/MLP mappings against actual GPU
coordinates, numerical behavior including no/all/checkerboard/calibrated masks,
tail padding and lifetimes, then all nine model smoke/profile cases and two
reverse-order timing rounds. Each round has nine position-balanced blocks,
10 warmup plus 20 timed optimizer steps per window. No operator-speed gate skips
the user-requested full-step comparisons.

```bash
python3 tools/a800/phase5/e2e/verify.py logs/a800/phase5/joint-e2e-NEW
python3 tools/a800/phase5/e2e/profile_components.py logs/a800/phase5/joint-e2e-NEW
```

Calibration passes 0/1 fit the mask; pass 2 only diagnoses held-out membership.
Both fitting passes must place a source within polling uncertainty of the last
arrival. The highest join-time scores receive at most 25% of each source's
remote-tile budget. Ties use physical tile index; no eligible candidate means
BF16. The reused dataset includes the exact MLP shape and both base orders;
mapping checks must match it. These are perturbed exploratory labels, not a
measured positive-net-benefit oracle. No post-quantization refitting, random
equal-budget control, or convergence claim is included. Comparing selective
with all-remote quantization alone cannot prove the calibration-based choice
is better than other masks of the same size.

New builds retain phase4 warp fusion (4 KiB staging); source and binary snapshots
are frozen separately. Logs/results go to `logs/a800/phase5`, build caches to
`outputs/a800/phase5`; both are ignored by Git. Maintained code lives here and
in `src/gemm_rs` / `python/flux`. Base objects and the standalone comparator
remain local prerequisites from earlier validated builds.

## Arrival-priority and native Megatron quantization extension

Optional policies are `remote_arrival_selective`, `interleaved_arrival_selective`,
and `native_taco` (12 groups total). Arrival variants reuse their base policy's
exact physical-tile selection mask. Only the logical production order changes,
using the existing v2 tail permutation within 16 legal destination slots.
GPU checks validate coordinates, destination/window constraints, and unchanged
physical quantization sets. Attention bypasses the arrival lookup. No mask
refitting after reordering is claimed.

```bash
python3 tools/a800/phase5/build.py \
  --out outputs/a800/phase5/arrival-joint-build-NEW \
  --arrival-plan outputs/a800/arrival-v2-tp4-20261003/artifacts/plan.json
python3 tools/a800/phase5/e2e/prepare.py "$PWD" \
  "$PWD/logs/a800/phase5/arrival-native-e2e-NEW" \
  "$PWD/outputs/a800/phase5/joint-build-20261003" \
  "$PWD/logs/a800/phase5/builds-20261003/selection-plan.json" \
  --arrival-build "$PWD/outputs/a800/phase5/arrival-joint-build-NEW" \
  --arrival-plan "$PWD/outputs/a800/arrival-v2-tp4-20261003/artifacts/plan.json" \
  --native-quant --job-id 179147
```

This measures 12 balanced blocks per round, two reversed rounds, 288 windows
and 5760 timed global optimizer steps. All original nine policies are rerun in
the same blocks. Omit `--native-quant` for eleven policies; omit both arrival
options to retain the original nine-policy preparation.

`native_taco` keeps native Megatron attention and RowParallelLinear GEMM, bias,
and linear backward. A per-module copy of the original forward function replaces
only its sequence-parallel RS call, without changing process-global functions.
PyTorch tensor TACO encoding packs remote contributions as uint8 payload plus
FP32 scales; NCCL all-to-all exchanges only remote packets, then tensor decoding
and BF16 ring-order addition produce the local output. Local contributions stay
BF16. Communication backward calls the original Megatron AllGather as an STE.
No Flux library is loaded. This intentionally simple baseline includes temporary
allocation, packing, Q/DQ and collective costs, and is not an optimized reference.
Value diagnostics/saturation counting are excluded from its timed path.

Native quantization checks cover CPU packet/reference/padding, GPU numerical
cases and lifetimes, exact STE gather gradients, original native-forward code
preservation, no loaded Flux library, and 48 encode/exchange/decode scopes per
full-step profile. These CPU annotations verify call counts, not GPU durations.
Flux-specific kernel columns are zero for both native policies; that does not
mean quantization has zero cost.
