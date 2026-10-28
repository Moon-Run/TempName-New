# MLP-only TACO full-step measurements

Four groups use original tile order: stock Flux, fused-TACO build with codec
disabled, separate TACO, and fused TACO. Only `mlp.linear_fc2` is quantized;
attention remains BF16 Flux with original mapping. The disabled-codec group
exposes compiler/resource changes that can also affect unquantized kernels.

The separate path stages BF16 contributions locally, then runs a standalone
encode/scatter kernel. The fused path writes FP8 packets from the GEMM epilogue.
Both use the same codec, FP8 packet layout, decode/ring reduction and reuse
barrier. No BF16 remote transfer precedes the compressed transfer. Backward
keeps BF16 AllGather and local dgrad/wgrad for both, a straight-through codec
approximation. All per-call validation, configuration, cloning and sync costs
remain inside the measured full training step.

Build each variant into a fresh directory:

```bash
python3 tools/a800/phase4/build.py --placement separate \
  --out outputs/a800/phase4/taco-separate-20261003
python3 tools/a800/phase4/build.py --placement fused \
  --out outputs/a800/phase4/taco-fused-20261003
```

The current preparation entry uses these two exact build directories and the
validated stock build, verifies all hashes, and creates an isolated bundle:

```bash
python3 tools/a800/phase4/e2e/prepare.py "$PWD" \
  "$PWD/logs/a800/phase4/taco-e2e-tp4-NEW" --job-id EXISTING_JOB_ID
```

Run its frozen `campaign.py` within that existing allocation only after other
GPU experiments finish. The campaign performs both codec preflights, model
smoke/profile checks, and two full-step rounds without an operator-speed gate.
Eight balanced blocks per round, four policies, 10 warmup + 20 measured steps
per window; round two reverses policy and block order. It never releases or
reconfigures the allocation.

```bash
python3 tools/a800/phase4/e2e/verify.py logs/a800/phase4/taco-e2e-tp4-NEW
```

The fused encoder now uses one warp per H128 group and four concurrent groups
per CTA, with a compact 4 KiB fragment workspace. Build into a new directory;
the previous artifacts must remain frozen. To compare the new build with the
previous fused implementation in the same timing blocks:

```bash
python3 tools/a800/phase4/build.py --placement fused \
  --out outputs/a800/phase4/taco-fused-warp-NEW
python3 tools/a800/phase4/e2e/prepare.py "$PWD" \
  "$PWD/logs/a800/phase4/taco-warp-e2e-tp4-NEW" --job-id EXISTING_JOB_ID \
  --fused-build outputs/a800/phase4/taco-fused-warp-NEW \
  --legacy-fused-build outputs/a800/phase4/taco-fused-20261003
```

This adds a fifth policy and uses ten position-balanced blocks per round.
`original_matched` loads the new fused build with quantization disabled;
`taco_fused_legacy` loads the frozen previous build with MLP quantization enabled.
The separate baseline can also be selected with `--separate-build`.
Both rounds and all numerical/model preflights are still required. Afterward,
`profile_components.py CAMPAIGN_ROOT` extracts per-rank kernel sums and launch
resource metadata; those sums are diagnostic, not full-step critical-path times.

Full-step timing includes forward, backward, collectives, optimizer and one
gradient clear per step. Fixed GPU-resident synthetic tokens exclude data
loading/checkpoint/evaluation. Finite losses do not establish convergence or
gradient-equivalence budgets. Independent repeats are fresh processes in one
allocation, not independent Slurm jobs. Runtime data stays in ignored `logs`
and `outputs`; maintained code stays in this directory and `src`/`python`.
