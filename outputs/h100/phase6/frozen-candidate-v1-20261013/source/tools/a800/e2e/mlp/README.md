# MLP-only TP4 end-to-end experiment

This experiment compares native Megatron, original Flux, MLP-only rank+1,
and MLP-only rank+1 plus frozen v2 tail arrival priority. Attention keeps the
original Flux mapping in all Flux groups. No compression is used.

The supported case is frozen: 12 layers, H=2048, FFN=8192, S=2048, TP4,
micro batch 1, global batch 4, BF16, `ring_reduction=True`. Its attention and MLP
GEMM-RS shapes have different local K values (512 and 2048). `build.py` selects
the rank offset in the host swizzle constructor. The arrival variant additionally
looks up the frozen MLP table using the physical source rank, before applying
the rank+1 base offset. Attention never matches that table; the shared kernel's
selection checks are included in full-step timing. This is a shape specialization,
not a generic module router.
`routing.py` checks every module name and observed shape to reject ambiguous or
unsupported use.

Maintained model code is in `model/`. Prepare fresh results under `logs/a800`;
builds and the copied calibration plan go under `outputs/a800`. Both runtime
roots are ignored by Git. Previously generated binaries/calibration data must
be available locally; source snapshots and data are not substitutes for code.

```bash
python3 tools/a800/e2e/mlp/prepare.py "$PWD" \
  "$PWD/logs/a800/e2e/mlp-arrival-tp4-NEW" --job-id EXISTING_JOB_ID
```

Execute the frozen `campaign.py` with `srun` inside the specified allocation.
The runner does not submit, cancel, release, or reconfigure allocations. When a
resident step holds the allocation, verify its GPU workloads are idle before
using an additional overlapping step.

The runner builds into a fresh directory, verifies the original parent assets
are unchanged, checks numerical output and actual kernels on all model targets,
and compares all GPU tile coordinates against previously validated original
and remote/arrival mappings. Only then does it run two rounds of 8 balanced blocks,
with 10 warmup and 20 timed optimizer steps per window. The second round
reverses both blocks and policy order. Both baselines are timed again. Results
are paired within blocks and remain scoped to one allocation.

CPU module-scope checks:

```bash
python3 -m unittest discover -s tools/a800/e2e/mlp -p 'test_*.py'
```

After both rounds complete, validate all records and cross-round signatures:

```bash
python3 tools/a800/e2e/mlp/verify.py logs/a800/e2e/mlp-arrival-tp4-NEW
```

A timing window is one fresh process group with 10 warmup and 20 measured full
optimizer steps. A paired block contains one window per policy. The 16-slot
tile window separately constrains offline reordering. Confidence intervals
bootstrap paired blocks, not the correlated steps inside a timing window.
The interval's upper endpoint is not the point estimate or a best observed run.
