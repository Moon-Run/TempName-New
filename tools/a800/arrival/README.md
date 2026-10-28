# BF16 arrival-priority experiment

Two new policies, `remote_arrival` and `interleaved_arrival`, layer a measured,
offline priority over the existing `remote_first` and `interleaved` mappings.
They retain the destination of every logical slot and permute only within
windows of 16 slots belonging to the same destination. The audited v2 default (`plan.py --score tail`) uses the median receiver-local
join time, after subtracting each invocation's start marker. A source is eligible
only if it is within two polling cycles of the last arrival on BOTH training
passes; near ties may admit several sources. `--score gap` is an ablation that
requires the same resolved last source on both passes. Stable ties retain order;
zero-score tiles keep their relative order but can be displaced by scored tiles.
No signal permits an identity mapping. This is a bounded heuristic, not an
optimal scheduler. Frozen historical results used the older gap-only algorithm.

Calibration covers every physical output tile in three passes on four A800s.
Passes 0/1 fit the mapping; pass 2 reports held-out last-source prediction.
All source comparisons for a tile use its receiver's clock. Sampling overhead
is recorded but is not treated as a verified low-perturbation mechanism result.
Only independent, uninstrumented jobs determine performance. A source becoming
last under a new order is not predicted by a fixed point solver in this first
prototype; one fitted mapping is frozen for validation, with no test-set tuning.

The swizzle remaps the logical tile index before applying the baseline mapping.
All Stream-K tasks for an index use the same remapping. Device constant tables
are selected by exact M/N/local-K/TP, with supported shapes checked on the host.
The GPU checker compares the compiled mapping with the independently recorded
baseline coordinates plus the fitted permutation, including full bijection.
Unsupported configurations fail preflight. The baseline CUDA/IPC libraries are
copied unchanged; only the two new CUDA variants are compiled, serially.

Current scope: single-node TP4, BF16, tile 128×128×32, Stream-K, three stages,
ring_reduction=True for calibration, operator and model. Four operator shapes:
(8192,4096,2048), (4096,4096,8192), (2048,2048,2048), (2048,2048,8192).
The last two are the actual 12-layer H=2048/S=2048 GPT target shapes.
These ring-reduction timings are a new cohort, not directly comparable to old
phase3 ring_reduction=False operator numbers.

From the TempName root:

```bash
python3 tools/a800/arrival/prepare.py
python3 tools/a800/arrival/test_plan.py
sbatch --export=ALL,ARRIVAL_STAGE=calibrate tools/a800/arrival/run.sbatch
# On the login host, passing the submitted calibration job ID:
python3 tools/a800/arrival/pipeline.py JOB_ID
```

The finite pipeline waits for calibration, fits/freeze-checks the inputs,
compiles the two variants, and submits a combined operator/model preflight.
Preflight covers five Flux policies, all four operator shapes, three seeds,
BF16 reference admission, diagnostic-only FP32 errors, mapping and actual kernel checks.
The user explicitly authorized FP32 tolerance failures to be recorded without
blocking this experiment; BF16 tolerances remain atol=rtol=0.02. Model smoke tests add
first-forward per-module BF16 GEMM+NCCL RS checks at atol=rtol=0.02 and full
optimizer updates. Profile runs are separate. This does not claim complete
paired full-model gradient accuracy or real-data convergence.

Only after preflight succeeds, the pipeline runs two independent operator jobs
and two independent end-to-end jobs, sequentially. The model has six controls:
native Megatron, original Flux, remote_first, interleaved, remote_arrival and
interleaved_arrival. Each job has six position-balanced blocks, ten warmup and
twenty timed optimizer steps per window. Operator jobs use five balanced blocks,
12 trials × 100 calls, and the second job reverses order. All ranks and slow
windows are retained. Candidate improvements are compared with their own base,
original Flux and, for the model, native Megatron. Paired bootstrap intervals
are within-job descriptive intervals; independent repetitions remain separate.

State: `logs/a800/arrival/pipeline.json`; logs: `pipeline.log`, `fit.log`,
`build.log` and per-job directories. A failure stops dependent submissions and
retains its artifacts. Resume with the same pipeline command only after
reviewing the failure; partial builds are deliberately not overwritten.
Do not run multiple pipeline processes for the same state file.

After all jobs pass, `report.py` writes `operator-comparison.csv`,
`model-comparison.csv`, `report.md`, and a marked result section in
`docs/design/design-a800/only-tile.md`. It reports regressions as well as gains.

`outputs/a800/arrival/compile-check` uses an explicitly labelled synthetic
identity-table fixture only to verify C++/CUDA compilation and linking while
GPUs are queued. It is never selected by the experiment pipeline and supplies
no numerical or performance evidence.

Audit (2026-10-03): run CPU checks with `python3 -m unittest discover -s tools/a800/arrival -p 'test_*.py'`.
Use fresh experiment output/build directories for v2; do not overwrite frozen
plans or update an old pipeline's script hashes to bypass its mismatch check.
`prepare.py` must regenerate model snapshots to include the new timing protocol
and reversed second repetition. The audited builder requires schema_version=2
and a matching planner hash. Next steps and historical tables are in
[only-tile.md](../../../docs/design/design-a800/only-tile.md).
