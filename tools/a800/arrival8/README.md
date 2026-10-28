# Single-node TP8 arrival-priority experiment

This directory independently extends the completed TP4 experiment to **one node,
eight A800 GPUs, TP=8, DP=PP=CP=1**. It does not use the two-node hierarchical path.
Historical TP4 maps, binaries and results are retained; source fixes apply to both entrypoints.

The audited v2 algorithm is a single offline fit. The default `--score tail`
uses median join time relative to each invocation's receiver-local start marker,
with sources required to remain near-critical on both training passes. Ties can
admit multiple sources. `--score gap` keeps only sources resolved as last on both
passes. Stable-sort groups of 16 slots for each destination; every destination
slot is preserved. With no reliable signal the mapping may remain identity.
Pass 2 reports critical-source-set hit rate under the BASE ordering, not proof
of tail ranking or of reordered performance. Post-reorder sampling and independent
uninstrumented timing are still required. Historical runs used gap-only v1.

Eight-source sampling uses one receiver GPU clock per tile, eight fragments per
source and a separate diagnostic library. Three passes cover every physical tile.
The TP8 priority tables are rebuilt from new samples, not from TP4 tables.
The runtime libraries use precomputed constant tables; index lookup cost remains
inside timing. Unsupported shapes, TP, or kernel dispatch fail validation.

The four global M/N/K shapes are (8192,4096,2048), (4096,4096,8192),
(2048,2048,2048), (2048,2048,8192). All phases use BF16 and ring_reduction=True.
The last two match the 12-layer H=2048/S=2048 model, with local K=256/1024 at TP8.
Model controls: native Megatron, original Flux, remote_first, interleaved,
remote_arrival, interleaved_arrival. Each model repetition uses six balanced
blocks, 10 warmup steps and 20 timed optimizer steps per window, 8192 tokens/step.
Operator repetitions use five balanced blocks, 12 trials of 100 calls, and a
separate profile. Both operator and model jobs are repeated independently twice.
Compare within TP8; TP4/TP8 padded vocabulary differs (8704/9216), so these runs
are not a strict strong-scaling test of identical parameter tensors.

From the repository root:

```bash
python3 tools/a800/arrival8/prepare.py
python3 tools/a800/arrival8/test_plan.py
python3 tools/a800/arrival8/test_tp8.py
python3 tools/a800/arrival8/test_summary_records.py
# Only when the isolated sampler directory does not already exist:
python3 tools/a800/arrival8/build_sampler.py
sbatch --qos=normal --export=ALL,ARRIVAL_STAGE=calibrate tools/a800/arrival8/run.sbatch
python3 tools/a800/arrival8/pipeline.py CALIBRATION_JOB_ID
```

Builds are serial on the login host. Once calibration and runtime build succeed,
the controller submits preflight and prequeues two operator/two model jobs with
serial afterok dependencies. No pending job consumes GPUs until scheduled.
Failure stops acceptance/reporting; dependent jobs cannot run until their
prerequisites succeed. The pipeline lock prevents duplicate controllers.
BF16 correctness, bijection, eight-rank completeness, actual kernels and finite
model training are required. FP32 errors are diagnostic-only, as authorized.

Outputs: outputs/a800/arrival8/; raw jobs, pipeline.json and reports:
logs/a800/arrival8/. The final report is appended to a distinct arrival8-results
section of docs/design/design-a800/only-tile.md, preserving all TP4 results.

Audit (2026-10-03): run CPU checks with `python3 -m unittest discover -s tools/a800/arrival8 -p 'test_*.py'`.
Use fresh experiment output/build directories for v2; do not overwrite frozen
plans or update an old pipeline's script hashes to bypass its mismatch check.
`prepare.py` must regenerate model snapshots to include the new timing protocol
and reversed second repetition. The audited builder requires schema_version=2
and a matching planner hash. Next steps and historical tables are in
[only-tile.md](../../../docs/design/design-a800/only-tile.md).
