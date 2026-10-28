# Frozen TP4/DP2 measurement

2026-10-10: new model preparation defaults to GPT 6.7B. Pass
`--source <fresh-6.7B-TP4-prepared-run>` and
`--model-config tools/a800/phase6/model-6.7b.json` to `prepare.py`.
The source model and calibration shape must match; DP2 doubles global batch
while preserving each replica's workload. Reports read the actual model.
No tests were run. The historical 0.64B workflows below retain their recorded
scope; the old H2048 source is not automatically substituted for a 6.7B run.

Two independent existing Slurm allocations each host one four-GPU TP group.
Global world size is eight; DP groups pair corresponding TP shards across nodes.
This runner does not change the single-node worker, adapter, CUDA code or frozen
libraries. It does not support TP8 or optimize communication/DP overlap.

`prepare.py OUT --jobs JOB0 JOB1 --nodes HOST0 HOST1` creates a new immutable
campaign using the final phase6 compact candidate builds and the unchanged
original-order fused-v2 required baseline. Defaults: S1024/mb8/global16 and
S2048/mb4/global8; both have 16384 global tokens per optimizer step and one
microbatch per replica. These are not equal-global-batch comparisons to DP1.

Run `OUT/scripts/run.py OUT` using the host CPU Python. The controller starts
two concurrent `srun --jobid ...` tasks per window, each running four workers.
It requires no other internal steps, and never cancels the allocation or writes
its RELEASE file. Failed windows terminate only their own newly started srun
processes. Frozen directories cannot resume or overwrite earlier results; fix
maintenance code and prepare a new directory after a failure.

The node launcher records GPU/transport settings, enables IB using the observed
four active HCAs, and removes inherited single-node NCCL restrictions. The network
probe must establish actual IB use. Adapt NIC/HCA selection deliberately for other
machines; these defaults describe the audited A800 allocations, not every cluster.

Each scenario first checks all six policies with smoke and profile windows,
including FP8 reference budgets, MLP-only kernel counts, expected decoder branches,
DP gradient/parameter equality, different DP token shards and all-rank identities.
Timing uses six position-balanced blocks in each of two reverse-order rounds,
10 warmup + 20 timed full optimizer steps per fresh process. Latency is the maximum
over eight ranks; bootstrap uses paired blocks (10000 samples). Results include
both required baselines and both native references; no tuning is performed to
reach the 4% threshold.

Run CPU topology/protocol checks with `python -m unittest discover -s
tools/a800/phase6/dp2 -p 'test_*.py'`. Formal outputs include per-rank fingerprints,
raw logs, profiles, round reports and acceptance JSON. `source-consistency.json`
certifies the protected single-node source hashes did not change during this run.
