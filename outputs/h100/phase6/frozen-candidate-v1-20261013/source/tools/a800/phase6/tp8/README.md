# TP8 BF16 measurement and TP4 regression

This is the user-selected existing BF16 hierarchy, **not** a TP8 port of
arrival-priority/selective TACO. Do not apply the six-policy quantization
acceptance criterion to these four BF16 policies.

Each node owns four contiguous TP ranks. The unchanged hierarchical adapter
splits the rows into two chunks, runs local TP4 Flux GEMM-RS on each, clones and
barriers before reuse, then exchanges and adds the remote node's partial result.
The new BF16 build changes only the MLP shape gate (M4096/N2048/K_local1024).
Attention K_local256 keeps original order. No kernel or communication tuning is
performed. Original libraries and all single-node entrypoints remain unchanged.

`build_bf16.py OUT` creates an isolated build. `prepare.py LOG --build OUT
--jobs JOB0 JOB1 --nodes HOST0 HOST1` freezes two scenarios (S1024/mb8 and
S2048/mb4), both TP8/DP1/global8192 tokens. Vocab padding is set explicitly so
the actual vocabulary remains 8704, matching the TP4 model. Run the frozen
`LOG/scripts/run.py LOG` only after other internal tests finish. It verifies IB,
GPU tile mappings, model outputs, loaded libraries, shapes and exact kernel
counts before four balanced blocks in each of two reversed rounds.

For user-requested TP4 impact checks, prepare each node with
`prepare_regression.py LOG --sequence 1024|2048 --job-id JOB`. Within that
allocation run `LOG/regression_stage.py before`, then run TP8 after both before
stages exit, and finally run `LOG/regression_stage.py after`. These stages reuse
the unmodified phase5/6 six-policy single-node protocol. They never overwrite
old results. Before/after timing is separated in time; interpret shared changes
as possible drift, not proof that the TP8 test changed the software.

Generate the combined report with `report.py TP8_LOG --regressions TP4_LOG_A
TP4_LOG_B`. It compares raw before/after latency, original-Flux-normalized
latency, initial fingerprints and full timed loss/gradient-norm trajectories.
Tests: `python -m unittest discover -s tools/a800/phase6/tp8 -p 'test_*.py'`.

For a repeat without implementation changes, prepare a fresh campaign with the
same build and configuration. Keep its internal `round-1/2` paths intact and
record `round-mapping.json` with `previous_root` and
`local_to_display_round: {"1": 3, "2": 4}`. After both campaigns complete,
`report_repeat.py PREVIOUS_LOG CURRENT_LOG` verifies frozen files, IB transport,
identical GPU scripts/configuration/build, and initialization fingerprints across
all four rounds. It writes four-round tables grouped by shape, policy, then
round, without pooling the rounds or rerunning the previous TP4 regression.

Only newly created internal steps are managed. Preserve both outer Slurm holds
and never create their RELEASE files. New immutable directories are required
after a failed preflight; do not silently relax a failed assertion.
