# Single-node TP8 BF16 measurement

This runner measures six existing BF16 strategies on one eight-A800 allocation:
Megatron, original Flux, remote/interleaved base order and their arrival-priority
variants. It does not implement or tune quantization. The existing TP8 plan is
reused at M2048/N2048/K_local1024, and an isolated build restricts candidate
mapping to MLP; attention keeps original Flux order. Kernel parameters are unchanged.

Build using `build_bf16.py OUTPUT/build`, then freeze a new directory with
`prepare.py LOG --build OUTPUT/build --job-id JOB --node HOST`. Run
`LOG/run.py LOG` inside that allocation with eight GPUs and sixteen CPUs.
Only one internal test may occupy the allocation. Protect the outer hold.

The model is the existing 12-layer H2048/FFN8192 S2048/mb1/global4 case,
TP8/DP1, four microbatches, 8192 tokens per optimizer step, padded vocabulary
9216. This differs from the mb4/vocab8704 phase6 cases; do not label the timings
as an identical-workload strong-scaling comparison. Six smoke and six profile
windows precede six balanced blocks in each of two reversed timing rounds,
10 warmup + 20 timed optimizer steps per window. Unlike the earlier operator
campaign, a speed threshold does not decide whether model timing runs.
Numerical, mapping and kernel preflight checks still apply.

`report.py LOG` verifies complete rank records, initialization across policies
and rounds, rank topology, loaded libraries and original attention mapping.
Fresh immutable result/build directories are required; never overwrite the old
arrival8 campaign or shared single-node TP4 code/libraries.
