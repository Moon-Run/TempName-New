# Isolated TP8 optimization experiments

These tools extend the original compatibility-only entrypoint. They do not
modify the shared TP4 runtime or frozen required/reference baseline binaries.
Use a fresh output directory for every build and run, and query the live Slurm
allocation before launching. All GPU timing on an allocation is serial.

- `rebuild_decoder.py BASE OUT --default V` clones both TP8 candidates and
  rebuilds the runtime with the isolated `src/gemm_rs/taco_decode_tp8.cuh`.
  Variant 0 retains the original decoder; 14–17 change single-kernel geometry;
  18 materializes sparse decoded sources and then reduces; 19 assigns one warp
  per source inside mixed CTAs, alongside dense BF16 CTAs in the same launch.
  The latter variants require valid compact tile metadata and the exact
  TP8/M2048/N2048 shape. Other shapes retain the original fallback.
  The builder defaults to variant 0; the new decoders are explicit experiments.
- `rebuild_gemm.py BASE OUT --stages 3 --sms 80` changes only the selected MLP
  M2048/N2048/K1024 launch. Attention, BF16 arithmetic, tile dimensions and wire
  format are preserved. K64 was rejected by the existing TACO geometry guard
  during screening and is not an exposed supported option.
- `decoder_diagnostic.py` and `gemm_diagnostic.py` freeze and run whole-operator
  screening. They verify numerical budgets and retain per-rank samples.
  `bench_decoder.py` additionally checks exact output bits across decoders.
  Operator timings are not full-step acceptance.
- `decoder_ablation.py` freezes a paired full-step comparison at one GEMM
  configuration. It reuses the established model initialization, 10 warmup +
  20 timed optimizer steps, all-rank maximum, balanced ordering and bootstrap.
- `locality_plan.py SOURCE PLAN --window 16` restricts the original stable tail
  ranking to smaller destination windows. It preserves the exact physical
  quantization mask and does not read the held-out samples to choose an order.
  Build the resulting plan with the original `build.py`.
- `locality_ablation.py` compares window64/default SM scheduling with
  window16/default scheduling and optionally window64/80 SM scheduling. All
  conditions explicitly select decoder 0. `configuration_id` is a condition
  identifier; the manifest records its actual window and SM setting.
- `sparse_plan.py` selects only the highest positive training-tail score per
  source, retaining the original 1/64 upper bound. `bind_mask.py` binds the
  new runtime mask to a cloned build after checking that all GPU mappings are
  identical. `mask_ablation.py` supplies the appropriate frozen plan to each
  worker and checks mask hashes and exact logical bytes in every rank report.
- `prepare_optimized.py` freezes a six-policy two-round campaign. It checks
  decoder launch counts, model scope, geometry and loaded libraries. Historical
  4% assessment remains in `acceptance.json`; the user's separate 3% goal is
  evaluated in `user-goal.json`. The complete campaign includes codec/lifetime
  validation, mappings, six model smoke/profile pairs and 72 timing windows.

All candidates retain one post-GEMM publication barrier per call, two complete
alternating workspaces, independent output ownership, BF16 local/GEMM/backward,
and the E4M3/H128 remote codec. CUDA Graphs are disabled. Arrival predictions are
still derived from the existing eight-rank BF16 calibration; neither execution
tuning nor a smaller window constitutes post-quantization recalibration.

The session results and selected reproducible configuration are recorded in
`docs/design/design-a800/optimize/tp8/optimize1.md` and `docs/design/design-a800/base-phase6-tp8.md`.
