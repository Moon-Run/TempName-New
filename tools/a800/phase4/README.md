# Original-order, all-remote TACO baseline

Implementation and protocol: [base-phase4.md](../../../docs/design/design-a800/base-phase4.md).

This is BF16 GEMM with TACO E4M3/128 adaptive scaling + normalized Hadamard in
the Stream-K epilogue, followed by fused decode/ring reduction. It keeps the
original Flux swizzle, compresses every remote contribution, and keeps local
contributions in BF16. It neither links COCCL collectives nor adds reordering
or selective quantization. Source attribution: TACO_LICENSE.txt.

Run the CPU reference tests with the existing `outputs/a800/venv/bin/python`:

```bash
outputs/a800/venv/bin/python -m unittest discover -s tools/a800/phase4 -p 'test_*.py'
python3 tools/a800/phase4/build.py --out outputs/a800/phase4/taco-build
```

The builder fails if the output directory exists; use a new directory to rebuild.
It does not modify frozen baseline libraries or submit GPU jobs. The guarded
source hooks require FLUX_TACO_BASELINE and are enabled only in this build.
Use `from flux.gemm_rs_taco import GemmRSTaco` from its isolated Python package.
The API is forward-only and stream-bound. The explicit MLP-only model autograd
adapter and full-step comparison are in [e2e/](e2e/README.md); backward uses the
same BF16 gather/dgrad/wgrad path as the baseline (straight-through codec).

`build.py --placement separate --out NEW_DIRECTORY` additionally builds the
unfused comparison: stage original-order Flux GEMM contributions locally,
encode/scatter packets in a standalone kernel, then decode/ring reduce.
`--placement fused` retains epilogue encoding. Both keep the same codec and
packet format; neither transmits a remote BF16 copy before quantization.
The build exports `taco_placement()` and the Python API can enforce the expected
placement. Experimental macros remain opt-in and parent builds are immutable.

Store generated logs/results under `logs/a800/phase4`, builds under
`outputs/a800/phase4`; both are ignored by Git. Keep only maintained source here,
in `src/gemm_rs`, and in `python/flux`.

Before performance testing, run `validate.py` via torchrun inside an explicitly
allocated Slurm job as documented in base-phase4.md. Compile/link and CPU tests do
not establish GPU correctness or performance. The complete operator must include
the post-decode workspace-reuse barrier. Report packet padding and both scales.
