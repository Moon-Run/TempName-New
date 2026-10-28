# Single-node TP8 arrival and selective TACO

2026-10-10: the CLI defaults in `plan.py` and `prepare.py` now select
[GPT 6.7B](../model-6.7b.json): 32 layers, H4096/FFN16384, 32 heads,
vocab32768, S2048/mb1/global4. Both accept `--model-config` explicitly.
The MLP calibration shape is [2048,4096,16384], padded vocab33792;
new calibration and candidate compilation are required. Mapping, codec
preflight and report shape/wording now derive from the selected profile.
See [new-model commands](../../../../docs/design/model-gpt-6.7b.md).
This update was not compiled or tested.

The remaining text and commands describe the historical 0.64B campaign.
Replay those results from their frozen scripts/configuration; do not pass the
old calibration into the new default model or reuse expired job IDs.

This isolated entrypoint adapts the complete six-policy comparison to one
eight-A800 node. It retains the existing kernels and phase6 window64/budget1/64
rules. It does not tune kernels, masks, budgets or decoder parameters.

The model matches the preceding TP8 BF16 run: 12 layers, H2048/FFN8192,
S2048/mb1/global4, four microbatches, TP8/DP1, 8192 tokens per optimizer step,
padded vocabulary 9216. Only MLP forward remote contributions use the
FP8 E4M3/H128 codec. Attention retains its original backend; GEMM, local
contributions and STE backward remain BF16. This is not cross-node TP8.

`plan.py CALIBRATION PLAN` fits TP8 arrival permutations and physical masks
from the existing eight-rank BF16 calibration. Passes 0/1 train, pass 2 only
diagnoses. Destination and legal-window assignments remain fixed. This is
not recalibration after reordering or quantization.

`build.py PLAN BUILD` compiles separate MLP-only candidate libraries. Shared
TP4 sources and libraries are untouched. `prepare.py` snapshots generic model
scripts, changes validation rank domains to eight and diagnostic M to 1024,
and admits TP8 in the copied native Megatron tensor codec. The required fused
baseline is the original frozen `phase4/taco-fused-warp-20261003` binary,
never a candidate rebuild. Candidate TP8 decoding uses the existing generic
path, without the TP4 compact specialization.

```bash
cd /data/run01/scyb672/hjr/TempName
python3 tools/a800/phase6/tp8_quant/plan.py \
  outputs/a800/arrival-v2-tp8-20261003/results/calibration-fit \
  outputs/a800/phase6/NEW-TP8-QUANT/selection-plan.json
python3 tools/a800/phase6/tp8_quant/build.py \
  outputs/a800/phase6/NEW-TP8-QUANT/selection-plan.json \
  outputs/a800/phase6/NEW-TP8-QUANT/build-candidates
python3 tools/a800/phase6/tp8_quant/prepare.py logs/a800/phase6/NEW-TP8-QUANT \
  --build outputs/a800/phase6/NEW-TP8-QUANT/build-candidates \
  --plan outputs/a800/phase6/NEW-TP8-QUANT/selection-plan.json \
  --job-id 179139 --node d1n41a15g02
squeue --steps -j 179139
srun --jobid=179139 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --gpus=8 --kill-on-bad-exit=1 \
  /data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python \
  "$PWD/logs/a800/phase6/NEW-TP8-QUANT/campaign.py" \
  > logs/a800/phase6/NEW-TP8-QUANT/driver.log 2>&1
```

Check the live allocation before using the example job. Always create fresh
build/result directories and preserve the outer hold. The controller requires
exclusive experiment use of the existing allocation and checks immutable input
hashes before each stage.

Preflight covers actual GPU mappings, all/none/checkerboard/calibrated masks,
N tails, zero/random/spiky/repeated inputs, skewed double-buffered reuse,
native quantized exchange and exact STE gradient gathering. Six model smoke
and six profile windows verify module routing, numerical budgets and kernel
counts. Two reverse rounds then measure six balanced blocks each, 10 warmup
and 20 timed optimizer steps per window (72 windows, 576 rank records).
Numerical gates apply; no performance gate selects which results are recorded.

`report.md`, `verification.json` and `acceptance.{json,md}` report completed
measurements and the unchanged 4% criterion. TP4 archives are indexed in
[tp4-archive.md](../../../../docs/design/design-a800/tp4-archive.md); audit them before and
after TP8 work. Finite synthetic training does not establish convergence.
