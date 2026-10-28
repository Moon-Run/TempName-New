import csv
import json
from pathlib import Path
import sys
out = Path(sys.argv[1])
data = json.loads((out/'results.json').read_text())
assert len(data['shapes']) == 3
rows = []
for shape in data['shapes']:
    checks = [rank for seed in shape['checks'] for rank in seed['ranks']]
    assert len(checks) == 6 and all(c['passed'] for c in checks)
    rows.append({
        'M':shape['M'],'N':shape['N'],'K_global':shape['K_global'],'K_local':shape['K_local'],
        'flux_us':shape['flux_ms']*1000,'torch_gemm_nccl_rs_us':shape['torch_ms']*1000,
        'speedup':shape['speedup'],'flux_tflops_per_gpu':shape['flux_tflops_per_gpu'],
        'max_abs_vs_bf16':max(c['max_abs_vs_bf16'] for c in checks),
        'max_abs_vs_fp32':max(c['max_abs_vs_fp32'] for c in checks),
        'max_relative_l2_vs_fp32':max(c['relative_l2_vs_fp32'] for c in checks),
        'correctness':'PASS'})
with (out/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
print(json.dumps(rows,indent=2))
