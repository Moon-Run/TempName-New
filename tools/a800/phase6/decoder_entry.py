"""Run the frozen worker without importing CUDA before its CPU-affinity setup."""
import json
import os
from pathlib import Path
import runpy

variant = int(os.environ['E2E_DECODE_VARIANT'])
assert variant in (0, 3, 4, 5, 6, 10, 11, 12, 13)
runpy.run_path(str(Path(__file__).with_name('worker.py')), run_name='__main__')
path = Path(os.environ['E2E_PROCESS_OUT'])/f"rank{os.environ['RANK']}.json"
report = json.loads(path.read_text())
report['decoder_variant'] = variant
path.write_text(json.dumps(report, indent=2)+'\n')
