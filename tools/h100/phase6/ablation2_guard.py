"""One startup availability check, called only after explicit GPU authorization.

No waiting, retrying, queueing, Slurm, or between-window occupancy queries.
"""
import csv
import datetime
import os
import socket
import subprocess

from common import require, write


def check_start(root, info):
    target = root/'start-gpu-check.json'
    require(not target.exists(), 'Suite already started; preserve it and prepare a new session')
    result = dict(run_id=info['run_id'], monitoring='before_each_suite_only', passed=False,
                  checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  host=socket.gethostname(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        query = lambda fields, kind: subprocess.check_output(
            ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True, timeout=30)
        result['gpu_csv'] = query('index,uuid,name,memory.used,utilization.gpu', 'gpu')
        result['apps_csv'] = query('gpu_uuid,pid,process_name,used_memory', 'compute-apps')
        rows = [[v.strip() for v in row] for row in csv.reader(result['gpu_csv'].splitlines())]
        visible = result['visible_devices']
        if visible is not None:
            order = visible.split(',')
            rows = [next((r for r in rows if key.strip() in (r[0], r[1])), []) for key in order]
        expected = [d['uuid'].removeprefix('GPU-') for d in info['calibration_hardware']['devices']]
        require(result['host'] == info['calibration_hardware']['host'] and len(rows) == 8 and all(rows) and
                [r[1].removeprefix('GPU-') for r in rows] == expected, 'Recalibrate for this host/GPU ordering')
        require(all('H100' in r[2] and int(r[3]) <= 1024 and int(r[4]) <= 5 for r in rows) and
                not result['apps_csv'].strip(), 'GPUs busy/unavailable; no automatic wait or retry')
        result['passed'] = True
    finally:
        write(target, result)
    return result
