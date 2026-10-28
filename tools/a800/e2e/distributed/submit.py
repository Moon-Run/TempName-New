"""Resolve a configurable experiment; submit only with --submit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from config import resolve

ROOT = Path(__file__).resolve().parents[4]

def main():
    args, config = resolve()
    if args.stage == 'timing' and not args.preflight:
        raise ValueError('--stage timing requires --preflight <completed result directory>')
    python = Path(args.python or ROOT.parents[1]/'conda_envs/flux-megatron-a800/bin/python').resolve()
    print(json.dumps(dict(config=config, stage=args.stage, python=str(python)), indent=2))
    if not args.submit:
        return
    if not python.is_file():
        raise ValueError(f'Python environment missing: {python}')
    out = ROOT/'logs/a800/e2e/distributed'/uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    env = dict(os.environ, E2E_DIST_OUT=str(out), E2E_DIST_STAGE=args.stage,
               E2E_DIST_PYTHON=str(python), E2E_DIST_ROOT=str(ROOT))
    if args.preflight:
        env['E2E_DIST_PREFLIGHT'] = str(Path(args.preflight).resolve())
    cmd = ['sbatch', '--parsable', '--partition=gpu_a800', f'--nodes={config["nodes"]}',
           '--ntasks-per-node=1', f'--gpus-per-node={config["gpus_per_node"]}',
           f'--cpus-per-task={args.cpus_per_task}', f'--time={args.time}',
           f'--output={out}/slurm-%j.log', '--export=ALL',
           str(Path(__file__).with_name('run.sbatch'))]
    # Site submit plugin requires this QoS for multi-node GPU allocations.
    if config['nodes'] > 1:
        cmd.insert(2, '--qos=gpugpu')
    result = subprocess.check_output(cmd, env=env, cwd=ROOT, text=True).strip()
    (out/'submission.json').write_text(json.dumps(dict(job=result, command=cmd), indent=2)+'\n')
    print(f'Submitted {result}: {out}')

if __name__ == '__main__':
    main()
