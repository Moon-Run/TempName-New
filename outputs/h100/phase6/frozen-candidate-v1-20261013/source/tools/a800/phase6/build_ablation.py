"""Pair complete training steps across immutable candidate builds, using the normal worker.

All conditions share model initialization, base order, arrival tables and masks.
This estimates an implementation increment, not the required-baseline target.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

POLICIES = ('remote_arrival_selective', 'interleaved_arrival_selective')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    source, root = args.source.resolve(), args.root.resolve()
    info = json.loads((source/'submission.json').read_text())
    repo = Path(info['repo'])
    assert json.loads((source/'results/preflight/preflight.json').read_text())['completed']
    for name, digest in info['files'].items():
        assert sha(source/name) == digest, name
    builds = dict(pair.split('=', 1) for pair in args.builds)
    assert len(builds) == len(args.builds) and 2 <= len(builds) <= 3
    conditions = [(p, label) for p in POLICIES for label in builds]
    assert args.blocks >= len(conditions) and args.blocks % len(conditions) == 0
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'scripts', root/'scripts', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__, root/'run.py')
    command = json.loads((source/'results/model-1/commands.json').read_text())[0]['command']
    worker = next(i for i, arg in enumerate(command) if arg.endswith('/worker.py'))
    frozen = {}
    for label, path in builds.items():
        build = Path(path).resolve()
        frozen[label] = dict(path=str(build), policies={})
        target = root/'builds'/label
        target.mkdir(parents=True)
        for policy in POLICIES:
            folder = policy.removesuffix('_selective')
            manifest = json.loads((build/folder/'manifest.json').read_text())
            assert manifest['arrival_plan_sha256'] == sha(source/'scripts/arrival-plan.json')
            for name, digest in manifest['files'].items():
                assert sha(build/folder/name) == digest, name
            (target/policy).symlink_to(build/folder/'taco')
            frozen[label]['policies'][policy] = dict(manifest=str(build/folder/'manifest.json'),
                manifest_sha256=sha(build/folder/'manifest.json'),
                library_sha256=sha(build/folder/'taco/libflux_cuda.so'),
                wrapper_sha256=sha(build/folder/'taco/libflux_cuda_ths_op.so'))
    orders = []
    for block in range(args.blocks):
        shift = block % len(conditions)
        order = conditions[shift:] + conditions[:shift]
        if block // len(conditions) % 2:
            order.reverse()
        orders.append(order)
    manifest = dict(source=str(source), repo=str(repo), builds=frozen, orders=orders,
        model_args=command[worker+1:], job_id=args.job_id,
        scope='Paired complete optimizer steps, identical model/order/mask; implementation ablation only.',
        files={str(p.relative_to(root)):sha(p) for p in (root/'scripts').rglob('*') if p.is_file()})
    manifest['files']['run.py'] = sha(root/'run.py')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(root)


def run(root):
    root = root.resolve()
    info = json.loads((root/'manifest.json').read_text())
    assert os.environ.get('SLURM_JOB_ID') == str(info['job_id'])
    for name, digest in info['files'].items():
        assert sha(root/name) == digest, name
    for build in info['builds'].values():
        for policy in build['policies'].values():
            path = Path(policy['manifest'])
            assert sha(path) == policy['manifest_sha256']
            for name, digest in json.loads(path.read_text())['files'].items():
                assert sha(path.parent/name) == digest, name
    repo = Path(info['repo'])
    config = json.loads((root/'scripts/config.json').read_text())
    signatures, trajectories, windows = {}, {}, []
    state = dict(status='running',job_id=info['job_id'],step_id=os.environ['SLURM_STEP_ID'])
    def save():
        state.update(finished_windows=len(windows),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for block, order in enumerate(info['orders']):
        for policy, label in order:
            window = len(windows)
            directory = root/f'window-{window:02d}-{policy}-{label}'
            directory.mkdir()
            build = root/'builds'/label
            library = build/policy
            env = dict(os.environ, CUDA_HOME='/data/apps/cuda/12.4', CUDA_DEVICE_MAX_CONNECTIONS='1',
                PATH=f'{Path(sys.executable).parent}:/data/apps/cuda/12.4/bin:'+os.environ['PATH'],
                MAX_JOBS='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
                NCCL_DEBUG='WARN',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='NVL',
                FLUX_FORCE_NVLINK='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                PYTHONPATH=f'{repo.parent}/Megatron-LM:{library}/python',
                LD_LIBRARY_PATH=f'{library}:/data/apps/cuda/12.4/lib64',E2E_POLICY=policy,E2E_MODE='timing',
                E2E_PROCESS_OUT=str(directory),E2E_BUILD=str(build),E2E_CASE=config['cases'][0]['name'],
                E2E_BLOCK=str(block),E2E_WINDOW=str(window))
            command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                       str(root/'scripts/worker.py'),*info['model_args']]
            print('START',block,policy,label,flush=True)
            with (directory/'process.log').open('w') as log:
                result = subprocess.run(command,cwd=repo.parent/'Megatron-LM',env=env,stdout=log,stderr=subprocess.STDOUT)
            (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
            if result.returncode:
                state.update(status='failed',failed_window=window);save()
                print((directory/'process.log').read_text()[-10000:],flush=True)
                raise RuntimeError(directory)
            rows = [json.loads((directory/f'rank{r}.json').read_text()) for r in range(4)]
            for rank, row in enumerate(rows):
                assert row['completed'] and row['passed'] and row['skips'] == row['fallback_calls'] == 0
                assert row['warmup_steps'] == 10 and row['timed_steps'] == 20
                assert row['timing_protocol'] == 'full-step-one-clear-v2'
                expected = info['builds'][label]['policies'][policy]
                assert row['libflux_cuda.so']['sha256'] == expected['library_sha256']
                assert row['libflux_cuda_ths_op.so']['sha256'] == expected['wrapper_sha256']
                signature = tuple(row[k] for k in ('initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'))
                assert signature == signatures.setdefault(rank,signature)
                trajectory = row['losses'] + row['grad_norms']
                reference = trajectories.setdefault((policy,label,rank),trajectory)
                assert all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7) for a,b in zip(trajectory,reference))
                assert row['module_policies'] == {name:policy if name.endswith('.mlp.linear_fc2') else 'original' for name in row['target_modules']}
            windows.append(dict(block=block,policy=policy,build=label,ms_per_step=max(r['ms_per_step'] for r in rows)))
            save()
            print('DONE',block,policy,label,windows[-1]['ms_per_step'],flush=True)
    labels = list(info['builds'])
    comparisons = {}
    for policy in POLICIES:
        pairs = [{w['build']:w['ms_per_step'] for w in windows if w['block']==b and w['policy']==policy}
                 for b in range(len(info['orders']))]
        for candidate in labels[1:]:
            logs = [math.log(p[candidate]/p[labels[0]]) for p in pairs]
            rng = random.Random(20261004)
            boot = sorted(100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000))
            comparisons[f'{policy}-{candidate}-vs-{labels[0]}'] = dict(reduction_percent=100*(1-math.exp(statistics.mean(logs))),
                ci95=[boot[249],boot[9749]],median_ms={label:statistics.median(p[label] for p in pairs) for label in (labels[0],candidate)},
                max_trajectory_difference=max(abs(a-b) for r in range(4) for a,b in zip(trajectories[(policy,labels[0],r)],trajectories[(policy,candidate,r)])))
    result = dict(completed=True,scope=info['scope'],comparisons=comparisons,windows=windows,rank_records=4*len(windows))
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    state['status']='completed';save()
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--builds',nargs='+')
    parser.add_argument('--blocks',type=int,default=8)
    parser.add_argument('--job-id',type=int)
    args=parser.parse_args()
    prepare(args) if args.source else run(args.root)
