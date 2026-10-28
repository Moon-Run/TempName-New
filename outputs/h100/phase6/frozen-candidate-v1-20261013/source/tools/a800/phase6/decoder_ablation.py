"""Freeze/run paired full-step decoder ablation with identical order and masks."""
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
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(source, root, blocks, variants=(0,3), build_override=None, job_id=179147):
    source, root = source.resolve(), root.resolve()
    info = json.loads((source/'submission.json').read_text())
    repo = Path(info['repo'])
    assert repo/'logs/a800' in root.parents
    assert 2 <= len(variants) <= 3 and len(set(variants)) == len(variants) and set(variants) <= {0,3,4,5,6,10,11,12,13}
    conditions = [(policy, variant) for policy in POLICIES for variant in variants]
    assert blocks >= len(conditions) and blocks % len(conditions) == 0
    preflight = json.loads((source/'results/preflight/preflight.json').read_text())
    assert preflight['completed']
    for name, digest in info['files'].items():
        assert sha(source/name) == digest, name
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'scripts', root/'scripts', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__).with_name('decoder_entry.py'), root/'scripts/decoder_entry.py')
    shutil.copy2(Path(__file__).with_name('decoder_control.py'), root/'scripts/decoder_control.py')
    # Add the diagnostic hook only to the new snapshot, before sealing hashes.
    # The standard worker pins its CPU before importing Torch/CUDA as usual.
    support = root/'scripts/taco_support.py'
    text = support.read_text()
    anchor = '        self.codec = GemmRSTaco('
    assert text.count(anchor) == 1
    support.write_text(text.replace(anchor,
        '        from decoder_control import configure_decoder_variant\n'
        '        configure_decoder_variant()\n'+anchor))
    shutil.copy2(__file__, root/'run.py')
    commands = json.loads((source/'results/model-1/commands.json').read_text())
    command = commands[0]['command']
    worker_index = next(i for i, arg in enumerate(command) if arg.endswith('/worker.py'))
    orders = []
    for block in range(blocks):
        shift = block % len(conditions)
        order = conditions[shift:] + conditions[:shift]
        if block // len(conditions) % 2:
            order = list(reversed(order))
        orders.append(order)
    build = build_override.resolve() if build_override else Path(info['artifact_root'])/'build'
    manifest = dict(source=str(source), repo=str(repo), build=str(build), variants=list(variants),
                    build_override=bool(build_override),cuda_initialization='standard worker order; variant set at warmup codec construction',
                    build_manifest_sha256=sha(build/'manifest.json'),
                    model_args=command[worker_index+1:], orders=orders,
                    job_id=job_id, scope='Full optimizer-step paired decoder ablation; same GEMM, base order, arrival table, physical mask and numerical protocol. Not a new baseline acceptance campaign.',
                    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(root)


def run(root):
    root = root.resolve()
    info = json.loads((root/'manifest.json').read_text())
    assert os.environ.get('SLURM_JOB_ID') == str(info['job_id'])
    for name, digest in info['files'].items():
        assert sha(root/name) == digest, name
    repo, build = Path(info['repo']), Path(info['build'])
    assert sha(build/'manifest.json') == info['build_manifest_sha256']
    manifest = json.loads((build/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(build/name) == digest, name
    config = json.loads((root/'scripts/config.json').read_text())
    signatures, trajectories, records, windows = {}, {}, [], []
    state = dict(status='running',job_id=info['job_id'],step_id=os.environ['SLURM_STEP_ID'])
    def save():
        state['finished_windows'] = len(windows)
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for block, order in enumerate(info['orders']):
        for policy, variant in order:
            window = len(windows)
            directory = root/f'window-{window:02d}-{policy}-v{variant}'
            directory.mkdir()
            library = build/policy
            env = dict(os.environ, CUDA_HOME='/data/apps/cuda/12.4', CUDA_DEVICE_MAX_CONNECTIONS='1',
                       PATH=f'{Path(sys.executable).parent}:/data/apps/cuda/12.4/bin:'+os.environ['PATH'],MAX_JOBS='2',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                       NCCL_DEBUG='WARN', NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='0', NCCL_P2P_LEVEL='NVL',
                       FLUX_FORCE_NVLINK='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                       PYTHONPATH=f'{repo.parent}/Megatron-LM:{library}/python',
                       LD_LIBRARY_PATH=f'{library}:/data/apps/cuda/12.4/lib64',
                       E2E_POLICY=policy, E2E_MODE='timing', E2E_PROCESS_OUT=str(directory),
                       E2E_BUILD=str(build), E2E_CASE=config['cases'][0]['name'],
                       E2E_BLOCK=str(block), E2E_WINDOW=str(window), E2E_DECODE_VARIANT=str(variant))
            command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                       str(root/'scripts/decoder_entry.py'),*info['model_args']]
            print('START', block, policy, variant, flush=True)
            with (directory/'process.log').open('w') as log:
                result = subprocess.run(command,cwd=repo.parent/'Megatron-LM',env=env,stdout=log,stderr=subprocess.STDOUT)
            (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
            if result.returncode:
                state.update(status='failed', failed_window=window)
                save()
                print((directory/'process.log').read_text()[-12000:],flush=True)
                raise RuntimeError(directory)
            rows = [json.loads((directory/f'rank{rank}.json').read_text()) for rank in range(4)]
            for rank, row in enumerate(rows):
                assert row['completed'] and row['passed'] and row['skips'] == row['fallback_calls'] == 0
                assert row['decoder_variant'] == variant and row['warmup_steps'] == 10 and row['timed_steps'] == 20
                assert row['timing_protocol'] == 'full-step-one-clear-v2'
                assert row['libflux_cuda.so']['sha256'] == manifest['policies'][policy]['library_sha256']
                signature = tuple(row[k] for k in ('initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'))
                assert signature == signatures.setdefault(rank, signature)
                trajectory = row['losses'] + row['grad_norms']
                reference = trajectories.setdefault((policy, rank), trajectory)
                assert all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7) for a,b in zip(trajectory,reference))
                assert row['module_policies'] == {'%s'%name: policy if name.endswith('.mlp.linear_fc2') else 'original' for name in row['target_modules']}
                records.append(row)
            windows.append(dict(block=block,policy=policy,variant=variant,
                                ms_per_step=max(r['ms_per_step'] for r in rows)))
            save()
            print('DONE', block, policy, variant, windows[-1]['ms_per_step'], flush=True)
    comparisons = {}
    for policy in POLICIES:
        pairs = [{w['variant']:w['ms_per_step'] for w in windows if w['block']==block and w['policy']==policy}
                 for block in range(len(info['orders']))]
        variants = info.get('variants',[0,3])
        for i, reference in enumerate(variants):
            for candidate in variants[i+1:]:
                logs = [math.log(pair[candidate]/pair[reference]) for pair in pairs]
                rng = random.Random(20261004)
                boot = sorted(100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000))
                comparisons[f'{policy}-v{candidate}-vs-v{reference}'] = dict(
                    policy=policy,reduction_percent=100*(1-math.exp(statistics.mean(logs))),
                    ci95=[boot[249],boot[9749]],reference_variant=reference,candidate_variant=candidate,
                    median_ms_previous=statistics.median(p[reference] for p in pairs),
                    median_ms_tiled=statistics.median(p[candidate] for p in pairs))
    result = dict(completed=True,scope=info['scope'],comparisons=comparisons,windows=windows,
                  rank_records=len(records),initialization_and_trajectories_checked=True)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    state['status'] = 'completed'
    save()
    print(json.dumps(comparisons,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--prepare-from',type=Path)
    parser.add_argument('--blocks',type=int,default=8)
    parser.add_argument('--variants',type=int,nargs='+',default=[0,3])
    parser.add_argument('--build',type=Path)
    parser.add_argument('--job-id',type=int,default=179147)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.prepare_from,args.root,args.blocks,args.variants,args.build,args.job_id)
    else:
        run(args.root)
