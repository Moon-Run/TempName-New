"""Coordinate two existing allocations; cancel only this runner's internal steps."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from protocol import orders, sha, write_json
from summarize import check_window, analyze


def model_args(model):
    args = []
    for flag, key in [('num-layers','layers'), ('hidden-size','hidden'), ('ffn-hidden-size','ffn'),
                      ('num-attention-heads','heads'), ('seq-length','sequence'),
                      ('max-position-embeddings','sequence'), ('micro-batch-size','micro_batch'),
                      ('global-batch-size','global_batch'), ('seed','seed'), ('vocab-size','vocab'),
                      ('tensor-model-parallel-size','tp')]:
        args += ['--'+flag, str(model[key])]
    args += ['--make-vocab-size-divisible-by','64',
             '--transformer-impl','local','--position-embedding-type','learned_absolute',
             '--untie-embeddings-and-output-weights','--no-persist-layer-norm',
             '--no-masked-softmax-fusion','--no-bias-gelu-fusion','--no-bias-dropout-fusion',
             '--no-rope-fusion','--train-iters','1000','--bf16','--optimizer','adam',
             '--lr','1e-4','--min-lr','1e-4','--lr-decay-style','constant','--weight-decay','0.01',
             '--adam-beta1','0.9','--adam-beta2','0.95','--clip-grad','1.0','--init-method-std','0.02',
             '--hidden-dropout','0','--attention-dropout','0','--no-gradient-accumulation-fusion',
             '--pipeline-model-parallel-size','1','--context-parallel-size','1','--sequence-parallel',
             '--mock-data','--tokenizer-type','NullTokenizer','--split','100,0,0','--num-workers','0',
             '--eval-iters','0','--timing-log-level','0','--no-check-for-nan-in-loss-and-grad']
    return args


def main(args):
    root = args.root.resolve()
    spec = json.loads((root/'submission.json').read_text())
    assert not (root/'state.json').exists(), 'New immutable campaign directory required'
    now = lambda: datetime.datetime.now().astimezone().isoformat()
    state = dict(status='running', started_at=now(), windows=[], active_steps=[], scope=spec['scope'])
    counter = 0
    def save():
        state['updated_at'] = now()
        write_json(root/'state.json', state)
    def check_sources():
        for name, digest in spec['files'].items():
            assert sha(root/name) == digest, name
        for name, digest in spec['protected_sources'].items():
            assert sha(name) == digest, name
        build = Path(spec['build'])
        assert sha(build/'manifest.json') == spec['original_build_manifest_sha256']
    def execute(out, policy, mode, cfg=None, config=None, block=-1, window=-1):
        nonlocal counter
        check_sources()
        out.mkdir(parents=True, exist_ok=False)
        counter += 1
        req = dict(out=str(out), policy=policy, mode=mode, block=block, window=window,
                   port=spec['port_base']+counter)
        if cfg:
            req.update(config=str(config), case=cfg['name'],args=model_args(cfg))
        request = out/'request.json'
        write_json(request, req)
        entry = dict(out=str(out), policy=policy, mode=mode, started_at=now())
        state['windows'].append(entry)
        state['current'] = str(out.relative_to(root))
        save()
        print(now(), 'START', state['current'], flush=True)
        procs, streams = [], []
        try:
            for node, job in enumerate(spec['jobs']):
                stream = (out/f'srun-node{node}.log').open('w')
                streams.append(stream)
                cmd = ['srun', f'--jobid={job}', '--overlap', '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=8', '--gpus=4', '--gpu-bind=none', '--kill-on-bad-exit=1',
                       spec['python'], str(root/'scripts/node.py'), str(root), str(request), str(node)]
                write_json(out/f'command-node{node}.json', cmd)
                procs.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True))
            deadline = time.monotonic() + (240 if policy == 'probe' else 360)
            while any(p.poll() is None for p in procs):
                state['active_steps'] = [json.loads(p.read_text()) for p in out.glob('node[01].json')]
                if any(p.poll() not in (None, 0) for p in procs):
                    raise RuntimeError(f'Node failed: {out}')
                if time.monotonic() > deadline:
                    raise TimeoutError(str(out))
                time.sleep(1)
            assert all(p.returncode == 0 for p in procs)
        finally:
            for p in procs:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p in procs:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            for stream in streams:
                stream.close()
            state['active_steps'] = []
            save()
        (out/'exit-code.txt').write_text('0\n')
        if policy == 'mapping':
            write_json(root/'mapping-check.json',dict(completed=True))
        elif policy == 'probe':
            rows = [json.loads((out/f'rank{r}.json').read_text()) for r in range(8)]
            assert all(r['passed'] for r in rows)
            network = '\n'.join(p.read_text() for p in out.glob('nccl-*.log'))
            assert 'NET/IB' in network and 'Using network IB' in network, 'Require observed IB transport'
            assert 'NET/Socket' not in network or 'Using network Socket' not in network
            write_json(root/'network-check.json', dict(passed=True, rows=rows, transport='IB'))
        else:
            check_window(out, cfg, spec, policy, mode)
        entry.update(passed=True, finished_at=now())
        save()
        print(now(), 'DONE', state['current'], flush=True)
    try:
        save()
        check_sources()
        # Refuse overlap with unrelated internal steps; batch/extern holds are allowed.
        snapshot = subprocess.check_output(['squeue','--steps','-h','-j',','.join(map(str,spec['jobs'])),'-o','%i'], text=True)
        assert all(x.endswith(('.batch','.extern')) for x in snapshot.split()), snapshot
        (root/'initial-slurm-steps.txt').write_text(snapshot)
        execute(root/'network-probe', 'probe', 'probe')
        execute(root/'mapping', 'mapping', 'mapping')
        if args.probe_only:
            state.update(status='probe_completed', finished_at=now())
            save()
            return
        for case in spec['cases']:
            config = Path(case['config'])
            cfg = json.loads(config.read_text())
            directory = config.parent
            for policy in spec['policies']:
                execute(directory/('smoke-'+policy), policy, 'smoke', cfg, config)
                execute(directory/('profile-'+policy), policy, 'profile', cfg, config)
            write_json(directory/'preflight.json', dict(completed=True, config=cfg))
            for repetition in (1, 2):
                w = 0
                for block, order in enumerate(orders(repetition)):
                    for policy in order:
                        execute(directory/f'round-{repetition}'/f'window-{w:02d}-{policy}',
                                policy, 'timing', cfg, config, block, w)
                        w += 1
                analyze(directory, spec, repetition)
        check_sources()
        state.update(status='completed', finished_at=now())
        write_json(root/'source-consistency.json', dict(passed=True, unchanged=spec['protected_sources']))
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), finished_at=now())
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--probe-only', action='store_true')
    main(p.parse_args())
