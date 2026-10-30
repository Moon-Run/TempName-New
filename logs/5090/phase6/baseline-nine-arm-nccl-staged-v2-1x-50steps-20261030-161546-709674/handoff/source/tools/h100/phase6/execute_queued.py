"""Execute an idle-gated queue item without polling GPU resources during the run.

The queue checks GPU availability before launching each item. This controller
only records process/stage progress; GPU correctness checks remain in preflight.
"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys

from common import read, require, sha, write


IMPLEMENTATION = 'a800-fused-v2-cta-v1'
GPU_MONITORING = 'before_each_run_only'


def execute(repo, build, logroot, candidates, plan, megatron, cuda_home, python):
    compare = logroot / 'compare'
    handoff = logroot / 'handoff'
    stage = 'prepare'
    child = None
    zone = datetime.timezone(datetime.timedelta(hours=8))

    def now():
        return datetime.datetime.now(zone).isoformat()

    def state(status, detail):
        folders = list(compare.glob('results/round*/window-*'))
        complete = sum(all((d / f'rank{r}.json').is_file() for r in range(8))
                       for d in folders)
        write(logroot / 'execution.json', dict(
            status=status, stage=stage, detail=detail, updated_beijing=now(),
            controller_pid=os.getpid(), child_pid=child.pid if child else None,
            compare=str(compare), started_windows=len(folders), completed_windows=complete,
            gpu_monitoring=GPU_MONITORING,
        ))

    def run(command, name):
        nonlocal child
        print(now(), stage, 'START', name, flush=True)
        with (logroot / name).open('x') as output:
            child = subprocess.Popen(command, cwd=repo, stdout=output,
                                     stderr=subprocess.STDOUT)
            state('running', name)
            while child.poll() is None:
                # A process heartbeat only: no nvidia-smi/NVML/resource query.
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                state('running', name)
            code = child.returncode
            child = None
            if code:
                raise RuntimeError(f'{name} exited with {code}; see {logroot / name}')
        print(now(), stage, 'COMPLETE', name, flush=True)

    try:
        baseline = read(build / 'base/manifest.json')
        require(baseline['status'] == 'built', 'Baseline build is incomplete')
        require(baseline['variants']['taco_fused']['implementation'] == IMPLEMENTATION,
                'The queue requires the A800-aligned fused-v2 baseline')
        require(sha(plan) == read(handoff / 'reuse-verification.json')['plan_sha256'],
                'The frozen selection plan changed')
        run([str(python), 'tools/h100/phase6/prepare.py', str(compare),
             '--base-build', str(build / 'base'), '--candidate-build', str(candidates),
             '--plan', str(plan), '--megatron', str(megatron)], 'prepare.log')
        require(read(compare / 'submission.json')['variants']['taco_fused']['implementation']
                == IMPLEMENTATION, 'Prepared comparison has the wrong baseline')

        # GPU availability is checked by queue.py before run-one.sh is launched.
        # Do not re-enter a GPU polling loop here, between stages or on failure.
        stage = 'preflight'
        run([str(python), 'tools/h100/phase6/run.py', str(compare), '--stage', 'preflight',
             '--python', str(python), '--cuda-home', str(cuda_home), '--execute'],
            'preflight.log')
        gate = read(compare / 'preflight.json')
        require(gate['completed'], 'Preflight did not complete')
        cta = {key: [n for n in names if 'taco_decode_ring_kernel(' in n]
               for key, names in gate['kernel_names'].items() if key.startswith('taco_fused/')}
        require(len(cta) == 8 and all(cta.values()), 'Missing fused-v2 CTA decoder evidence')
        write(handoff / 'cta-trace-verification.json', dict(passed=True, ranks=cta))
        stage = 'timing'
        run([str(python), 'tools/h100/phase6/run.py', str(compare), '--stage', 'timing',
             '--python', str(python), '--cuda-home', str(cuda_home), '--execute'], 'timing.log')
        stage = 'report'
        run([str(python), 'tools/h100/phase6/report.py', str(compare)], 'report.log')
        result = read(compare / 'report.json')
        require(result['completed'] and result['fused_baseline_implementation'] == IMPLEMENTATION,
                'Final report is incomplete or has the wrong baseline')
        state('completed', 'All preflight and 72 timing windows verified; report generated')
        print(now(), 'COMPLETE', compare, 'accepted', result['accepted'], flush=True)
    except BaseException as error:
        state('failed', str(error))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('GPU execution is disabled. Use the idle-gated queue on the H100 host.')
        return
    execute(*(Path(os.environ[key]) for key in (
        'REPO', 'H100_BUILD', 'H100_LOG', 'H100_CANDIDATES', 'H100_PLAN',
        'MEGATRON_DIR', 'CUDA124')), python=Path(sys.executable))


if __name__ == '__main__':
    main()
