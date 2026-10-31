#!/usr/bin/env python3
"""Single-candidate fullstep runs using the original MA16--MA19 workers.

The source/ directory is an immutable worker template, not a completed paired
experiment. Only the outer plan schedules measurements. All original per-rank
smoke, mechanism, gradient coverage and profile checks remain active; no paired
equivalence, MA11 speedup or 2% target is inferred without a measured control.
"""
import argparse
import gzip
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import shutil
import statistics
import sys
from types import SimpleNamespace


EXPERIMENTS = ('ma16', 'ma17', 'ma18', 'ma19')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def backend(experiment):
    require(experiment in EXPERIMENTS, 'Unsupported single-candidate experiment')
    if Path(__file__).name == 'candidate-controller.py':
        scripts = Path(__file__).resolve().parent / 'source/scripts'
        sys.path.insert(0, str(scripts))
        return SimpleNamespace(**runpy.run_path(str(scripts / 'fullstep-controller.py')))
    return importlib.import_module(experiment + '_fullstep')


def selected_jobs(full, template, stage):
    candidate = full.bench.ARMS[1]
    return [job for job in full.jobs(template, stage) if job['arm'] == candidate]


def prepare(args):
    require(Path(__file__).name != 'candidate-controller.py', 'Frozen candidate controller is read-only')
    root = args.root.resolve()
    full = backend(args.experiment)
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    full.prepare(argparse.Namespace(root=source, build=args.build, megatron=args.megatron,
                                    trials=args.trials, blocks=8))
    template = full.verify(source)
    candidate = full.bench.ARMS[1]
    shutil.copy2(__file__, root / 'candidate-controller.py')
    plan = dict(protocol=f'h100-{args.experiment}-candidate-only-fullstep-v1',
                experiment=args.experiment, comparison_mode='candidate_only',
                run_id=template['run_id'], arms=[candidate], candidate=candidate,
                trials=args.trials, blocks=8, warmup_steps=10, timed_steps=50,
                source_plan_sha256=sha(source / 'plan.json'),
                controller_sha256=sha(__file__),
                schedule=selected_jobs(full, template, 'timing'),
                source_scope='Frozen paired worker template only; MA11 control jobs are never launched.',
                target_evaluated=False, paired_equivalence_evaluated=False)
    write(root / 'plan.json', plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    full = backend(plan['experiment'])
    require(plan['protocol'] == f"h100-{plan['experiment']}-candidate-only-fullstep-v1" and
            plan['comparison_mode'] == 'candidate_only' and
            plan['arms'] == [full.bench.ARMS[1]] and plan['candidate'] == full.bench.ARMS[1],
            'Changed candidate-only protocol or arm')
    require(sha(__file__) == sha(root / 'candidate-controller.py') == plan['controller_sha256'],
            'Candidate controller changed')
    source = root / 'source'
    require(sha(source / 'plan.json') == plan['source_plan_sha256'], 'Worker template changed')
    template = full.verify(source, offline=offline)
    require(plan['run_id'] == template['run_id'] and plan['trials'] == template['trials'] and
            (plan['blocks'], plan['warmup_steps'], plan['timed_steps']) == (8, 10, 50) and
            plan['schedule'] == selected_jobs(full, template, 'timing') and
            len(plan['schedule']) == plan['trials'] * 8 and
            plan['target_evaluated'] is False and plan['paired_equivalence_evaluated'] is False,
            'Changed single-candidate schedule or measurement contract')
    return plan, template, full


def rank_files(root, job):
    directory = Path(root) / 'source/results' / job['directory']
    return {str(path.relative_to(root)): sha(path)
            for path in (directory / f'rank{rank}.json' for rank in range(8))}


def preflight(root, offline=False):
    root = Path(root).resolve()
    plan, template, full = verify(root, offline)
    job, = selected_jobs(full, template, 'preflight')
    rows = full.window(root / 'source', template, job)
    result = dict(protocol=plan['protocol'], run_id=plan['run_id'], completed=True,
                  candidate=plan['candidate'], comparison_mode='candidate_only',
                  per_rank_smoke_validated=True, mechanism_state_validated=True,
                  full_parameter_gradient_coverage_validated=True,
                  paired_equivalence_evaluated=False,
                  rank_identities=full.identity(rows), evidence_files=rank_files(root, job))
    path = root / 'preflight.json'
    if path.exists():
        require(read(path) == result, 'Candidate preflight differs from raw evidence')
    else:
        write(path, result)
    return result


def profile_summary(root, offline=False):
    root = Path(root).resolve()
    plan, template, full = verify(root, offline)
    gate = preflight(root, offline)
    job, = selected_jobs(full, template, 'profile')
    rows = full.window(root / 'source', template, job)
    require(full.identity(rows) == gate['rank_identities'], 'Profile hardware/input identity changed')
    profiles, evidence = {}, rank_files(root, job)
    for rank in range(8):
        path = root / 'source/results' / job['directory'] / f'rank{rank}-profile.json.gz'
        with gzip.open(path, 'rt') as handle:
            profiles[f"{plan['candidate']}/rank{rank}"] = full.trace_projection_records(
                json.load(handle)['traceEvents'], plan['candidate'], tuning=template['tuning'])
        evidence[str(path.relative_to(root))] = sha(path)
    return dict(protocol=plan['protocol'], run_id=plan['run_id'], completed=True,
                execution_validated=True, comparison_mode='candidate_only', profiles=profiles,
                evidence_files=evidence, plan_sha256=sha(root / 'plan.json'),
                limitation='Candidate execution evidence only; profile durations are not MA11 speedups.')


def report(root, offline=False, save=True):
    root = Path(root).resolve()
    plan, template, full = verify(root, offline)
    gate = preflight(root, offline)
    samples, evidence, numerics = [], dict(gate['evidence_files']), []
    for job in plan['schedule']:
        rows = full.window(root / 'source', template, job)
        require(full.identity(rows) == gate['rank_identities'], 'Timing hardware/input identity changed')
        samples.append(dict(job, ms_per_step=max(row['elapsed_seconds'] for row in rows) * 1000 / 50))
        numerics.append(dict(mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1],
                             final_grad_norm=rows[0]['grad_norms'][-1]))
        evidence.update(rank_files(root, job))
    summaries = {}
    for trial in range(1, plan['trials'] + 1):
        values = [sample['ms_per_step'] for sample in samples if sample['trial'] == trial]
        require(len(values) == 8, 'Missing candidate timing windows')
        summaries[f'trial{trial}'] = dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                                         min_ms=min(values), max_ms=max(values), windows=len(values))
    candidate = plan['candidate']
    result = dict(protocol=plan['protocol'], run_id=plan['run_id'], completed=True,
                  comparison_mode='candidate_only', executed_arms=[candidate],
                  formal_optimizer_steps=len(samples) * 50, timing_rank_records=len(samples) * 8,
                  all_samples=samples, trial_summaries=summaries,
                  training_summary={key: statistics.mean(row[key] for row in numerics) for key in numerics[0]},
                  per_rank_smoke_validated=True, full_parameter_gradient_coverage_validated=True,
                  paired_equivalence_evaluated=False, target_evaluated=False,
                  target_validated=False, target_point_observed=False,
                  increment_over_MA11_validated={candidate: False},
                  acceptance_status={candidate: 'candidate_measured_without_MA11_comparison'},
                  evidence_files=evidence,
                  limitation='No MA11 control was measured or borrowed. No paired equivalence, speedup, CI or 2% target claim.')
    if save:
        path = root / 'report.json'
        if path.exists():
            require(read(path) == result, 'Candidate report differs from raw evidence')
        else:
            write(path, result)
        lines = [f"# {plan['experiment'].upper()}：仅候选组完整训练", '',
                 f"候选：`{candidate}`。正式窗口 {len(samples)}，optimizer step {len(samples)*50}。", '',
                 '| Trial | 平均 ms/step | 中位 ms/step |', '| --- | ---: | ---: |']
        lines += [f"| {name} | {value['mean_ms']:.3f} | {value['median_ms']:.3f} |"
                  for name, value in summaries.items()]
        lines += ['', '仅验证候选自身的 smoke、机制状态与梯度覆盖；profile 另存。',
                  '未运行或借用 MA11 完整训练结果，不计算配对等价性、加速比、置信区间或 2% 目标。']
        (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return result


def check_complete(root, trials, profile=False):
    root = Path(root).resolve()
    result = report(root, save=False)
    require(result == read(root / 'report.json') and result['formal_optimizer_steps'] == trials * 400 and
            result['timing_rank_records'] == trials * 64, 'Incomplete candidate measurement')
    if profile:
        require(profile_summary(root) == read(root / 'profile-summary.json'),
                'Candidate profile differs from execution evidence')
    return result


def run(args):
    root = args.root.resolve()
    plan, template, full = verify(root)
    source = root / 'source'
    jobs = selected_jobs(full, template, args.stage)
    python = args.python.expanduser().absolute()
    command = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               str(source / 'scripts/worker.py'), *full.model_arguments(full.MODEL)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True, command=command, jobs=jobs), indent=2))
        return
    require(args.worker_timeout > 0, 'Positive worker timeout required')
    marker = root / f'{args.stage}-started.json'
    require(not marker.exists(), 'Use a fresh candidate stage')
    write(marker, dict(run_id=plan['run_id'], jobs=jobs))
    prefix = plan['experiment'].upper()
    for job in jobs:
        if job['mode'] != 'smoke':
            preflight(root)
        verify(root)
        arm = job['arm']
        env = full.bench.environment(source, template, arm, python, args.cuda_home, profile=False)
        env = full.communication.apply(env, template['communication_record'])
        library = Path(template['libraries'][arm]['root'])
        destination = source / 'results' / job['directory']
        destination.mkdir(parents=True, exist_ok=False)
        env.update({f'H100_{prefix}_EXECUTE': '1', f'H100_{prefix}_ARM': arm})
        env.update(H100_TP='8', H100_SM_COUNT='132',
                   H100_RUN_ID=template['trial_ids'].get(str(job['trial']), template['run_id']),
                   E2E_POLICY=full.policy_for_arm(arm), E2E_MODE=job['mode'], E2E_CASE='h100-phase6',
                   E2E_BUILD=str(source / 'build' / arm), E2E_PROCESS_OUT=str(destination),
                   E2E_BLOCK=str(job['block']), E2E_WINDOW=str(job['index']),
                   PYTHONPATH=f"{source}/scripts:{template['megatron']}:{library}/python")
        write(destination / 'command.json', dict(command=command, job=job, comparison_mode='candidate_only',
                                                shared_environment=template['shared_environment']))
        print('START', job['directory'], flush=True)
        try:
            with (destination / 'process.log').open('x') as output:
                full.bench.run_worker(command, template['megatron'], env, output, args.worker_timeout)
            full.window(source, template, job)
        except BaseException as error:
            write(destination / 'failed.json', dict(status='failed', retry=False, job=job,
                                                   error=f'{type(error).__name__}: {error}'))
            raise
    if args.stage == 'preflight':
        preflight(root)
    elif args.stage == 'profile':
        write(root / 'profile-summary.json', profile_summary(root))
    else:
        report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('root', type=Path)
    p.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--megatron', type=Path, required=True)
    p.add_argument('--trials', type=int, choices=(1, 2), default=1)
    p = sub.add_parser('run')
    p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('all', 'preflight', 'timing', 'profile'), default='all')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--cuda-home', type=Path)
    p.add_argument('--worker-timeout', type=int, default=1800)
    p.add_argument('--execute', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('root', type=Path)
    p.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    if Path(__file__).name == 'candidate-controller.py':
        require(args.action == 'report' and args.offline, 'Frozen candidate controller supports offline reports only')
        require((args.root/'preflight.json').is_file() and (args.root/'report.json').is_file(),
                'Frozen replay requires existing smoke and timing reports')
        print(json.dumps(report(args.root, offline=True, save=False), indent=2, ensure_ascii=False))
        return
    if args.action == 'report':
        report(args.root, args.offline)
    else:
        globals()[args.action](args)


if __name__ == '__main__':
    main()
