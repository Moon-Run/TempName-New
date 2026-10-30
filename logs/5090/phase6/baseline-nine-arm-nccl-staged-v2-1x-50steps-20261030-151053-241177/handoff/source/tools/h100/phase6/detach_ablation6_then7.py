#!/usr/bin/env python3
"""Run ablation6 three times, then ablation7 three times, at 10+50 steps/window.

Default: read-only preview. --execute detaches under container PID 1.
Each fresh comparison must pass its own preflight and strict report before
the next comparison is prepared. Any failure stops the whole batch.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import ablation4_repeat as repeat
import ablation7_protocol as a7
import detach_ablation6_run_two as six
import detach_ablation5_run_two as launcher
from common import REPO, read, require, sha, write

TRIALS = ('compare1', 'compare2', 'compare3')
FAMILIES = ('ablation6', 'ablation7')
ORDER = [f'{family}/{trial}' for family in FAMILIES for trial in TRIALS]
BLOCKS = {'ablation6': 5, 'ablation7': 6}
WINDOWS = {'ablation6': 50, 'ablation7': 36}
TIMING = dict(warmup_steps=10, timed_steps=50)
TOTAL_WINDOWS = sum(WINDOWS.values()) * len(TRIALS)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=repeat.DEFAULT_BUILD, help='Existing ablation4 parent build for both experiments')
    p.add_argument('--flux-build', type=Path, help='Reuse a compiled ablation6 GemmOnly build')
    p.add_argument('--ablation7-build', type=Path, help='Reuse a compiled ablation7 build derived from --build')
    p.add_argument('--out', type=Path, help='Fresh batch directory; existing paths are rejected')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--nccl-root', type=Path, help='NCCL development root; default: verified files recorded in parent fused manifest')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true', help='Keep the supervisor attached for diagnostics')
    args = p.parse_args(argv)
    require(args.jobs > 0, 'jobs must be positive')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/('ablation6-then7-3x-50steps-'+stamp)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def commands(args):
    """Build each new library once; never start ablation7 before all three a6 reports."""
    steps = []
    compile_flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
    if args.nccl_root:
        compile_flags += ['--nccl-root', str(args.nccl_root)]
    for family in FAMILIES:
        cli = [str(args.python), str(Path(__file__).with_name(family+'.py'))]
        root = args.out/family
        if family == 'ablation6':
            build = args.flux_build or root/'flux-build'
            prepare_flags = ['--build', str(args.build), '--flux-build', str(build)]
            build_flags = []
            reuse = args.flux_build
        else:
            build = args.ablation7_build or root/'build'
            prepare_flags = ['--build', str(build)]
            build_flags = ['--parent-build', str(args.build)]
            reuse = args.ablation7_build
        if reuse is None:
            steps += [(family, 'build-prepare', cli+['build-prepare', str(build), *build_flags]),
                      (family, 'compile', cli+['compile', str(build), *compile_flags])]
        for trial in TRIALS:
            target = str(root/trial)
            run = cli+['run', target, '--python', str(args.python), '--cuda-home', str(args.cuda_home)]
            steps += [(family, trial+'-prepare', cli+['prepare', target, *prepare_flags,
                        '--megatron', str(args.megatron), '--blocks', str(BLOCKS[family]), '--rounds', '2', '--nvls-enable', '0']),
                      (family, trial+'-dry-run', run),
                      (family, trial+'-controller', run+['--stage', 'all', '--execute']),
                      (family, trial+'-report', cli+['report', target])]
    return steps


def check_inputs(root, family):
    if family == 'ablation6':
        return six.check_inputs(root)
    info = read(root/'submission.json')
    require(info['protocol'] == a7.PROTOCOL and info['execution_timing'] == TIMING and
            info['executable'] and not info['source_only'] and info['rounds'] == 2 and
            info['blocks_per_round'] == 6 and info['timing_windows'] == 36 and
            info['schedule'] == a7.schedule(6, rounds=2) and
            info['budget_fractions'] == {p: a7.definition(p)['budget_fraction'] for p in a7.POLICIES} and
            info['communication']['environment'].get('NCCL_NVLS_ENABLE') == '0',
            'Expected a fresh C(1/16)/A/improved-B ablation7 comparison at 10+50 steps')
    return info


def check_report(root, family, info):
    if family == 'ablation6':
        return six.check_report(root, info)
    result = read(root/'report.json'); gate = read(root/'preflight.json')
    require(read(root/'all-state.json')['status'] == 'completed' and result['completed'] and result['correctness_passed'] and
            result['protocol'] == a7.PROTOCOL and result['run_id'] == info['run_id'] and result['execution_timing'] == TIMING and
            result['round_count'] == 2 and result['blocks_per_round'] == 6 and result['timing_windows'] == 36 and
            len(result['all_samples']) == 36 and result['formal_optimizer_steps'] == 1800 and
            result['timing_rank_records'] == 288 and result['preflight_rank_records'] == 48 and
            result['profile_traces'] == 24 and result['candidate_shape_mappings'] == 6,
            'Incomplete ablation7 strict report; next comparison will not start')
    require(gate['completed'] and gate['protocol'] == a7.PROTOCOL and gate['execution_timing'] == TIMING and
            gate['submission_sha256'] == sha(root/'submission.json') and
            result['evidence']['preflight_sha256'] == sha(root/'preflight.json') and
            result['communication'] == gate['communication'] == info['communication'], 'Report/preflight identity mismatch')
    return result


def resolve_nccl(args):
    if args.flux_build is not None and args.ablation7_build is not None:
        return
    if args.nccl_root is None:
        recorded = read(args.build/'fused/manifest.json').get('nccl_files', {})
        headers = [Path(p) for p in recorded if p.endswith('/include/nccl.h')]
        require(len(headers) == 1, 'Pass --nccl-root: no unique recorded NCCL development directory')
        require(all(Path(p).is_file() and sha(p) == h for p, h in recorded.items()), 'Recorded NCCL files changed')
        args.nccl_root = headers[0].parent.parent
    require(all((args.nccl_root/p).is_file() for p in ('include/nccl.h', 'lib/libnccl_static.a')),
            'NCCL headers/static library missing')


def supervise(args):
    path = args.out/'batch-state.json'
    state = dict(status='preparing', order=ORDER, execution_timing=TIMING, rounds=2, nvls_enable='0',
                 expected_timing_windows=TOTAL_WINDOWS, expected_formal_optimizer_steps=TOTAL_WINDOWS*50,
                 experiments={f: dict(status='pending', blocks_per_round=BLOCKS[f],
                    expected_timing_windows=WINDOWS[f]*3, expected_formal_optimizer_steps=WINDOWS[f]*150,
                    comparisons={}) for f in FAMILIES})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    inputs = {}; reports = {}; run_ids = set()
    try:
        resolve_nccl(args)
        state['nccl_root'] = str(args.nccl_root) if args.nccl_root else None
        for family, label, cmd in commands(args):
            batch = args.out/family; batch.mkdir(exist_ok=True)
            experiment = state['experiments'][family]
            experiment['status'] = 'running'
            state.update(status='running', stage=f'{family}/{label}')
            write(path, state); launcher.invoke(cmd, batch/(label+'.log'), env)
            if label.endswith('-dry-run'):
                trial = label.removesuffix('-dry-run'); info = check_inputs(batch/trial, family)
                require(info['run_id'] not in run_ids, 'All six comparisons require distinct run_ids')
                run_ids.add(info['run_id'])
                if family in inputs:
                    repeat.same_inputs(inputs[family], info)
                    if family == 'ablation6':
                        for key in ('flux_build_manifest', 'naive_codec_identity'):
                            require(inputs[family][key] == info[key], 'Repeat baseline identity changed: '+key)
                else:
                    inputs[family] = info
                inputs[f'{family}/{trial}'] = info
                experiment['comparisons'][trial] = dict(run_id=info['run_id'], status='prepared')
            if label.endswith('-report'):
                trial = label.removesuffix('-report')
                result = check_report(batch/trial, family, inputs[f'{family}/{trial}'])
                prior = reports.setdefault(family, [])
                if prior:
                    for key in ('measurement_identity', 'role_plan_sha256'):
                        require(prior[0][key] == result[key], 'Repeat runtime/plan changed: '+key)
                prior.append(result)
                experiment['comparisons'][trial].update(status='completed', timing_windows=WINDOWS[family],
                    formal_optimizer_steps=WINDOWS[family]*50, report=str(batch/trial/'report.json'),
                    report_sha256=sha(batch/trial/'report.json'))
                if len(prior) == len(TRIALS):
                    experiment.update(status='completed', timing_windows=WINDOWS[family]*3,
                        formal_optimizer_steps=WINDOWS[family]*150, timing_rank_records=WINDOWS[family]*24)
            write(path, state)
        require(all(len(reports.get(f, [])) == 3 for f in FAMILIES), 'All six strict reports are required')
        state.update(status='completed', stage='done', timing_windows=TOTAL_WINDOWS,
                     formal_optimizer_steps=TOTAL_WINDOWS*50, timing_rank_records=TOTAL_WINDOWS*8)
        write(path, state)
        print('Completed: ablation6 x3, then ablation7 x3; 258 windows / 12900 formal steps.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        if state.get('stage'):
            state['experiments'][state['stage'].split('/')[0]]['status'] = 'failed'
        write(path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    print('Batch:', args.out, flush=True)
    print('Order: '+' -> '.join(ORDER), flush=True)
    print('Each: 10 warmup + 50 timed steps/window; rounds=2, NVLS=0.', flush=True)
    print('Ablation6: 5 blocks/round, 50 windows/run x3. Ablation7: 6 blocks/round, 36 windows/run x3.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, fork, compilation, subprocesses or GPU queries.')
        for _, _, cmd in commands(args):
            print(shlex.join(cmd))
        print('Total: 6 fresh preflights/reports, 258 formal windows / 2064 rank records / 12900 formal steps.')
        return
    require(args.python.is_file() and (args.build/'ablation4-manifest.json').is_file(), 'Training Python / parent build missing')
    if args.flux_build is None or args.ablation7_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(), 'Build CUDA Toolkit missing')
    if args.flux_build:
        require((args.flux_build/'manifest.json').is_file(), 'GemmOnly build manifest missing')
    if args.ablation7_build:
        record = read(args.ablation7_build/'ablation7-manifest.json')
        require(Path(record['parent_build']).resolve() == args.build, 'Reused ablation7 build has a different parent')
    args.out.mkdir(parents=True, exist_ok=False); (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', order=ORDER, execution_timing=TIMING,
          expected_timing_windows=TOTAL_WINDOWS, expected_formal_optimizer_steps=TOTAL_WINDOWS*50))
    write(args.out/'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        if args.foreground:
            supervise(args)
        else:
            print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
            print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.', flush=True)
    except Exception as error:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
