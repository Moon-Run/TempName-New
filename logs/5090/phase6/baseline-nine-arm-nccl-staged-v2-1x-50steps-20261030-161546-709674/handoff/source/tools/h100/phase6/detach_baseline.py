#!/usr/bin/env python3
"""A/B/C unchanged + Flux/Megatron upstream CUDA TACO, three fresh comparisons.

Default is a read-only preview. --execute builds and detaches the full batch.
Each comparison has its own smoke, profile, 50 timing windows and strict report.
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
import baseline_protocol as routing
import detach_ablation5_run_two as launcher
from common import REPO, read, require, sha, write

ENTRY = Path(__file__).with_name('baseline.py')
TRIALS = ('compare1', 'compare2', 'compare3')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=repeat.DEFAULT_BUILD, help='The unchanged ablation6 A/B/C parent build')
    p.add_argument('--flux-build', type=Path, help='Optional existing ablation6 GemmOnly build')
    p.add_argument('--codec-build', type=Path, help='Optional existing standalone upstream CUDA codec build')
    p.add_argument('--out', type=Path)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--nccl-root', type=Path, help='Development root for Flux; defaults to the parent manifest')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true')
    args = p.parse_args(argv)
    require(args.jobs > 0, 'jobs must be positive')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/('baseline-taco-cuda-3x-50steps-'+stamp)
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.expanduser().resolve())
    return args


def commands(args):
    cli = [str(args.python), str(ENTRY)]
    flux = args.flux_build or args.out/'flux-build'
    codec = args.codec_build or args.out/'codec-build'
    steps = []
    if args.flux_build is None:
        steps += [('flux-prepare', cli+['build-prepare', str(flux)]),
                  ('flux-compile', cli+['compile', str(flux), '--cuda-home', str(args.build_cuda_home),
                    '--jobs', str(args.jobs)]+(['--nccl-root', str(args.nccl_root)] if args.nccl_root else []))]
    if args.codec_build is None:
        steps += [('codec-prepare', cli+['codec-prepare', str(codec)]),
                  ('codec-compile', cli+['codec-compile', str(codec), '--cuda-home', str(args.build_cuda_home)])]
    for trial in TRIALS:
        root = str(args.out/trial)
        run = cli+['run', root, '--python', str(args.python), '--cuda-home', str(args.cuda_home)]
        steps += [(trial+'-prepare', cli+['prepare', root, '--build', str(args.build),
                    '--flux-build', str(flux), '--codec-build', str(codec), '--megatron', str(args.megatron),
                    '--blocks', '5', '--rounds', '2', '--nvls-enable', '0']),
                  (trial+'-dry-run', run), (trial+'-controller', run+['--stage', 'all', '--execute']),
                  (trial+'-report', cli+['report', root])]
    return steps


def check_inputs(root):
    info = read(root/'submission.json')
    require(info['protocol'] == routing.PROTOCOL and info['execution_timing'] == routing.TIMING and
            info['executable'] and not info['source_only'] and info['rounds'] == 2 and
            info['blocks_per_round'] == 5 and info['timing_windows'] == 50 and
            info['schedule'] == routing.schedule() and info['codec_library'] and
            info['communication']['environment'].get('NCCL_NVLS_ENABLE') == '0',
            'Expected a fresh five-arm standalone CUDA baseline experiment')
    return info


def check_report(root, info):
    result, gate = read(root/'report.json'), read(root/'preflight.json')
    require(read(root/'all-state.json')['status'] == 'completed' and result['completed'] and
            result['correctness_passed'] and result['protocol'] == routing.PROTOCOL and
            result['run_id'] == info['run_id'] and result['execution_timing'] == routing.TIMING and
            result['round_count'] == 2 and result['blocks_per_round'] == 5 and
            result['timing_windows'] == len(result['all_samples']) == 50 and
            result['formal_optimizer_steps'] == 2500 and result['timing_rank_records'] == 400 and
            result['preflight_rank_records'] == 80 and result['profile_traces'] == 40 and
            result['candidate_shape_mappings'] == 6, 'Incomplete strict baseline report')
    require(gate['completed'] and gate['protocol'] == routing.PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and
            result['evidence']['preflight_sha256'] == sha(root/'preflight.json') and
            all(result[k] == gate[k] == info[k] for k in ('communication', 'codec_identity', 'codec_library')),
            'Preflight/report identity mismatch')
    return result


def supervise(args):
    path = args.out/'batch-state.json'
    state = dict(status='preparing', protocol=routing.PROTOCOL, execution_timing=routing.TIMING,
                 expected_timing_windows=150, expected_formal_optimizer_steps=7500, comparisons={})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    inputs, reports = {}, []
    try:
        if args.flux_build is None and args.nccl_root is None:
            recorded = read(args.build/'fused/manifest.json').get('nccl_files', {})
            headers = [Path(p) for p in recorded if p.endswith('/include/nccl.h')]
            require(len(headers) == 1, 'Pass --nccl-root: no unique parent NCCL development root')
            require(all(Path(p).is_file() and sha(p) == h for p, h in recorded.items()), 'Recorded NCCL files changed')
            args.nccl_root = headers[0].parent.parent
        for label, cmd in commands(args):
            state.update(status='running', stage=label)
            write(path, state); launcher.invoke(cmd, args.out/(label+'.log'), env)
            if label.endswith('-dry-run'):
                trial = label.removesuffix('-dry-run'); info = check_inputs(args.out/trial)
                if inputs:
                    first = next(iter(inputs.values())); repeat.same_inputs(first, info)
                    for key in ('flux_build_manifest', 'codec_build_manifest', 'codec_identity', 'codec_library'):
                        require(first[key] == info[key], 'Repeat input changed: '+key)
                    require(info['run_id'] not in {i['run_id'] for i in inputs.values()}, 'Duplicate comparison run ID')
                inputs[trial] = info
                state['comparisons'][trial] = dict(status='prepared', run_id=info['run_id'])
            if label.endswith('-report'):
                trial = label.removesuffix('-report'); result = check_report(args.out/trial, inputs[trial])
                if reports: require(result['measurement_identity'] == reports[0]['measurement_identity'], 'Repeat runtime changed')
                reports.append(result)
                state['comparisons'][trial].update(status='completed', timing_windows=50,
                    formal_optimizer_steps=2500, report=str(args.out/trial/'report.json'),
                    report_sha256=sha(args.out/trial/'report.json'))
            write(path, state)
        require(len(reports) == 3, 'All three strict reports are required')
        state.update(status='completed', stage='done', timing_windows=150,
                     timing_rank_records=1200, formal_optimizer_steps=7500)
        write(path, state)
        print('Completed: three comparisons, 150 windows / 7500 formal optimizer steps.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(path, state); raise


def main(argv=None):
    args = arguments(argv)
    print('Batch:', args.out, flush=True)
    print('A/B/C unchanged; Flux/Megatron + standalone upstream TACO CUDA codec; NCCL All-to-All.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, fork, compilation, subprocesses or GPU queries.')
        for _, cmd in commands(args): print(shlex.join(cmd))
        print('Three fresh comparisons, 10 warmup + 50 timed steps; 150 windows / 7500 formal steps.')
        return
    require(args.python.is_file() and (args.build/'ablation4-manifest.json').is_file(),
            'Training Python / unchanged A/B/C parent build missing')
    if args.flux_build is None or args.codec_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(), 'Build CUDA Toolkit missing')
    for root in (args.flux_build, args.codec_build):
        if root: require((root/'manifest.json').is_file(), 'Requested reused build manifest missing')
    args.out.mkdir(parents=True, exist_ok=False); (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', protocol=routing.PROTOCOL,
          expected_timing_windows=150, expected_formal_optimizer_steps=7500))
    write(args.out/'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.foreground: return supervise(args)
    print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
    print('Started, not completed. See batch-state.json and handoff/container-supervisor.log.', flush=True)


if __name__ == '__main__':
    main()
