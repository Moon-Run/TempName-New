"""Strict ablation evidence gates and all-sample paired-block statistics (CPU only)."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import random
import statistics

from ablation_plan import PROTOCOL, definitions, mask_sha, schedule, experiment
from ablation_routing import module_routes
from common import read, require, sha, verify_files, write
from report import fingerprints, paired


def verify_inputs(root, offline=False):
    root = Path(root)
    info = read(root/'submission.json')
    spec = experiment(info['protocol'])
    definitions, schedule = spec.definitions, spec.schedule
    verify_files(root, info['files'])
    expected = [row for base in info['bases'] for row in schedule(
        base, definitions(base, info['seeds']), info['blocks_per_round'], info['schedule_seed'])]
    require(info['schedule'] == read(root/'schedule.json') == expected and
            info['timing_windows'] == len(expected), 'Schedule changed')
    if not offline:
        verify_files('/', info['source_manifests'])
        verify_files('/', info['build_files'])
    if info['protocol'] != PROTOCOL:
        from ablation2_report import verify_protocol
        verify_protocol(root, info, offline)
    return info


def window(root, directory, base, key, mode, block=-1, index=-1):
    info = read(root/'submission.json')
    c = info['config']; world = c['tp']
    v = info['variants'][base+'-'+key]
    definition = experiment(info['protocol']).definitions(base, info['seeds'])[key]
    records = [read(directory/f'rank{r}.json') for r in range(world)]
    model = {k: c[k] for k in ('layers', 'hidden', 'ffn', 'heads', 'sequence', 'tp',
                              'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')}
    model['name'] = 'h100-phase6'
    plan = read(root/'scripts'/base/'selection-plan.json')
    for rank, r in enumerate(records):
        require(r['completed'] and r['passed'] and r['skips'] == r['fallback_calls'] == 0,
                'Failed/incomplete/skipped/fallback window')
        require(r['run_id'] == info['run_id'] and r['policy'] == key and r['mode'] == mode and
                r['ablation'] == definition, 'Window group/base identity mismatch')
        require(r['derived_plan_sha256'] == v.get('derived_plan_sha256') and
                r['parent_plan_sha256'] == info['parent_plan_sha256'], 'Rank plan hash mismatch')
        require(r['rank'] == r['local_rank'] == r['tp_rank'] == rank and r['dp_rank'] == 0 and
                r['tp_group_ranks'] == list(range(world)) and r['dp_group_ranks'] == [rank] and
                r['world_size'] == world, 'Not a complete local TP8 group')
        require(r['hardware']['host'] == info['calibration_hardware']['host'] and
                r['hardware']['devices'] == info['calibration_hardware']['devices'] and
                r['hardware']['local_rank'] == rank and
                r['gpu_uuid'] == info['calibration_hardware']['devices'][rank]['uuid'],
                'Recalibrate on this exact H100 allocation/device ordering')
        require(r['model'] == model and r['block'] == block and r['window'] == index, 'Model/pairing mismatch')
        if info['protocol'] != PROTOCOL:
            require(r['padded_vocab_size'] == 33792, 'Wrong actual padded vocabulary')
        require(r['timing_protocol'] == 'full-step-one-clear-v2' and not r['sampling'] and
                r['profiling'] == (mode == 'profile'), 'Wrong timing/execution path')
        require(all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']]), 'Non-finite trajectory')
        cfg = read(root/'scripts'/base/'config.json')
        warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else 10
        count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else 20
        require(r['warmup_steps'] == warmup and len(r['losses']) == len(r['grad_norms']) == count,
                'Incomplete full-step trajectory')
        if mode != 'profile':
            require(r['timed_steps'] == count and math.isfinite(r['elapsed_seconds']) and
                    r['elapsed_seconds'] > 0, 'Truncated/invalid measurement')
        require(r['module_policies'] == module_routes(r['observed_shapes'], key, c['layers'], model),
                'Actual model routing mismatch')
        names = set(r['module_policies'])
        require(set(r['target_modules']) == names and set(r['warmup_calls']) == names and
                all(n == warmup*c['global_batch']//c['micro_batch'] for n in r['warmup_calls'].values()),
                'Missing target module execution')
        require(r['replaced_modules'] == (0 if key == 'native' else c['layers'] if key == 'native_taco'
                                          else 2*c['layers']), 'Wrong adapter path')
        if key == 'native':
            require(r['flux_library_loaded'] is False and not r['selection_checks'], 'Native loaded a candidate path')
            continue
        if key == 'native_taco':
            from ablation2_report import native_window
            native_window(r, names, c, mode)
            continue
        for lib, expected in v['libraries'].items():
            require(r[lib] == expected, 'Wrong loaded library path/hash')
        require(len(v['libraries']) == 2, 'Missing library provenance')
        mlp = {n for n in names if n.endswith('.mlp.linear_fc2')}
        require(set(r['selection_checks']) == mlp, 'Missing/extra MLP wire checks')
        require(r['codec_placements'] == {n: 'fused' if n in mlp else 'bf16' for n in names},
                'BF16 ablation must retain enabled V2 MLP and V3 attention')
        mask = plan['entries'][key]['mask']
        bf16 = (world-1)*(c['sequence']*c['micro_batch']//world)*c['hidden']*2
        expected = dict(mask_sha256=mask_sha(mask), wire_bytes=bf16+sum(mask[rank])*128*(136-256),
                        bf16_wire_bytes=bf16)
        require(all(check == expected for check in r['selection_checks'].values()), 'Wrong physical mask/wire bytes')
        if mode == 'smoke':
            require(set(r['forward_checks']) == names, 'Missing numerical checks')
            for name, check in r['forward_checks'].items():
                require(math.isfinite(check['max_abs']) and math.isfinite(check['relative_l2']), 'Non-finite forward error')
                if name in mlp:
                    require(check['passed'] and check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01 and
                            check['relative_l2'] <= .05 and math.isfinite(check['protocol_relative_l2']) and
                            check['protocol_relative_l2'] <= .01, 'Codec numerical budget failed')
    return records


def preflight(root, offline=False):
    root = Path(root)
    info = verify_inputs(root, offline)
    definitions = experiment(info['protocol']).definitions
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU execution')
    reference = None; kernels = {}; evidence = []
    if info['protocol'] != PROTOCOL:
        start = read(root/'start-gpu-check.json')
        require(start['passed'] and start['run_id'] == info['run_id'], 'Missing successful suite startup check')
        evidence.append(root/'start-gpu-check.json')
    for base in info['bases']:
        selected = read(root/'scripts'/base/'selection-plan.json')['entries']
        for key in definitions(base, info['seeds']):
            for mode in ('smoke', 'profile'):
                directory = root/'results'/base/(mode+'-'+key)
                rows = window(root, directory, base, key, mode)
                fp = fingerprints(rows)
                if reference is None:
                    reference = fp
                require(fp == reference, 'Different initialization/RNG/tokens/software/hardware across groups')
                evidence += [p for p in directory.rglob('*') if p.is_file()]
            for rank in range(8):
                path = root/'results'/base/('profile-'+key)/f'rank{rank}-profile.json.gz'
                with gzip.open(path, 'rt') as f:
                    events = json.load(f)['traceEvents']
                names = sorted({e['name'] for e in events if e.get('cat') == 'kernel'})
                require(names, 'Empty GPU profile')
                if key in ('native', 'native_taco'):
                    require(not any('flux_bf16' in n or 'taco_decode' in n for n in names), 'Native executed candidate kernel')
                    if key == 'native_taco':
                        from ablation2_report import native_profile
                        native_profile(events)
                else:
                    require(any('flux_bf16' in n and 'sm80' in n.lower() for n in names), 'Candidate V2 MLP kernel missing')
                    require(any('flux_bf16' in n and 'sm90' in n.lower() for n in names), 'V3 attention kernel missing')
                    # This kernel also performs pure BF16 reduction when all mask bits are zero.
                    require(any('taco_decode_ring_kernel' in n for n in names), 'Frozen TP8 mixed-wire reduction missing')
                kernels[f'{base}/{key}/rank{rank}'] = names
            if key in ('native', 'native_taco'):
                continue
            path = root/'results'/base/f'mapping-{key}.csv'
            with path.open() as f:
                rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
            entry = selected[key]; tiles = entry['tiles']; nt = info['config']['hidden']//128
            require(len(rows) == 8*tiles and {r['rank'] for r in rows} == set(range(8)), 'Incomplete GPU mapping')
            for rank in range(8):
                rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
                require([r['index'] for r in rr] == list(range(tiles)), 'Missing/duplicate mapping slots')
                require([r['m']*nt+r['n'] for r in rr] == entry['coords'][rank] and
                        all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 for r in rr),
                        'Executed physical mapping differs from plan')
            evidence.append(path)
    result = dict(protocol=info['protocol'], completed=True, submission_sha256=sha(root/'submission.json'),
                  fingerprints=reference, kernel_names=kernels,
                  files={str(p.relative_to(root)): sha(p) for p in evidence})
    write(root/'preflight.json', result)
    return result


def contrasts(values, seeds):
    comparisons = [(g, 'native') for g in values if g != 'native']
    comparisons += [('G1', 'G0'), ('G2', 'G1'), ('G4', 'G3'), ('G3', 'G1'), ('G4', 'G2'), ('G4', 'G6')]
    comparisons += [('G4', f'G5-s{s}') for s in seeds]
    result = {f'{a}/{b}': [math.log(x/y) for x, y in zip(values[a], values[b])]
              for a, b in comparisons}
    result['G4/G5-seed-mean'] = [math.log(values['G4'][i])-statistics.mean(
        math.log(values[f'G5-s{s}'][i]) for s in seeds) for i in range(len(values['G4']))]
    result['arrival_quantization_interaction'] = [
        math.log(a/b)-math.log(c/d) for a, b, c, d in zip(values['G4'], values['G3'], values['G2'], values['G1'])]
    return result


def summarize(values, seeds):
    lengths = {len(v) for v in values.values()}
    require(len(lengths) == 1 and min(lengths) >= 2 and
            all(math.isfinite(x) and x > 0 for v in values.values() for x in v), 'Incomplete paired blocks')
    values = dict(values)
    values['G5-seed-mean'] = [math.exp(statistics.mean(math.log(values[f'G5-s{s}'][i]) for s in seeds))
                              for i in range(min(lengths))]
    return dict(ms_per_step=values, median_ms={k: statistics.median(v) for k, v in values.items()},
                tokens_per_second={k: [8192*1000/x for x in v] for k, v in values.items()},
                comparisons={k: dict(paired(v), log_ratios=v) for k, v in contrasts(values, seeds).items()})


def measurement_identity(info, gate):
    """Bind independent repeats to the same runtime, worker and loaded code.

    Paths may differ between session directories; content and runtime identities
    must agree. Source hashes alone do not identify compiler outputs or workers.
    """
    return dict(
        fingerprints=gate['fingerprints'],
        worker_files={name: digest for name, digest in info['files'].items()
                      if name.startswith('scripts/')},
        libraries={key: {name: library['sha256'] for name, library in variant.get('libraries', {}).items()}
                   for key, variant in info['variants'].items()},
        controllers={Path(name).name: digest for name, digest in info['controllers'].items()})


def report(root, offline=False):
    root = Path(root)
    if read(root/'submission.json')['protocol'] != PROTOCOL:
        from ablation2_report import report as five_group_report
        return five_group_report(root, offline)
    info = verify_inputs(root, offline)
    gate = read(root/'preflight.json')
    require(info['executable'] and gate['completed'] and gate['submission_sha256'] == sha(root/'submission.json'),
            'Verified GPU preflight required')
    verify_files(root, gate['files'])
    result = dict(protocol=PROTOCOL, completed=False, run_id=info['run_id'], config=info['config'],
                  seeds=info['seeds'], bases={}, all_samples=[], timing_windows=info['timing_windows'],
                  parent_plan_sha256=info['parent_plan_sha256'], parent_freeze=info['parent_freeze'],
                  source_hashes={k: v.get('source_sha256') for k, v in info['variants'].items()},
                  measurement_identity=measurement_identity(info, gate),
                  sample_policy='all samples; no slow-window removal', acceptance_threshold=None,
                  offline=offline, source='frozen rank library hashes' if offline else 'verified on-disk libraries',
                  interaction='log(G4/G3)-log(G2/G1); percent=100*(1-exp(mean(log interaction)))')
    lines = ['# H100 GPT 6.7B TP8 完整 optimizer step 消融', '',
             '保留全样本；每窗口 10+20 步，取全部 rank 的最大 elapsed。95% 区间按完整配对块 bootstrap。',
             'G5 各 seed 单独报告，seed-mean 在每块内取几何平均后配对；没有单项 4% 门槛。', '',
             '| 基础顺序 | 轮次 | 比较 | 耗时下降 % | 95% 区间 | 结论 |',
             '| --- | --- | --- | ---: | --- | --- |']
    for base in info['bases']:
        summaries = {}
        for rep in (1, 2):
            values = {k: [] for k in definitions(base, info['seeds'])}
            for block in range(info['blocks_per_round']):
                jobs = [j for j in info['schedule'] if (j['base'], j['round'], j['block']) == (base, rep, block)]
                require({j['policy'] for j in jobs} == set(values), 'Incomplete block')
                for job in jobs:
                    directory = root/'results'/job['directory']
                    rows = window(root, directory, base, job['policy'], 'timing', block, job['index'])
                    require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing inputs differ from preflight')
                    elapsed = [r['elapsed_seconds'] for r in rows]
                    ms = max(elapsed)*1000/20
                    values[job['policy']].append(ms)
                    result['all_samples'].append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms))
            summaries[str(rep)] = summarize(values, info['seeds'])
        combined = {k: summaries['1']['ms_per_step'][k]+summaries['2']['ms_per_step'][k]
                    for k in definitions(base, info['seeds'])}
        summaries['combined'] = summarize(combined, info['seeds'])
        result['bases'][base] = summaries
        for rep, summary in summaries.items():
            for label, stats in summary['comparisons'].items():
                low, high = stats['ci95']
                status = '确认正增益' if low > 0 else '尚未确认正增益'
                lines.append(f'| {base} | {rep} | {label} | {stats["percent"]:.3f} | [{low:.3f}, {high:.3f}] | {status} |')
    require(len(result['all_samples']) == info['timing_windows'], 'Missing formal windows')
    lines += ['', '| 基础顺序 | 轮次 | 组别/seed | 中位 ms/step | 中位 tokens/s |',
              '| --- | --- | --- | ---: | ---: |']
    for base, summaries in result['bases'].items():
        for rep, summary in summaries.items():
            for key, ms in summary['median_ms'].items():
                tps = statistics.median(summary['tokens_per_second'][key])
                lines.append(f'| {base} | {rep} | {key} | {ms:.3f} | {tps:.2f} |')
    result['completed'] = True
    write(root/'report.json', result)
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def aggregate(roots, out, offline=False):
    """Predefined equal-session mean; hierarchical session/whole-block bootstrap."""
    require(len(roots) >= 2, 'At least two independent sessions required')
    reports = [report(Path(r), offline) for r in roots]
    require(len({r['run_id'] for r in reports}) == len(reports), 'Duplicate session/run_id')
    require(all((r['config'], r['seeds'], list(r['bases'])) ==
                (reports[0]['config'], reports[0]['seeds'], list(reports[0]['bases'])) for r in reports),
            'Different experiment configuration; do not pool')
    require(all((r['parent_plan_sha256'], r['parent_freeze'], r['source_hashes']) ==
                (reports[0]['parent_plan_sha256'], reports[0]['parent_freeze'], reports[0]['source_hashes'])
                for r in reports), 'Different frozen plans/implementations; report sessions separately')
    require(all(r['measurement_identity'] == reports[0]['measurement_identity'] for r in reports),
            'Different measurement environment/workers/libraries/controllers; report sessions separately')
    result = dict(protocol=PROTOCOL, sessions=[dict(root=str(r), report_sha256=sha(Path(r)/'report.json')) for r in roots],
                  weighting='equal independent sessions; resample sessions then complete paired blocks',
                  measurement_identity=reports[0]['measurement_identity'], bases={})
    for base in reports[0]['bases']:
        result['bases'][base] = {}
        for key in reports[0]['bases'][base]['combined']['comparisons']:
            sessions = [r['bases'][base]['combined']['comparisons'][key]['log_ratios'] for r in reports]
            rng = random.Random(20261013)
            boot = []
            for _ in range(10000):
                sampled = [rng.choice(sessions) for _ in sessions]
                logmean = statistics.mean(statistics.mean(rng.choice(s) for _ in s) for s in sampled)
                boot.append(100*(1-math.exp(logmean)))
            boot.sort()
            result['bases'][base][key] = dict(percent=100*(1-math.exp(statistics.mean(map(statistics.mean, sessions)))),
                                              ci95=[boot[250], boot[9750]], sessions=len(sessions))
    require(not Path(out).exists(), 'Do not overwrite a session summary')
    write(out, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('roots', nargs='+', type=Path); p.add_argument('--preflight', action='store_true')
    p.add_argument('--offline', action='store_true', help='Returned evidence: verify frozen hashes without local H100 libraries')
    p.add_argument('--aggregate', type=Path)
    a = p.parse_args()
    if a.aggregate:
        aggregate(a.roots, a.aggregate, a.offline)
    else:
        require(len(a.roots) == 1, 'Use --aggregate to combine independent sessions')
        (preflight if a.preflight else report)(a.roots[0], a.offline)
