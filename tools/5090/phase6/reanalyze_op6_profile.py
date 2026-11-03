#!/usr/bin/env python3
"""Read-only diagnostics of the first OP4/OP6 pair, including failed profiles.

This does not run GPUs, verify absent build binaries, resume training, or
change the saved batch state. Profile timing is never a formal step result.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path

from rtx5090_loader import REPO
import op3_protocol as p
import op6_evidence
import op6_profile


def profile_rank(task):
    root, rank = task
    path = Path(root)/f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz'
    events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    result = op6_profile.analyze(events, p.COMBINED, rank, 'op6')
    p.require(result == json.loads(json.dumps(result)), 'Profile result is not JSON-stable')
    graph = result['attention_graph']
    order = graph['timestamp_order_check']
    old_rejected = 0
    by_identity = {}
    for event in events:
        if event.get('cat') == 'kernel':
            by_identity[(event['args'].get('correlation'), event['ts'])] = event
    for overlap in order['tolerated_overlaps']:
        first = by_identity[(overlap['correlation'], overlap['predecessor_ts_us'])]
        second = by_identity[(overlap['correlation'], overlap['successor_ts_us'])]
        old_rejected += overlap['observed_overlap_us'] > min(.5, .05*min(first['dur'], second['dur']))
    return dict(rank=rank, trace=os.path.relpath(path, REPO), trace_sha256=p.sha(path),
        complete_profile_path_reanalyzed=result['execution_validated'],
        forward_graph_launches=graph['forward_graph_launches'],
        backward_graph_launches=graph['backward_graph_launches'],
        timestamp_overlaps=order['tolerated_overlap_count'],
        old_timestamp_rule_rejections=old_rejected,
        max_observed_overlap_us=order['max_observed_overlap_us'],
        timestamp_policy={key: order[key] for key in ('policy', 'graph_kernel_tolerance_us',
            'max_fraction_of_shorter_kernel', 'strict_boundary_tolerance_us')})


def analyze(batch, jobs):
    batch = Path(batch).resolve()
    control, candidate = (batch/'compare1'/f'block00-{arm}' for arm in ('op4', 'op6'))
    report = p.read(control/'report.json')
    p.require(report['completed'] and report['timing_windows'] == 1
              and report['formal_optimizer_steps'] == 50, 'Control has no complete 50-step window')
    files = {}
    for name, expected in report['worker_records'].items():
        digest = p.sha(control/name)
        p.require(digest == expected, 'Changed control worker record: '+name)
        files['op4/'+name] = digest
    numerical = []
    for mode in ('smoke', 'profile'):
        rows = []
        for arm, root in (('op4', control), ('op6', candidate)):
            own = []
            for rank in range(8):
                name = f'results/{mode}-{p.COMBINED}/rank{rank}.json'
                path = root/name
                own.append(p.read(path))
                p.require(own[-1]['completed'] and own[-1]['mode'] == mode,
                          'Incomplete rank record: '+str(path))
                files[arm+'/'+name] = p.sha(path)
            rows.append(own)
        numerical.append(op6_evidence.compare_rows(*rows, mode))
    job = p.schedule(1, 1)[0]
    rows = [p.read(control/f"results/{job['directory']}/rank{rank}.json") for rank in range(8)]
    sample = p.summarize(rows, job)
    p.require(sample == report['all_samples'][0], 'Control timing differs from raw elapsed')
    profiles = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for result in pool.map(profile_rank, [(str(candidate), rank) for rank in range(8)]):
            profiles.append(result)
            print(json.dumps(result), flush=True)
    return dict(schema='op6-first-pair-profile-reanalysis-v1',
        batch=os.path.relpath(batch, REPO), saved_batch_state=p.read(batch/'batch-state.json'),
        op4_formal_ms=sample['ms_per_step'],
        candidate_formal_report_present=(candidate/'report.json').is_file(),
        numerical_checks=numerical, profiles=profiles, rank_records_sha256=files,
        saved_status_modified=False, gpu_work_rerun=False, build_binaries_revalidated=False,
        profile_auditor_sha256=p.sha(Path(op6_profile.__file__)),
        limitation='Offline numerical comparison and raw-trace path reanalysis only. '
        'No completed pair, new formal timings, binary replay or speedup is inferred.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    p.require(1 <= args.jobs <= 8, 'Use 1–8 CPU analysis workers')
    p.require(out != batch and batch not in out.parents, 'Output must be outside the source batch')
    result = analyze(batch, args.jobs)
    p.write(out, result)


if __name__ == '__main__':
    main()
