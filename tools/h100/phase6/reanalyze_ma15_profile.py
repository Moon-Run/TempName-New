#!/usr/bin/env python3
"""Recheck completed MA15 evidence without CUDA, compilation, or training.

Default: validate and print a concise receipt. --apply additionally saves the
missing profile summary, preserves the failed state, and completes the batch
with explicit offline-reanalysis provenance. Frozen sources are never edited.
"""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root):
    plan = read(root / 'plan.json')
    for name, digest in plan['source_files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'Frozen source escaped experiment root: ' + name)
        require(sha(path) == digest, 'Frozen source changed: ' + name)
    return plan


def reader_changes(old, new):
    def functions(path):
        return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    before, after = functions(old), functions(new)
    require(before.keys() == after.keys(), 'Unexpected profile reader structure change')
    changed = sorted(name for name in before if before[name] != after[name])
    require(set(changed) == {'_same_stream_order', 'trace_attention_graph_records'},
            'Reanalysis is restricted to the MA15 graph timestamp-order fix')
    return changed


def child(stage, root, reader):
    """A fresh interpreter isolates each experiment's frozen module namespace."""
    verify_sources(root)
    scripts = root / 'scripts'
    sys.path.insert(0, str(scripts))
    entry = scripts / ('bench.py' if stage == 'operator' else 'fullstep-controller.py')
    with contextlib.redirect_stdout(io.StringIO()):
        controller = runpy.run_path(str(entry))
        controller['verify'](root, offline=True)
        report = controller['report'](root, offline=True, save=False)
        require(report == read(root / 'report.json'), 'Saved timing report differs from raw results')
        if stage == 'operator':
            profile = controller['profile_summary'](root, offline=True)
            require(profile == read(root / 'profile-summary.json'), 'Saved operator profile changed')
            result = dict(timing_verified=True, profile_verified=True,
                          report_sha256=sha(root / 'report.json'),
                          profile_sha256=sha(root / 'profile-summary.json'))
        else:
            frozen = controller['ma15_profile']
            changed = reader_changes(Path(frozen.__file__), reader)
            spec = importlib.util.spec_from_file_location('_ma15_graph_timestamp_reanalysis', reader)
            corrected = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(corrected)
            # Validate the original frozen files, then use the explicitly
            # recorded reader correction in memory. Hash checks stay enabled;
            # all other frozen validators and the measured worker stay intact.
            original = frozen.trace_attention_graph_records
            try:
                frozen.trace_attention_graph_records = corrected.trace_attention_graph_records
                profile = controller['profile_summary'](root, offline=True)
            finally:
                frozen.trace_attention_graph_records = original
            require(profile['execution_validated'] and len(profile['profiles']) == 16,
                    'Both arms and all eight profile ranks must pass')
            result = dict(timing_verified=True, profile=profile, report=report,
                          changed_reader_functions=changed,
                          original_reader_sha256=sha(frozen.__file__))
    require('torch' not in sys.modules, 'Offline analysis unexpectedly imported Torch')
    return result


def invoke(stage, root, reader):
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), str(root),
               '--_stage', stage, '--reader', str(reader)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=600)
    require(process.returncode == 0, stage + ' offline validation failed:\n' + process.stderr)
    return json.loads(process.stdout)


def reanalyze(batch, reader):
    state_path = batch / 'batch-state.json'
    state = read(state_path)
    require(state['protocol'] == 'h100-ma15-attention-graph-detached-v1' and
            state['status'] == 'failed' and state['stage'] == 'fullstep-profile',
            'Expected a MA15 batch that stopped during fullstep-profile')
    require(state.get('timing_report_verified') is True,
            'Formal training must already have completed')
    identity_paths = [state_path, batch / 'fullstep/plan.json', batch / 'operator/plan.json',
                      batch / 'fullstep/report.json', batch / 'operator/report.json',
                      batch / 'operator/profile-summary.json', reader, Path(__file__).resolve()]
    before = {str(path): sha(path) for path in identity_paths}
    operator = invoke('operator', batch / 'operator', reader)
    full = invoke('fullstep', batch / 'fullstep', reader)
    report = full['report']
    require(report['formal_optimizer_steps'] == state['expected_formal_optimizer_steps'] and
            len(report['all_samples']) == state['expected_timing_windows'] and
            report['timing_rank_records'] == len(report['all_samples']) * 8,
            'Missing formal timing windows or ranks')
    evidence = dict(before)
    for name, digest in full['profile']['evidence_files'].items():
        evidence[str(batch / 'fullstep' / name)] = digest
    # Recheck that results did not change while the independent validators ran.
    require(all(sha(path) == digest for path, digest in evidence.items()),
            'Evidence changed during reanalysis')
    overlaps = {key: value['attention_graph'].get('timestamp_order_check', {})
                for key, value in full['profile']['profiles'].items()
                if value['attention_graph'].get('installed')}
    receipt = dict(protocol='ma15-offline-profile-reanalysis-v1',
                   completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   operator=operator, fullstep_execution_validated=True,
                   formal_optimizer_steps=report['formal_optimizer_steps'],
                   timing_windows=len(report['all_samples']), profile_rank_records=16,
                   changed_reader_functions=full['changed_reader_functions'],
                   original_reader_sha256=full['original_reader_sha256'],
                   corrected_reader_sha256=sha(reader), reanalysis_tool_sha256=sha(__file__),
                   input_sha256=evidence, timestamp_order_checks=overlaps,
                   comparisons=report['comparisons'],
                   limitation='Offline validation of archived rank records and traces. No new GPU run, '
                              'library rebuild, host-state measurement, or affinity retest.')
    return state, full, receipt


def write_new(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def apply(batch, state, full, receipt):
    archive = batch / 'profile-reanalysis'
    profile_path = batch / 'fullstep/profile-summary.json'
    require(not archive.exists() and not profile_path.exists(), 'Reanalysis output already exists')
    archive.mkdir()
    # Preserve original bytes (including the failed state); never edit traces,
    # raw ranks, frozen scripts, manifests, timing reports, or failure logs.
    (archive / 'batch-state.before.json').write_bytes((batch / 'batch-state.json').read_bytes())
    write_new(archive / 'receipt.json', receipt)
    profile = full['profile']
    profile['analysis_provenance'] = dict(mode='offline_profile_reanalysis',
                                        receipt='../profile-reanalysis/receipt.json',
                                        corrected_reader_sha256=receipt['corrected_reader_sha256'])
    write_new(profile_path, profile)
    report = full['report']
    updated = dict(state, status='completed', stage='done', completion_mode='offline_profile_reanalysis',
                   previous_failure=dict(status=state['status'], stage=state['stage'], error=state.get('error'),
                                         backup='profile-reanalysis/batch-state.before.json'),
                   fullstep_execution_path_validated=True, operator_correctness_validated=True,
                   operator_execution_path_validated=True, timing_report_verified=True,
                   formal_optimizer_steps=report['formal_optimizer_steps'], timing_windows=len(report['all_samples']),
                   profile_reanalysis='profile-reanalysis/receipt.json')
    updated.pop('error', None)
    for key in ('acceptance_status', 'target_evaluated', 'target_validated',
                'target_point_observed', 'increment_over_MA11_validated'):
        updated[key] = report[key]
    updated['reports'] = dict(state['reports'], fullstep_profile='fullstep/profile-summary.json')
    temporary = archive / 'batch-state.completed.json'
    write_new(temporary, updated)
    os.replace(temporary, batch / 'batch-state.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--apply', action='store_true', help='Save derived summary and complete the batch offline')
    parser.add_argument('--reader', type=Path, default=HERE / 'ma15_profile.py')
    parser.add_argument('--_stage', choices=('operator', 'fullstep'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    batch, reader = args.batch.resolve(), args.reader.resolve()
    if args._stage:
        print(json.dumps(child(args._stage, batch, reader)))
        return
    state, full, receipt = reanalyze(batch, reader)
    if args.apply:
        apply(batch, state, full, receipt)
    print(json.dumps(dict(validated=True, applied=args.apply, batch=str(batch),
                          timing_windows=receipt['timing_windows'],
                          formal_optimizer_steps=receipt['formal_optimizer_steps'],
                          profile_rank_records=16,
                          tolerated_timestamp_overlaps=sum(v.get('tolerated_overlap_count', 0)
                                                           for v in receipt['timestamp_order_checks'].values())), indent=2))


if __name__ == '__main__':
    main()
