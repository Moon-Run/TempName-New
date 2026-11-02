#!/usr/bin/env python3
"""Read-only OP4 attention submission evidence, with complete OP3 validation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import median

from rtx5090_loader import REPO
import op3_profile
import ma11_profile
import ma15_profile
from ma9_profile import Index, contains, end, require


def analyze_trace(path, rank):
    events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    validated = op3_profile.analyze(events, 'op3-combined', rank)
    index = Index(events)
    cpu = ma11_profile.CPUIndex(index.cpu)
    result, all_scopes, all_envelopes, all_work, all_launches = {}, [], [], [], []
    for phase, marker in [('forward', ma11_profile.FORWARD_MARKER),
                          ('backward', ma11_profile.BACKWARD_MARKER)]:
        scopes, envelopes, work, launches = [], [], [], []
        markers = [e for e in events if e.get('cat') == 'user_annotation' and e['name'] == marker]
        require(len(markers) == 128, 'Historical attention invocation coverage changed')
        for scope in markers:
            owner_name = 'MA11MaskedSoftmax' if phase == 'forward' else 'MA11MaskedSoftmaxBackward'
            owners = [e for e in index.cpu if e['name'] == owner_name and contains(e, scope)]
            require(len(owners) == 1, 'Original MA11 autograd owner is ambiguous')
            if phase == 'forward':
                first = cpu.neighbor(scope, 'aten::baddbmm', True)
                pv = cpu.neighbor(scope, 'aten::bmm', False)
                last = cpu.neighbor(pv, 'aten::contiguous', False)
                selected = [first, owners[0], pv, last]
            else:
                first = cpu.neighbor(scope, 'BmmBackward0', True)
                last = cpu.neighbor(scope, 'BaddbmmBackward0', False)
                selected = [first, owners[0], last]
            envelope = dict(tid=first['tid'], ts=first['ts'], dur=end(last) - first['ts'])
            require(all(contains(envelope, s) for s in selected), 'Attention envelope contains wrong scopes')
            scopes.extend(selected)
            envelopes.append(envelope)
            for selected_scope in selected:
                device, calls = index.work(selected_scope)
                work.extend(device)
                launches.extend(calls.values())
        require(len({id(e) for e in work}) == len(work), 'GPU accounting duplicated an attention operation')
        result[phase] = dict(
            calls=len(markers),
            selected_cpu_scope_sum_ms=sum(e['dur'] for e in scopes) / 1000,
            selected_cpu_scope_union_ms=ma15_profile.interval_union_ms(scopes),
            enclosing_cpu_span_union_ms=ma15_profile.interval_union_ms(envelopes),
            cuda_api_union_ms=ma15_profile.interval_union_ms(launches),
            cuda_api_calls=len(launches),
            gpu_work_ms=sum(e['dur'] for e in work) / 1000,
            gpu_kernel_work_ms=sum(e['dur'] for e in work if e['cat'] == 'kernel') / 1000,
            gpu_work_union_ms=ma15_profile.interval_union_ms(work),
            gpu_kernel_calls=sum(e['cat'] == 'kernel' for e in work),
            gpu_memset_calls=sum(e['cat'] == 'gpu_memset' for e in work),
            gpu_memcpy_calls=sum(e['cat'] == 'gpu_memcpy' for e in work))
        all_scopes.extend(scopes)
        all_envelopes.extend(envelopes)
        all_work.extend(work)
        all_launches.extend(launches)
    require(len({id(e) for e in all_work}) == len(all_work), 'Forward/backward GPU accounting overlapped')
    result['combined'] = dict(
        selected_cpu_scope_union_ms=ma15_profile.interval_union_ms(all_scopes),
        enclosing_cpu_span_union_ms=ma15_profile.interval_union_ms(all_envelopes),
        cuda_api_union_ms=ma15_profile.interval_union_ms(all_launches),
        gpu_work_ms=sum(e['dur'] for e in all_work) / 1000,
        gpu_work_union_ms=ma15_profile.interval_union_ms(all_work))
    return dict(rank=rank, trace=os.path.relpath(path.resolve(), REPO.parent), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                control_composed_profile_validated=True,
                cache_backward_allgathers=validated['activation_cache']['backward_allgathers'],
                zero_count_kernels=validated['zero_count']['compute_kernel_calls'],
                softmax_kernels=validated['attention_softmax']['softmax_compute_kernel_calls'],
                measurements=result)



def analyze(batch):
    from op5_evidence import reference_record
    batch = Path(batch).resolve()
    reference = reference_record(batch)
    windows = []
    for child in reference['children']:
        directory = batch/child['directory']/'results/profile-op3-combined'
        rows = []
        for rank in range(8):
            rows.append(analyze_trace(directory/f'rank{rank}-profile.json.gz', rank))
            print(f"Validated {child['directory']} rank{rank}", flush=True)
        aggregate = {phase: {key: median(row['measurements'][phase][key] for row in rows)
                            for key in rows[0]['measurements'][phase]}
                     for phase in ('forward', 'backward', 'combined')}
        windows.append(dict(directory=child['directory'], formal_ms=child['historical_mean_ms'],
                            per_rank=rows, median_over_ranks=aggregate))
    return dict(schema='op6-op4-attention-submission-v1',
        source_batch=os.path.relpath(batch, REPO.parent),
        source_report_sha256=reference['report_sha256'],
        historical_mean_ms=reference['historical_mean_ms'], candidate_gpu_measured=False,
        definition=dict(
            forward_scopes=['aten::baddbmm', 'MA11MaskedSoftmax', 'aten::bmm', 'context aten::contiguous'],
            backward_scopes=['BmmBackward0', 'MA11MaskedSoftmaxBackward', 'BaddbmmBackward0'],
            cpu_scope='Union of selected scopes within one rank.',
            cpu_envelope='Union of complete attention submission spans including gaps.',
            gpu_work='Actual CUDA correlation-owned kernels, memcpy and memset, counted once.',
            aggregate='Median of per-rank sums/unions, separately in each historical window.'),
        windows=windows,
        caveat='Profiler changes submission and rank arrival. CPU scopes, spans, API and GPU work '
               'overlap; neither subtraction nor their sum is unprofiled critical-path savings. '
               'This artifact contains no OP6 GPU correctness or performance result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    if out == batch or batch in out.parents:
        parser.error('--out must be outside the read-only historical batch')
    result = analyze(batch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps([dict(directory=w['directory'], **w['median_over_ranks']['combined'])
                      for w in result['windows']], indent=2))


if __name__ == '__main__':
    main()
