#!/usr/bin/env python3
"""Read the original MA11 traces to quantify the MA18 transport boundary.

No candidate runs and no input files are modified. Reuses MA17's independently
checked column attribution, not its submission mechanism or saved results.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import analyze_ma17_baseline as original_column


def analyze(paths):
    rows = [original_column.analyze_trace(path, rank) for rank, path in enumerate(paths)]
    return summarize_rows(rows)


def summarize_rows(rows):
    kernels = Counter(call['allgather_kernel'] for row in rows for call in row['per_call'])
    original_column.require(sum(kernels.values()) == 8 * 256,
                            'Expected eight ranks and 256 target gathers per rank')
    original_column.require(all('AllGather_RING_LL(' in name for name in kernels),
                            'Historical transport differs from the observed NCCL RING LL path')
    local_bytes, ranks, calls = 1048576 * 2, 8, 256
    return dict(
        schema='ma18-historical-ma11-column-transport-evidence-v1',
        source_arm=original_column.SOURCE_ARM,
        candidate_gpu_measured=False,
        source_analysis_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(original_column.__file__))},
        definition=dict(
            topology='Per rank and optimizer step: 128 QKV and 128 FC1 forward gathers; '
                     'four vocabulary heads excluded. Full GEMM starts after its gather.',
            attribution='Original MA11 composed checker plus forward/backward autograd pairing, '
                        'CPU scope containment and CUDA launch correlation; all eight original '
                        'traces are reparsed, not copied from an earlier summary.',
            observed_transport='All 2048 target invocations launch NCCL AllGather RING LL '
                               'kernels on a communication stream distinct from GEMM.',
            proposed_transport='Stage local BF16 input in one of two IPC-exported slots, '
                               'shared across all 64 modules, use the existing GPU publication '
                               'barrier, and enqueue cyclic-source peer copies into rank-ordered '
                               'destinations in the original independent full input allocation. '
                               'Keep the full GEMM and retained input for backward.',
            bytes='Logical payload/copy bytes only. A read plus write is not counted twice '
                  'here. These are not measured HBM, NVLink, PCIe or NCCL wire traffic.',
            aggregation='First sum or take medians within a rank, then take the median of eight '
                        'rank statistics. Nested CPU/API scopes and GPU intervals must not '
                        'be added together or interpreted as removable time.'),
        logical_bytes=dict(
            dtype='BF16', bytes_per_element=2, group_size=ranks,
            input_per_call=local_bytes, output_per_call=local_bytes * ranks,
            remote_input_per_call=local_bytes * (ranks - 1),
            candidate_stage_per_call=local_bytes,
            candidate_pull_all_ranks_per_call=local_bytes * ranks,
            remote_input_per_rank_step=local_bytes * (ranks - 1) * calls,
            candidate_stage_per_rank_step=local_bytes * calls,
            candidate_pull_all_ranks_per_rank_step=local_bytes * ranks * calls,
            candidate_shared_staging_slots_per_rank=2 * local_bytes,
            candidate_all_rank_staging_mappings=ranks * 2 * local_bytes),
        observed_kernel_counts=dict(sorted(kernels.items())),
        caveat='NCCL GPU duration contains rank arrival skew and profiler effects. This analysis '
               'does not measure copy-engine use, SM occupancy, unprofiled communication cost '
               'or a removable budget. MA18 adds a staging copy and seven remote reads per '
               'invocation, plus publication and submission overhead. Whether the driver uses '
               'copy engines and whether full optimizer steps improve require candidate GPU '
               'traces and paired unprofiled trials.',
        aggregate=original_column.aggregate(rows), per_rank=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.trace_dir / f'rank{rank}-profile.json.gz' for rank in range(8)]
    original_column.require(args.out.resolve() not in {p.resolve() for p in paths},
                            'Output would overwrite an input trace')
    result = analyze(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: result[key] for key in
                     ('schema', 'candidate_gpu_measured', 'logical_bytes',
                      'observed_kernel_counts', 'aggregate')}, indent=2))


if __name__ == '__main__':
    main()
