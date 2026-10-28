"""Extract diagnostic kernel sums; these are not full-step critical-path times."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def extract(root):
    plan = json.loads((root/'scripts/config.json').read_text())
    rows = []
    for policy in plan['policies']:
        for rank in range(4):
            path = root/f'results/preflight/profile-{policy}/rank{rank}-profile.json.gz'
            with gzip.open(path, 'rt') as stream:
                kernels = [e for e in json.load(stream)['traceEvents'] if e.get('cat') == 'kernel']
            row = dict(policy=policy, rank=rank)
            for kind, marker in (('gemm', 'flux_bf16'), ('encode', 'taco_encode_scatter_kernel'),
                                 ('decode', 'taco_decode_ring_kernel')):
                selected = [e for e in kernels if marker in e['name']]
                row[kind+'_calls'] = len(selected)
                row[kind+'_sum_us'] = sum(e['dur'] for e in selected)
                row[kind+'_registers'] = ';'.join(map(str, sorted({e['args']['registers per thread'] for e in selected})))
                row[kind+'_shared_bytes'] = ';'.join(map(str, sorted({e['args']['shared memory'] for e in selected})))
            rows.append(row)
    path = root/'profile-components.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    extract(parser.parse_args().root)
