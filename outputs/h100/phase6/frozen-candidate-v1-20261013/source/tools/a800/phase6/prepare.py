"""Freeze six required phase6 comparisons, reusing audited phase5 measurement."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'phase5/e2e'))
from prepare import main

if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--arrival-plan', type=Path)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--graph', action='store_true')
    p.add_argument('--double-buffered', action='store_true')
    p.add_argument('--job-id', type=int, default=179147)
    p.add_argument('--target-percent',type=float,default=4.)
    a = p.parse_args()
    assert 0 < a.target_percent < 100
    repo = HERE.parents[2]
    policies = ['native', 'original', 'native_taco', 'taco_fused',
                'remote_arrival_selective', 'interleaved_arrival_selective']
    if a.pilot:
        policies = [x for x in policies if x not in ('native','native_taco')]
    main(repo, a.out, a.job_id, repo/'outputs/a800/phase5/joint-build-20261003',
         a.plan or repo/'logs/a800/phase5/builds-20261003/selection-plan.json',
         a.build, a.arrival_plan or repo/'outputs/a800/arrival-v2-tp4-20261003/artifacts/plan.json',
         native_quant=not a.pilot,
         fused_build=repo/'outputs/a800/phase4/taco-fused-warp-20261003',
         only_policies=policies, blocks=2 if a.pilot else None, graph_forward=a.graph,
         double_buffered=a.double_buffered,
         acceptance=dict(version='paired-full-step-v2-20261004',target_percent=a.target_percent))
