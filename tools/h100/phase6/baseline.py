"""Five-policy 50-step ablation; default run is CPU-only command preview."""
import argparse
from pathlib import Path
import sys

import ablation6_build as builder
import baseline_codec_build as codec_builder
import baseline_prepare as preparation
import baseline_protocol as routing
import baseline_report as reporter
import ablation4_runtime as communication
import scope_run
from calibrate import environment
from common import REPO, read


def jobs(info, stage):
    return scope_run.jobs(info,stage,routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root,info,routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    record=read(Path(root)/'submission.json')['communication']
    return scope_run.run(root,stage,execute,python,cuda_home,routing,reporter,'H100_BASELINE_EXECUTE',
        environment_fn=lambda *args:communication.apply(environment(*args),record))


def main():
    p=argparse.ArgumentParser(__doc__); sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('build-prepare'); q.add_argument('out',type=Path);q.add_argument('--source-only',action='store_true')
    q=sub.add_parser('compile');q.add_argument('root',type=Path);q.add_argument('--cuda-home',type=Path,required=True)
    q.add_argument('--jobs',type=int,default=2);q.add_argument('--nccl-root',type=Path)
    q=sub.add_parser('codec-prepare');q.add_argument('out',type=Path);q.add_argument('--source-only',action='store_true')
    q=sub.add_parser('codec-compile');q.add_argument('root',type=Path);q.add_argument('--cuda-home',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('out',type=Path);q.add_argument('--build',type=Path,required=True)
    q.add_argument('--flux-build',type=Path,required=True);q.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    q.add_argument('--codec-build',type=Path,required=True)
    q.add_argument('--blocks',type=int,default=5);q.add_argument('--rounds',type=int,choices=(1,2),default=2)
    q.add_argument('--nvls-enable',choices=('0','auto'),default='0');q.add_argument('--source-only',action='store_true')
    q=sub.add_parser('run');q.add_argument('root',type=Path);q.add_argument('--stage',choices=('preflight','timing','all'),default='all')
    q.add_argument('--python',type=Path,default=Path(sys.executable));q.add_argument('--cuda-home',type=Path);q.add_argument('--execute',action='store_true')
    q=sub.add_parser('report');q.add_argument('root',type=Path);q.add_argument('--preflight',action='store_true');q.add_argument('--offline',action='store_true')
    a=p.parse_args()
    if a.action=='build-prepare':print(builder.prepare(a.out,a.source_only))
    elif a.action=='compile':builder.compile_build(a.root,a.cuda_home,a.jobs,a.nccl_root)
    elif a.action=='codec-prepare':print(codec_builder.prepare(a.out,a.source_only))
    elif a.action=='codec-compile':codec_builder.compile_build(a.root,a.cuda_home)
    elif a.action=='prepare':print(preparation.prepare(a.out,a.build,a.flux_build,a.megatron,a.blocks,a.rounds,a.nvls_enable,a.source_only,a.codec_build))
    elif a.action=='run':run(a.root,a.stage,a.execute,a.python,a.cuda_home)
    else:(reporter.preflight if a.preflight else reporter.report)(a.root,a.offline)


if __name__=='__main__':main()
