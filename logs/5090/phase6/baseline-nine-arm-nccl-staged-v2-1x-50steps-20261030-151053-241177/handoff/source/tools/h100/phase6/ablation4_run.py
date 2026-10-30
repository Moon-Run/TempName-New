"""Explicit execution using a frozen communication configuration for every group."""
from pathlib import Path
import sys

import ablation4_protocol as routing
import ablation4_report as reporter
import ablation4_runtime as communication
import scope_run
from calibrate import environment
from common import read


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    record = read(Path(root)/'submission.json')['communication']
    def env(*args):
        return communication.apply(environment(*args), record)
    return scope_run.run(root, stage, execute, python, cuda_home, routing, reporter,
                         'H100_ABLATION4_EXECUTE', environment_fn=env)
