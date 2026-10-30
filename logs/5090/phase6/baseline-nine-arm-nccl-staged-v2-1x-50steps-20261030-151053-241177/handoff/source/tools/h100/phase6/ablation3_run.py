"""Nine-policy binding of the shared dry-run/explicit-execution controller."""
from pathlib import Path
import sys
import scope_run
import ablation3_protocol as routing
import ablation3_report as reporter


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    return scope_run.run(root, stage, execute, python, cuda_home, routing, reporter, 'H100_ABLATION3_EXECUTE')
