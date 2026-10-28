"""Use the audited scope runner with explicit Phase7 routing and evidence gates."""
from pathlib import Path
import sys

import fused_support
import fused_protocol as routing
import fused_report as reporter
import scope_run


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    return scope_run.run(root, stage, execute, python, cuda_home,
                         routing, reporter, 'H100_PHASE7_FUSED_EXECUTE')
