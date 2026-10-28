"""Keep the existing TP4 allocation alive after its experiment exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(root, repo, experiment):
    submission = json.loads((root/'submission.json').read_text())
    assert str(submission['job_id']) == os.environ['SLURM_JOB_ID']
    state = root/'allocation-state.txt'
    state.write_text('running_campaign\n')
    result = subprocess.run([sys.executable, str(experiment), str(root), str(repo)])
    (root/'campaign-exit-code.txt').write_text(str(result.returncode)+'\n')
    state.write_text('campaign_completed_holding\n' if result.returncode == 0 else 'campaign_failed_holding\n')
    # The submitted batch waits for this step, so its allocation stays alive.
    # Slurm still enforces the one-day limit. RELEASE explicitly ends the hold.
    while not (root/'RELEASE').exists():
        time.sleep(30)
    state.write_text('release_requested\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('repo', type=Path)
    args = parser.parse_args()
    run(args.run_root, args.repo, Path(__file__).with_name('campaign_experiment.py'))
