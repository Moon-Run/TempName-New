import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import resident_campaign


class ResidentTests(unittest.TestCase):
    def test_success_and_failure_both_hold_until_release(self):
        for code, expected in [(0, 'campaign_completed_holding'), (7, 'campaign_failed_holding')]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'submission.json').write_text(json.dumps(dict(job_id=179147)))
                experiment = root/'experiment.py'
                experiment.write_text(f'raise SystemExit({code})\n')

                def release(seconds):
                    self.assertEqual(seconds, 30)
                    self.assertEqual((root/'allocation-state.txt').read_text().strip(), expected)
                    (root/'RELEASE').touch()

                with patch.dict(os.environ, SLURM_JOB_ID='179147'), patch.object(resident_campaign.time, 'sleep', release):
                    resident_campaign.run(root, root, experiment)
                self.assertEqual((root/'campaign-exit-code.txt').read_text().strip(), str(code))
                self.assertEqual((root/'allocation-state.txt').read_text().strip(), 'release_requested')


if __name__ == '__main__':
    unittest.main()
