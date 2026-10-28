"""CPU checks for all-rank timing aggregation and rejection of incomplete runs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('e2e_summary',Path(__file__).with_name('summarize.py'))
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
CONFIG=json.loads(Path(__file__).with_name('config.json').read_text())

class SummaryChecks(unittest.TestCase):
    def fixture(self,out):
        (out/'protocol.json').write_text(json.dumps(dict(stage='timing',config=CONFIG)))
        times=dict(native=.020,original=.010,remote_first=.009,interleaved=.012)
        window=0
        for block,order in enumerate(CONFIG['orders']):
            for policy in order:
                path=out/f'window-{window:02d}-{policy}';path.mkdir()
                (path/'exit-code.txt').write_text('0\n')
                for rank in range(4):
                    elapsed=times[policy]*(1+rank*.1)*CONFIG['timed_steps']
                    r=dict(completed=True,passed=True,policy=policy,mode='timing',rank=rank,sampling=False,
                      fallback_calls=0,target_modules=[str(i) for i in range(8)],replaced_modules=0 if policy=='native' else 8,
                      observed_shapes={str(i):dict(M=1024,N=1024,K_global=1024 if i%2==0 else 4096) for i in range(8)},
                      gpu_uuid=str(rank),tokens_sha256='tokens',initial_parameters_sha256=str(rank),initial_rng_sha256=str(rank),
                      window=window,block=block,timed_steps=CONFIG['timed_steps'],warmup_steps=CONFIG['warmup_steps'],
                      warmup_calls={str(i):CONFIG['warmup_steps']*4 for i in range(8)},profiling=False,elapsed_seconds=elapsed,
                      losses=[1.]*CONFIG['timed_steps'],grad_norms=[1.]*CONFIG['timed_steps'],skips=0)
                    (path/f'rank{rank}.json').write_text(json.dumps(r))
                window+=1

    def test_slowest_rank_and_single_counted_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            with contextlib.redirect_stdout(io.StringIO()):report=summary.analyze(out)
            rows={r['policy']:r for r in report['summary']}
            self.assertAlmostEqual(rows['native']['median_ms_per_step'],26)
            self.assertAlmostEqual(rows['native']['median_tokens_per_second'],4096/.026)
            self.assertAlmostEqual(rows['remote_first']['latency_reduction_vs_original_percent'],10)
            self.assertAlmostEqual(rows['interleaved']['latency_reduction_vs_original_percent'],-20)

    def test_missing_rank_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            (out/'window-00-native/rank3.json').unlink()
            with self.assertRaises(FileNotFoundError):summary.analyze(out)

    def test_changed_initial_state_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            path=out/'window-01-original/rank0.json'
            r=json.loads(path.read_text());r['initial_parameters_sha256']='different';path.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):summary.analyze(out)

if __name__=='__main__':unittest.main()
