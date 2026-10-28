"""CPU regression checks: TP8 includes every rank and counts tokens only once."""
import contextlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('scale_summary',Path(__file__).with_name('summarize.py'))
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
PLAN=json.loads(Path(__file__).with_name('config.json').read_text())
class ScaleSummaryChecks(unittest.TestCase):
    def fixture(self,out):
        case=PLAN['cases'][-1];model=dict(PLAN['common'],**case);tp=case['tp']
        assert tp==8 and model['layers']==12
        (out/'protocol.json').write_text(json.dumps(dict(plan=PLAN,case=case,model=model,stage='timing')))
        times=dict(native=.032,original=.024,remote_first=.020,interleaved=.028)
        window=0
        for block,order in enumerate(PLAN['orders']):
            for policy in order:
                path=out/f'window-{window:02d}-{policy}';path.mkdir();(path/'exit-code.txt').write_text('0\n')
                for rank in range(tp):
                    r=dict(case=case['name'],model=model,world_size=tp,rank=rank,policy=policy,mode='timing',
                        completed=True,passed=True,sampling=False,profiling=False,skips=0,fallback_calls=0,
                        target_modules=[str(i) for i in range(24)],observed_shapes={str(i):dict(M=case['sequence'],N=case['hidden'],K_global=k) for i,k in enumerate([case['hidden'],case['ffn']])},
                        tokens_per_optimizer_step=case['sequence']*4,losses=[1.]*20,grad_norms=[1.]*20,initial_loss=9.,
                        gpu_uuid=str(rank),initial_parameters_sha256=str(rank),initial_rng_sha256=str(rank),tokens_sha256='tokens',
                        window=window,block=block,warmup_steps=10,timed_steps=20,warmup_calls={str(i):40 for i in range(24)},
                        elapsed_seconds=times[policy]*(1+rank*.01)*20,peak_allocated_gib=4.,peak_reserved_gib=5.)
                    (path/f'rank{rank}.json').write_text(json.dumps(r))
                window+=1
    def test_eight_rank_max_and_token_count(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out)
            with contextlib.redirect_stdout(io.StringIO()):result=summary.analyze(out)
            rows={r['policy']:r for r in result['summary']}
            self.assertAlmostEqual(rows['native']['median_ms_per_step'],34.24)
            self.assertAlmostEqual(rows['native']['median_tokens_per_second'],16384/.03424)
            self.assertAlmostEqual(rows['remote_first']['latency_reduction_vs_original_percent'],100*(1-.020/.024))
    def test_missing_eighth_rank_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out);(out/'window-00-native/rank7.json').unlink()
            with self.assertRaises(FileNotFoundError):summary.analyze(out)
    def test_nan_on_last_rank_cannot_hide_behind_max(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);self.fixture(out);path=out/'window-00-native/rank7.json'
            row=json.loads(path.read_text());row['elapsed_seconds']=float('nan');path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):summary.analyze(out)
if __name__=='__main__':unittest.main()
