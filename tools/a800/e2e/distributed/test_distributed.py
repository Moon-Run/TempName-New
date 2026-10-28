"""CPU regression tests for topology, sample accounting and distributed result integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from config import resolve, validate
from summarize import check_window, analyze

class DistributedTests(unittest.TestCase):
    def config(self, *args):
        return resolve(list(args))[1]

    def test_topology_and_microsteps(self):
        c = self.config()
        self.assertEqual((c['world_size'],c['dp'],c['microsteps'],c['tokens_per_step']), (8,2,2,8192))
        c = self.config('--tp','8','--flux-transport','hierarchical')
        self.assertEqual((c['tp_nodes'],c['local_tp'],c['dp'],c['microsteps']), (2,4,1,4))
        c = self.config('--nodes','4','--tp','8','--flux-transport','hierarchical')
        self.assertEqual((c['tp_nodes'],c['dp'],c['microsteps']), (2,2,2))
        c = self.config('--nodes','1','--tp','4','--micro-batch','2')
        self.assertEqual((c['microsteps'],c['tokens_per_step']), (2,8192))

    def test_invalid_plans(self):
        for args in [('--tp','8'), ('--tp','3'), ('--global-batch','3'),
                     ('--nodes','0'), ('--blocks','3'), ('--tp','2')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.config(*args)

    def test_json_override(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.json'
            p.write_text(json.dumps(dict(nodes=4,global_batch=8)))
            c=self.config('--config',str(p),'--nodes','2')
            self.assertEqual((c['nodes'],c['global_batch']), (2,8))

    def test_resolved_config_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'config.json'
            p.write_text(json.dumps(self.config()))
            c=self.config('--config',str(p),'--nodes','4','--global-batch','8')
            self.assertEqual((c['world_size'],c['dp'],c['microsteps']), (16,4,2))

    def fixture(self, out, c, policy='native', mode='timing', block=0):
        out.mkdir(parents=True,exist_ok=True)
        (out/'exit-code.txt').write_text('0')
        for n in range(c['nodes']):
            (out/f'node{n}-exit-code.txt').write_text('0')
        count=3 if mode=='smoke' else c['timed_steps']
        warmup=2 if mode=='smoke' else c['warmup_steps']
        for rank in range(c['world_size']):
            r=dict(rank=rank,world_size=c['world_size'],model=c,policy=policy,mode=mode,completed=True,passed=True,
                parameters_updated=True,skips=0,fallback_calls=0,local_rank=rank%c['gpus_per_node'],
                tp_rank=rank%c['tp'],dp_rank=rank//c['tp'],tokens_per_optimizer_step=c['tokens_per_step'],
                target_modules=list(range(c['layers']*2)),observed_shapes={str(k):dict(M=c['sequence']*c['micro_batch'],N=c['hidden'],K_global=k) for k in (c['hidden'],c['ffn'])},
                warmup_steps=warmup,timed_steps=count,losses=[1.]*count,grad_norms=[1.]*count,initial_loss=2.,
                warmup_calls={str(k):warmup*c['microsteps'] for k in range(c['layers']*2)},
                elapsed_seconds=(1+rank)*count/1000,gpu_uuid=str(rank),host=str(rank//c['gpus_per_node']),
                global_tokens_sha256='global',tokens_sha256=str(rank//c['tp']),
                initial_parameters_sha256=str(rank%c['tp']),final_parameters_sha256='final'+str(rank%c['tp']),
                initial_rng_sha256=str(rank%c['tp']),peak_allocated_gib=4.,block=block)
            (out/f'rank{rank}.json').write_text(json.dumps(r))

    def test_full_world_max_and_tokens(self):
        c=self.config('--policies','native')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            (out/'protocol.json').write_text(json.dumps(dict(config=c)))
            for b in range(c['blocks']):
                self.fixture(out/f'block-{b:03d}-native',c,block=b)
            result=analyze(out)['summary'][0]
            self.assertEqual(result['median_ms_per_step'],8.)
            self.assertEqual(result['median_tokens_per_second'],8192/.008)

    def test_missing_rank_and_dp_divergence(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            self.fixture(out,c)
            p=out/'rank7.json';r=json.loads(p.read_text());p.unlink()
            with self.assertRaises(FileNotFoundError):check_window(out,c,'native','timing')
            r['final_parameters_sha256']='diverged';p.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):check_window(out,c,'native','timing')

    def test_hierarchical_destination_math(self):
        # Emulate every node contributing a separate partial to each output rank.
        for nodes in (2,3,4):
            local=4
            for node in range(nodes):
                for lr in range(local):
                    rank=node*local+lr
                    sources=[node]
                    for d in range(1,nodes):
                        source=((rank-d*local)%(nodes*local))//local
                        self.assertEqual((source+d)%nodes,node)
                        sources.append(source)
                    self.assertEqual(sorted(sources),list(range(nodes)))

if __name__=='__main__':unittest.main()
