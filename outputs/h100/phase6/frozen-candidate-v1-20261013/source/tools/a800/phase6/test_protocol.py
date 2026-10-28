import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('routing',Path(__file__).resolve().parents[1]/'phase5/e2e/routing.py')
routing=importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


class ProtocolTest(unittest.TestCase):
    def test_required_fused_is_original_order(self):
        self.assertEqual(routing.codec_placement('taco_fused'),'fused')
        self.assertEqual(routing.mapping_policy('taco_fused'),'original')
        self.assertNotEqual(routing.mapping_policy('remote_all'),'original')
        self.assertNotEqual(routing.mapping_policy('interleaved_all'),'original')

    def test_six_policy_attention_and_physical_masks(self):
        observed={'layer.self_attention.linear_proj':dict(M=2048,N=2048,K_global=2048,K_local=512),
                  'layer.mlp.linear_fc2':dict(M=2048,N=2048,K_global=8192,K_local=2048)}
        for policy in ('native','original','native_taco','taco_fused','remote_arrival_selective','interleaved_arrival_selective'):
            routes=routing.module_routes(observed,policy,layers=1)
            self.assertEqual(routes['layer.self_attention.linear_proj'],
                             'native' if policy in ('native','native_taco') else 'original')
            self.assertEqual(routes['layer.mlp.linear_fc2'],policy)
        self.assertEqual(routing.base_policy('remote_arrival_selective'),'remote_first')
        self.assertEqual(routing.base_policy('interleaved_arrival_selective'),'interleaved')

    def test_batched_compact_model_scope_and_remote_leading_names(self):
        model=dict(hidden=512,ffn=2048,sequence=2048,micro_batch=4,tp=4)
        observed={'layer.self_attention.linear_proj':dict(M=8192,N=512,K_global=512,K_local=128),
                  'layer.mlp.linear_fc2':dict(M=8192,N=512,K_global=2048,K_local=512)}
        for base in ('interleaved_remote','interleaved_remote_group'):
            policy=base+'_arrival_selective'
            self.assertEqual(routing.base_policy(policy),base)
            self.assertEqual(routing.mapping_policy(policy),base+'_arrival')
            routes=routing.module_routes(observed,policy,layers=1,model=model)
            self.assertEqual(routes['layer.self_attention.linear_proj'],'original')
            self.assertEqual(routes['layer.mlp.linear_fc2'],policy)
        observed['layer.mlp.linear_fc2']['M']=2048
        with self.assertRaises(AssertionError):routing.module_routes(observed,'remote_arrival_selective',1,model)

    def test_logical_bytes_include_metadata_and_tail(self):
        wire,bf16=routing.logical_bytes(None,0,8192,512,4)
        self.assertEqual(bf16,6*1024**2)
        self.assertEqual(wire,bf16*136//256)
        mask=[[0]*8 for _ in range(4)]
        mask[0][3]=1  # Remote N-tail: 8 valid BF16 values but 128 FP8 coefficients.
        wire,bf16=routing.logical_bytes(mask,0,512,136,4)
        self.assertEqual(wire-bf16,128*(136-16))


if __name__=='__main__':unittest.main()
