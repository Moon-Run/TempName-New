import unittest
import torch
from reference import hadamard, roundtrip, wire_bytes, encode, decode


class TacoReferenceTests(unittest.TestCase):
    def test_hadamard_is_orthonormal_and_self_inverse(self):
        x=torch.randn(8,128,generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(hadamard(hadamard(x)),x,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(hadamard(x).square().sum(-1),x.square().sum(-1))

    def test_zero_and_impulse(self):
        for n in (8,64,128,136,256):
            x=torch.zeros(2,n,dtype=torch.bfloat16)
            y,_=roundtrip(x)
            self.assertTrue(torch.equal(x,y))
            x[0,0]=1; x[1,-1]=-2
            y,_=roundtrip(x)
            torch.testing.assert_close(x,y,atol=.002,rtol=.01)

    def test_tail_coefficients_must_not_be_dropped(self):
        x=torch.zeros(1,136,dtype=torch.bfloat16);x[0,128]=1
        payload,qs,adaptive,_=encode(x)
        correct=decode(payload,qs,adaptive,136)
        corrupted=payload.float();corrupted[:,-1,8:]=0
        wrong=decode(corrupted.to(torch.float8_e4m3fn),qs,adaptive,136)
        self.assertGreater((correct.float()-wrong.float()).abs().max().item(),.5)

    def test_random_error_bound_across_scales(self):
        gen=torch.Generator().manual_seed(3)
        for scale in (1e-5,.1,1.,1e3):
            x=(torch.randn(7,136,generator=gen)*scale).to(torch.bfloat16)
            y,_=roundtrip(x)
            self.assertLess(((y.float()-x.float()).norm()/x.float().norm()).item(),.05)

    def test_nonfinite_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):roundtrip(torch.full((1,128),value))

    def test_wire_accounting_includes_both_scales_and_tail(self):
        for world in (2,4,8):
            d=wire_bytes(128*world,128,world)
            self.assertEqual(d['per_rank_remote']/d['per_rank_bf16_remote'],136/256)
            self.assertEqual(wire_bytes(128*world,136,world)['per_source'],128*2*136)
        with self.assertRaises(ValueError):wire_bytes(129,128,4)


if __name__=='__main__':unittest.main()
