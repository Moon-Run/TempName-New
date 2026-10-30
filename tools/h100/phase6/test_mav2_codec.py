"""CPU checks of the independent MAV2 protocol and numerical edge cases."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import mav2_build as build
import mav2_codec as codec
import mav2_reference as reference


def packet(q, qs, tag):
    chunks = q.shape[0]
    return torch.cat((q.contiguous().view(torch.uint8).reshape(chunks,-1),
        qs.contiguous().view(torch.uint8).reshape(chunks,-1),
        tag.contiguous().view(torch.uint8).reshape(chunks,-1)), dim=1)


class CudaScalarDivisionOnCPU(TorchDispatchMode):
    """Model PyTorch 2.6 CUDA's CPU-scalar divisor optimization on a CPU test."""
    def __init__(self):
        super().__init__(); self.scalar_divisions = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (func in (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar) and
                len(args) >= 2 and isinstance(args[0], torch.Tensor) and
                args[0].dtype == torch.float32 and isinstance(args[1], (int,float))):
            self.scalar_divisions += 1
            reciprocal = torch.tensor(1.0/args[1], dtype=torch.float32, device=args[0].device)
            return args[0]*reciprocal
        return func(*args, **(kwargs or {}))


def midpoint_fixture(device='cpu'):
    # Exact BF16 values: x/scale = 368 is the E4M3 tie between 352 and 384.
    # RN-even picks 384; a one-ULP larger scale moves the value below the tie.
    x=torch.full((1,128),2.0563602447509766e-6,dtype=torch.bfloat16,device=device)
    x[:,0]=2.5033950805664062e-6
    return x


class ReferenceTests(unittest.TestCase):
    def assert_midpoint_protocol(self, device='cpu'):
        x=midpoint_fixture(device)
        _,_,q,s,y=reference._direct_quantize(x)
        expected_scale=torch.full_like(s,3.0*2.**-29)
        expected_q=torch.full(q.shape,384.0,dtype=torch.float32,device=device)
        expected_q[:,:,0]=448.0; expected_q=expected_q.to(torch.float8_e4m3fn)
        expected_y=torch.full_like(x,2.1457672119140625e-6); expected_y[:,0]=x[:,0]
        self.assertTrue(torch.equal(s.view(torch.int32),expected_scale.view(torch.int32)))
        self.assertTrue(torch.equal(q.view(torch.uint8),expected_q.view(torch.uint8)))
        self.assertTrue(torch.equal(y.reshape_as(x).view(torch.int16),expected_y.view(torch.int16)))

    def test_cuda_scalar_division_emulation_reproduces_old_midpoint_bug(self):
        x=midpoint_fixture(); g=x.float().reshape(1,1,128)
        with CudaScalarDivisionOnCPU() as emulated:
            old_scale=g.abs().amax(-1)/448
            old_q=(g/old_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        self.assertEqual(emulated.scalar_divisions,1)
        self.assertEqual(float(old_q[0,0,1]),352.0)
        self.assertGreater(float(old_scale),3.0*2.**-29)
        old_y=(old_q.float()*old_scale.unsqueeze(-1)).to(torch.bfloat16).reshape_as(x)
        expected=torch.full_like(x,2.1457672119140625e-6); expected[:,0]=x[:,0]
        self.assertGreater(float((old_y.double()-expected.double()).norm()/x.double().norm()),.005)

    def test_direct_reference_survives_cuda_scalar_division_behavior(self):
        with CudaScalarDivisionOnCPU() as emulated:
            self.assert_midpoint_protocol()
        # The old scalar-divisor reference fails both this and the exact payload check.
        self.assertEqual(emulated.scalar_divisions,0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_cuda_direct_reference_uses_rn_scale_at_fp8_midpoint(self):
        self.assert_midpoint_protocol('cuda')

    def test_zero_group_is_direct_and_has_positive_scale(self):
        x = torch.zeros(3,256,dtype=torch.bfloat16)
        q,s,y,d = reference.direct(x)
        self.assertTrue(d.all()); self.assertTrue((s > 0).all())
        self.assertTrue(torch.equal(x,y)); self.assertTrue((q.float() == 0).all())

    def test_exact_fp8_grid_roundtrip(self):
        x = torch.arange(-64,64,dtype=torch.float32).clamp(-16,16).to(torch.bfloat16).reshape(1,128)
        _,_,y,_ = reference.direct(x)
        # This grid includes both signs; no assumption that every integer is exactly representable.
        self.assertTrue(torch.isfinite(y).all())
        self.assertLess(float((y.double()-x.double()).norm()/x.double().norm()), .03)

    def test_subnormal_bf16_survives_direct_codec(self):
        x = torch.zeros(4,128,dtype=torch.bfloat16)
        x[:,::2] = 2.**-133; x[:,1::4] = -2.**-126
        q,s,y,d = reference.direct(x)
        self.assertTrue((s > 0).all()); self.assertTrue(d.all())
        self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(reference.decode_packets(packet(q,s,torch.full_like(s,-1)),1,128).reshape_as(x),x))

    def test_nonfinite_input_is_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            x=torch.zeros(1,128,dtype=torch.bfloat16); x[0,0]=value
            with self.assertRaises(ValueError): reference.roundtrip(x)
            with self.assertRaises(ValueError): reference.direct_roundtrip(x)

    def test_formal_d8_ignores_selective_rejection_and_never_calls_h8(self):
        x=torch.full((2,128),.51953125,dtype=torch.bfloat16); x[:,0]=1
        _,_,expected,selected=reference.direct(x)
        self.assertFalse(selected.any())
        with mock.patch.object(reference,'direct',side_effect=AssertionError('selection ran')), \
             mock.patch.object(reference,'h8',side_effect=AssertionError('H8 fallback ran')):
            actual,stats=reference.direct_roundtrip(x)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual((stats['d8'],stats['h8']),(2,0))

    def test_formal_policies_map_to_fixed_native_modes(self):
        self.assertIsNone(codec.encoder_mode('basic'))
        self.assertEqual(codec.encoder_mode('mav2'),-1)
        self.assertEqual(codec.encoder_mode('d8'),1)
        for policy in ('basic','d8'):
            for mode in (0,1):
                with self.assertRaises(ValueError): codec.encoder_mode(policy,mode)
                with self.assertRaises(ValueError): codec.pack_rows(object(),1,policy=policy,force_mode=mode)
        with self.assertRaises(ValueError): codec.encoder_mode('missing')
        with self.assertRaises(ValueError): codec.encoder_mode('mav2',True)

    def test_formal_d8_diagnostics_reject_h8_fallback(self):
        x=torch.ones(1,128,dtype=torch.bfloat16); q,s,_,_=reference.direct(x)
        op=codec.TensorTacoRS.__new__(codec.TensorTacoRS)
        op._diagnostic_enabled=True; op.policy='d8'; op.rows=1; op.n=128; op.calls=1
        op._diagnostic_packet=packet(q,s,torch.full_like(s,-1))
        evidence=op.transform_diagnostics()
        self.assertEqual((evidence['d8'],evidence['h8']),(1,0))
        self.assertFalse(evidence['quality_selection']); self.assertFalse(evidence['h8_fallback_allowed'])
        op._diagnostic_packet=packet(q,s,torch.ones_like(s))
        with self.assertRaises(RuntimeError): op.transform_diagnostics()

    def test_modes_are_group_local(self):
        x = torch.zeros(2,256,dtype=torch.bfloat16)
        x[1,128:] = .51953125; x[1,128] = 1
        q,s,y,d = reference.direct(x)
        self.assertEqual(d.tolist(),[[True,True],[True,False]])
        original=y[0].clone(); x[1] *= 32
        self.assertTrue(torch.equal(original,reference.direct(x)[2][0]))

    def test_h8_branch_matches_existing_tensor_reference(self):
        path=Path(__file__).resolve().parents[2]/'a800/phase4/reference.py'
        spec=importlib.util.spec_from_file_location('upstream_reference',path)
        existing=importlib.util.module_from_spec(spec); spec.loader.exec_module(existing)
        gen=torch.Generator().manual_seed(81)
        x=torch.randn((5,256),generator=gen,dtype=torch.bfloat16)
        expected,_=existing.roundtrip(x)
        self.assertTrue(torch.equal(reference.h8(x),expected))

    def test_mixed_roundtrip_changes_only_admitted_groups(self):
        x=torch.zeros(4,128,dtype=torch.bfloat16)
        x[2:]=.51953125; x[2:,0]=1
        y,stats=reference.roundtrip(x)
        self.assertEqual((stats['d8'],stats['h8']),(2,2))
        self.assertTrue(torch.equal(y[:2],reference.direct(x)[2][:2]))
        self.assertTrue(torch.equal(y[2:],reference.h8(x)[2:]))

    def test_packet_modes_and_wire_size(self):
        x=torch.ones(2,128,dtype=torch.bfloat16)
        q,s,expected,_=reference.direct(x)
        p=packet(q,s,torch.full_like(s,-1))
        self.assertEqual(p.numel(),2*136)
        decoded=reference.decode_packets(p,1,128).reshape_as(x)
        self.assertTrue(torch.equal(decoded,expected))
        d=codec.packet_diagnostics(p,128)
        self.assertEqual((d['total'],d['d8'],d['h8']),(2,2,0))

    def test_illegal_tags_and_scales_fail(self):
        x=torch.ones(1,128,dtype=torch.bfloat16); q,s,_,_=reference.direct(x)
        for tag in (0.,-0.,-2.,float('nan'),float('inf')):
            p=packet(q,s,torch.full_like(s,tag))
            with self.assertRaises(ValueError): reference.decode_packets(p,1,128)
            with self.assertRaises(RuntimeError): codec.packet_diagnostics(p,128)
        for scale in (0.,-1.,float('nan')):
            p=packet(q,torch.full_like(s,scale),torch.full_like(s,-1))
            with self.assertRaises(ValueError): reference.decode_packets(p,1,128)

    def test_selector_contract_is_frozen_across_components(self):
        self.assertEqual(reference.RELATIVE_L2,codec.D8_REL_L2)
        self.assertEqual(reference.MAX_ABS_OVER_AMAX,codec.D8_MAX_ABS)
        self.assertEqual(build.SELECTOR['relative_l2'],codec.D8_REL_L2)
        self.assertEqual(build.SELECTOR['packet_bytes_per_group'],136)
        self.assertEqual(build.SELECTOR['quantized_remote_fraction'],1.)


if __name__ == '__main__': unittest.main()
