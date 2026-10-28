"""Reject unsupported layouts before allocating IPC or entering collectives."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch  # Load once, before the temporary sys.modules package fixture.

ROOT=Path(__file__).resolve().parents[3]


class ApiGuardTests(unittest.TestCase):
    def load(self):
        package=types.ModuleType('test_flux');package.__path__=[]
        package.cpp_mod=types.ModuleType('test_flux.cpp_mod')
        spec=importlib.util.spec_from_file_location('test_flux.gemm_rs_taco',ROOT/'python/flux/gemm_rs_taco.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'test_flux':package,'test_flux.cpp_mod':package.cpp_mod}):
            spec.loader.exec_module(module)
        return module

    def test_invalid_shape_rejected_before_cuda_or_ipc(self):
        mod=self.load()
        with patch.object(mod.dist,'get_rank',return_value=0), patch.object(mod.dist,'get_world_size',return_value=4):
            with self.assertRaisesRegex(ValueError,'M%'):
                mod.GemmRSTaco(None,513,128,256)

    def test_invalid_placement_rejected_before_cuda_or_ipc(self):
        mod=self.load()
        with self.assertRaisesRegex(ValueError,'placement'):
            mod.GemmRSTaco(None,512,128,256,placement='after_completed_rs')

    def test_subgroup_rank_cannot_reuse_global_swizzle(self):
        mod=self.load()
        with patch.object(mod.dist,'get_rank',return_value=0), patch.object(mod.dist,'get_world_size',return_value=4), \
                patch.object(mod.torch.cuda,'get_device_capability',return_value=(8,0)), \
                patch.dict(os.environ,LOCAL_RANK='2',LOCAL_WORLD_SIZE='4'):
            with self.assertRaisesRegex(ValueError,'subgroup'):
                mod.GemmRSTaco(None,512,128,256)


if __name__=='__main__':unittest.main()
