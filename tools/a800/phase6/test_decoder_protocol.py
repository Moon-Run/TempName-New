"""A formal run must launch exactly the kernels implied by its frozen mask."""
import importlib.util
from pathlib import Path
import sys
import unittest

COMMON = Path(__file__).resolve().parents[1]/'phase5/e2e'
sys.path.insert(0,str(COMMON))
spec = importlib.util.spec_from_file_location('decoder_summary',COMMON/'summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class DecoderProtocolTest(unittest.TestCase):
    def setUp(self):
        self.model = dict(sequence=1024,micro_batch=8,hidden=2048,ffn=8192,tp=4)
        self.mask = [[0]*1024 for _ in range(4)]
        self.plan = dict(selective_decoder='compact-v1')
        self.selection = dict(schema=2,policies={'remote_first':[dict(shape=[8192,2048,8192],mask=self.mask)]})

    def counts(self,policy='remote_arrival_selective',rank=1):
        return summary.expected_decoder_counts(self.plan,self.selection,self.model,policy,rank,24)

    def test_zero_and_sparse_destinations(self):
        self.assertEqual(self.counts(),dict(legacy=0,bf16=12,compact=0))
        self.mask[2][260]=1
        self.assertEqual(self.counts(),dict(legacy=0,bf16=12,compact=12))
        self.assertEqual(self.counts(rank=0),dict(legacy=0,bf16=12,compact=0))
        self.mask[3][260]=1  # Two quantized sources still identify one tile.
        self.assertEqual(self.counts(),dict(legacy=0,bf16=12,compact=12))

    def test_dense_and_unmeasured_shapes_keep_legacy(self):
        self.mask[2][256:288]=[1]*32
        self.assertEqual(self.counts(),dict(legacy=0,bf16=12,compact=12))
        self.mask[2][256:289]=[1]*33
        self.assertEqual(self.counts(),dict(legacy=12,bf16=0,compact=0))
        self.model['micro_batch']=2
        self.assertEqual(self.counts(),dict(legacy=12,bf16=0,compact=0))

    def test_baselines_and_legacy_contract(self):
        self.assertEqual(self.counts('taco_fused'),dict(legacy=12,bf16=0,compact=0))
        self.assertEqual(self.counts('original'),dict(legacy=0,bf16=0,compact=0))
        self.assertEqual(self.counts('native_taco'),dict(legacy=0,bf16=0,compact=0))
        self.plan.clear()
        self.assertEqual(self.counts(),dict(legacy=12,bf16=0,compact=0))


if __name__ == '__main__':
    unittest.main()
