import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).parent/'e2e'))
sys.path.insert(0,str(Path(__file__).parent.parent/'phase4'))
from megatron_taco import pack_rows,unpack_rows
from reference import roundtrip


class TensorPacketTests(unittest.TestCase):
    def test_packet_roundtrip_matches_reference_with_padding(self):
        gen=torch.Generator().manual_seed(34)
        for n in (128,136,256):
            x=torch.randn(3*8,n,generator=gen).to(torch.bfloat16)
            packets=pack_rows(x,3)
            self.assertEqual(packets.dtype,torch.uint8)
            self.assertEqual(tuple(packets.shape),(3,8*((n+127)//128)*136))
            actual=unpack_rows(packets,8,n).reshape_as(x)
            expected,_=roundtrip(x)
            self.assertTrue(torch.equal(actual,expected))

    def test_source_packets_remain_separate(self):
        x=torch.zeros(3*8,136,dtype=torch.bfloat16)
        x[0,0]=1;x[8,-1]=-2;x[16,64]=3
        packets=pack_rows(x,3)
        decoded=unpack_rows(packets.flip(0),8,136)
        expected,_=roundtrip(x)
        self.assertTrue(torch.equal(decoded,expected.reshape(3,8,136).flip(0)))


if __name__=='__main__':unittest.main()
