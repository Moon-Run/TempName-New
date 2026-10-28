import subprocess
import tempfile
import unittest
from pathlib import Path
from build import arrival_header


class HeaderTests(unittest.TestCase):
    def test_device_and_host_tables_compile_together_and_agree(self):
        shape = dict(maps=[[(i+17*r)%256 for i in range(256)] for r in range(4)],
                     coords=[[(255-i+13*r)%256 for i in range(256)] for r in range(4)])
        unit = '#define __device__\n#define __constant__\n#define __forceinline__ inline\n'
        unit += arrival_header(shape, device=True)
        unit += arrival_header(shape)+arrival_header(shape, coord=True)
        unit += '''
int main() {
  for (int rank=0; rank<4; ++rank) for (int i=0; i<256; ++i) {
    if(flux_arrival_index(16,16,2048,4,rank,i)!=(i+17*rank)%256) return 1;
    if(flux_arrival_host_index(16,16,2048,4,rank,i)!=(i+17*rank)%256) return 2;
    if(flux_arrival_host_coord(16,16,2048,4,rank,i)!=(255-i+13*rank)%256) return 3;
    if(flux_arrival_index(16,16,512,4,rank,i)!=-1) return 4;
  }
  if(flux_arrival_index(16,16,2048,4,4,0)!=-1) return 5;
  if(flux_arrival_index(16,16,2048,4,0,256)!=-1) return 6;
  return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='mlp-header-test-') as tmp:
            source = Path(tmp)/'test.cc'
            binary = Path(tmp)/'test'
            source.write_text(unit)
            subprocess.run(['g++','-std=c++17',str(source),'-o',str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
