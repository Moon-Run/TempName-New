"""Permutation, exact-A preservation, and executable host mapping checks."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import arrival_pair_order as pair
import scope_build
from ablation_plan import FREEZE


def baseline_sources():
    entries = {}
    for role, k in (('attn', 512), ('mlp', 2048)):
        entries[role] = dict(pair.role_mapping(32), M=2048, N=4096, K_local=k, tiles=512)
    swizzle = scope_build.swizzle((FREEZE / 'source' / pair.SWIZZLE).read_text(), entries)
    return swizzle, '#pragma once\n' + scope_build.tables(entries)


class MappingTests(unittest.TestCase):
    def test_all_spans_preserve_destination_windows_and_cover_every_physical_tile(self):
        for span in pair.SPANS:
            result = pair.role_mapping(span)
            order = pair.indices(span)
            with self.subTest(span=span):
                self.assertEqual(sorted(order), list(range(512)))
                self.assertTrue(all(new % 8 == old % 8 for new, old in enumerate(order)))
                for rank, coords in enumerate(result['coords']):
                    self.assertEqual(sorted(coords), list(range(512)))
                    self.assertTrue(all(coord // 64 == (slot % 8 + rank) % 8
                                        for slot, coord in enumerate(coords)))
                    # Physical full-remote mask remains selected for exactly 448
                    # tiles, including after this new production order.
                    self.assertEqual(sum(coord // 64 != rank for coord in coords), 448)
                    self.assertEqual(coords, [result['base_coords'][rank][i] for i in order])

    def test_independent_small_example_and_identity(self):
        self.assertEqual([i // 8 for i in pair.indices(4)[::8]],
                         [block * 4 + row * 32 + col
                          for block in range(8) for row in range(2) for col in range(4)])
        self.assertEqual(pair.indices(32), list(range(512)))
        self.assertFalse(pair.diagnostics(4)['identity_fallback'])
        self.assertTrue(pair.diagnostics(32)['identity_fallback'])

    def test_pair_distance_and_no_changes_to_logical_destination(self):
        for span in pair.SPANS:
            order = pair.indices(span)
            inverse = {old: new for new, old in enumerate(order)}
            self.assertEqual({inverse[old + 256] - inverse[old] for old in range(256)},
                             {8 * span})
            self.assertEqual(pair.diagnostics(span)['b_row_pair_slot_gap'], 8 * span)

    def test_role_spans_are_independent_and_return_values_do_not_alias(self):
        result = pair.mapping({'attn': 1, 'mlp': 16})
        self.assertNotEqual(result['attn']['maps'], result['mlp']['maps'])
        before = copy.deepcopy(result)
        result['attn']['maps'][0][0] = -1
        self.assertEqual(result['attn']['maps'][1], before['attn']['maps'][1])
        self.assertEqual(result['mlp'], before['mlp'])

    def test_invalid_spans_and_missing_roles_rejected(self):
        for span in (True, False, None, '4', 4.0, 0, 2, 3, 64):
            with self.subTest(span=span), self.assertRaises(ValueError):
                pair.indices(span)
        for spans in ({}, {'attn': 4}, {'attn': 4, 'mlp': 8, 'extra': 1}, [4, 8]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                pair.mapping(spans)


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.swizzle, cls.host = baseline_sources()
        cls.changed = pair.sources(cls.swizzle, cls.host)

    def test_exactly_two_files_with_unchanged_original_a_tables(self):
        self.assertEqual(set(self.changed), set(pair.ALLOWED_SOURCE_CHANGES))
        for role in pair.ROLES:
            for key, prefix, original in ((pair.SWIZZLE, 'phase6_device_', self.swizzle),
                                           (pair.HOST, 'phase6_host_', self.host)):
                name = prefix + role + '_data'
                self.assertEqual(pair._array(self.changed[key], name), pair._array(original, name))
        self.assertIn('phase6_device_lookup(16,32,scope_k,local_rank,tile_idx)',
                      self.changed[pair.SWIZZLE])
        self.assertIn('scope_pair_span = phase6_op2_pair_span(scope_k)', self.changed[pair.SWIZZLE])
        self.assertNotIn('taco_', ''.join(self.changed.values()))

    def test_nonbaseline_sources_and_duplicate_transformation_rejected(self):
        with self.assertRaises(ValueError):
            pair.sources(self.swizzle.replace('data[] = {0,1,', 'data[] = {1,0,', 1), self.host)
        with self.assertRaises(ValueError):
            pair.sources(self.changed[pair.SWIZZLE], self.changed[pair.HOST])
        with self.assertRaises(ValueError):
            pair.sources(self.swizzle.replace('  bool scope_shape = false;', ''), self.host)

    @unittest.skipUnless(shutil.which('g++'), 'g++ is needed for the host mapping check')
    def test_compiled_host_mapping_parser_role_dispatch_and_invalid_shapes(self):
        code = r'''
#include "phase6_host.hpp"
int main() {
  unsetenv("H100_OP2_ATTN_SPAN"); unsetenv("H100_OP2_MLP_SPAN");
  if(phase6_host_lookup(8,32,512,0,0)!=-1 ||
     phase6_host_lookup(16,16,512,0,0)!=-1 ||
     phase6_host_lookup(16,32,1024,0,0)!=-1 ||
     phase6_host_lookup(16,32,512,-1,0)!=-1 ||
     phase6_host_lookup(16,32,512,8,0)!=-1 ||
     phase6_host_lookup(16,32,512,0,-1)!=-1 ||
     phase6_host_lookup(16,32,512,0,512)!=-1) return 1;
  bool rejected=false;
  try { phase6_host_lookup(16,32,512,0,0); }
  catch(const std::invalid_argument &) { rejected=true; }
  if(!rejected) return 2;
  const char* bad[]={"","2","0","-1","4x"," 4","4 ","04","4.0","999999999999999999"};
  for(const char* value:bad) {
    setenv("H100_OP2_ATTN_SPAN",value,1); rejected=false;
    try { phase6_host_lookup(16,32,512,0,0); }
    catch(const std::invalid_argument &) { rejected=true; }
    if(!rejected) return 3;
  }
  const int spans[]={1,4,8,16,32};
  const char* strings[]={"1","4","8","16","32"};
  for(int a=0;a<5;++a)for(int b=0;b<5;++b) {
    setenv("H100_OP2_ATTN_SPAN",strings[a],1);
    setenv("H100_OP2_MLP_SPAN",strings[b],1);
    for(int role=0;role<2;++role) {
      const int span=spans[role?b:a],k=role?2048:512;
      for(int rank=0;rank<8;++rank) {
        int seen[512]={};
        int slot=0;
        // Independent nested-loop enumeration of a pair of M-row spans.
        for(int column=0;column<32;column+=span)
          for(int row=0;row<2;++row)
            for(int n=column;n<column+span;++n)
              for(int d=0;d<8;++d,++slot) {
                const int expected=((d+rank)%8)*64+row*32+n;
                const int got=phase6_host_lookup(16,32,k,rank,slot);
                if(got!=expected || ++seen[got]!=1) return 4;
              }
      }
    }
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='op2-pair-host-') as temporary:
            root = Path(temporary)
            (root / 'phase6_host.hpp').write_text(self.changed[pair.HOST])
            (root / 'test.cc').write_text(code)
            subprocess.run(['g++', '-std=c++17', '-O2', str(root / 'test.cc'),
                            '-o', str(root / 'check')], check=True, capture_output=True, text=True)
            subprocess.run([str(root / 'check')], check=True, capture_output=True, text=True,
                           env=dict(os.environ))


if __name__ == '__main__':
    unittest.main()
