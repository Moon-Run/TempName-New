// Host-side audit of the exact CUTLASS epilogue fragment layout used by TACO.
#include <cstdio>
#include <set>
#include <map>
// Uninstantiated permutation helpers in this CUTLASS revision refer to
// blockIdx even under the host compiler. The partition audited below takes
// its tile coordinate explicitly and never uses this parsing-only stub.
#ifndef __CUDACC__
static const struct { int x = 0, y = 0, z = 0; } blockIdx;
#endif
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"

int main() {
  using namespace cute;
  using Map = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      cutlass::gemm::GemmShape<128,128,32>, cutlass::gemm::GemmShape<64,64,32>,
      cutlass::bfloat16_t, 8, 1>;
  static_assert(Map::Base::kThreads == 128);
  std::set<std::pair<int,int>> total;
  for (int step = 0; step < 8; ++step) {
    std::map<int, std::set<int>> rows;
    std::set<std::pair<int,int>> fragment;
    for (int tid = 0; tid < 128; ++tid) {
      auto shape = make_shape(128,128,1);
      auto identity = make_identity_tensor(shape);
      auto coords = outer_partition(group_modes<3,6>(Map::partition(identity, tid, {0,0,0})),
                                    Shape<Int<8>>{}, (_0{}));
      if (size<3>(coords) != 8) return 1;
      auto v = filter(coords(_,_,_,step));
      for (int i = 0; i < size(v); ++i) {
        int row = get<0>(v(i)), col = get<1>(v(i));
        for (int j = 0; j < 8; ++j) {
          if (row < 0 || row >= 128 || col+j >= 128) return 2;
          if (!fragment.insert({row,col+j}).second || !total.insert({row,col+j}).second) return 3;
          rows[row].insert(col+j);
        }
      }
    }
    if (rows.size() != 16 || fragment.size() != 2048) return 4;
    std::set<int> compact_rows;
    for (auto& row : rows) {
      if (row.second.size() != 128) return 5;
      int compact = row.first % 8 + (row.first / 64) * 8;
      if (!compact_rows.insert(compact).second) return 7;
      if (row.first != step * 8 + compact % 8 + (compact / 8) * 64) return 8;
    }
    if (compact_rows.size() != 16 || *compact_rows.begin() != 0 || *compact_rows.rbegin() != 15) return 9;
  }
  if (total.size() != 128*128) return 6;
  puts("PASS: 8 disjoint fragments; compact 16-row staging and inverse coordinates verified");
}
