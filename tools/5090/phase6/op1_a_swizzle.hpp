// RTX5090 OP1: only the CTA-to-output-tile mapping changes.
// The rank is captured on the host in CUTLASS kernel parameters; no mutable
// device-global rank, tensor permutation or communication-order change is used.
#pragma once

#include <cstdlib>
#include <stdexcept>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"

namespace cutlass::gemm::threadblock {

struct Op1AInterleavedSwizzle : public ThreadblockSwizzleStreamK {
  using Base = ThreadblockSwizzleStreamK;
  int op1_rank = -1;
  bool op1_target = false;

  CUTLASS_HOST_DEVICE
  Op1AInterleavedSwizzle() {}

  static int rank_value(const char* name) {
    const char* text = std::getenv(name);
    if (!text || !*text) throw std::runtime_error("OP1 swizzle requires torchrun rank environment");
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*end || value < 0 || value > 8)
      throw std::runtime_error("Invalid OP1 swizzle torchrun rank environment");
    return static_cast<int>(value);
  }

  Op1AInterleavedSwizzle(
      GemmUniversalMode mode, GemmCoord problem, GemmCoord tile,
      int batch_split, int occupancy, int device_sms, int avail_sms,
      size_t a_bytes, size_t b_bytes, size_t c_bytes, int accumulator_fragments)
      : Base(mode, problem, tile, batch_split, occupancy, device_sms, avail_sms,
             a_bytes, b_bytes, c_bytes, accumulator_fragments) {
    op1_target = mode == GemmUniversalMode::kGemm && batch_split == 1 &&
                 problem.m() == 2048 && problem.n() == 4096 &&
                 (problem.k() == 512 || problem.k() == 2048) &&
                 tile.m() == 128 && tile.n() == 128 && tile.k() == 32;
    if (op1_target) {
      // Keep the baseline's StreamK schedule. Changing tile assignment can
      // change which output coordinates receive split K reductions; this is
      // explicitly allowed by OP1 and is not claimed to be bitwise equivalent.
      op1_rank = rank_value("LOCAL_RANK");
      if (rank_value("LOCAL_WORLD_SIZE") != 8 || op1_rank >= 8)
        throw std::runtime_error("OP1 interleaved projection requires eight local ranks");
    }
  }

  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int tile_index) const {
    // Keep the parent's padding and non-target behavior. Both target shapes
    // have exactly 16 x 32 tiles, hence no partial tile or extra tile mapping.
    if (!op1_target || tile_index < 0 || tile_index >= 512)
      return Base::get_tile_offset(tile_index);
    const int destination = (tile_index % 8 + op1_rank) % 8;
    const int within = tile_index / 8;
    return GemmCoord(destination * 2 + within / 32, within % 32, get_batch_idx());
  }
};

}  // namespace cutlass::gemm::threadblock
