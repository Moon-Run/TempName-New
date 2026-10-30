// Shared CPU/CUDA wire contract. It performs no quantization or reduction.
#pragma once
#include <cstdint>
#include <stdexcept>

namespace rtx5090_local_layout {
constexpr int kWorld = 8, kM = 2048, kN = 4096, kRows = 256;
constexpr int kLocalTiles = 64, kTiles = 512, kGroups = 8192;
constexpr int64_t kChunk = int64_t(kRows) * kN;
constexpr int64_t kPacket = int64_t(kGroups) * 136;
constexpr int64_t kBF16Tile = 128 * 128 * 2;
constexpr int64_t kQuantTile = 128 * 128 + 128 * 4 * 2;

inline void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

inline void validate(int rank, const unsigned char* mask,
    const int64_t* tx, const int64_t* rx, int64_t tx_bytes, int64_t rx_bytes) {
  require(rank >= 0 && rank < kWorld && tx && rx, "Invalid local wire identity");
  if (mask) {
    for (int source = 0; source < kWorld; ++source)
      for (int tile = 0; tile < kTiles; ++tile) {
        const int value = mask[source * kTiles + tile];
        require(value == 0 || value == 1, "Non-binary TACO selection mask");
        require(tile / kLocalTiles != source || value == 0, "Local contributions cannot be quantized");
      }
  }
  int64_t tx_cursor = 0, rx_cursor = 0;
  for (int peer = 0; peer < kWorld; ++peer) {
    for (int tile = 0; tile < kLocalTiles; ++tile) {
      const int index = peer * kLocalTiles + tile;
      if (peer == rank) {
        require(tx[index] == -1 && rx[index] == -1, "Self transmission must be excluded");
        continue;
      }
      require(tx[index] == tx_cursor && rx[index] == rx_cursor, "Compact wire tile offsets changed");
      const bool tx_quant = !mask || mask[rank * kTiles + index];
      const bool rx_quant = !mask || mask[peer * kTiles + rank * kLocalTiles + tile];
      tx_cursor += tx_quant ? kQuantTile : kBF16Tile;
      rx_cursor += rx_quant ? kQuantTile : kBF16Tile;
    }
  }
  require(tx_cursor == tx_bytes && rx_cursor == rx_bytes, "Compact wire buffer length changed");
}
}  // namespace rtx5090_local_layout
