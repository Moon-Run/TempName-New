"""Optional K64 producer tile for MA5; M/N and epilogue ownership stay fixed.

Changing only the CTA K tile would leave the automatic warp K at 32 and create
two K partitions / eight warps. The K64 variant explicitly changes warp K too,
preserving one K partition, four warps and the original output thread layout.
"""
from pathlib import Path

from common import replace_once, require

GENERATOR = 'src/generator/gen_gemm_rs.cc'
EPILOGUE = 'src/gemm_rs/epilogue_evt.hpp'


def validate_tile_k(tile_k):
    require(type(tile_k) is int and tile_k in (32, 64), 'MA5 GEMM tile K must be 32 or 64')
    return tile_k


def sources(reference, tile_k=32):
    """Return only the isolated K64 edits; K32 leaves both files byte-identical."""
    validate_tile_k(tile_k)
    if tile_k == 32:
        return {}
    reference = Path(reference)
    generator = replace_once((reference / GENERATOR).read_text(), '''  auto v2hp = make_space_gemm_hparams(make_tuple(Auto{}), make_tuple(Auto{}),
      make_tuple(Shape<_128,_128,_32>{}), make_tuple(Auto{}), make_tuple(_3{}));''', '''  // MA5 K64: warp K follows CTA K, preserving 4 warps and one K partition.
  auto v2hp = make_space_gemm_hparams(make_tuple(make_gemm_v2_hparams(
      Shape<_64,_64,_64>{}, Shape<_16,_8,_16>{})), make_tuple(Auto{}),
      make_tuple(Shape<_128,_128,_64>{}), make_tuple(Auto{}), make_tuple(_3{}));''')
    epilogue = replace_once((reference / EPILOGUE).read_text(), '''  using TacoThreadMap = OutputTileThreadLayout<
      cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::bfloat16_t, 8, 1>;''', '''  using TacoOriginalThreadMap = OutputTileThreadLayout<
      cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::bfloat16_t, 8, 1>;
  using TacoThreadMap = OutputTileThreadLayout<
      cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::bfloat16_t, 8, 1>;
  // The encoder indexes four warps and 16 rows per step. Its complete output
  // layout must remain the original layout, including thread and step order.
  static_assert(std::is_same_v<typename TacoThreadMap::Base,
                               typename TacoOriginalThreadMap::Base>);
  static_assert(std::is_same_v<typename TacoThreadMap::ThreadMapShape,
                               typename TacoOriginalThreadMap::ThreadMapShape>);''')
    epilogue = replace_once(epilogue,
        'TACO requires BF16 SM80 NVLink 128x128x32/128 threads and exact shape/rank',
        'MA5 K64 requires BF16 SM80 NVLink 128x128x64/128 threads and exact shape/rank')
    return {GENERATOR: generator, EPILOGUE: epilogue}


def compile_probe_source(tile_k=32):
    """Instantiate the real producer device, CUTLASS kernel and fused encoder."""
    validate_tile_k(tile_k)
    impl = ('Auto{}' if tile_k == 32 else
            'make_gemm_v2_hparams(Shape<_64, _64, _64>{}, Shape<_16, _8, _16>{})')
    return '''#include "gemm_rs/gemm_v2_reduce_scatter.hpp"
namespace bytedance::flux {
using namespace cute;
using Meta = decltype(make_gemm_meta(
    make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
    _Sm80{}, _A100{}, _ReduceScatter{}, _RCR{}, _GemmV2{}, None{},
    make_reduce_scatter_meta(_False{}, _IntraNode{})));
using HP = decltype(materialize_hparams(Meta{}, make_gemm_hparams(
    IMPL_SPEC, Auto{}, Shape<_128, _128, _TILE_K>{}, Auto{}, _3{})));
using Builder = GemmV2ReduceScatter_Kernel<Meta, HP>;
using StoreD = decltype(Builder().evt_store_d(Builder().default_kernel_params()));
static_assert(StoreD::kTacoSupported, "MA5 tile must enable the actual fused encoder");
using Kernel = decltype(Builder().gemm_kernel());
using Device = GemmV2ReduceScatter_Device<Meta, HP, Kernel>;
static_assert(Kernel::kThreadCount == 128, "MA5 producer must retain four warps");
extern "C" void ma5_gemm_compile_probe(std::any const &args, void *workspace,
                                      cudaStream_t stream) {
  Device op;
  op.run(args, workspace, stream);
}
}
'''.replace('IMPL_SPEC', impl).replace('_TILE_K', '_' + str(tile_k))
