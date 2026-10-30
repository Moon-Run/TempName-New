// RTX5090 BF16 RCR V2 registry. Actual CUDA machine code targets SM120.
// Narrowed from src/generator/gen_comm_none.cc; no Hopper GMMA registration.
#include "flux/flux.h"
#include "generator_utils.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
int main(int argc, char const **argv) {
  using namespace cute;
  using namespace bytedance::flux;
  using namespace bytedance::flux::generator;
  Options options;
  options.parse(argc, argv);
  if (options.help) { options.print_usage(std::cout); return 0; }
  auto meta = make_space_gemm_meta(
      make_tuple(make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      make_tuple(_Sm89{}), make_tuple(_RTX5090{}), make_tuple(_CommNone{}),
      make_tuple(_RCR{}), make_tuple(_GemmV2{}));
  auto hp = make_space_gemm_hparams(make_tuple(Auto{}), make_tuple(Auto{}),
      make_tuple(Shape<_128,_128,_32>{}), make_tuple(Auto{}), make_tuple(_3{}));
  return main_template(options, {cute::make_tuple(build_gen_space(meta, hp),
      std::string("comm_none/gemm_v2_comm_none.hpp"), std::string("GemmV2CommNone"))});
}
