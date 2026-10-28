// Narrow, reproducible BF16 RCR registry for the H100 Phase6 experiment.
// V2 retains Ampere algorithm metadata but is compiled to SM90 machine code.
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
  auto v2 = make_space_gemm_meta(
      make_tuple(make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      make_tuple(_Sm80{}), make_tuple(_A100{}), make_tuple(_ReduceScatter{}),
      make_tuple(_RCR{}), make_tuple(_GemmV2{}), make_tuple(None{}),
      make_tuple(make_reduce_scatter_meta(_False{}, _IntraNode{})));
  auto v2hp = make_space_gemm_hparams(make_tuple(Auto{}), make_tuple(Auto{}),
      make_tuple(Shape<_128,_128,_32>{}), make_tuple(Auto{}), make_tuple(_3{}));
  auto v3 = make_space_gemm_meta(
      make_tuple(make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      make_tuple(_Sm90{}), make_tuple(H100_PROFILE{}), make_tuple(_ReduceScatter{}),
      make_tuple(_RCR{}), make_tuple(_GemmV3{}),
      make_tuple(make_gemm_v3_meta(_False{})),
      make_tuple(make_reduce_scatter_meta(_False{}, _IntraNode{})));
  // Preserve the upstream Hopper BF16 cluster choices.
  auto v3hp = make_space_gemm_hparams(make_tuple(
      make_gemm_v3_hparams(Shape<_2,_1,_1>{}),
      make_gemm_v3_hparams(Shape<_1,_2,_1>{})));
  return main_template(options, {
      cute::make_tuple(build_gen_space(v2, v2hp), std::string("gemm_rs/gemm_v2_reduce_scatter.hpp"), std::string("GemmV2ReduceScatter")),
      cute::make_tuple(build_gen_space(v3, v3hp), std::string("gemm_rs/gemm_v3_reduce_scatter.hpp"), std::string("GemmV3ReduceScatter"))});
}
