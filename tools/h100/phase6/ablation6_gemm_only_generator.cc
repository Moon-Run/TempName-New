// Native Hopper BF16 RCR GEMM-only registry for the two ablation6 shapes.
// Uses the upstream cooperative/ping-pong and default/StreamK choices.
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
      make_tuple(_Sm90{}), make_tuple(_H800{}), make_tuple(_CommNone{}),
      make_tuple(_RCR{}), make_tuple(_GemmV3{}), make_tuple(make_gemm_v3_meta(_False{})));
  auto hp = tuple_filter(make_space_gemm_hparams(
      tuple_transform(tuple_cartesian_product(
          make_tuple(Shape<_2,_1,_1>{}, Shape<_1,_2,_1>{}),
          make_tuple(_Cooperative{}, _PingPong{})),
          [](auto tup) { auto [cluster, schedule] = tup; return make_gemm_v3_hparams(cluster, schedule); }),
      make_tuple(Auto{}), make_tuple(Auto{}), make_tuple(_GemmDefault{}, _GemmStreamK{})),
      [](auto tuple) {
        auto h = to_gemm_hparams(tuple);
        return !(to_gemm_v3_hparams(h.impl_spec()).kernel_schedule() == _PingPong{} &&
                 h.gemm_kind() == _GemmStreamK{});
      });
  // std::vector/std::string arguments let ADL select std::make_tuple here.
  // main_template requires a cute::tuple, as in the upstream generators.
  return main_template(options, {cute::make_tuple(build_gen_space(meta, hp),
      std::string("comm_none/gemm_v3_comm_none.hpp"), std::string("GemmV3CommNone"))});
}
