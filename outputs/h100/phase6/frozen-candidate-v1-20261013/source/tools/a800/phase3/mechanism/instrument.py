"""Reviewed phase2 fragment protocol, parameterized with four-source configuration."""
def instrument_epilogue(source):
    s = source
    s = s.replace('#pragma once', '#pragma once\n#include "sampler.h"', 1)
    s = s.replace('    bool has_barrier_ptr;', '    MechConfig mech;\n    bool has_barrier_ptr;', 1)
    s = s.replace('  static constexpr Params\n  to_underlying_arguments', '  static Params\n  to_underlying_arguments', 1)
    s = s.replace('    int col_stride[MAX_RANK_SIZE];', '    int col_stride[MAX_RANK_SIZE];\n    unsigned p2_step_mask = 0;', 1)
    s = s.replace('    params.dAux = args.dAux;', '    params.mech = mech_config();\n    params.dAux = args.dAux;', 1)
    needle = '    CUTLASS_DEVICE void\n    end_step(int step_idx) {\n'
    assert s.count(needle) == 1
    s = s.replace(needle, needle + "      bool p2 = !kPcieMode && params.mech.mode && params.mech.stride > 0 &&\n                tile_idx < MECH_TILES && step_idx < MECH_FRAGMENTS &&\n                tile_idx % params.mech.stride == params.mech.offset;\n      unsigned long long* rec = nullptr;\n      if(p2){\n        p2_step_mask |= 1u << step_idx;\n        // All threads have completed this fragment's conversion before the ready marker.\n        __syncthreads();\n        if(thread_idx==0){\n          rec=params.mech.producer+(tile_idx*MECH_FRAGMENTS+step_idx)*MECH_FIELDS;\n          rec[0]=mech_clock(); rec[2]=ThreadblockShape::kM; rec[3]=ThreadblockShape::kN;\n          rec[4]=size<3>(tC_cAux); rec[5]=dst_rank; rec[6]=blockIdx.x;\n          atomicAdd(rec+7,1ull);\n        }\n      }\n", 1)
    needle = '      }\n    }\n\n    ////\n    CUTLASS_DEVICE void\n    end_epilogue()'
    assert s.count(needle) == 1
    s = s.replace(needle, '      }\n      if(p2 && params.mech.mode==1){\n        __syncthreads();\n        if(thread_idx==0)rec[1]=mech_clock();\n      }\n    }\n\n    ////\n    CUTLASS_DEVICE void\n    end_epilogue()', 1)
    s = s.replace('    end_epilogue() {', '    end_epilogue() {\n      if(p2_step_mask && params.mech.mode==2){\n        // One system fence per epilogue callback, not per fragment. Separate Stream-K\n        // reduction callbacks publish their own fragment; no tile is declared complete here.\n        __threadfence_system();\n        __syncthreads();\n        if(thread_idx==0){\n          auto stamp=mech_clock();\n          for(int f=0;f<MECH_FRAGMENTS;++f)if(p2_step_mask & (1u<<f)){\n            auto rec=params.mech.producer+(tile_idx*MECH_FRAGMENTS+f)*MECH_FIELDS;\n            rec[1]=stamp;\n            mech_publish(params.mech.flags[dst_rank]+params.rank*MECH_SLOTS+\n                           tile_idx*MECH_FRAGMENTS+f,params.mech.epoch);\n          }\n        }\n      }\n', 1)
    return s
