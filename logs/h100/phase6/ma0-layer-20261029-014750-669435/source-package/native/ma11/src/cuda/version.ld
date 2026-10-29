{
global:
    ma11_softmax_protocol;
    ma11_softmax_forward_cuda;
    ma11_softmax_backward_cuda;
    ma9_zero_count_protocol;
    ma9_zero_count_chunk_size;
    ma9_zero_count_cuda;
    taco_*;
    h100_v2_occupancy;
    extern "C++" {
        bytedance::flux::*;
        nvshmemi_init_thread*;
        nvshmemi_finalize*;
    };

local:
    *;
};
