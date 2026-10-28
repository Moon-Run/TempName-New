{
global:
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
