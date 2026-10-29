__device__ __forceinline__ float safe_divide(float a, float b) {
    return (fabsf(b) < 1e-12f) ? 0.0f : (a / b);
}

__device__ __forceinline__ float clampf(float x, float min_val, float max_val) {
    return fmaxf(fminf(x, max_val), min_val);
}

__device__ __forceinline__ float check_nan_inf(float x) {
    return (isnan(x) || isinf(x)) ? 0.0f : x;
}

__device__ __forceinline__ float clamp_fp8(float x) {
    return fmaxf(fminf(x, 448.0f), -448.0f);
}

// Warp Reduction Sum
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp Reduction Max
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}


// Super-Fused Kernel: Stats + Scale + Hadamard + Quantization
template<typename T, int GROUP_SIZE>
__global__ void __launch_bounds__(GROUP_SIZE) fused_ash_compress_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    float* __restrict__ quant_scales,    // Not strictly needed
    float* __restrict__ ada_scales_out,  // Not strictly needed
    int orgChunkCount,
    size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes,
    size_t scale_chunk_stride_bytes,
    float target_range,
    float lambda,
    float fp8_max_val,
    __nv_saturation_t saturate,
    __nv_fp8_interpretation_t fp8_type)
{
    int group_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    const char* in_base = (const char*)input + chunk_id * in_chunk_stride_bytes;
    char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;

    float* chunk_q_scales = (float*)(out_base + orgChunkCount);
    float* chunk_ada_scales = (float*)(out_base + orgChunkCount + scale_chunk_stride_bytes);

    int start_idx = group_id * GROUP_SIZE;
    int idx = start_idx + tid;

    float val = 0.0f;
    if (idx < orgChunkCount) {
        if (std::is_same<T, float>::value) {
            val = reinterpret_cast<const float*>(in_base)[idx];
        } else {
            val = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_base)[idx]);
        }
    }

    float val_sq = val * val;
    val_sq = warpReduceSum(val_sq); 

    static __shared__ float shared_stats[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_stats[warp_id] = val_sq;
    __syncthreads();

    float group_sum_sq = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_sum_sq = warpReduceSum(group_sum_sq);

    __shared__ float ada_scale;
    if (tid == 0) {
        int count = min(GROUP_SIZE, orgChunkCount - start_idx);
        float var = (group_sum_sq / (float)count) + lambda;
        if (var < 1e-12f) var = 1e-12f;
        
        float s = fp8_max_val * rsqrtf(var);
        ada_scale = clampf(s, 1e-3f, 1e3f);

        chunk_ada_scales[group_id] = ada_scale;
    }
    __syncthreads();

    val *= ada_scale;

    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

    if (GROUP_SIZE > 32) {
        extern __shared__ float smem_data[];
        smem_data[tid] = val;
        __syncthreads();

        for (int s = 32; s < GROUP_SIZE; s *= 2) {
            int neighbor_idx = tid ^ s;
            float neighbor = smem_data[neighbor_idx];

            float new_val;
            if (tid & s) new_val = neighbor - val;
            else         new_val = val + neighbor;

            __syncthreads();
            val = new_val;
            smem_data[tid] = val;
            __syncthreads();
        }
    }

    val *= rsqrtf((float)GROUP_SIZE);

    float abs_val = fabsf(val);
    abs_val = warpReduceMax(abs_val);

    if (lane == 0) shared_stats[warp_id] = abs_val;
    __syncthreads();

    float group_max = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_max = warpReduceMax(group_max);

    __shared__ float quant_scale;
    if (tid == 0) {
        float mv = (group_max < 1e-12f) ? 1e-12f : group_max;
        float s = safe_divide(mv, target_range);
        quant_scale = clampf(s, 1e-12f, 1e6f);
        chunk_q_scales[group_id] = quant_scale;
    }
    __syncthreads();

    val = check_nan_inf(val);
    float scaled = clamp_fp8(safe_divide(val, quant_scale));
    unsigned char q8 = (unsigned char)__nv_cvt_float_to_fp8(scaled, saturate, fp8_type);

    unsigned int packed = (unsigned int)q8;
    unsigned int b1 = __shfl_down_sync(0xffffffff, packed, 1);
    unsigned int b2 = __shfl_down_sync(0xffffffff, packed, 2);
    unsigned int b3 = __shfl_down_sync(0xffffffff, packed, 3);

    if ((tid & 3) == 0 && (idx + 3 < orgChunkCount)) {
        unsigned int final_int = packed | (b1 << 8) | (b2 << 16) | (b3 << 24);
        reinterpret_cast<unsigned int*>(out_base)[idx / 4] = final_int;
    }
    else if (idx < orgChunkCount && (tid & 3) != 0 && (idx / 4 != (idx + 3) / 4)) {
        ((__nv_fp8_storage_t*)out_base)[idx] = (__nv_fp8_storage_t)q8;
    }
}

template<typename T, int GROUP_SIZE>
__global__ void __launch_bounds__(GROUP_SIZE) fused_ash_decompress_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    const float* __restrict__ q_scales_base,   // Not strictly needed
    const float* __restrict__ ada_scales_base, // Not strictly needed
    int orgChunkCount,
    size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes,
    size_t scale_chunk_stride_bytes,
    __nv_fp8_interpretation_t fp8_type)
{
    int group_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    const char* in_base = (const char*)input + chunk_id * in_chunk_stride_bytes;
    char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;

    // Correct offsets for scales in the compressed buffer
    // Layout per chunk: [FP8 Data] [QuantScales] [AdaScales]
    const float* chunk_q_scales = (const float*)(in_base + orgChunkCount);
    const float* chunk_ada_scales = (const float*)(in_base + orgChunkCount + scale_chunk_stride_bytes);

    int idx = group_id * GROUP_SIZE + tid;

    float val = 0.0f;
    if (idx < orgChunkCount) {
        __nv_fp8_storage_t raw = ((const __nv_fp8_storage_t*)in_base)[idx];
        __half_raw hr = __nv_cvt_fp8_to_halfraw(raw, fp8_type);
        float f = __half2float(*reinterpret_cast<__half*>(&hr));
        float qs = chunk_q_scales[group_id];
        val = f * qs;
    }

    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

    if (GROUP_SIZE > 32) {
        extern __shared__ float smem_data[];
        smem_data[tid] = val;
        __syncthreads();

        for (int s = 32; s < GROUP_SIZE; s *= 2) {
            int neighbor_idx = tid ^ s;
            float neighbor = smem_data[neighbor_idx];
            
            float new_val;
            if (tid & s) new_val = neighbor - val;
            else         new_val = val + neighbor;
            
            __syncthreads();
            val = new_val;
            smem_data[tid] = val;
            __syncthreads();
        }
    }
    val *= rsqrtf((float)GROUP_SIZE);

    if (idx < orgChunkCount) {
        float ada_s = chunk_ada_scales[group_id];
        if (ada_s < 1e-12f) ada_s = 1e-12f;
        val = check_nan_inf(val) / ada_s;

        if (std::is_same<T, float>::value) {
            reinterpret_cast<float*>(out_base)[idx] = val;
        } else {
            reinterpret_cast<__nv_bfloat16*>(out_base)[idx] = __float2bfloat16(val);
        }
    }
}


