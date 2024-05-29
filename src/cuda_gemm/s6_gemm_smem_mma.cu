#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include "helpers.cu"

__device__ __forceinline__ uint32_t lane_id() { uint32_t res; asm("mov.u32 %0, %laneid;" : "=r"(res) ); return res; }
__device__ float f1(uint32_t val) { return __half2float( ((half*)&val)[0] ); }
__device__ float f2(uint32_t val) { return __half2float( ((half*)&val)[1] ); }

__device__ void cp_async16B_cache_all(void *smem_ptr, const void *gmem_ptr, bool predicate=true) {
    constexpr int32_t kCopySize = 16;
    int32_t src_size = predicate? kCopySize : 0;
    uint32_t smem_ptr_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)); // Cast smem_ptr to u32

    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"
        :: "r"(smem_ptr_u32), "l"(gmem_ptr), "n"(kCopySize), "r"(src_size));
}

__device__ void cp_async16B_cache_global(void *smem_ptr, const void *gmem_ptr, bool predicate=true) {
    constexpr int32_t kCopySize = 16;
    int32_t src_size = predicate? kCopySize : 0;
    uint32_t smem_ptr_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)); // Cast smem_ptr to u32

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;"
        :: "r"(smem_ptr_u32), "l"(gmem_ptr), "n"(kCopySize), "r"(src_size));
}

__device__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}


// One thread per output element [kMatSizeM, kMatSizeN] = [kMatSizeM, kMatSizeK] * [kMatSizeK, kMatSizeN]
// mat_a (row-major) and mat_b (col-major) allows continuous memory access
template <
    int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK                // Matrix A & B sizes
>
__global__ void gemm_kernel_m16n8k16(const half* __restrict__ mat_a, const half* __restrict__ mat_b, half* __restrict__ mat_out) {

    constexpr int32_t kKernelSizeMK = 16;
    constexpr int32_t kKernelSizeN = 8;

    // 2D thread group block in 2D dispatch grid
    int32_t row = blockIdx.x * kKernelSizeMK;
    int32_t col = blockIdx.y * kKernelSizeN;

    // Handle output matrix (kMatSizeM x kMatSizeN) not evenly divisible by kernel size
    if (row >= kMatSizeM || col >= kMatSizeN) {
        return; // Early exit
    }

    uint32_t laneid = lane_id();
    uint32_t shift_k = laneid >> 2;
    uint32_t shift_elem = laneid % 4;

    constexpr int32_t kBufferCount = 2; // Triple buffering
    __shared__ uint32_t smem_mat_a[kKernelSizeMK*kKernelSizeMK*kBufferCount/2];
    __shared__ uint32_t smem_mat_b[kKernelSizeMK*kKernelSizeN*kBufferCount/2];

    // MatOut 16x8 tile
    // Calculate 4xf32 p/thread
    float acc[4] = {};

    for (int32_t k=0; k < kMatSizeK; k += kKernelSizeMK) {
        // MatA 16x16 tile
        // 8xf16 across 4 32b register (2xf16 p/register)
        #pragma unroll
        for (int cp = 0; cp < 16; cp++) {
            uint32_t* mat_a_u32 = (uint32_t*)&mat_a[(row + cp) * kMatSizeK + k];
            cp_async16B_cache_global(&smem_mat_a[cp * 8 + 0], &mat_a_u32[0]); // copy 8x
            cp_async16B_cache_global(&smem_mat_a[cp * 8 + 4], &mat_a_u32[4]); // copy 8x
        }

        // MatB 16x8 tile
        // 4xf16 across 2 32b register (2xf16 p/register)
        #pragma unroll
        for (int cp = 0; cp < 8; cp++) {
            uint32_t* mat_b_u32 = (uint32_t*)&mat_b[(col + cp) * kMatSizeK + k];
            cp_async16B_cache_global(&smem_mat_b[cp * 8 + 0], &mat_b_u32[0]);
            cp_async16B_cache_global(&smem_mat_b[cp * 8 + 4], &mat_b_u32[4]);
        }

        cp_async_wait_all();

        uint32_t a0 = smem_mat_a[shift_k * 8 + shift_elem];
        uint32_t a1 = smem_mat_a[shift_k * 8 + shift_elem + 64];
        uint32_t a2 = smem_mat_a[shift_k * 8 + shift_elem + 4];
        uint32_t a3 = smem_mat_a[shift_k * 8 + shift_elem + 64 + 4];
        uint32_t b0 = smem_mat_b[shift_k * 8 + shift_elem];
        uint32_t b1 = smem_mat_b[shift_k * 8 + shift_elem + 4];

        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7}, {%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
          :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3), "r"(b0),  "r"(b1),
             "f"(acc[0]),  "f"(acc[1]),  "f"(acc[2]),  "f"(acc[3]));

#if 0 // Debug
        if (laneid == 0) {
            printf("\na %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f", f1(a0), f2(a0), f1(a1), f2(a1), f1(a2), f2(a2), f1(a3), f2(a3));
            printf("\nb %.3f %.3f %.3f %.3f", f1(b0), f2(b0), f1(b1), f2(b1));
            printf("\nacc %.3f %.3f %.3f %.3f\n\n", acc[0], acc[1], acc[2], acc[3]);
        }
#endif
    }

    half* mat_out_top = &mat_out[(row + shift_k + 0) * kMatSizeN + col];
    half* mat_out_bot = &mat_out[(row + shift_k + 8) * kMatSizeN + col];
    mat_out_top[shift_elem * 2 + 0] = __float2half_rn(acc[0]);
    mat_out_top[shift_elem * 2 + 1] = __float2half_rn(acc[1]);
    mat_out_bot[shift_elem * 2 + 0] = __float2half_rn(acc[2]);
    mat_out_bot[shift_elem * 2 + 1] = __float2half_rn(acc[3]);
}


template <typename T, typename Acc>
int gemm_main(float EPSILON = 0.001f) {
    // MxK and NxK matrices
    constexpr int32_t kMatSizeM = 2048; // 2k context
    constexpr int32_t kMatSizeK = 4096; // 4k token dimension
    constexpr int32_t kMatSizeN = 2048; // 2k context

    // Per-Kernel Computation Size
    constexpr int32_t kKernelSizeMK = 16;
    constexpr int32_t kKernelSizeN = 8;

    // Keep CPU copy of data for validation later
    T *cpu_mat_a, *cpu_mat_b, *cpu_mat_out;

    // Alloc CUDA device memory with random data for mat_a, mat_b and zeros for mat_out
    T* mat_a = deviceTensorRand<T>(1, kMatSizeM, kMatSizeK, 2.0f, &cpu_mat_a);     // [-2, 2] rand values
    T* mat_b = deviceTensorRand<T>(1, kMatSizeK, kMatSizeN, 2.0f, &cpu_mat_b);     // [-2, 2] rand values
    // Output is matRows x matRows
    T* mat_out = deviceTensorRand<T>(1, kMatSizeM, kMatSizeN, 0.0f, &cpu_mat_out); // [0, 0] Zero it
    if (mat_a == nullptr || mat_b == nullptr || mat_out == nullptr) {
        return -1; // error
    }

    // Must be warp-size (32) multiple due to mma instruction
    dim3 thread_group_size = dim3(32, 1, 1);
    dim3 thread_groups = dim3(
        CEIL_DIV(kMatSizeM, kKernelSizeMK),
        CEIL_DIV(kMatSizeN, kKernelSizeN));
    int32_t dynamic_smem_size = 0;

    // Calculate on GPU
    cudaStream_t stream;
    cudaEvent_t kernel_start, kernel_stop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    cudaEventRecord(kernel_start, stream);
    gemm_kernel_m16n8k16
        <kMatSizeM, kMatSizeN, kMatSizeK>
        <<<thread_groups, thread_group_size, dynamic_smem_size, stream>>>
        (mat_a, mat_b, mat_out);
    cudaEventRecord(kernel_stop, stream);

    // Wait for GPU (just for correctness as CPU is much slower)
    cudaError_t status = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Calculate runtime (ideally avg over many runs)
    float kernel_elapsed_ms = 0.0f;
    cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    printf("Kernel runtime: %.2fms\n", kernel_elapsed_ms);

#ifdef CPU_MATH_VALIDATION_ENABLED
    // Calculate on CPU
    for (int i=0; i < kMatSizeM; ++i) {
        for (int j=0; j < kMatSizeN; ++j) {
            Acc acc = 0.0f;
            for (int k=0; k < kMatSizeK; ++k) {
                float temp = static_cast<float>(cpu_mat_a[i * kMatSizeK + k]) * static_cast<float>(cpu_mat_b[j * kMatSizeK + k]);
                if (sizeof(Acc) == 2) { // half
                    acc = __float2half_rn(static_cast<float>(acc) + temp);
                } else { // float
                    acc += temp;
                }
            }
            cpu_mat_out[i * kMatSizeN + j] = acc;
        }
    }

    // Validate CPU vs GPU computation
    T* mat_out_cpu_copied;
    auto [diffs, mse] = debugCompare(cpu_mat_out, mat_out, &mat_out_cpu_copied, kMatSizeM * kMatSizeN, EPSILON);
    printf("Epsilon-diffs: count %d, perc %.3f, MSE %.4f\n", diffs, diffs/(float)(kMatSizeM * kMatSizeN), mse);

    // Debug small matrices
    if (kMatSizeM <= 32 && kMatSizeN <= 32) {
        printTensor("cpu_mat_a\n", cpu_mat_a, kMatSizeM, kMatSizeK);
        printTensor("cpu_mat_b\n", cpu_mat_b, kMatSizeN, kMatSizeK);
        printTensor("cpu_mat_out\n", cpu_mat_out, kMatSizeM, kMatSizeN);
        printTensor("cuda_mat_out\n", mat_out_cpu_copied, kMatSizeM, kMatSizeN);
    }
    SAFE_FREE(mat_out_cpu_copied);
#endif

    SAFE_FREE(cpu_mat_a);
    SAFE_FREE(cpu_mat_b);
    SAFE_FREE(cpu_mat_out);
    SAFE_CUDA_FREE(mat_a);
    SAFE_CUDA_FREE(mat_b);
    SAFE_CUDA_FREE(mat_out);
    return 0;
}

int main() {
    return gemm_main<half, float>(0.1f);       // 6.2ms on 3060TI for 2k-4k-gemm, MSE 0.0007 or 0.02 (k=32)
}