#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "gemm_main.cuh"

__device__ __forceinline__ uint32_t lane_id() { uint32_t res; asm("mov.u32 %0, %laneid;" : "=r"(res) ); return res; }
__device__ float f1(uint32_t val) { return __half2float( ((half*)&val)[0] ); }
__device__ float f2(uint32_t val) { return __half2float( ((half*)&val)[1] ); }

__device__ __forceinline__ void cp_async16B_cache_global(void *smem_ptr, const void *gmem_ptr, bool predicate=true) {
    constexpr int32_t kCopySize = 16;
    uint32_t src_size = predicate ? kCopySize : 0;
    uint32_t smem_ptr_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)); // Cast smem_ptr to u32

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
        :: "r"(smem_ptr_u32), "l"(gmem_ptr), "n"(kCopySize), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// One thread per output element [kMatSizeM, kMatSizeN] = [kMatSizeM, kMatSizeK] * [kMatSizeK, kMatSizeN]
// mat_a (row-major) and mat_b (col-major) allows continuous memory access
template <
    int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK,                // Matrix A & B sizes
    int32_t kKernelSizeMK, int32_t kKernelSizeN, int32_t kKernelCount
>
__global__ void gemm_kernel_m16n8k16(const half* __restrict__ mat_a_f16, const half* __restrict__ mat_b_f16, half* __restrict__ mat_out_f16) {

    const uint32_t* __restrict__ mat_a_u32 = (const uint32_t*)mat_a_f16;
    const uint32_t* __restrict__ mat_b_u32 = (const uint32_t*)mat_b_f16;

    // 2D thread group block in 2D dispatch grid
    int32_t kernel_index = threadIdx.x / 32;
    int32_t col = blockIdx.x * kKernelCount * kKernelSizeN + kernel_index * kKernelSizeN;
    int32_t row = blockIdx.y * kKernelSizeMK;

    // Handle output matrix (kMatSizeM x kMatSizeN) not evenly divisible by kernel size
    if (row >= kMatSizeM || col >= kMatSizeN) {
        return; // Early exit
    }

    constexpr int32_t kBufferCount = 2; // Multi buffering
    int32_t read_index = 0;
    int32_t write_index = 0;

    // Kernel M is (16 * 2B) 32B, 4xM == 128B (cache line)
    constexpr int32_t kMatrixBatch = 4;
    __shared__ uint32_t smem_mat_a_u32[kKernelCount][kBufferCount][kKernelSizeMK * kKernelSizeMK * kMatrixBatch / 2];    // 2xf16 p/reg
    __shared__ uint32_t smem_mat_b_u32[kKernelCount][kBufferCount][kKernelSizeMK * kKernelSizeN * kMatrixBatch / 2];    // 2xf16 p/reg

    // Calculate 4xf32 p/thread
    float acc[4] = {};

    uint32_t laneid = lane_id();
    int32_t shift_k = laneid >> 2;      // [0, 7]
    int32_t shift_elem = laneid % 4;    // [0, 3]
    int32_t tid_k   = laneid / 8;       // [0, 3]
    int32_t tid_ele = laneid % 8;       // [0, 7]

    constexpr int32_t kSmemLengthK_U32 = 32;                // 128B
    constexpr int32_t kGmemLengthK_U32 = kMatSizeK / 2;     // 2xF16 per U32
    constexpr int32_t kCopyLength_U32 = 4;                  // 16B

    // Warp - Copy 16x rows of 128B (64 elements, or 4x m16n8k16)
    #pragma unroll
    for (int32_t i=0; i < 4; i++) {
        int32_t local_row = tid_k + i * 4;
        cp_async16B_cache_global(&smem_mat_a_u32[kernel_index][write_index][local_row * kSmemLengthK_U32 + tid_ele * kCopyLength_U32],
            &mat_a_u32[(row + local_row) * kGmemLengthK_U32 + tid_ele * kCopyLength_U32]);
    }

    // Warp - Copy 8x cols of 128B (64 elements, or 4x m16n8k16)
    #pragma unroll
    for (int32_t i = 0; i < 2; i++) {
        int32_t local_col = tid_k + i * 4;
        cp_async16B_cache_global(&smem_mat_b_u32[kernel_index][write_index][local_col * kSmemLengthK_U32 + tid_ele * kCopyLength_U32],
            &mat_b_u32[(col + local_col) * kGmemLengthK_U32 + tid_ele * kCopyLength_U32]);
    }
    cp_async_commit_group();

    for (int32_t count=0, k=0; k < kMatSizeK; count++, k += kKernelSizeMK) {

        if (count % kMatrixBatch == 0) {
            read_index = write_index;
            write_index = (write_index + 1) % kBufferCount;

            // Warp - Copy 16x rows of 128B (64 elements, or 4x m16n8k16)
            #pragma unroll
            for (int32_t i = 0; i < 4; i++) {
                int32_t local_row = tid_k + i * 4;
                cp_async16B_cache_global(&smem_mat_a_u32[kernel_index][write_index][local_row * kSmemLengthK_U32 + tid_ele * kCopyLength_U32],
                    &mat_a_u32[(row + local_row) * kGmemLengthK_U32 + tid_ele * kCopyLength_U32 + (k + kKernelSizeMK * kMatrixBatch) / 2]); // todo skip initial
            }

            // Warp - Copy 8x cols of 128B (64 elements, or 4x m16n8k16)
            #pragma unroll
            for (int32_t i = 0; i < 2; i++) {
                int32_t local_col = tid_k + i * 4;
                cp_async16B_cache_global(&smem_mat_b_u32[kernel_index][write_index][local_col * kSmemLengthK_U32 + tid_ele * kCopyLength_U32],
                    &mat_b_u32[(col + local_col) * kGmemLengthK_U32 + tid_ele * kCopyLength_U32 + (k + kKernelSizeMK * kMatrixBatch) / 2]);
            }

            cp_async_commit_group();
            cp_async_wait_group<1>();
        }
        //__syncthreads();

        int32_t mat_shift = (count % kMatrixBatch) * 8; // Row shift is 8x u32 == 16x half
        uint32_t* a_u32 = &smem_mat_a_u32[kernel_index][read_index][mat_shift];
        uint32_t* b_u32 = &smem_mat_b_u32[kernel_index][read_index][mat_shift];

        uint32_t a0 = a_u32[(shift_k + 0) * kSmemLengthK_U32 + shift_elem + 0];
        uint32_t a2 = a_u32[(shift_k + 0) * kSmemLengthK_U32 + shift_elem + 4];
        uint32_t a1 = a_u32[(shift_k + 8) * kSmemLengthK_U32 + shift_elem + 0];
        uint32_t a3 = a_u32[(shift_k + 8) * kSmemLengthK_U32 + shift_elem + 4];
        uint32_t b0 = b_u32[shift_k * kSmemLengthK_U32 + shift_elem];
        uint32_t b1 = b_u32[shift_k * kSmemLengthK_U32 + shift_elem + 4];

        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7}, {%8,  %9},"
          "{%10, %11, %12, %13};"
          : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
          :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3), "r"(b0),  "r"(b1),
             "f"(acc[0]),  "f"(acc[1]),  "f"(acc[2]),  "f"(acc[3]));

#if 0 // Debug
        if (laneid == 0) {
            printf("\na %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f", f1(a0), f2(a0), f1(a1), f2(a1), f1(a2), f2(a2), f1(a3), f2(a3));
            printf("\nb %.2f %.2f %.2f %.2f", f1(b0), f2(b0), f1(b1), f2(b1));
            printf("\nacc %.2f %.2f %.2f %.2f\n\n", acc[0], acc[1], acc[2], acc[3]);
        }
#endif
    }

    half* mat_out_top = &mat_out_f16[(row + shift_k + 0) * kMatSizeN + col];
    half* mat_out_bot = &mat_out_f16[(row + shift_k + 8) * kMatSizeN + col];
    mat_out_top[shift_elem * 2 + 0] = __float2half_rn(acc[0]);
    mat_out_top[shift_elem * 2 + 1] = __float2half_rn(acc[1]);
    mat_out_bot[shift_elem * 2 + 0] = __float2half_rn(acc[2]);
    mat_out_bot[shift_elem * 2 + 1] = __float2half_rn(acc[3]);
}


template <typename T, typename Acc, int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK>
void kernel_main(cudaStream_t stream, cudaEvent_t kernel_start, cudaEvent_t kernel_stop,
    T* mat_a, T* mat_b, T* mat_out) {
    // Must be warp-size (32) multiple due to mma instruction
    constexpr int32_t kKernelDispatchCount = 1;
    constexpr int32_t kKernelSizeMK = 16;
    constexpr int32_t kKernelSizeN = 8;

    dim3 thread_group_size = dim3(kKernelDispatchCount * 32, 1, 1);
    dim3 thread_groups = dim3(
        CEIL_DIV(kMatSizeN, kKernelDispatchCount * kKernelSizeN),
        CEIL_DIV(kMatSizeM, kKernelSizeMK)
    );

    cudaEventRecord(kernel_start, stream);
    int32_t dynamic_smem_size = 0;
    gemm_kernel_m16n8k16
        <kMatSizeM, kMatSizeN, kMatSizeK,
        kKernelSizeMK, kKernelSizeN, kKernelDispatchCount>
        <<<thread_groups, thread_group_size, dynamic_smem_size, stream>>>
        (mat_a, mat_b, mat_out);
    cudaEventRecord(kernel_stop, stream);
}

int main() {
    constexpr int32_t kMatSizeM = 2048; // 2k context
    constexpr int32_t kMatSizeK = 4096; // 4k token dimension
    constexpr int32_t kMatSizeN = 2048; // 2k context

    // 6.2ms on 3060TI for 2k-4k-gemm, MSE 0.0007 or 0.02 (k=32)
    return gemm_main<half, float, kMatSizeM, kMatSizeN, kMatSizeK>();
}
