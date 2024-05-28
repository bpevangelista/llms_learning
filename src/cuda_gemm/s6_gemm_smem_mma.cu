#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include "helpers.cu"

__device__ __forceinline__ uint32_t lane_id() { uint32_t res; asm("mov.u32 %0, %laneid;" : "=r"(res) ); return res; }
__device__ float f1(uint32_t val) { return __half2float( ((half*)&val)[0] ); }
__device__ float f2(uint32_t val) { return __half2float( ((half*)&val)[1] ); }

__device__ float cp_async16B(void* smem, void *gmem) {
    asm("cp.async.ca.shared.global")
}

// One thread per output element [matSizeM, matSizeN] = [matSizeM, matSizeK] * [matSizeK, matSizeN]
// matA (row-major) and matB (col-major) allows continuous memory access
template <
    int32_t matSizeM, int32_t matSizeN, int32_t matSizeK                // Matrix A & B sizes
>
__global__ void gemm_kernel_m16n8k16(const half* __restrict__ matA, const half* __restrict__ matB, half* __restrict__ matOut) {

    constexpr int32_t kernelSizeMK = 16;
    constexpr int32_t kernelSizeN = 8;

    // 2D thread group block in 2D dispatch grid
    int32_t row = blockIdx.x * kernelSizeMK;
    int32_t col = blockIdx.y * kernelSizeN;

    // Handle output matrix (matSizeM x matSizeN) not evenly divisible by kernel size
    if (row >= matSizeM || col >= matSizeN) {
        return; // Early exit
    }

    // MatOut 16x8 tile
    // Calculate 4xf32 p/thread
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    __shared__ T sharedMatA[kernelSizeMK*kernelSizeMK];
    __shared__ T sharedMatB[kernelSizeMK*kernelSizeN];

    uint32_t laneid = lane_id();
    uint32_t k_shift = laneid >> 2;
    uint32_t lane_shift = laneid % 4;

    constexpr int32_t kernelSizeK = 16;
    for (int32_t k=0; k < matSizeK; k += kernelSizeK) {
        // MatA 16x16 tile
        // 8xf16 across 4 32b register (2xf16 p/register)
        uint32_t* matA_top_u32 = (uint32_t*)&matA[(row + k_shift + 0) * matSizeK + k];
        uint32_t* matA_bot_u32 = (uint32_t*)&matA[(row + k_shift + 8) * matSizeK + k];
        uint32_t a0 = matA_top_u32[lane_shift];
        uint32_t a1 = matA_bot_u32[lane_shift];
        uint32_t a2 = matA_top_u32[lane_shift + 4]; // step 8 - 4x(2xf16)
        uint32_t a3 = matA_bot_u32[lane_shift + 4]; // step 8 - 4x(2xf16)

        // MatB 16x8 tile
        // 4xf16 across 2 32b register (2xf16 p/register)
        uint32_t* matB_u32 = (uint32_t*)&matB[(col + k_shift + 0) * matSizeK + k];
        uint32_t b0 = matB_u32[lane_shift];
        uint32_t b1 = matB_u32[lane_shift + 4]; // step 8 - 4x(2xf16)

        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(acc0), "=f"(acc1), "=f"(acc2), "=f"(acc3)
          :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
             "r"(b0),  "r"(b1),
             "f"(acc0),  "f"(acc1),  "f"(acc2),  "f"(acc3));

        // Debug
        //if (laneid == 0) {
        //    printf("\na %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f", f1(a0), f2(a0), f1(a1), f2(a1), f1(a2), f2(a2), f1(a3), f2(a3));
        //    printf("\nb %.3f %.3f %.3f %.3f", f1(b0), f2(b0), f1(b1), f2(b1));
        //    printf("\nacc %.3f %.3f %.3f %.3f\n\n", acc0, acc1, acc2, acc3);
        //}
    }

    half* matOut_top = &matOut[(row + k_shift + 0) * matSizeN + col];
    half* matOut_bot = &matOut[(row + k_shift + 8) * matSizeN + col];
    matOut_top[lane_shift * 2 + 0] = __float2half_rn(acc0);
    matOut_top[lane_shift * 2 + 1] = __float2half_rn(acc1);
    matOut_bot[lane_shift * 2 + 0] = __float2half_rn(acc2);
    matOut_bot[lane_shift * 2 + 1] = __float2half_rn(acc3);
}


template <typename T, typename Acc>
int gemm_main(float EPSILON = 0.001f) {
    // MxK and NxK matrices
//#define USE_SMALL_MAT_FOR_DEBUG
#ifdef USE_SMALL_MAT_FOR_DEBUG
    constexpr int32_t matSizeM = 16;
    constexpr int32_t matSizeK = 16;
    constexpr int32_t matSizeN = 8;
#else
    constexpr int32_t matSizeM = 2048; // 2k context
    constexpr int32_t matSizeK = 4096; // 4k token dimension
    constexpr int32_t matSizeN = 2048; // 2k context
#endif
    // Per-Kernel Computation Size
    constexpr int32_t kernelSizeM = 16;
    constexpr int32_t kernelSizeN = 8;

    // Keep CPU copy of data for validation later
    T *cpuMatA, *cpuMatB, *cpuMatOut;

    // Alloc CUDA device memory with random data for matA, matB and zeros for matOut
    T* matA = deviceTensorRand<T>(1, matSizeM, matSizeK, 2.0f, &cpuMatA);     // [-2, 2] rand values
    T* matB = deviceTensorRand<T>(1, matSizeK, matSizeN, 2.0f, &cpuMatB);     // [-2, 2] rand values
    // Output is matRows x matRows
    T* matOut = deviceTensorRand<T>(1, matSizeM, matSizeN, 0.0f, &cpuMatOut); // [0, 0] Zero it
    if (matA == nullptr || matB == nullptr || matOut == nullptr) {
        return -1; // error
    }

    // Must be warp-size (32) multiple due to mma instruction
    dim3 threadGroupSize = dim3(32, 1, 1);
    dim3 threadGroupsCount = dim3(
        CEIL_DIV(matSizeM, kernelSizeM),
        CEIL_DIV(matSizeN, kernelSizeN)
        );
    int32_t dynamicSharedMemSize = 0; // Still not using it

    // Calculate on GPU
    cudaStream_t stream;
    cudaEvent_t kernelStart, kernelStop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart, 0);
    gemm_kernel_m16n8k16
        <matSizeM, matSizeN, matSizeK>
        <<<threadGroupsCount, threadGroupSize, dynamicSharedMemSize, stream>>>
        (matA, matB, matOut);
    cudaEventRecord(kernelStop, 0);

    // Calculate on CPU
    for (int i=0; i<matSizeM; ++i) {
        for (int j=0; j<matSizeN; ++j) {
            Acc acc = 0.0f;
            for (int k=0; k<matSizeK; ++k) {
                float temp = static_cast<float>(cpuMatA[i * matSizeK + k]) * static_cast<float>(cpuMatB[j * matSizeK + k]);
                if (sizeof(Acc) == 2) { // half
                    acc = __float2half_rn(static_cast<float>(acc) + temp);
                } else { // float
                    acc += temp;
                }
            }
            cpuMatOut[i * matSizeN + j] = acc;
        }
    }

    // Wait for GPU (just for correctness as CPU is much slower)
    cudaError_t cudaStatus = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Calculate runtime (ideally avg over many runs)
    float kernelMs = 0.0f;
    cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    printf("Kernel runtime: %.2fms\n", kernelMs);

    // Validate CPU vs GPU computation
    T* matOutCpuPtr;
    auto [diffs, mse] = debugCompare(cpuMatOut, matOut, &matOutCpuPtr, matSizeM * matSizeN, EPSILON);
    printf("Epsilon-diffs: count %d, perc %.3f, MSE %.4f\n", diffs, diffs/(float)(matSizeM * matSizeN), mse);

    // Debug small matrices
    if (matSizeM <= 32 && matSizeN <= 32) {
        printf("cpuMatA\n");
        printTensor(cpuMatA, matSizeM, matSizeK);
        printf("cpuMatB\n");
        printTensor(cpuMatB, matSizeN, matSizeK);
        printf("cpuMatOut\n");
        printTensor(cpuMatOut, matSizeM, matSizeN);
        printf("cudaMatOut\n");
        printTensor(matOutCpuPtr, matSizeM, matSizeN);
    }

    SAFE_FREE(cpuMatA);
    SAFE_FREE(cpuMatB);
    SAFE_FREE(cpuMatOut);
    SAFE_FREE(matOutCpuPtr);
    SAFE_CUDA_FREE(matA);
    SAFE_CUDA_FREE(matB);
    SAFE_CUDA_FREE(matOut);
    return 0;
}

int main() {
    return gemm_main<half, float>(0.1f);       // 6.2ms on 3060TI for 2k-4k-gemm, MSE 0.0007 or 0.02 (k=32)
}