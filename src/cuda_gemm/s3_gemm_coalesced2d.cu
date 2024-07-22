#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "gemm_main.cuh"


// matA (row-major) and matB (col-major) allows continuous memory access
template <
    typename T, typename Acc,
    int32_t matSizeM, int32_t matSizeN, int32_t matSizeK                // Matrix A & B sizes
>
__global__ void gemm_kernel1x1(const T* __restrict__ matA, const T* __restrict__ matB, T* __restrict__ matOut) {

    // 2D thread group block in 2D dispatch grid
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t col = blockIdx.y * blockDim.y + threadIdx.y;
    // Handle output matrix (matSizeM x matSizeN) not evenly divisible by kernel size
    if (row >= matSizeM || col >= matSizeN) {
        return; // Early exit
    }

    if (row < matSizeM && col < matSizeN) {
        int32_t index = row * matSizeN + col;

        Acc acc = {0};
        for (int32_t k=0; k < matSizeK; k++) {
            // Row-major * Column-major
            acc += static_cast<Acc>(
                matA[row * matSizeK + k] *
                matB[col * matSizeK + k]);
        }

        if (sizeof(T) == 2 && sizeof(Acc) == 4) { // compile-time expr
            matOut[index] = __float2half_rn(acc);
        } else {
            matOut[index] = static_cast<T>(acc);
        }
    }
}

template <typename T, typename Acc, int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK>
void kernel_main(cudaStream_t stream, cudaEvent_t kernel_start, cudaEvent_t kernel_stop,
    T* mat_a, T* mat_b, T* mat_out) {

    constexpr int32_t kDynamicSmemSize = 0;
    constexpr int32_t kKernelTileM = 8;
    constexpr int32_t kKernelTileN = 16;

    dim3 thread_group_size = dim3(kKernelTileM, kKernelTileN, 1);
    dim3 thread_groups = dim3(
        CEIL_DIV(kMatSizeM, kKernelTileM),
        CEIL_DIV(kMatSizeN, kKernelTileN)
    );

    cudaEventRecord(kernel_start, stream);
    gemm_kernel1x1
        <T, Acc, kMatSizeM, kMatSizeN, kMatSizeK>
        <<<thread_groups, thread_group_size, kDynamicSmemSize, stream>>>
        (mat_a, mat_b, mat_out);
    cudaEventRecord(kernel_stop, stream);
}

int main() {
    constexpr int32_t kMatSizeM = 2048; // 2k context
    constexpr int32_t kMatSizeK = 4096; // 4k token dimension
    constexpr int32_t kMatSizeN = 2048; // 2k context

    //return gemm_main<float, float, kMatSizeM, kMatSizeN, kMatSizeK>();    // 100ms on 3060TI for 2k-4k-gemm, MSE 0
    return gemm_main<half, float, kMatSizeM, kMatSizeN, kMatSizeK>();       // 100ms on 3060TI for 2k-4k-gemm, MSE 0.0007 or 0.02 (k=32)
}
