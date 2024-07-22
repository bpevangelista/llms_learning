#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "gemm_main.cuh"


// One thread per output element [matSizeM, matSizeN]
// matA (row-major) and matB (col-major) allows continuous memory access
template <
    typename T, typename Acc,
    int32_t matSizeM, int32_t matSizeN, int32_t matSizeK        // Matrix A & B sizes
>
__global__ void gemm_kernel1x1(const T* __restrict__ matA, const T* __restrict__ matB, T* __restrict__ matOut) {

    // 1D thread group block in 1D dispatch grid
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // Assumes output matrix (matSizeM x matSizeN) evenly divides by kernel size
    int32_t row = index / matSizeN;
    int32_t col = index % matSizeN;

    Acc acc = static_cast<Acc>(0.0f);
    for (int32_t k=0; k < matSizeK; k++) {
        // Row-major * Column-major
        acc += static_cast<Acc>(
            matA[row * matSizeK + k] *
            matB[col * matSizeK + k]);
    }

    matOut[index] = static_cast<T>(acc);
}

template <typename T, typename Acc, int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK>
void kernel_main(cudaStream_t stream, cudaEvent_t kernel_start, cudaEvent_t kernel_stop,
    T* mat_a, T* mat_b, T* mat_out) {

    constexpr int32_t kDynamicSmemSize = 0;
    constexpr int32_t kKernelSize = 128;

    dim3 thread_group_size = dim3(kKernelSize, 1, 1);
    dim3 thread_groups = dim3(
        CEIL_DIV(kMatSizeM * kMatSizeN, kKernelSize)
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

    //return gemm_main<float, float, kMatSizeM, kMatSizeN, kMatSizeK>();    // 950ms on 3060TI for 2k-4k-gemm
    return gemm_main<half, float, kMatSizeM, kMatSizeN, kMatSizeK>();       // 300ms on 3060TI for 2k-4k-gemm
}
