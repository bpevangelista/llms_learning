#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include "helpers.cu"

// One thread per output element [matSizeM, matSizeN] = [matSizeM, matSizeK] * [matSizeK, matSizeN]
// matA (row-major) and matB (col-major) allows continuous memory access
template <
    typename T, typename Acc,
    int32_t matSizeM, int32_t matSizeN, int32_t matSizeK,               // Matrix A & B sizes
    int32_t kernelSizeM, int32_t kernelSizeN                            // Kernel computation size
>
__global__ void gemm_kernel_mn(const T* __restrict__ matA, const T* __restrict__ matB, T* __restrict__ matOut) {

    // 2D thread group block in 2D dispatch grid
    int32_t row = (blockIdx.x * blockDim.x + threadIdx.x) * kernelSizeM;
    int32_t col = (blockIdx.y * blockDim.y + threadIdx.y) * kernelSizeN;
    // Handle output matrix (matSizeM x matSizeN) not evenly divisible by kernel size
    if (row >= matSizeM || col >= matSizeN) {
        return; // Early exit
    }

    // Accumulator registers
    constexpr int32_t kernelSizeK = 16;
    Acc acc[kernelSizeM*kernelSizeN] = {0};

    #pragma unroll // Too many registers?
    for (int32_t k=0; k < matSizeK; k += kernelSizeK) {
        // Compute kernelSizeM x kernelSizeN x kernelSizeK
        for (int ki=0; ki < kernelSizeM; ki++) {
            for (int kj=0; kj < kernelSizeN; kj++) {
                for (int kk=0; kk < kernelSizeK; kk++) {
                    acc[ki * kernelSizeN + kj] += static_cast<Acc>(
                        matA[(row + ki) * matSizeK + k + kk] *
                        matB[(col + kj) * matSizeK + k + kk]);
                }
            }
        }
    }

    #pragma unroll
    for (int i=0; i < kernelSizeM; i++) {
        for (int j=0; j < kernelSizeN; j++) {
            matOut[(row + i) * matSizeN + col + j] = acc[i * kernelSizeN + j];
        }
    }
}


template <typename T, typename Acc>
int gemm_main(float EPSILON = 0.001f) {
    // MxK and NxK matrices
    constexpr int32_t matSizeM = 2048; // 2k context
    constexpr int32_t matSizeK = 4096; // 4k token dimension
    constexpr int32_t matSizeN = 2048; // 2k context
    // Per-Kernel Computation Size
    constexpr int32_t kernelSizeM = 16;
    constexpr int32_t kernelSizeN = 8;

    // Keep CPU copy of data for validation later
    T *cpuMatA, *cpuMatB, *cpuMatOut;

    // Alloc CUDA device memory with random data for matA, matB and zeros for matOut
    T* matA = deviceTensorRand<T>(1, matSizeM, matSizeK, 2.0f, &cpuMatA);     // [-2, 2] rand values
    T* matB = deviceTensorRand<T>(1, matSizeK, matSizeN, 2.0f, &cpuMatB);     // [-2, 2] rand values
    // Output is matRows x matRows
    T* matOut = deviceTensorRand<T>(1, matSizeM, matSizeN, 0.0f, &cpuMatOut); // [ 0, 0] rand values
    if (matA == nullptr || matB == nullptr || matOut == nullptr) {
        return -1; // error
    }

    // Empiric dispatch block size 8x16 == 128 threads (match previous sample)
    dim3 threadGroupSize = dim3(8, 16, 1);
    dim3 threadGroupsCount = dim3(
        CEIL_DIV(matSizeM, threadGroupSize.x * kernelSizeM),
        CEIL_DIV(matSizeN, threadGroupSize.y * kernelSizeN));
    int32_t dynamicSharedMemSize = 0; // Still not using it

    // Calculate on GPU
    cudaStream_t stream;
    cudaEvent_t kernelStart, kernelStop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart, 0);
    gemm_kernel_mn
        <T, Acc, matSizeM, matSizeN, matSizeK, kernelSizeM, kernelSizeN>
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
    printf("Epsilon-diffs: count %d, perc %.3f. MSE %.4f\n", diffs, diffs/(float)(matSizeM * matSizeN), mse);

    // Debug small matrices
    if (matSizeM <= 32 && matSizeN <= 32) {
        printTensor(cpuMatA, matSizeM, matSizeK);
        printTensor(cpuMatB, matSizeN, matSizeK);
        printTensor(cpuMatOut, matSizeM, matSizeN);
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
    //return gemm_main<float, float>(0.001);    // 100ms on 3060TI for 2k-4k-gemm, MSE 0
    return gemm_main<half, float>(0.1f);       // 100ms on 3060TI for 2k-4k-gemm, MSE 0.0007 or 0.02 (k=32)
}