#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "helpers.cu"


// One thread per output element [matSizeM, matSizeN] = [matSizeM, matSizeK] * [matSizeK, matSizeN]
// matA (row-major) and matB (col-major) allows continuous memory access
__global__ void gemm_kernel1x1(const float* matA, const float* matB, float* matOut,
    const int32_t matSizeM, const int32_t matSizeN, const int32_t matSizeK) {

    // Using 1D block
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // Out is Rows x Rows
    int32_t row = index / matSizeN;
    int32_t col = index % matSizeN;

    float acc = 0.0f;
    for (int32_t k=0; k < matSizeK; k++) {
        // Row-major * Column-major
        acc += matA[row * matSizeK + k] * matB[col * matSizeK + k];
    }

    matOut[index] = acc;
}


int main() {
    // MxK and NxK matrices
    constexpr int32_t matSizeM = 2048;      // 2k context
    constexpr int32_t matSizeK = 4096;      // 4k token dimension
    constexpr int32_t matSizeN = 2048;      // 2k context

    // Keep CPU copy of data for validation later
    float *cpuMatA, *cpuMatB, *cpuMatOut;

    // Alloc CUDA device memory with random data for matA, matB and zeros for matOut
    float* matA = deviceTensorRand<float>(1, matSizeM, matSizeK, 2.0f, &cpuMatA);       // [-2, 2] rand values
    float* matB = deviceTensorRand<float>(1, matSizeN, matSizeK, 2.0f, &cpuMatB);       // [-2, 2] rand values
    // Output is matRows x matRows
    float* matOut = deviceTensorRand<float>(1, matSizeM, matSizeN, 0.0f, &cpuMatOut);     // Zeroed
    if (matA == nullptr || matB == nullptr || matOut == nullptr) {
        return -1; // error
    }

    // Empiric block size of 128 threads (rational, SM can dispatch 4xWarps of 32 threads)
    dim3 blockSize = dim3(128, 1, 1);
    dim3 blocksCount = dim3((matSizeM * matSizeN) / blockSize.x);
    // For simplicity, matrix size must be multiple of block size
    assert((matSizeM * matSizeN) % blockSize.x == 0);
    int32_t sharedMemorySize = 0;

    // Calculate GEMM on GPU
    cudaStream_t stream;
    cudaEvent_t kernelStart, kernelStop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart, 0);
    gemm_kernel1x1<<<blocksCount, blockSize, sharedMemorySize, stream>>>(matA, matB, matOut, matSizeM, matSizeN, matSizeK);
    cudaEventRecord(kernelStop, 0);

    // Calculate GEMM on CPU
    for (int i=0; i<matSizeM; ++i) {
        for (int j=0; j<matSizeN; ++j) {
            float acc = 0.0f;
            for (int k=0; k<matSizeK; ++k) {
                acc = cpuMatA[i * matSizeK + k] * cpuMatB[j * matSizeK + k] + acc;
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
    auto [diffs, mse] = debugCompare<T>(cpuMatOut, matOut, nullptr, matSizeM * matSizeN, EPSILON);
    printf("Epsilon-diffs: count %d, perc %.3f. MSE %.4f\n", diffs, diffs/(float)(matSizeM * matSizeN), mse);

    SAFE_FREE(cpuMatA);
    SAFE_FREE(cpuMatB);
    SAFE_FREE(cpuMatOut);
    SAFE_CUDA_FREE(matA);
    SAFE_CUDA_FREE(matB);
    SAFE_CUDA_FREE(matOut);
    return 0;
}