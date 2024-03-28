#include <stdint.h>
#include <stdio.h>
#include "helpers.cu"


// Parallel on output index [Rows, Rows] = [Rows, Cols] * [Cols, Rows]
// matA is row-major and matB col-major for continuous memory access
__global__ void gemm_kernel(const float* matA, const float* matB, float* matOut, const int32_t matRows, const int32_t matCols) {
    // Using 1D block
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // Out is Rows x Rows
    int32_t row = index / matRows;
    int32_t col = index % matRows;

    float acc = 0.0f;
    for (int32_t k=0; k < matCols; k++) {
        // Row-major * Column-major
        acc = matA[row * matCols + k] * matB[col * matCols + k] + acc;
    }

    matOut[index] = acc;
}


int main() {
     constexpr int32_t matRows = 2048; // 2k context
     constexpr int32_t matCols = 4096; // 4k token dimension

    // Keep CPU copy of data for validation later
    float *cpuMatA, *cpuMatB, *cpuMatOut;

    // Alloc CUDA device memory with random data for matA, matB and zeros for matOut
    float* matA = deviceTensorRand<float>(1, matRows, matCols, 2.0f, &cpuMatA);     // [-2, 2] rand values
    float* matB = deviceTensorRand<float>(1, matCols, matRows, 2.0f, &cpuMatB);     // [-2, 2] rand values
    // Output is matRows x matRows
    float* matOut = deviceTensorRand<float>(1, matRows, matRows, 0.0f, &cpuMatOut); // [ 0, 0] rand values
    if (matA == nullptr || matB == nullptr || matOut == nullptr) {
        return -1; // error
    }

    // Empiric block size of 128 threads (rational, SM can dispatch 4xWarps of 32 threads)
    dim3 blockSize = dim3(128, 1, 1);
    dim3 blocksCount = dim3(ceil(matRows * matRows / float(blockSize.x)));
    int32_t sharedMemorySize = 0;

    // Calculate GEMM on GPU
    cudaStream_t stream;
    cudaEvent_t kernelStart, kernelStop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart, 0);
    gemm_kernel<<<blocksCount, blockSize, sharedMemorySize, stream>>>(matA, matB, matOut, matRows, matCols);
    cudaEventRecord(kernelStop, 0);

    // Calculate GEMM on CPU
    for (int i=0; i<matRows; ++i) {
        for (int j=0; j<matRows; ++j) {
            float acc = 0.0f;
            for (int k=0; k<matCols; ++k) {
                acc = cpuMatA[i * matCols + k] * cpuMatB[j * matCols + k] + acc;
            }
            cpuMatOut[i * matRows + j] = acc;
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
    debugCompareAndPrint(cpuMatOut, matOut, matRows * matRows);

    SAFE_FREE(cpuMatA);
    SAFE_FREE(cpuMatB);
    SAFE_FREE(cpuMatOut);
    SAFE_CUDA_FREE(matA);
    SAFE_CUDA_FREE(matB);
    SAFE_CUDA_FREE(matOut);
    return 0;
}