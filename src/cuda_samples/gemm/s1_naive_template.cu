#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include "helpers.cu"

#include <cutlass/cutlass.h>

// Parallel on output index [Rows, Rows] = [Rows, Cols] * [Cols, Rows]
// matA is row-major and matB col-major for continuous memory access
template <typename T, typename Acc>
__global__ void gemm_kernel(const T* __restrict__ matA, const T* __restrict__ matB, T* __restrict__ matOut, const int32_t matRows, const int32_t matCols) {
    // Using 1D block
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // Out is Rows x Rows
    int32_t row = index / matRows;
    int32_t col = index % matRows;

    Acc acc = static_cast<Acc>(0.0f);
    for (int32_t k=0; k < matCols; k++) {
        // Row-major * Column-major
        acc = static_cast<Acc>(matA[row * matCols + k] * matB[col * matCols + k]) + acc;
    }

    matOut[index] = static_cast<T>(acc);
}


template <typename T, typename Acc>
int gemm_main(float EPSILON = 0.001f) {
     constexpr int32_t matRows = 2048; // 2k context
     constexpr int32_t matCols = 4096; // 4k token dimension

    // Keep CPU copy of data for validation later
    T *cpuMatA, *cpuMatB, *cpuMatOut;

    // Alloc CUDA device memory with random data for matA, matB and zeros for matOut
    T* matA = deviceTensorRand<T>(1, matRows, matCols, 2.0f, &cpuMatA);     // [-2, 2] rand values
    T* matB = deviceTensorRand<T>(1, matCols, matRows, 2.0f, &cpuMatB);     // [-2, 2] rand values
    // Output is matRows x matRows
    T* matOut = deviceTensorRand<T>(1, matRows, matRows, 0.0f, &cpuMatOut); // [ 0, 0] rand values
    if (matA == nullptr || matB == nullptr || matOut == nullptr) {
        return -1; // error
    }

    // Empiric block size of 128 threads (rational, SM can dispatch 4xWarps of 32 threads)
    dim3 blockSize = dim3(128, 1, 1);
    dim3 blocksCount = dim3(ceil(matRows * matRows / float(blockSize.x)));
    int32_t sharedMemorySize = 0;

    // Calculate on GPU
    cudaStream_t stream;
    cudaEvent_t kernelStart, kernelStop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart, 0);
    gemm_kernel<T, Acc><<<blocksCount, blockSize, sharedMemorySize, stream>>>(matA, matB, matOut, matRows, matCols);
    cudaEventRecord(kernelStop, 0);

    // Calculate on CPU
    for (int i=0; i<matRows; ++i) {
        for (int j=0; j<matRows; ++j) {
            Acc acc = 0.0f;
            for (int k=0; k<matCols; ++k) {
                float temp = static_cast<float>(cpuMatA[i * matCols + k]) * static_cast<float>(cpuMatB[j * matCols + k]);
                if (sizeof(Acc) == 2) { // half
                    acc = __float2half_rn(static_cast<float>(acc) + temp);
                } else { // float
                    acc += temp;
                }
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
    debugCompareAndPrint(cpuMatOut, matOut, matRows * matRows, EPSILON);

    // Debug
    //printMat(cpuMatA, matRows, matCols);
    //printMat(cpuMatB, matRows, matCols);
    //printMat(cpuMatOut, matRows, matRows);
    //printMat(gpuToCpuMatOut, matRows, matRows);

    SAFE_FREE(cpuMatA);
    SAFE_FREE(cpuMatB);
    SAFE_FREE(cpuMatOut);
    SAFE_CUDA_FREE(matA);
    SAFE_CUDA_FREE(matB);
    SAFE_CUDA_FREE(matOut);
    return 0;
}

int main() {
    //return gemm_main<float, float>(0.001);    // 950ms on 3060TI for 2k-4k-gemm
    return gemm_main<half, float>(0.25f);       // 300ms on 3060TI for 2k-4k-gemm
    //return gemm_main<half, half>(0.75f);      // 360ms on 3060TI for 2k-4k-gemm
}