#include "helpers.cuh"

template <typename T, typename Acc, int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK>
void kernel_main(cudaStream_t stream,cudaEvent_t kernel_start, cudaEvent_t kernel_stop, T* mat_a, T* mat_b, T* mat_out);

template <typename T, typename Acc, int32_t kMatSizeM, int32_t kMatSizeN, int32_t kMatSizeK>
int gemm_main(float EPSILON = 0.001f) {
    // Keep the source CPU data for validation later
    T *cpu_mat_a, *cpu_mat_b;

    // Alloc CUDA device memory with random data for mat_a, mat_b and zeros for mat_out
    T* mat_a = deviceTensorRand<T>(1, kMatSizeM, kMatSizeK, 2.0f, &cpu_mat_a);      // [-2, 2] rand values
    T* mat_b = deviceTensorRand<T>(1, kMatSizeK, kMatSizeN, 2.0f, &cpu_mat_b);      // [-2, 2] rand values
    // Output is matRows x matRows
    T* mat_out = deviceTensorRand<T>(1, kMatSizeM, kMatSizeN, 0.0f);                // [0, 0] Zero it
    if (mat_a == nullptr || mat_b == nullptr || mat_out == nullptr) {
        return -1; // error
    }

    // Run on GPU
    cudaStream_t stream;
    cudaEvent_t kernel_start, kernel_stop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    cudaEventRecord(kernel_start, stream);
    kernel_main<T, Acc, kMatSizeM, kMatSizeN, kMatSizeK>(stream, kernel_start, kernel_stop,
        mat_a, mat_b, mat_out);
    cudaEventRecord(kernel_stop, stream);

    // Wait for GPU (just for correctness as CPU is much slower)
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Calculate runtime (ideally avg over many runs)
    float kernel_elapsed_ms = 0.0f;
    cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    printf("Kernel runtime: %.2fms\n", kernel_elapsed_ms);

    // Compare CPU and GPU results
    #ifdef GEMM_MATH_VALIDATION_ENABLED
    compare_cpu_gpu_gemm<T, Acc, kMatSizeM, kMatSizeN, kMatSizeK>(mat_out, cpu_mat_a, cpu_mat_b);
    #endif

    SAFE_FREE(cpu_mat_a);
    SAFE_FREE(cpu_mat_b);
    SAFE_CUDA_FREE(mat_a);
    SAFE_CUDA_FREE(mat_b);
    SAFE_CUDA_FREE(mat_out);
    return 0;
}
