#include <tuple>

#ifndef SAFE_FREE
#define SAFE_FREE(MPTR) [MPTR] { if (MPTR != nullptr) { free(MPTR); } }()
#endif

#ifndef SAFE_CUDA_FREE
#define SAFE_CUDA_FREE(MPTR) [MPTR] { if (MPTR != nullptr) { cudaFree(MPTR); } }()
#endif

inline int32_t CEIL_DIV(const int32_t num, const int32_t div) {
    return (num + div - 1) / div;
}


template <typename T>
T* deviceTensorRand(int batches, int rows, int columns, float randScale = 1.0f, T** optOutCpuPtr = nullptr) {
    int32_t resultElements = batches * rows * columns;
    int32_t resultSizeInBytes = resultElements * sizeof(T);

    T* device = nullptr;
    T* host = reinterpret_cast<T*>(malloc(resultSizeInBytes));
    if (host != nullptr) {
        if (cudaMalloc(&device, resultSizeInBytes) == cudaSuccess) {
            for (int32_t i=0; i<resultElements; ++i) {
                float floatValue = (2.0 * std::rand() / (float)RAND_MAX) - 1.0;
                T value = static_cast<T>(floatValue * randScale);
                host[i] = value;
            }
            cudaMemcpy(device, host, resultSizeInBytes, cudaMemcpyHostToDevice);
        }
        if (optOutCpuPtr != nullptr) {
            *optOutCpuPtr = host;
        } else {
            SAFE_FREE(host);
        }
    }

    return device;
}

template <typename T>
void printTensor(const char* labelName, T* tensor, int32_t rows, int32_t columns) {
    printf("%s", labelName);
    printTensor(tensor, rows, columns);
}

template <typename T>
void printTensor(T* tensor, int32_t rows, int32_t columns) {
    for (int32_t i=0; i<rows; ++i) {
        printf("[");
        for (int32_t j=0; j < columns; ++j) {
            printf("% 7.2f", static_cast<float>(tensor[i * columns + j]));
        }
        printf(" ]\n");
    }
    printf("\n");
}

template <typename T>
std::tuple<uint32_t, double> debugCompare(T* cpuTensorPtr, T* gpuTensorPtr, T** optGpuTensorCpuPtr,
    int32_t numElements, float EPSILON = 0.001f, bool printDeltas = false) {
    int32_t sizeInBytes = numElements * sizeof(T);
    T* gpuTensorCpuMapped = reinterpret_cast<T*>(malloc(sizeInBytes));
    cudaMemcpy(gpuTensorCpuMapped, gpuTensorPtr, sizeInBytes, cudaMemcpyDeviceToHost);

    double mse = 0.0;
    uint32_t diffCount = 0;
    for (int i=0; i<numElements; i++) {
        float cpuVal = static_cast<float>(cpuTensorPtr[i]);
        float gpuVal = static_cast<float>(gpuTensorCpuMapped[i]);
        mse += pow(cpuVal - gpuVal, 2);
        if (fabs(cpuVal - gpuVal) > EPSILON) {
            if (printDeltas) {
                printf("Epsilon-err @%8d: %9.4f\t%9.4f\n", i, cpuVal, gpuVal);
            }
            diffCount++;
        }
    }

    mse /= numElements;

    if (optGpuTensorCpuPtr != nullptr) {
        *optGpuTensorCpuPtr = gpuTensorCpuMapped;
    } else {
        SAFE_FREE(gpuTensorCpuMapped);
    }
    return {diffCount, mse};
}
