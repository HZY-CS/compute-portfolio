#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// CUDA Kernel：每个线程计算一个元素
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 修改这里的数字做实验A
    int numElements = 1000000; 
    size_t size = numElements * sizeof(float);
    std::cout << "[Vector addition of " << numElements << " elements]" << std::endl;

    // 1. 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 2. 分配设备（GPU）内存
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_C, size));

    // 3. 数据从主机拷贝到设备
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 4. 启动 Kernel
    // 修改这里的数字做实验B
    int threadsPerBlock = 512; 
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 5. 结果从设备拷贝回主机
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 6. 验证结果
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Test PASSED" << std::endl;

    // 7. 释放内存
    free(h_A); free(h_B); free(h_C);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    return 0;
}
