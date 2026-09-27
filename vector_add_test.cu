#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 1000000;  // 100万，让CPU和GPU差距更明显
    size_t size = numElements * sizeof(float);
    std::cout << "[Vector addition of " << numElements << " elements]" << std::endl;

    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    // 初始化
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // ========== CPU 计时 ==========
    const int CPU_RUNS = 10;
    double cpu_total_ms = 0.0;
    for (int r = 0; r < CPU_RUNS; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numElements; ++i) {
            h_C_cpu[i] = h_A[i] + h_B[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        cpu_total_ms += elapsed.count();
    }
    double cpu_avg_ms = cpu_total_ms / CPU_RUNS;
    std::cout << "CPU average time: " << cpu_avg_ms << " ms" << std::endl;

    // ========== GPU 准备 ==========
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_C, size));

    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

    // 预热一次（不计时）
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CHECK(cudaDeviceSynchronize());

    // ========== GPU 纯 kernel 计时 ==========
    const int GPU_RUNS = 20;
    float gpu_total_ms = 0.0f;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int r = 0; r < GPU_RUNS; ++r) {
        CHECK(cudaEventRecord(start));
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_total_ms += ms;
    }
    float gpu_avg_ms = gpu_total_ms / GPU_RUNS;
    std::cout << "GPU kernel average time: " << gpu_avg_ms << " ms" << std::endl;

    // ========== GPU 总时间（含拷贝）计时 ==========
    float gpu_total_with_copy_ms = 0.0f;
    for (int r = 0; r < GPU_RUNS; ++r) {
        CHECK(cudaEventRecord(start));
        CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_total_with_copy_ms += ms;
    }
    float gpu_total_avg_ms = gpu_total_with_copy_ms / GPU_RUNS;
    std::cout << "GPU total (copy+kernel) average time: " << gpu_total_avg_ms << " ms" << std::endl;

    // 验证 GPU 结果
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C_gpu[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Test PASSED" << std::endl;

    // 加速比
    std::cout << "Speedup (CPU / GPU kernel): " << cpu_avg_ms / gpu_avg_ms << "x" << std::endl;
    std::cout << "Speedup (CPU / GPU total): " << cpu_avg_ms / gpu_total_avg_ms << "x" << std::endl;

    // 释放
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return 0;
}
//the running result:

// [Vector addition of 1000000 elements]
// CPU average time: 5.20842 ms
// CUDA kernel launch with 3907 blocks of 256 threads
// GPU kernel average time: 0.0106544 ms
// GPU total (copy+kernel) average time: 1.99443 ms
// Test PASSED
// Speedup (CPU / GPU kernel): 488.852x
// Speedup (CPU / GPU total): 2.61148x