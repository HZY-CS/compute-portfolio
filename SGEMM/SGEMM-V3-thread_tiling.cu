#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 64 // Block 负责 64x64 的大块

// 错误检查宏
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// 随机初始化矩阵
void init_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() % 100) / 100.0f; // 0.0 ~ 1.0
    }
}

// CPU 端验证参考（Naive方式）
void cpu_sgemm(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];   //a[i][k]*b[k][j]  对k 0~K-1遍历 得到c[i][j]
            }
            C[i * N + j] = sum;
        }
    }
}

// ==========================================
// Kernel 1: Naive 版本
// ==========================================
__global__ void sgemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;    //注意行是y(笛卡尔坐标系)
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {      //row col是在指针A,B下的 全局index
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;   //得到C[row][col]
    }
}

// ==========================================
// Kernel 3: register 加强 版本
// ==========================================
__global__ void sgemm_v3_simple(float* A, float* B, float* C, int M, int N, int K) {
    // 去掉 [2] 维度，单缓冲。64x64 的大块，共享内存也是 64x64
    __shared__ float As[TILE_SIZE][BLOCK_SIZE + 1];     //A矩阵是多存了一倍的行 64*32
    __shared__ float Bs[BLOCK_SIZE][TILE_SIZE + 1];     //B是多存了一倍的列  32*64

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row0 = blockIdx.y * TILE_SIZE + ty;
    int row1 = row0 + BLOCK_SIZE; // 负责下半部分的行
    int col0 = blockIdx.x * TILE_SIZE + tx;
    int col1 = col0 + BLOCK_SIZE; // 负责右半部分的列

    float sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;

    // K 轴按 32 步进
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        
        // 1. 搬运 A (共 64 行，分两次搬)
        if (row0 < M && (t * BLOCK_SIZE + tx) < K) 
            As[ty][tx] = A[row0 * K + t * BLOCK_SIZE + tx]; 
        else As[ty][tx] = 0;
        
        if (row1 < M && (t * BLOCK_SIZE + tx) < K) 
            As[ty + BLOCK_SIZE][tx] = A[row1 * K + t * BLOCK_SIZE + tx]; 
        else As[ty + BLOCK_SIZE][tx] = 0;

        // 2. 搬运 B (共 64 列，分两次搬)
        if ((t * BLOCK_SIZE + ty) < K && col0 < N) 
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col0]; 
        else Bs[ty][tx] = 0;
        
        if ((t * BLOCK_SIZE + ty) < K && col1 < N) 
            Bs[ty][tx + BLOCK_SIZE] = B[(t * BLOCK_SIZE + ty) * N + col1]; 
        else Bs[ty][tx + BLOCK_SIZE] = 0;

        __syncthreads();

        // 3. 核心计算：2x2 寄存器累加
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // 一次读 4 个，做 4 次乘加
            float a0 = As[ty][k];
            float a1 = As[ty + BLOCK_SIZE][k];
            float b0 = Bs[k][tx];
            float b1 = Bs[k][tx + BLOCK_SIZE];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads();
    }

    // 4. 写回
    if (row0 < M && col0 < N) C[row0 * N + col0] = sum00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = sum01;
    if (row1 < M && col0 < N) C[row1 * N + col0] = sum10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = sum11;
}
// ==========================================
// Main 函数
// ==========================================
int main() {
    int M = 8192, N = 8192, K = 8192;
    size_t size_A = M * K * sizeof(float);     //a矩阵大小 
    size_t size_B = K * N * sizeof(float);     //b矩阵
    size_t size_C = M * N * sizeof(float);     //answer矩阵

    // Host 分配与初始化
    float *h_A = (float*)malloc(size_A);        //a[i][j]  为h_A[i*K+j]
    float *h_B = (float*)malloc(size_B);        //b[i][j]  为h_b[i*N+j]
    float *h_C_naive = (float*)malloc(size_C);  //naive版的 answer矩阵
    float *h_C_tiled = (float*)malloc(size_C);  //分块tiling+shared的answer
    float *h_C_ref = (float*)malloc(size_C);    //Host端的 answer

    //矩阵的初始生成
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);

    // Device 分配与拷贝Host端
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, size_A));
    CHECK(cudaMalloc((void**)&d_B, size_B));
    CHECK(cudaMalloc((void**)&d_C, size_C));
    CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // 配置 2D Grid 和 Block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);    //2D的block  size*size
    dim3 grid_naive((N+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE);    //naive版的分配
    dim3 grid_v3((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE); //向上取整

    // 计时工具
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_naive = 0, ms_tiled = 0;

    // --- 运行 Naive ---
    sgemm_naive<<<grid_naive, block>>>(d_A, d_B, d_C, M, N, K); // 预热
    cudaDeviceSynchronize();            //等 kernel 真正跑完 再进入下一步
    cudaEventRecord(start);             //预热完 开始计时
    for(int i=0; i<10; ++i) sgemm_naive<<<grid_naive, block>>>(d_A, d_B, d_C, M, N, K);
    //运行10次取平均时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);       //保证stop在GPU的操作完成后再往下
    cudaEventElapsedTime(&ms_naive, start, stop);
    ms_naive /= 10.0f;                //求平均时间 /10
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);   //把naive的answer传回Host

    // --- 运行 Tiled ---
    sgemm_v3_simple<<<grid_v3, block>>>(d_A, d_B, d_C, M, N, K); // 预热
    cudaDeviceSynchronize();          
    cudaEventRecord(start);
    for(int i=0; i<10; ++i) sgemm_v3_simple<<<grid_v3, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_tiled, start, stop);
    ms_tiled /= 10.0f;
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);

    // --- CPU版本去验证 --- 本版本加大了数据强度去分析Naive 和 Tiling为效率信任其正确率
    // cpu_sgemm(h_A, h_B, h_C_ref, M, N, K);     //得到CPU端计算的结果 h_C_ref
    // int errors = 0;
    // for (int i = 0; i < M * N; i++) {
    //     if (fabs(h_C_tiled[i] - h_C_ref[i]) > 1e-3) errors++;    //float判断是否相等
    // }

    // --- 输出结果 ---
    printf("=== SGEMM Result (M=N=K=%d) ===\n", M);
    // printf("Verification errors: %d / %d\n", errors, M * N);
    //add: 显示GFLOPS (每秒十亿次浮点运算)
    double gflops_naive = (2.0 * M * N * K) / (ms_naive * 1e6);
    double gflops_tiled = (2.0 * M * N * K) / (ms_tiled * 1e6);
    printf("Naive GFLOPS: %.2f\n", gflops_naive);
    printf("V3-register GFLOPS: %.2f\n", gflops_tiled);
    printf("Naive Time:  %.3f ms\n", ms_naive);
    printf("V3-register Time:  %.3f ms\n", ms_tiled);
    printf("Speedup:     %.2fx\n", ms_naive / ms_tiled);

    // 清理空间
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}