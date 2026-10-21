#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 32

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
// Kernel 2: Tiled 版本
// ==========================================
__global__ void sgemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 声明共享内存      存对应Tiling矩阵的小方块元素 
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];    //较原版对列的长度+1 
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];    //以避免在shared的bank中同时访问同一bank的多个地址的串行问题 

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 2. 步进循环
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // 【TODO 1：搬运 A】 把算block内thread所需的A矩阵的元素 按相对的index存入shared
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {   //index的y x 不超边界
            As[threadIdx.y][threadIdx.x] = A[row*K+t*TILE_SIZE+threadIdx.x];   //即存入A[row][t*TILE_SIZE+threadIdx.x]
        } else {      //超出了边界
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 【TODO 2：搬运 B】
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE+threadIdx.y)*N+col]; //存入b[t*TILE_SIZE+threadIdx.y][col]
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 等待所有线程搬运完毕
        __syncthreads();

        // 【TODO 3：计算】
        // c[i][j]=sum(a[i][k]*b[k][j])  途中i,j不变  对应的就是threadIdx的 y,x
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 等待所有线程计算完毕，防止提前覆盖共享内存
        __syncthreads();
    }

    // 3. 写回
    if (row < M && col < N) {     //排除掉多余的
        C[row * N + col] = sum;
    }
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
    dim3 block(TILE_SIZE, TILE_SIZE);    //2D的block  size*size
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE); //向上取整

    // 计时工具
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_naive = 0, ms_tiled = 0;

    // --- 运行 Naive ---
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K); // 预热
    cudaDeviceSynchronize();            //等 kernel 真正跑完 再进入下一步
    cudaEventRecord(start);             //预热完 开始计时
    for(int i=0; i<10; ++i) sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    //运行10次取平均时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);       //保证stop在GPU的操作完成后再往下
    cudaEventElapsedTime(&ms_naive, start, stop);
    ms_naive /= 10.0f;                //求平均时间 /10
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);   //把naive的answer传回Host

    // --- 运行 Tiled ---
    sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K); // 预热
    cudaDeviceSynchronize();          
    cudaEventRecord(start);
    for(int i=0; i<10; ++i) sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("Tiled GFLOPS: %.2f\n", gflops_tiled);
    printf("Naive Time:  %.3f ms\n", ms_naive);
    printf("Tiled Time:  %.3f ms\n", ms_tiled);
    printf("Speedup:     %.2fx\n", ms_naive / ms_tiled);

    // 清理空间
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}