//global 表示kernel核函数的keyword
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define N 512
#define M 20
__global__ void add(int *a,int *b,int *t,int n)
{
    //Dim 是 一个block含的thread数
    int index=blockIdx.x*blockDim.x+threadIdx.x;  //标准index
    if(index<n)t[index]=a[index]+b[index];    //保证index有效
}
void get_num(int *x,int num)
{
    int cnt=0;
    while(cnt<N)
    {
        x[cnt]=num;
        num+=2;
        cnt++;
    }
}
int main(void)
{
    int *a,*b,*c;         //主机端数据
    int *d_a,*d_b,*d_c;   //the device
    int size=N*sizeof(int);
    cudaMalloc((void**)&d_a,size);  //device 端分配内存
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);
    a=(int*)malloc(size);
    b=(int*)malloc(size);
    get_num(a,40);
    get_num(b,100);    //处理数据 实际应该是input
    c=(int *)malloc(size);
    //the first step: take data from host to device
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice); //目标地址 源地址 复制大小 复制方向
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    //the second step: launch the kernel in GPU: add N blocks
    add<<<(N+M-1)/M,M>>>(d_a,d_b,d_c,N);   //向上取整
    //step 3: copy the result to host
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    // 验证结果
    for (int i = 0; i < 5; i++) 
    { // 只打印前5个
        printf("a[%d]=%d, b[%d]=%d, c[%d]=%d\n", i, a[i], i, b[i], i, c[i]);
    }
    free(a); free(b); free(c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
}
