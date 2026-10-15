// Type your code here, or load an example.
//global 表示kernel核函数的keyword 
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;
#define BLOCK_SIZE 256
#define RADIUS 100
#define n 100000000
const int RUNS=10;
//函数内部虽然未用循环 但是每个thread都会用
__global__ void stencil_1d(int *in,int *out)
{
    __shared__ int temp[BLOCK_SIZE+RADIUS*2];    //开辟共享内存空间 limitation: 48KB a block
    //主要目的: 是为了存 隔壁block的部分边界元素
    int gindex=blockIdx.x*blockDim.x+threadIdx.x;   //全局内存的坐标
    int lindex=RADIUS+threadIdx.x;               //共享内存的坐标   留好两边的空白
    
    //接下来是吧 数据从 全局搬到共享
    if(gindex<n)temp[lindex]=in[gindex];                //把block内 数据直接搬到 shared的temp上
    else temp[lindex]=0;               //这是到block中 多余元素 
    //接下来 加载这个 radius内容
    if(threadIdx.x<RADIUS)       //加载两侧光晕
    {
        //R`BLOCK`R
        if(gindex-RADIUS<0)temp[lindex-RADIUS]=0;         //出GPU边界了 变0处理
        else  temp[lindex-RADIUS]=in[gindex-RADIUS];      //加载左边的radius
        if(gindex+BLOCK_SIZE>=n)temp[lindex+BLOCK_SIZE]=0;    //!!!不能return,出右边界了  变0处理
        else temp[lindex+BLOCK_SIZE]=in[gindex+BLOCK_SIZE];  //右边的
    }
    //the next action: read 7 elements from the shared memory every thread
    __syncthreads();   //保证block中全部读完了再去计算(当block中有空余元素的时候 也要保证其能到这)
    int result=0;
    for(int i=-RADIUS;i<=RADIUS;i++)
    {
        result+=temp[i+RADIUS+threadIdx.x];
    }
    if(gindex<n)out[gindex]=result;    //把合法结果传出去
}
// void syncthreads();    //强制一个block中所有thread都到达syncthread()后才会执行后面的

//And then we will make one without using the shared 
//compare the first and second in running time
__global__ void constant(int *in,int *out)
{
    int gindex=blockIdx.x*blockDim.x+threadIdx.x;
    if(gindex>=n)return;        //空闲元素不用操作d
    int result=0;
    for(int i=gindex-RADIUS;i<=gindex+RADIUS;i++)
    {
        if(i>=0&&i<n)result+=in[i];                  //add legal element
    }
    out[gindex]=result;
}
void random_int(int *a)
{
    for(int i=0;i<n;i++)
    {
        a[i]=i%100+3;           //弄个数据生成器
    }
}
int main()
{
    int *h_in,*h_out;             //host端 input and output
    int size1=n*sizeof(int);
    h_in=(int*)malloc(size1);
    random_int(h_in);
    h_out=(int*)malloc(size1);

    //下面是 : device 端的 分配
    int *d_in,*d_out;
    cudaMalloc((void**)&d_in,size1);
    cudaMalloc((void**)&d_out,size1);
    
    //transfer the data from host to device
    cudaMemcpy(d_in,h_in,size1,cudaMemcpyHostToDevice);
    int blocks=(n+BLOCK_SIZE-1)/BLOCK_SIZE;   //block number
    
    //计时的工具
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热（不计时）  为了防止前期 CUDA 上下文初始化、JIT 编译、驱动加载等开销混杂运行时间
    constant<<<blocks, BLOCK_SIZE>>>(d_in, d_out);
    stencil_1d<<<blocks, BLOCK_SIZE>>>(d_in, d_out);
    cudaDeviceSynchronize();
    
    //先弄 constant的
    float time1=0.0,total=0,temp=0;    //total存10次总时长 temp取接取每次的用时
    for(int i=0;i<RUNS;i++)            //运行RUNS次 取均值 减少误差
    {
        cudaEventRecord(start);
        constant<<<blocks,BLOCK_SIZE>>>(d_in,d_out);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);    //保证是所有的线程都结束了才stop计时
        cudaEventElapsedTime(&temp,start,stop);
        total+=temp;
    }
    time1=total/RUNS;
    cudaMemcpy(h_out, d_out, size1, cudaMemcpyDeviceToHost);  // 拷回主机
    cout << "测试常规的前10个: "<<endl;
    for(int i=0;i<10;i++) cout << h_out[i] << " ";
    cout << endl;

    // 接下来 是运用了 shared内存的
    cudaMemset(d_out, 0, size1);               //清空原来的d_out 去存shared版的
    total=0;    temp=0;
    float time2=0.0;
    for(int i=0;i<RUNS;i++)
    {
        cudaEventRecord(start);
        stencil_1d<<<blocks,BLOCK_SIZE>>>(d_in,d_out);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp,start,stop);
        total+=temp;
    }
    time2=total/RUNS;
    cout<<"测试shared的前10个"<<endl;
    cudaMemcpy(h_out, d_out, size1, cudaMemcpyDeviceToHost);  // 拷回主机
    for(int i=0;i<10;i++) cout << h_out[i] << " ";
    cout << endl;
    
    //and then check the data out
    int wrong=0;
    for(int i=0;i<n;i++)
    {
        int sum=0;
        for(int j=-RADIUS;j<=RADIUS;j++)
        {
            if(j+i<0||j+i>=n)continue;
            sum+=h_in[j+i];
        }
        if(h_out[i]!=sum)wrong++;
    }
    printf("calculate error: %d sites\n",wrong);
    float f=wrong*1.0*100/n;
    printf("the error percentage: %.5f%%\n",f);
    //现在得到了 constant and shared 的数据 开始分析
    printf("constant\t耗时: %.3f ms\n", time1);
    printf("shared\t耗时: %.3f ms\n", time2);
    printf("加速比 (constant / shared) = %.2fx\n", time1 / time2);

    //last: free all memory
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);    //time object also need to free
    return 0;
}
