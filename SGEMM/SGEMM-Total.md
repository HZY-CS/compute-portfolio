# 模块 03：SGEMM 矩阵乘法与深度优化

本模块记录了单精度矩阵乘法（SGEMM）从基础实现到共享内存分块（Tiling），再到架构限制测试的完整演进过程。

[⬅️ 返回项目主目录](../README.md)

## 📑 实验报告

### 📘 基础篇
- [SGEMM 基础实现与 16x16 分块初探](./SGEMM-V1-report.md) 
  *(记录了第一次跑通 Naive 和 Tiled 算法，验证了 2048 规模下的 1.30x 加速比)*

### 📕 进阶篇
- [SGEMM 性能瓶颈深度分析与排坑](./SGEMM-V2-report.md) 
  *(记录了 RTX 3090/4090 上的缓存机制反直觉现象、同步开销跷跷板理论及 TILE_SIZE=64 的硬件死线排坑)*

## 💻 源码文件

- [V1: Naive 与 Tiled 16x16 基础版](./SGEMM-V1.cu)
- [V2: Tiled 32x32 + Padding 进阶版](./SGEMM-V2-tiled_32.cu)