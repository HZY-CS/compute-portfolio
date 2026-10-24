# 模块 03：SGEMM 矩阵乘法与深度优化

本模块记录了单精度矩阵乘法（SGEMM）从基础实现到共享内存分块（Tiling），再到寄存器分块（Thread Tiling）的完整演进过程。

[⬅️ 返回项目主目录](../README.md)

## 📑 实验报告

### 📊 终极总结（推荐阅读）
- [SGEMM 深度优化与架构剖析总报告](./SGEMM-Total.md) 
  *(记录了 V1 到 V3 的完整性能对比，深度剖析了 L1 缓存反直觉现象、同步跷跷板效应，以及最终突破 2.12x 加速比的寄存器分块原理)*

### 📘 过程记录
- [V1：基础实现与 16x16 分块初探](./SGEMM-V1-report.md) 
  *(跑通 Naive 和 Tiled，验证 2048 规模下的基础表现)*
- [V2：性能瓶颈分析与排坑记录](./SGEMM-V2-report.md) 
  *(记录了缓存机制反直觉现象及 TILE_SIZE=64 的硬件物理死线)*
- [V3：寄存器分块(Thread Tiling)实验记录](./SGEMM-V3.md)
  *(记录了 2x2 寄存器分块的具体推导与实现)*

## 💻 源码文件

- [V1: Naive 与 Tiled 16x16 基础版](./SGEMM-V1.cu)
- [V2: Tiled 32x32 + Padding 进阶版](./SGEMM-V2-tiled_32.cu)
- [V3: 2x2 寄存器分块 (Thread Tiling) 终极版](./SGEMM-V3-thread_tiling.cu)