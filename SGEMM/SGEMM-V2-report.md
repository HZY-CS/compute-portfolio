# CUDA SGEMM 矩阵乘法优化实验报告

[⬅️ 返回总项目主页](../README.md) | [💻 查看 V2 源码 (Naive + Tiled 32x32)](./SGEMM-V2-tiled_32.cu) | [⬅️ 返回SGEMM项目主页](./SGEMM-Total.md)

## 一、实验环境
- GPU: NVIDIA GeForce RTX 3090 (Ampere 架构)
- CUDA: 11.8
- 矩阵规模: M = N = K = 8192 (单精度浮点)
- 评估指标: 耗时 (ms)、GFLOPS、加速比

## 二、版本迭代与性能数据

| 版本 | 优化策略 | 耗时 (ms) | GFLOPS | 较 Naive 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| v1 | Naive (无分块，直接读全局内存) | 508.20 | 2163.53 | 1.00x |
| v2 | Tiled 16x16 (Shared Memory) | 513.11 | 2142.83 | 0.99x |
| v3 | Tiled 32x32 + Padding (消除 Bank Conflict) | 527.38 | 2084.85 | 0.96x |
| v4(测试) | Tiled 64x64 | 触发 Kernel 启动失败 | - | - |

## 三、反直觉现象深度分析

1. **为什么 v2 和 v3 的 Tiled 版本，反而比 Naive 慢？**
   - **L1/L2 缓存的胜利**：RTX 3090 具有极大的 L2 缓存（6MB）和优秀的 L1 缓存广播机制。Naive 版虽然理论访存量大，但相邻线程访问连续内存，硬件通过 Cache Line 预取，实际极大缓解了 HBM 带宽压力。Naive 版并没有被真正“卡死”。
   - **同步开销（跷跷板效应）**：K=8192，当 TILE_SIZE=16 时，每个 Block 需要执行 512 次 `__syncthreads()`；当 TILE_SIZE=32 时，需要执行 256 次。大 Block 虽然减少了同步次数，但 1024 个线程“齐步走”的单次对齐代价极高，且占满了 SM 资源，导致 Occupancy（占用率）降低，GPU 无法隐藏延迟。
   
2. **为什么 TILE_SIZE=64 时，跑出了 2000 亿的虚假 GFLOPS？**
   - Block 线程数达到 64x64 = 4096。NVIDIA GPU 规定单个 Block 最大只支持 1024 个线程。
   - 启动配置无效导致 Kernel 瞬间崩溃退出，`cudaEvent` 捕获到了近乎 0 的耗时。这暴露出缺少 `cudaGetLastError()` 错误检查的隐患。

## 四、核心结论与下一步计划
- **结论**：在单精度 SGEMM 中，单纯依赖 Shared Memory 在部分现代架构上收益有限。必须引入更高级的优化。
- **下一步 (v4 计划)**：
  1. 引入 **Thread Tiling（寄存器分块）**：例如使用 32x32 的 Block，但让每个线程计算 4 个 C 元素，大幅降低访存和同步频率。
  2. 引入 **Double Buffering（双缓冲）**：让数据搬运和计算并行（计算当前块的同时，预取下一个块）。