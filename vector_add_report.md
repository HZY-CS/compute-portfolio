# CUDA vector_add 控制变量实验报告

## 一、实验环境
- 平台：AutoDL 云GPU
- GPU型号：NVIDIA GeForce RTX 4080 SUPER
- 编译命令：`nvcc -o vector_add vector_add.cu`
- 运行命令：`./vector_add`

## 二、实验假设
改变数据规模（numElements）和线程块大小（threadsPerBlock），会改变网格大小（blocksPerGrid），但结果应始终正确（Test PASSED）。

## 三、实验设计与数据记录

| 实验编号 | 数据量 (numElements) | 每块线程数 (threadsPerBlock) | 网格块数 (blocksPerGrid) | 验证结果 |
| :---: | :---: | :---: | :---: | :---: |
| Baseline | 50000 | 256 | 196 | Test PASSED |
| 实验A (数据量↑) | 1000000 | 256 | 3907 | Test PASSED |
| 实验B (线程数↑) | 1000000 | 512 | 1954 | Test PASSED |

## 四、结果分析与结论
1. **公式推导**：`blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock`。
2. **实验A结论**：当数据量增大（5万 → 100万），每个班级人数不变的情况下，总活儿变多，因此所需的班级总数（blocks）也按比例增大（196 → 3907）。
3. **实验B结论**：在数据量（100万）不变的情况下，将每个班级的线程数增大（256 → 512），每个班能容纳更多的线程，因此所需的班级总数（blocks）减半（3907 → 1954）。
4. **全局验证**：所有实验均输出 `Test PASSED`，说明该核函数具备良好的**可扩展性**，能处理不同规模的数据，且逻辑正确。

## 五、下一步计划
在代码中引入 `cudaEvent` 计时，量化对比 CPU 与 GPU 的计算耗时，计算加速比（Speedup）。





# CUDA vector_add 性能量化与加速比分析报告

## 一、实验环境
- 平台：AutoDL 云GPU
- GPU型号：NVIDIA GeForce RTX 4080 SUPER
- CUDA版本：12.x
- 编译命令：`nvcc -o vector_add_test vector_add_test.cu`
- 数据规模：1,000,000 个 float 元素（约 3.8 MB 数据量）
- 测试方法：CPU（主机端）循环计算 10 次取平均，GPU kernel 运行 20 次取平均。

## 二、实验数据记录

| 测试项 | 耗时 (ms) | 说明 |
| :--- | :--- | :--- |
| CPU 纯计算耗时 | 5.20842 | 主机端完成 100 万次浮点加法 |
| GPU 纯 Kernel 耗时 | 0.0106544 | 仅核函数执行时间，不含内存拷贝 |
| GPU 总耗时（含拷贝） | 1.99443 | 包含 Host->Device 和 Device->Host 的数据拷贝 |
| 网格块数 (blocks) | 3907 | 100万 / 256 |
| 每块线程数 (threads) | 256 | 固定值 |

## 三、加速比计算
- **计算加速比（纯 Kernel）**：CPU 耗时 / GPU Kernel 耗时 = 5.20842 / 0.0106544 ≈ **488.85x**
- **端到端加速比（含拷贝）**：CPU 耗时 / GPU 总耗时 = 5.20842 / 1.99443 ≈ **2.61x**

## 四、现象分析与思考

1. **为什么纯 Kernel 能快 488 倍？**
   这证明了 GPU 的吞吐量极其恐怖。CPU 相当于博士生（单线程，算得快但只能一次算一个），而 GPU 同时调度了 3907 个班级（每个班级 256 个线程），将近 100 万个线程并行计算，瞬间就完成了任务。这显示了**大规模并行计算**在无依赖简单运算上的绝对优势。

2. **为什么端到端加速比骤降至 2.61 倍？**
   瓶颈在 **PCIe 数据搬运**。CPU 5.2 毫秒算完，而 GPU 算完只要 0.01 毫秒，但数据从内存搬运到显存（Host->Device）、再从显存搬回内存（Device->Host），消耗了超过 1.9 毫秒，直接吞掉了 99% 的纯计算优势。
   这正好印证了 CUDA 优化的黄金法则：“**计算是廉价的，搬运是昂贵的**”。

3. **这对以后写复杂算子的启示**：
   - 不能只看 Kernel 的加速比，必须看**端到端时间**。
   - 未来写 GEMM 或 Attention 时，必须使用 **Shared Memory (共享内存)** 做 Tiling（分块），让数据搬到显卡后反复使用，避免频繁访问全局显存（Global Memory），从而掩盖搬运延迟。

## 五、下一步计划
1. 在 Kernel 内引入 `__syncthreads()`，测试 Shared Memory 对数据复用率的影响。
2. 使用 `cudaMemcpyAsync` 和 Stream（多流）实现计算与数据传输的重叠（Overlap），尝试缩小端到端时间。
3. 引入 Roofline 模型计算此 kernel 的算术强度（Arithmetic Intensity），证明其属于 Memory-Bound（内存受限）型算子。

## 六、实验结论
- 纯 Kernel 计算：**GPU 完胜，加速比 488x**。
- 端到端性能：**受限于 PCIe 拷贝，加速比 2.6x**。
- 核心结论：**算力不是瓶颈，数据搬运才是瓶颈。未来的优化重点在于提高数据复用率（Tiling/Shared Memory）和减少数据搬运（Kernel Fusion）。**