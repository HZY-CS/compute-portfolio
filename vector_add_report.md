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