# run_benchmarks.py 使用指南

## 概述

`run_benchmarks.py` 是一个量子模拟器基准测试运行器，用于比较不同量子计算框架的性能。它支持多种量子电路和模拟器的性能比较，包括执行时间、内存使用、CPU利用率和状态保真度等指标的测量。

## 功能特点

- 支持多种量子电路（如QFT、Grover）的基准测试
- 支持多个量子计算框架的性能比较（Qibo、Qiskit、PennyLane等）
- 自动生成参考态用于保真度计算
- 详细的性能指标收集和分析
- 自动生成可视化报告和摘要
- 支持缓存机制以提高测试效率
- 支持多次运行和统计分析

## 系统要求

- Python 3.12.0 或更高版本
- 依赖库：pandas, numpy, matplotlib, seaborn, psutil
- 至少一个量子计算框架（Qibo、Qiskit或PennyLane）

## 安装和设置

### 基本安装

```bash
# 1. 创建虚拟环境
conda create -n qibo-benchmark python=3.12 -y
conda activate qibo-benchmark

# 2. 安装基本依赖
pip install pandas numpy matplotlib seaborn psutil

# 3. 安装量子计算框架（以Qibo为例）
pip install qibo
```

### 多框架安装

```bash
# 安装Qiskit
pip install qiskit

# 安装PennyLane
pip install pennylane
```

## 使用方法

### 基本语法

```bash
python run_benchmarks.py [选项]
```

### 命令行参数

#### 电路选择参数

- `--circuits`: 指定要运行的基准测试电路列表
  - 可选值：`qft`, `grover`
  - 默认值：`["qft"]`
  - 示例：`--circuits qft grover`

#### 量子比特数参数

- `--qubits`: 指定要测试的量子比特数列表
  - 类型：整数列表
  - 默认值：`[2, 3, 4]`
  - 示例：`--qubits 2 3 4 5 6`

#### 模拟器选择参数

- `--simulators`: 指定要测试的模拟器列表，格式为"platform-backend"
  - 默认值：`["qibo-qibojit"]`
  - 示例：`--simulators qibo-numpy qibo-qibojit qiskit-aer_simulator`

#### 黄金标准参数

- `--golden-standard`: 用于生成参考态的模拟器
  - 默认值：`"qibo-qibojit"`
  - 示例：`--golden-standard qibo-numpy`

#### 输出目录参数

- `--output-dir`: 结果输出目录
  - 默认值：`"results"`
  - 示例：`--output-dir my_results`

#### 详细输出参数

- `--verbose`: 启用详细输出模式
  - 类型：布尔标志
  - 示例：`--verbose`

#### 缓存相关参数

- `--enable-cache`: 启用参考态缓存（默认启用）
- `--no-cache`: 禁用缓存（覆盖--enable-cache）
- `--cache-type`: 缓存类型选择
  - 可选值：`memory`, `disk`, `hybrid`
  - 默认值：`"hybrid"`
- `--cache-dir`: 磁盘缓存目录
  - 默认值：`".benchmark_cache"`
- `--memory-cache-size`: 内存缓存最大条目数
  - 默认值：`64`
- `--clear-cache`: 开始前清空缓存
- `--cache-stats`: 显示缓存统计信息

#### 重复运行参数

- `--repeat`: 每个电路重复运行的次数
  - 默认值：`1`
  - 示例：`--repeat 5`
- `--warmup-runs`: 正式测量前的预热运行次数
  - 默认值：`0`
  - 示例：`--warmup-runs 2`
- `--statistical-analysis`: 启用统计分析，计算标准差、置信区间等

## 使用示例

### 基本用法

```bash
# 使用默认设置运行基准测试
python run_benchmarks.py

# 指定量子比特数和启用详细输出
python run_benchmarks.py --qubits 2 3 4 5 --verbose
```

### 多框架比较

```bash
# 比较Qibo和Qiskit框架
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator --qubits 2 3 4 --verbose

# 比较多个后端
python run_benchmarks.py --simulators qibo-numpy qibo-qibojit qiskit-aer_simulator pennylane-lightning.qubit --qubits 2 3 4
```

### 高级用法

```bash
# 多次运行并启用统计分析
python run_benchmarks.py --repeat 5 --warmup-runs 2 --statistical-analysis --verbose

# 使用自定义缓存设置
python run_benchmarks.py --cache-type disk --cache-dir ./my_cache --clear-cache --cache-stats

# 测试多种电路
python run_benchmarks.py --circuits qft grover --qubits 2 3 4 --simulators qibo-numpy qibo-qibojit
```

## 输入说明

### 电路类型

#### 量子傅里叶变换 (QFT)

- 名称：`qft`
- 描述：实现量子傅里叶变换电路，用于测试量子算法的基本性能
- 复杂度：O(n²) 量子门，其中n是量子比特数

#### Grover搜索算法

- 名称：`grover`
- 描述：实现Grover量子搜索算法，用于测试搜索类算法的性能
- 复杂度：O(√N) 量子门，其中N=2ⁿ是搜索空间大小

### 模拟器配置

模拟器配置格式为"platform-backend"，支持的平台包括：

#### Qibo平台

- `qibo-numpy`: 使用NumPy后端的Qibo模拟器
- `qibo-qibojit`: 使用JIT编译的Qibo模拟器（推荐）
- `qibo-qiboml_jax`: 使用JAX后端的Qibo机器学习模块
- `qibo-qiboml_pytorch`: 使用PyTorch后端的Qibo机器学习模块
- `qibo-qiboml_tensorflow`: 使用TensorFlow后端的Qibo机器学习模块

#### Qiskit平台

- `qiskit-aer_simulator`: Qiskit Aer模拟器
- `qiskit-statevector_simulator`: Qiskit状态向量模拟器

#### PennyLane平台

- `pennylane-default.qubit`: PennyLane默认量子模拟器
- `pennylane-lightning.qubit`: PennyLane高性能量子模拟器

## 输出说明

### 目录结构

基准测试结果保存在指定输出目录下的时间戳子目录中：

```
results/
└── benchmark_YYYYMMDD_HHMMSS/
    ├── raw_results.csv          # 原始结果数据
    ├── detailed_runs.csv         # 详细运行数据（当repeat>1时）
    ├── summary_report.md         # 摘要报告
    ├── wall_time_scaling.png     # 执行时间扩展性图
    ├── cpu_time_scaling.png      # CPU时间扩展性图
    ├── memory_scaling.png        # 内存使用扩展性图
    ├── cpu_utilization.png       # CPU利用率图
    ├── fidelity.png              # 保真度比较图
    ├── execution_stability.png   # 执行稳定性图（当repeat>1时）
    └── confidence_intervals.png  # 置信区间图（当启用统计分析时）
```

### 数据文件

#### raw_results.csv

包含所有基准测试结果的原始数据，主要字段：

- `simulator`: 模拟器名称
- `backend`: 后端名称
- `circuit_name`: 电路名称
- `n_qubits`: 量子比特数
- `wall_time_sec`: 墙上时钟时间（秒）
- `cpu_time_sec`: CPU时间（秒）
- `peak_memory_mb`: 峰值内存使用（MB）
- `cpu_utilization_percent`: CPU利用率（百分比）
- `state_fidelity`: 状态保真度
- `circuit_depth`: 电路深度
- `total_gates`: 总门数

#### detailed_runs.csv

当`--repeat > 1`时生成，包含每次运行的详细数据，额外字段：

- `run_id`: 运行ID
- `wall_time_mean`: 平均墙上时钟时间
- `wall_time_std`: 墙上时钟时间标准差
- `wall_time_min`: 最小墙上时钟时间
- `wall_time_max`: 最大墙上时钟时间
- `cpu_time_mean`: 平均CPU时间
- `cpu_time_std`: CPU时间标准差
- `memory_mean`: 平均内存使用
- `memory_std`: 内存使用标准差
- `fidelity_mean`: 平均保真度
- `fidelity_std`: 保真度标准差
- `confidence_interval`: 95%置信区间

### 可视化图表

#### 扩展性图表

- `wall_time_scaling.png`: 显示执行时间随量子比特数的变化
- `cpu_time_scaling.png`: 显示CPU时间随量子比特数的变化
- `memory_scaling.png`: 显示内存使用随量子比特数的变化

#### 性能比较图表

- `cpu_utilization.png`: 显示不同模拟器的CPU利用率
- `fidelity.png`: 显示不同模拟器的状态保真度比较

#### 统计分析图表

- `execution_stability.png`: 显示多次运行的执行稳定性（变异系数）
- `confidence_intervals.png`: 显示性能指标的置信区间

### 摘要报告

`summary_report.md` 包含以下内容：

1. **基本统计信息**
   - 总测试次数
   - 测试的模拟器和电路
   - 量子比特数范围

2. **电路信息**
   - 电路复杂度（深度、门数）
   - 电路摘要

3. **性能指标**
   - 最快执行
   - 内存使用最少
   - 平均保真度排名

4. **稳定性分析**
   - 最稳定执行
   - 最不稳定执行

5. **扩展性分析**
   - 执行时间随量子比特数的变化

6. **建议**
   - 基于测试结果的模拟器选择建议

## 最佳实践

### 测试设计

1. **从小规模开始**：先测试2-4个量子比特，确保系统正常工作
2. **逐步增加规模**：根据系统性能逐步增加量子比特数
3. **多次运行**：对于关键测试，使用`--repeat`参数进行多次运行
4. **预热运行**：使用`--warmup-runs`进行预热，减少初始化影响

### 性能优化

1. **使用缓存**：启用缓存可以显著减少重复计算时间
2. **选择合适的后端**：根据硬件选择最优的后端（如GPU加速）
3. **资源监控**：监控系统资源使用，避免资源竞争

### 结果分析

1. **关注扩展性**：重点关注性能随量子比特数的变化趋势
2. **平衡指标**：综合考虑执行时间、内存使用和保真度
3. **统计分析**：使用统计分析评估结果的可靠性

## 常见问题

### 问题1：ImportError: No module named 'qibo'

**解决方案**：
```bash
# 确保已激活正确环境
conda activate qibo-benchmark

# 重新安装Qibo
pip install qibo
```

### 问题2：内存不足

**解决方案**：
```bash
# 减少量子比特数
python run_benchmarks.py --qubits 2 3

# 或使用更小的电路
python run_benchmarks.py --circuits qft --qubits 2
```

### 问题3：模拟器初始化失败

**解决方案**：
```bash
# 检查模拟器名称是否正确
python run_benchmarks.py --simulators qibo-numpy --verbose

# 确保相应的框架已安装
pip install qiskit  # 对于Qiskit
pip install pennylane  # 对于PennyLane
```

### 问题4：图表不显示

**解决方案**：
```bash
# 安装matplotlib后端
pip install matplotlib tkinter

# 或在Jupyter中运行
jupyter notebook
```

## 扩展和自定义

### 添加新电路

1. 在`benchmark_harness/circuits/`目录下创建新电路模块
2. 实现`BenchmarkCircuit`接口
3. 在`run_benchmarks.py`中注册新电路

### 添加新模拟器

1. 在`benchmark_harness/simulators/`目录下创建新模拟器包装器
2. 实现`SimulatorInterface`接口
3. 在`simulators/__init__.py`中导入新包装器

### 自定义指标

1. 修改`BenchmarkResult`数据结构
2. 更新模拟器包装器以收集新指标
3. 修改后处理代码以处理新指标

## 参考资料

- [Qibo文档](https://qibo.science/)
- [Qiskit文档](https://qiskit.org/)
- [PennyLane文档](https://pennylane.ai/)
- [基准测试框架源码](../benchmark_harness/)