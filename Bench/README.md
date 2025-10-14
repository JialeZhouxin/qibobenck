# 量子模拟器基准测试平台

一个模块化、可扩展的基准测试平台，用于系统性地评估Qibo、Qiskit、PennyLane等量子计算框架的性能。

## 项目概述

本项目旨在通过严谨的横向对比，识别Qibo量子模拟器在不同后端下的性能瓶颈，并为未来的优化工作提供可量化的数据支持。

### 主要功能

- **多框架支持**：支持Qibo、Qiskit、PennyLane等主流量子计算框架
- **多后端测试**：每个框架支持多种后端配置（如Qibo的numpy、qibojit等）
- **全面性能指标**：测量执行时间、内存使用、CPU利用率等关键指标
- **正确性验证**：通过保真度计算验证模拟结果的正确性
- **可视化分析**：生成直观的图表和报告，便于性能对比分析
- **模块化设计**：易于扩展新的量子计算框架和基准电路
- **环境隔离**：支持为不同框架创建专用conda环境，避免依赖冲突

## 安装与设置

### 环境要求

- Python 3.12.0 或更高版本
- Conda 或 Miniconda（推荐用于多框架支持）

### 快速开始

🚀 **新用户？** 查看[快速开始指南](QUICK_START.md)获取5分钟快速安装教程。

### 安装选项

#### 选项1：仅Qibo框架（推荐新手）

```bash
git clone <repository-url>
cd Bench
conda create -n qibo-benchmark python=3.12 -y
conda activate qibo-benchmark
pip install -r requirements.txt
```

#### 选项2：多框架环境（高级用户）

对于需要同时测试多个框架的用户，我们强烈建议使用专用环境以避免依赖冲突：

```bash
# 创建Qiskit专用环境
conda env create -f environment-qiskit.yml
conda activate qibo-benchmark-qiskit

# 创建PennyLane专用环境
conda env create -f environment-pennylane.yml
conda activate qibo-benchmark-pennylane
```

📖 **详细指南**：查看[多环境使用指南](MULTI_ENVIRONMENT_GUIDE.md)了解完整的环境配置和管理。

### 依赖包说明

- **量子计算框架**：qibo, qiskit, pennylane
- **数据处理**：numpy, pandas, matplotlib, seaborn
- **系统监控**：psutil
- **开发工具**：pytest, black, flake8, isort

### 多框架环境管理

为了确保不同量子计算框架之间的隔离性，我们提供了专门的环境配置：

| 环境名称 | 用途 | 包含框架 |
|---------|------|---------|
| `qibo-benchmark-qibo` | 仅Qibo测试 | Qibo |
| `qibo-benchmark-qiskit` | Qibo + Qiskit测试 | Qibo, Qiskit |
| `qibo-benchmark-pennylane` | Qibo + PennyLane测试 | Qibo, PennyLane |

📖 **详细指南**：查看[多环境使用指南](MULTI_ENVIRONMENT_GUIDE.md)了解完整的环境配置和管理。

## 使用方法

### 基本用法

运行基准测试的最简单方式：

```bash
# 在Qibo环境中
python run_benchmarks.py

# 在Qiskit环境中
conda activate qibo-benchmark-qiskit
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator

# 在PennyLane环境中
conda activate qibo-benchmark-pennylane
python run_benchmarks.py --simulators qibo-numpy pennylane-default.qubit
```

这将使用默认设置运行QFT电路在指定后端上的基准测试。

### 命令行参数

`run_benchmarks.py`支持多种命令行参数来自定义测试：

```bash
python run_benchmarks.py [选项]
```

#### 可用选项

- `--circuits`: 指定要测试的电路列表（默认：["qft"]）
- `--qubits`: 指定要测试的量子比特数列表（默认：[2, 3, 4]）
- `--simulators`: 指定要测试的模拟器列表，格式为platform-backend（默认：["qibo-numpy"]）
- `--golden-standard`: 指定用于生成参考态的模拟器（默认："qibo-numpy"）
- `--output-dir`: 指定结果输出目录（默认："results"）
- `--verbose`: 启用详细输出

#### 示例命令

1. 测试多个量子比特数：
```bash
python run_benchmarks.py --qubits 2 3 4 5
```

2. 测试多个模拟器：
```bash
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator
```

3. 完整测试示例：
```bash
python run_benchmarks.py \
  --circuits qft \
  --qubits 2 3 4 5 \
  --simulators qibo-numpy qiskit-aer_simulator \
  --golden-standard qibo-numpy \
  --output-dir my_results \
  --verbose
```

### 输出结果

基准测试完成后，结果将保存在指定的时间戳目录中，包含以下文件：

- `raw_results.csv`: 原始测试数据
- `summary_report.md`: 摘要报告
- `fidelity.png`: 保真度对比图
- `wall_time_scaling.png`: 墙上时间扩展性图
- `memory_scaling.png`: 内存使用扩展性图
- `cpu_time_scaling.png`: CPU时间扩展性图
- `cpu_utilization.png`: CPU利用率对比图

## 项目结构

```
Bench/
├── benchmark_harness/          # 核心模块
│   ├── abstractions.py         # 抽象接口定义
│   ├── metrics.py              # 性能指标收集
│   ├── post_processing.py      # 结果后处理与可视化
│   ├── simulators/             # 模拟器封装
│   │   ├── __init__.py
│   │   ├── qibo_wrapper.py     # Qibo封装器
│   │   ├── qiskit_wrapper.py   # Qiskit封装器
│   │   └── pennylane_wrapper.py # PennyLane封装器
│   └── circuits/               # 基准电路
│       ├── __init__.py
│       └── qft.py              # 量子傅里叶变换电路
├── tests/                      # 测试文件
│   ├── test_abstractions.py
│   ├── test_metrics.py
│   ├── test_simulators.py
│   ├── test_integration.py
│   └── test_post_processing.py
├── results/                    # 测试结果目录
├── run_benchmarks.py           # 主运行脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 本文档
```

## 扩展开发

### 添加新的模拟器

1. 在`benchmark_harness/simulators/`目录下创建新的封装器文件
2. 实现`SimulatorInterface`接口
3. 在`__init__.py`中导入新封装器
4. 更新运行器以支持新模拟器

### 添加新的基准电路

1. 在`benchmark_harness/circuits/`目录下创建新电路文件
2. 实现`BenchmarkCircuit`接口
3. 在`__init__.py`中导入新电路类
4. 更新运行器的电路选择逻辑

## 测试

运行所有测试：

```bash
pytest tests/ -v
```

运行特定测试：

```bash
pytest tests/test_abstractions.py -v
```

## 代码质量

项目使用以下工具确保代码质量：

- **格式化**：black
- **导入排序**：isort
- **静态检查**：flake8

运行代码质量检查：

```bash
black .
isort .
flake8 .
```

## 性能指标说明

本平台测量以下性能指标：

- **墙上时间（wall_time_sec）**：实际经过的时间
- **CPU时间（cpu_time_sec）**：CPU实际执行时间
- **峰值内存（peak_memory_mb）**：测试过程中的最大内存使用量
- **CPU利用率（cpu_utilization_percent）**：CPU使用百分比
- **状态保真度（state_fidelity）**：与参考态的相似度

## 故障排除

### 常见问题

1. **导入错误**：确保所有依赖包已正确安装
2. **模拟器不可用**：某些模拟器可能需要额外安装步骤
3. **内存不足**：对于大型电路，可能需要更多内存
4. **环境冲突**：不同框架的依赖可能冲突，使用专用环境解决

### 调试技巧

1. 使用`--verbose`参数获取详细输出
2. 检查生成的日志文件
3. 从小规模测试开始，逐步增加复杂度
4. 验证环境隔离性：
   ```bash
   conda activate qibo-benchmark-qiskit
   python -c "import qiskit; print('Qiskit available')"
   python -c "import pennylane" 2>&1 | grep "No module named" && echo "Environment isolation working"
   ```

### 环境相关问题

如果遇到环境相关的问题，请参考：
- [多环境使用指南](MULTI_ENVIRONMENT_GUIDE.md)
- [快速开始指南](QUICK_START.md)

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 编写测试
4. 确保所有测试通过
5. 提交Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至项目维护者

## 致谢

感谢以下开源项目的支持：

- [Qibo](https://github.com/qiboteam/qibo)
- [Qiskit](https://github.com/Qiskit/qiskit)
- [PennyLane](https://github.com/PennyLaneAI/pennylane)