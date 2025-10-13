# Qibo量子计算框架完整测试环境

这是一个基于Qibo的完整量子计算框架测试环境，集成了QASMBench基准测试集，提供全面的性能测试、正确性验证和高级分析功能。

## 🚀 项目概述

本项目为Qibo量子计算框架提供了完整的测试和基准测试解决方案，支持：

- **QASMBench集成**: 完整的量子电路基准测试集（small/medium/large规模）
- **多后端测试**: 支持所有Qibo后端的性能对比和正确性验证
- **高级指标测量**: 包含执行时间、内存占用、JIT编译、吞吐率等关键指标
- **多格式报告**: 自动生成CSV、Markdown、JSON格式的详细报告
- **问题诊断**: 专门的后端问题分析和解决方案

## 📁 完整项目结构

```
qiboenv/
├── qibovenv/                          # Python虚拟环境
├── QASMBench/                         # QASMBench量子电路基准测试集
│   ├── small/                         # 小规模电路 (2-10量子比特)
│   ├── medium/                        # 中等规模电路 (11-30量子比特)
│   ├── large/                         # 大规模电路 (30+量子比特)
│   ├── metrics/                       # 电路度量工具
│   └── interface/                     # Qiskit接口
├── qibobench/                         # 主要测试工具目录
│   ├── qasmbench_runner.py            # 🎯 QASMBench通用基准测试工具
│   ├── example_usage.py               # 使用示例脚本
│   ├── example_usage.ipynb            # Jupyter notebook示例
│   ├── USAGE_GUIDE.md                 # 详细使用指南
│   ├── QASMBench_Runner_Technical_Report.md # 技术报告
│   ├── qasmbench_runner_backend_selection.py # 后端选择工具
│   ├── test_qasmbench_runner.py       # 测试脚本
│   ├── qft/                           # QFT专用测试框架
│   │   ├── benchmark_advanced.py      # 高级基准测试框架
│   │   ├── run_qft_18.py             # QFT-18专用测试
│   │   └── advanced_benchmark_report.* # 高级测试报告
│   ├── reports/                       # 📊 自动生成的报告目录
│   │   ├── [circuit_name]/           # 每个电路的专属报告
│   │   │   ├── benchmark_report.csv   # CSV格式报告
│   │   │   ├── benchmark_report.md    # Markdown格式报告
│   │   │   ├── benchmark_report.json  # JSON格式报告
│   │   │   └── [circuit_name]_diagram.png # 电路图
│   │   └── ...
│   └── qibobench/                     # 内部工具目录
├── test/                              # 测试脚本和文档
│   ├── strict_validation_test.py      # 严格正确性验证
│   ├── pytorch_backend_analysis.py    # PyTorch后端分析
│   ├── qibotn_warning_test.py         # qibotn警告分析
│   ├── final_backend_test.py          # 最终功能测试
│   ├── backend_test_summary.md        # 测试总结报告
│   ├── qibotn_warning_analysis.md     # 警告分析报告
│   └── test_backends.*               # 基础后端测试
├── results/                           # 测试结果存储
├── requirements.txt                   # Python依赖包
└── README.md                          # 项目说明文档
```

## ✨ 核心功能特性

### 🎯 QASMBench通用基准测试工具
- **全面电路支持**: 自动发现并测试QASMBench中的所有电路
- **智能文件选择**: 优先使用transpiled版本避免兼容性问题
- **多后端并行测试**: 一次运行测试所有可用后端
- **详细性能指标**: 执行时间、内存使用、加速比、吞吐率
- **正确性验证**: 状态向量对比验证计算结果准确性

### 🔬 高级基准测试框架
- **优先级指标设计**: 按照重要性排序的核心指标测量
- **JIT编译分析**: 测量即时编译开销和优化效果
- **内存精确测量**: 峰值内存使用和内存效率分析
- **环境信息记录**: 完整的测试环境和硬件配置

### 📊 多格式报告生成
- **CSV报告**: 便于数据分析和处理
- **Markdown报告**: 便于阅读和分享，包含图表和分析
- **JSON报告**: 便于程序化处理和API集成
- **电路图生成**: 自动生成电路结构图

### 🛠️ 问题诊断工具
- **后端兼容性分析**: 识别和解决后端特定问题
- **性能瓶颈诊断**: 定位性能问题和优化建议
- **数据类型处理**: 自动处理不同框架的数据转换

## 🔧 支持的Qibo后端

### 基本后端
| 后端 | 描述 | 性能特点 | 适用场景 |
|------|------|----------|----------|
| **numpy** | 基础后端 | 性能基准，兼容性最好 | 基准测试，简单电路 |
| **qibojit (numba)** | JIT编译后端 | 🥇 **性能最优** (20x+加速) | 性能优先场景 |
| **qibotn (qutensornet)** | 张量网络后端 | 🥇 **内存效率最高** | 大规模电路，内存敏感 |
| **clifford** | Clifford电路专用 | Clifford电路特化 | 特定算法优化 |
| **hamming_weight** | 汉明权重专用 | 特定计算模式 | 特定问题域 |

### 机器学习后端
| 后端 | 描述 | 性能特点 | 数据处理 |
|------|------|----------|----------|
| **qiboml (jax)** | 基于JAX | GPU加速，自动微分 | 需要转换 |
| **qiboml (pytorch)** | 基于PyTorch | 深度学习集成 | 需要`detach().cpu().numpy()` |
| **qiboml (tensorflow)** | 基于TensorFlow | 生产环境部署 | 需要转换 |

### 性能特性对比
| 后端 | 性能排名 | 内存效率 | 数据类型 | 特殊处理 |
|------|----------|----------|----------|----------|
| qibojit | 🥇 1st | ⭐⭐⭐⭐ | numpy数组 | 直接使用 |
| qibotn | 🥈 2nd | ⭐⭐⭐⭐⭐ | numpy数组 | 直接使用 |
| qiboml (pytorch) | 🥉 3rd | ⭐⭐ | torch.Tensor | 需要转换 |
| qiboml (jax) | 4th | ⭐⭐⭐ | jax.Array | 需要转换 |
| numpy | 5th | ⭐⭐⭐⭐ | numpy数组 | 基准后端 |
| qiboml (tensorflow) | 6th | ⭐⭐⭐ | tf.Tensor | 需要转换 |

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活虚拟环境
.\qibovenv\Scripts\Activate.ps1

# 安装依赖（如果需要）
pip install -r requirements.txt
```

### 2. 列出可用电路
```bash
cd qibobench
python qasmbench_runner.py --list
```

### 3. 测试特定电路
```bash
# 通过完整路径测试
python qasmbench_runner.py --circuit QASMBench/medium/qft_n18/qft_n18_transpiled.qasm

# 通过电路名称测试（自动查找文件）
python qasmbench_runner.py --circuit medium/qft_n18
```

### 4. 运行高级基准测试
```bash
python qibobench/qft/benchmark_advanced.py
```

### 5. 查看使用示例
```bash
python qibobench/example_usage.py
```

## 📖 详细使用示例

### 示例1: 测试QASMBench电路
```bash
# 列出所有可用电路
cd qibobench
python qasmbench_runner.py --list

# 测试小规模电路
python qasmbench_runner.py --circuit QASMBench/small/adder_n10/adder_n10_transpiled.qasm

# 测试中等规模电路
python qasmbench_runner.py --circuit QASMBench/medium/qft_n18/qft_n18_transpiled.qasm

# 测试大规模电路（需要更多时间和内存）
python qasmbench_runner.py --circuit QASMBench/large/qft_n63/qft_n63_transpiled.qasm
```

### 示例2: Python代码使用
```python
from qasmbench_runner import list_available_circuits, run_benchmark_for_circuit

# 列出所有电路
circuits = list_available_circuits()

# 测试特定电路
results = run_benchmark_for_circuit("QASMBench/medium/qft_n18/qft_n18_transpiled.qasm")

# 查看结果
for backend, metrics in results.items():
    print(f"{backend}: {metrics.execution_time_mean:.4f}s")
```

### 示例3: 自定义配置测试
```python
from qasmbench_runner import QASMBenchConfig, QASMBenchRunner

# 自定义配置
config = QASMBenchConfig()
config.num_runs = 3           # 减少运行次数
config.warmup_runs = 1        # 预热次数
config.output_formats = ['markdown']  # 只生成Markdown报告

# 运行测试
runner = QASMBenchRunner(config)
results = runner.run_benchmark_for_circuit("my_circuit", "path/to/circuit.qasm")
runner.generate_reports(results, "my_circuit")
```

### 示例4: 批量测试
```python
# 批量测试多个电路
circuits_to_test = [
    "QASMBench/small/adder_n10/adder_n10_transpiled.qasm",
    "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm",
    "QASMBench/small/qft_n4/qft_n4_transpiled.qasm"
]

for circuit_path in circuits_to_test:
    print(f"测试电路: {circuit_path}")
    results = run_benchmark_for_circuit(circuit_path)
```

## 📊 性能指标说明

### 🔴 核心指标（最高优先级）
- **执行时间（均值 ± 标准差）**: 最重要的性能指标
- **峰值内存占用**: 最重要的资源使用指标
- **正确性验证**: 计算结果准确性检查

### 🟡 高优先级指标
- **加速比**: 相对于numpy基准的性能提升
- **电路参数**: 量子比特数、深度、门数量
- **后端信息**: 后端类型和平台信息

### 🟢 中优先级指标
- **吞吐率**: 每秒处理的门操作数量
- **JIT编译时间**: 即时编译开销（适用于JIT后端）
- **环境信息**: 测试环境的硬件和软件配置

### 🔵 低优先级指标
- **电路构建时间**: 电路对象创建时间
- **报告元数据**: 测试配置和状态信息

## 🎯 测试结果示例

### QFT-18电路测试结果（18量子比特，820个门）
| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| qibojit (numba) | 0.383 ± 0.067 | 0.0 | 21.2x | ✅ Passed | 2,141 |
| qibotn (qutensornet) | 1.048 ± 0.025 | 0.6 | 7.7x | ✅ Passed | 783 |
| qiboml (pytorch) | 2.812 ± 0.276 | 1734.7 | 2.9x | ✅ Passed | 292 |
| qiboml (jax) | 3.413 ± 0.037 | 7.6 | 2.4x | ✅ Passed | 240 |
| numpy | 8.104 ± 0.102 | 0.0 | N/A | ✅ Passed | 101 |
| qiboml (tensorflow) | 21.698 ± 1.632 | 8.0 | 0.4x | ✅ Passed | 38 |

### 关键发现
- **🏆 性能最优**: qibojit (numba) - 21.2倍加速，几乎零内存开销
- **💾 内存最优**: qibotn (qutensornet) - 仅0.6MB内存占用，适合大规模电路
- **🔬 正确性**: 所有后端计算结果与numpy基准完全一致
- **⚠️ 特殊处理**: PyTorch后端需要`detach().cpu().numpy()`数据转换

## 💡 使用建议

### 性能优先场景
```python
from qibo import set_backend
set_backend("qibojit", platform="numba")
# 适用：大规模计算，性能敏感应用
```

### 内存敏感场景
```python
set_backend("qibotn", platform="qutensornet")
# 适用：内存受限环境，大规模量子电路
```

### 机器学习集成
```python
# JAX集成（推荐）
set_backend("qiboml", platform="jax")

# PyTorch集成（注意数据转换）
set_backend("qiboml", platform="pytorch")
result = circuit()
state = result.state().detach().cpu().numpy()  # 重要：数据类型转换
```

### 基准测试和验证
```python
set_backend("numpy")  # 作为性能基准和正确性参考
```

## 🔧 支持的QASMBench电路类型

### Small规模电路（2-10量子比特）
- **基础算法**: QFT, Grover搜索, 量子相位估计
- **算术电路**: 加法器, 乘法器, 平方根
- **量子协议**: 量子传态, BB84, 密钥分发
- **变分算法**: VQE, QAOA小规模实例

### Medium规模电路（11-30量子比特）
- **实用算法**: HHL线性求解器, 量子RAM
- **机器学习**: k-NN, 神经网络, 量子GAN
- **优化算法**: QAOA中等规模, VQE分子模拟
- **错误校正**: 量子纠错码小实例

### Large规模电路（30+量子比特）
- **大规模优化**: QAOA大规模实例
- **量子化学**: VQE分子模拟大体系
- **量子优势演示**: 量子体积, 随机电路
- **复杂算法**: Shor算法大数分解

## 🛠️ 故障排除

### 常见问题

#### 1. "找不到电路文件"错误
```bash
# 解决方案：检查QASMBench目录结构
python qasmbench_runner.py --list  # 确认电路是否存在
```

#### 2. "导入错误"问题
```bash
# 解决方案：确保依赖完整安装
pip install qibo qibojit qiboml qibotn
pip install numpy torch jax tensorflow
```

#### 3. 内存不足错误
```bash
# 解决方案：使用内存高效的后端
set_backend("qibotn", platform="qutensornet")
# 或减少运行次数
config.num_runs = 1
```

#### 4. JIT编译警告
```bash
# 解决方案：增加预热次数
config.warmup_runs = 2  # 确保JIT编译完成
```

### 性能优化建议

#### 1. 选择合适的后端
- **性能敏感**: 使用qibojit (numba)
- **内存受限**: 使用qibotn (qutensornet)
- **ML集成**: 使用qiboml (jax)

#### 2. 优化测试配置
- **快速测试**: 减少运行次数`config.num_runs = 1`
- **精确测试**: 增加运行次数`config.num_runs = 10`
- **避免JIT影响**: 增加预热次数`config.warmup_runs = 2`

#### 3. 硬件优化
- **CPU密集**: 使用更多CPU核心
- **GPU加速**: 使用qiboml (pytorch/jax)配合GPU
- **内存优化**: 监控内存使用，选择合适的电路规模

## 📋 环境要求

### 系统要求
- **操作系统**: Windows 10+, Linux, macOS
- **Python**: 3.8+ (推荐3.9+)
- **内存**: 8GB+ (大规模测试需要16GB+)
- **存储**: 5GB+ 可用空间

### 核心依赖
```
qibo==0.2.21              # 量子计算框架
qibojit==0.1.12           # JIT编译后端
qiboml==0.0.2             # 机器学习后端
qibotn==0.0.3             # 张量网络后端
numpy>=2.0.0              # 数值计算
torch>=2.0.0              # PyTorch支持
jax>=0.4.0                # JAX支持
tensorflow>=2.0.0         # TensorFlow支持
psutil>=5.0.0             # 系统监控
```

## 📄 许可证

本项目基于QASMBench基准测试集和Qibo量子计算框架，遵循相应的开源许可证。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个测试框架：

1. **Bug报告**: 请提供详细的错误信息和复现步骤
2. **功能请求**: 描述新功能的用途和预期行为
3. **代码贡献**: 确保代码质量，添加适当的测试和文档

## 📞 技术支持

- **文档**: 查看`qibobench/USAGE_GUIDE.md`获取详细使用指南
- **示例**: 运行`qibobench/example_usage.py`查看使用示例
- **技术报告**: 参考`qibobench/QASMBench_Runner_Technical_Report.md`

---

**最后更新**: 2025-10-13  
**项目版本**: 3.0 (完整版)  
**Qibo版本**: 0.2.21  
**QASMBench版本**: Latest

🚀 **开始您的量子计算基准测试之旅！**
