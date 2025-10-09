# Qibo 后端测试框架

这是一个完整的 Qibo 量子计算框架后端测试环境，包含全面的基准测试、正确性验证和性能分析功能。

## 项目结构

```
qiboenv/
├── qibovenv/                 # Python 虚拟环境
├── QASMBench/                # QASMBench 量子电路基准测试集
├── qibobench/qft/            # QFT基准测试框架
│   ├── run_qft_18.py         # 完整基准测试框架
│   ├── benchmark_advanced.py # 高级指标测试
│   ├── benchmark_report.*    # 多种格式基准报告
│   └── advanced_benchmark_report.* # 高级指标报告
├── test/                     # 测试脚本和文档
│   ├── strict_validation_test.py # 严格正确性验证
│   ├── pytorch_backend_analysis.py # PyTorch后端分析
│   ├── qibotn_warning_test.py     # qibotn警告分析
│   ├── final_backend_test.py      # 最终功能测试
│   ├── backend_test_summary.md    # 测试总结报告
│   ├── qibotn_warning_analysis.md # 警告分析报告
│   └── test_backends.*       # 基础后端测试脚本
├── requirements.txt          # Python 依赖包
└── README.md                 # 项目说明文档
```

## 功能特性

- **完整后端测试**: 支持所有 Qibo 可用后端的全面测试
- **高级基准测试**: 包含执行时间、内存占用、加速比等核心指标
- **正确性验证**: 严格的状态向量对比验证
- **问题诊断**: 专门的 PyTorch 和 qibotn 后端分析
- **多格式报告**: CSV、Markdown、JSON 格式的详细报告
- **QASMBench 集成**: 包含完整的量子电路基准测试集
- **虚拟环境**: 独立的 Python 环境配置

## 支持的后端

### 基本后端
1. **numpy** - 基础后端，性能基准
2. **qibojit (numba)** - JIT编译后端，**性能最优** (21.2倍加速)
3. **qibotn (qutensornet)** - 张量网络后端，**内存效率最高**
4. **clifford** - Clifford电路专用后端
5. **hamming_weight** - 汉明权重专用后端

### 机器学习后端
6. **qiboml (jax)** - 基于 JAX 的机器学习后端
7. **qiboml (pytorch)** - 基于 PyTorch 的机器学习后端
8. **qiboml (tensorflow)** - 基于 TensorFlow 的机器学习后端

### 后端特性对比
| 后端 | 性能排名 | 内存效率 | 数据类型 | 特殊处理 |
|------|----------|----------|----------|----------|
| qibojit | 🥇 1st | ⭐⭐⭐⭐ | numpy数组 | 直接使用 |
| qibotn | 🥈 2nd | ⭐⭐⭐⭐⭐ | numpy数组 | 直接使用 |
| qiboml (pytorch) | 🥉 3rd | ⭐⭐ | torch.Tensor | 需要转换 |
| qiboml (jax) | 4th | ⭐⭐⭐ | jax.Array | 需要转换 |
| numpy | 5th | ⭐⭐⭐⭐ | numpy数组 | 基准后端 |
| qiboml (tensorflow) | 6th | ⭐⭐⭐ | tf.Tensor | 需要转换 |

## 快速开始

### 1. 激活虚拟环境
```bash
.\qibovenv\Scripts\Activate.ps1
```

### 2. 运行完整基准测试
```bash
python qibobench/qft/benchmark_advanced.py
```

### 3. 验证后端正确性
```bash
python test/strict_validation_test.py
```

### 4. 分析特定后端问题
```bash
# PyTorch后端分析
python test/pytorch_backend_analysis.py

# qibotn警告分析
python test/qibotn_warning_test.py

# 最终功能测试
python test/final_backend_test.py
```

### 5. 查看测试报告
报告文件位于 `qibobench/qft/` 和 `test/` 目录：
- `advanced_benchmark_report.md` - 高级基准测试报告
- `backend_test_summary.md` - 测试总结报告
- `qibotn_warning_analysis.md` - 警告分析报告

## 环境要求

- Python 3.8+
- Windows/Linux/macOS
- Git

## 安装依赖

```bash
pip install -r requirements.txt
```

## 测试结果示例 (QFT电路 - 18量子比特, 820个门)

### 性能基准测试结果
| 后端 | 执行时间(秒) | 内存占用(MB) | 加速比 | 正确性 |
|------|-------------|-------------|--------|--------|
| qibojit (numba) | 0.383 ± 0.067 | 0.0 | 21.2x | ✅ |
| qibotn (qutensornet) | 1.048 ± 0.025 | 0.6 | 7.7x | ✅ |
| qiboml (pytorch) | 2.812 ± 0.276 | 1734.7 | 2.9x | ✅ |
| qiboml (jax) | 3.413 ± 0.037 | 7.6 | 2.4x | ✅ |
| numpy | 8.104 ± 0.102 | 0.0 | N/A | ✅ |
| qiboml (tensorflow) | 21.698 ± 1.632 | 8.0 | 0.4x | ✅ |

### 关键发现
- **性能最优**: qibojit (numba) - 21.2倍加速
- **内存最优**: qibotn (qutensornet) - 仅0.6MB内存占用
- **正确性**: 所有后端计算结果与numpy基准一致
- **特殊处理**: PyTorch后端需要数据类型转换

### 问题诊断结果
- **qibotn警告**: 良性警告，不影响核心计算功能
- **PyTorch数据类型**: 需要 `detach().cpu().numpy()` 转换
- **验证精度**: 状态向量差异 < 1e-15

## 核心测试脚本说明

### 基准测试框架 (`qibobench/qft/`)
- **`run_qft_18.py`**: 完整的基准测试框架，支持多格式报告生成
- **`benchmark_advanced.py`**: 高级指标测试，包含核心性能指标
- **BenchmarkReporter类**: 自动生成CSV、Markdown、JSON报告

### 功能验证脚本 (`test/`)
- **`strict_validation_test.py`**: 严格正确性验证，状态向量对比
- **`pytorch_backend_analysis.py`**: PyTorch后端数据类型分析
- **`qibotn_warning_test.py`**: qibotn后端警告诊断
- **`final_backend_test.py`**: 最终功能完整性测试

## 使用建议

### 性能优先场景
```python
set_backend("qibojit", platform="numba")
```

### 内存敏感场景  
```python
set_backend("qibotn", platform="qutensornet")
```

### 机器学习集成
```python
# JAX集成
set_backend("qiboml", platform="jax")

# PyTorch集成（注意转换）
set_backend("qiboml", platform="pytorch")
result = circuit()
state = result.state().detach().cpu().numpy()  # 重要：转换数据类型
```

## 许可证

本项目基于 QASMBench 基准测试集，遵循相应的开源许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个测试框架。

---

**最后更新**: 2025-09-30  
**测试框架版本**: 2.0 (完整版)