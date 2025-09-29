# Qibo 后端基准测试报告

**生成时间**: 2025-09-29 16:56:59

## 测试电路参数

### QFT (Quantum Fourier Transform) 电路

| 参数 | 值 | 描述 |
|------|----|------|
| 电路类型 | QFT | 量子傅里叶变换电路 |
| 量子比特数 | 18 | 电路的宽度 |
| 电路深度 | 138 | 电路的层数 |
| 门数量 | 820 | 总门操作数 |
| 电路来源 | QASMBench/medium/qft_n18/qft_n18_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同QFT电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 核心性能指标

| 优先级 | 指标 | 描述 | 示例 |
|--------|------|------|------|
| 核心 | 执行时间 (均值 ± 标准差) | 最重要的性能指标 | 1.56 ± 0.05 秒 |
| 核心 | 峰值内存占用 | 最重要的资源指标 | 128.5 MB |
| 高 | 加速比 | 相对于基线的性能提升 | 31.2x |
| 高 | 正确性验证 | 计算结果准确性验证 | Passed |

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 8.0470 ± 0.2425 | 6.0 | N/A | Passed | 102 |
| qibojit (numba) | 0.3798 ± 0.0790 | 6.0 | 21.19x | Passed | 2159 |
| qibotn (qutensornet) | 0.9782 ± 0.0363 | 4.6 | 8.23x | Passed | 838 |
| qiboml (jax) | 3.2064 ± 0.0457 | 8.5 | 2.51x | Passed | 256 |
| qiboml (pytorch) | 3.3243 ± 1.0719 | 2019.3 | 2.42x | Passed | 247 |
| qiboml (tensorflow) | 21.1916 ± 1.4195 | 8.5 | 0.38x | Passed | 39 |

## 测试环境

### numpy 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: numpy
- Platform: default

### qibojit (numba) 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: qibojit
- Platform: numba

### qibotn (qutensornet) 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: qibotn
- Platform: qutensornet

### qiboml (jax) 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: qiboml
- Platform: jax

### qiboml (pytorch) 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: qiboml
- Platform: pytorch

### qiboml (tensorflow) 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: qiboml
- Platform: tensorflow

## 性能分析

### 执行时间分析
- **最佳性能**: qibojit (numba) 后端，相比numpy基准有显著加速
- **稳定性能**: qibotn (qutensornet) 后端，标准差较小，性能稳定
- **机器学习后端**: qiboml (jax) 表现最佳

### 内存使用分析
- **最低内存**: qibotn (qutensornet) 内存使用最优化
- **常规内存**: 其他后端内存使用在合理范围内

### 吞吐率分析
- **最高吞吐**: qibojit (numba) 达到最高门操作吞吐率
- **基准吞吐**: numpy 后端作为性能比较基准

## 结论与建议

### 性能排名（从优到劣）
1. **qibojit (numba)** - 推荐用于高性能计算场景
2. **qibotn (qutensornet)** - 推荐用于内存敏感场景
3. **qiboml (jax)** - 推荐用于机器学习集成
4. **numpy** - 稳定的基准后端
5. **qiboml (pytorch)** - 存在性能稳定性问题
6. **qiboml (tensorflow)** - 性能较差，不推荐使用

### 使用建议
- **生产环境**: 优先选择 qibojit (numba) 或 qibotn (qutensornet)
- **研究开发**: 可根据具体需求选择合适后端
- **内存限制**: 使用 qibotn (qutensornet) 以获得最佳内存效率
- **性能优先**: 使用 qibojit (numba) 以获得最快执行速度

## 测试方法说明
所有测试均在相同硬件环境下进行，使用相同的QFT电路，确保结果的可比性。测试采用多次运行取平均值的方法，以消除单次运行的随机性影响。
