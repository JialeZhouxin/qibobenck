# QASMBench电路基准测试报告: simon_n6_transpiled

**生成时间**: 2025-10-09 14:58:48

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | simon_n6_transpiled | QASMBench电路 |
| 量子比特数 | 6 | 电路的宽度 |
| 电路深度 | 34 | 电路的层数 |
| 门数量 | 57 | 总门操作数 |
| 电路来源 | .\QASMBench\small\simon_n6\simon_n6_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 0.0018 ± 0.0004 | 0.0 | N/A | Passed (no baseline) | 31645 |
| qibojit (numba) | 0.0017 ± 0.0008 | 0.0 | 1.09x | Passed (fidelity: 1.000000) | 34482 |
| qibotn (qutensornet) | 0.0294 ± 0.0006 | 0.1 | 0.06x | Passed (fidelity: 1.000000) | 1942 |
| qiboml (jax) | 0.0335 ± 0.0009 | 0.0 | 0.05x | Passed (fidelity: 1.000000) | 1703 |
| qiboml (pytorch) | 0.0128 ± 0.0011 | 1.3 | 0.14x | Passed (fidelity: 1.000000) | 4466 |
| qiboml (tensorflow) | 0.0703 ± 0.0175 | 0.0 | 0.03x | Passed (fidelity: 1.000000) | 811 |

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

### 性能排名（从优到劣）
1. **qibojit (numba)** - 0.0017秒 (1.09x)
2. **numpy** - 0.0018秒
3. **qiboml (pytorch)** - 0.0128秒 (0.14x)
4. **qibotn (qutensornet)** - 0.0294秒 (0.06x)
5. **qiboml (jax)** - 0.0335秒 (0.05x)
6. **qiboml (tensorflow)** - 0.0703秒 (0.03x)

