# QASMBench电路基准测试报告: hhl_n7_transpiled

**生成时间**: 2025-10-09 14:48:13

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | hhl_n7_transpiled | QASMBench电路 |
| 量子比特数 | 7 | 电路的宽度 |
| 电路深度 | 828 | 电路的层数 |
| 门数量 | 991 | 总门操作数 |
| 电路来源 | .\QASMBench\small\hhl_n7\hhl_n7_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 0.0312 ± 0.0020 | 0.0 | N/A | Passed (no baseline) | 31766 |
| qibojit (numba) | 0.0182 ± 0.0012 | 0.0 | 1.71x | Passed (fidelity: 1.000000) | 54356 |
| qibotn (qutensornet) | 0.6124 ± 0.0731 | 0.7 | 0.05x | Passed (fidelity: 1.000000) | 1618 |
| qiboml (jax) | 0.4975 ± 0.0019 | 0.1 | 0.06x | Passed (fidelity: 1.000000) | 1992 |
| qiboml (pytorch) | 0.1944 ± 0.0037 | 19.2 | 0.16x | Passed (fidelity: 1.000000) | 5097 |
| qiboml (tensorflow) | 0.7072 ± 0.0037 | 0.0 | 0.04x | Passed (fidelity: 1.000000) | 1401 |

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
1. **qibojit (numba)** - 0.0182秒 (1.71x)
2. **numpy** - 0.0312秒
3. **qiboml (pytorch)** - 0.1944秒 (0.16x)
4. **qiboml (jax)** - 0.4975秒 (0.06x)
5. **qibotn (qutensornet)** - 0.6124秒 (0.05x)
6. **qiboml (tensorflow)** - 0.7072秒 (0.04x)

