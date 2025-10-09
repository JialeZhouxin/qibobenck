# QASMBench电路基准测试报告: adder_n10_transpiled

**生成时间**: 2025-10-09 16:41:59

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | adder_n10_transpiled | QASMBench电路 |
| 量子比特数 | 10 | 电路的宽度 |
| 电路深度 | 120 | 电路的层数 |
| 门数量 | 167 | 总门操作数 |
| 电路来源 | .\QASMBench\small\adder_n10\adder_n10_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 0.0127 ± 0.0011 | 0.0 | N/A | Passed (no baseline) | 13158 |
| qibojit (numba) | 0.0030 ± 0.0006 | 0.0 | 4.24x | Passed (fidelity: 1.000000) | 55814 |
| qibotn (qutensornet) | 0.1192 ± 0.0063 | 0.3 | 0.11x | Passed (fidelity: 1.000000) | 1401 |
| qiboml (jax) | 0.0925 ± 0.0024 | 0.0 | 0.14x | Passed (fidelity: 1.000000) | 1806 |
| qiboml (pytorch) | 0.0457 ± 0.0023 | 5.1 | 0.28x | Passed (fidelity: 1.000000) | 3654 |
| qiboml (tensorflow) | 0.1493 ± 0.0030 | 0.0 | 0.09x | Passed (fidelity: 1.000000) | 1119 |
| qulacs | 0.0014 ± 0.0005 | 0.0 | 9.02x | Passed (fidelity: 1.000000) | 118668 |

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

### qulacs 环境
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel
- RAM_GB: 15.909721374511719
- Python: 3.12.0
- Qibo: 0.2.21
- Backend: qulacs
- Platform: default

## 性能分析

### 性能排名（从优到劣）
1. **qulacs** - 0.0014秒 (9.02x)
2. **qibojit (numba)** - 0.0030秒 (4.24x)
3. **numpy** - 0.0127秒
4. **qiboml (pytorch)** - 0.0457秒 (0.28x)
5. **qiboml (jax)** - 0.0925秒 (0.14x)
6. **qibotn (qutensornet)** - 0.1192秒 (0.11x)
7. **qiboml (tensorflow)** - 0.1493秒 (0.09x)

