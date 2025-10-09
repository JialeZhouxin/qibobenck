# QASMBench电路基准测试报告: dnn_n16_transpiled

**生成时间**: 2025-10-09 16:47:04

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | dnn_n16_transpiled | QASMBench电路 |
| 量子比特数 | 16 | 电路的宽度 |
| 电路深度 | 246 | 电路的层数 |
| 门数量 | 2833 | 总门操作数 |
| 电路来源 | .\QASMBench\medium\dnn_n16\dnn_n16_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 4.7218 ± 0.3157 | 1.5 | N/A | Passed (no baseline) | 600 |
| qibojit (numba) | 0.2241 ± 0.0227 | 1.5 | 21.07x | Passed (fidelity: 1.000000) | 12641 |
| qibotn (qutensornet) | 1.3229 ± 0.1472 | 0.3 | 3.57x | Passed (fidelity: 1.000000) | 2142 |
| qiboml (jax) | 3.0781 ± 0.1102 | 2.9 | 1.53x | Passed (fidelity: 1.000000) | 920 |
| qiboml (pytorch) | 3.2749 ± 2.1760 | 1597.1 | 1.44x | Passed (fidelity: 1.000000) | 865 |
| qiboml (tensorflow) | 14.9809 ± 0.0621 | 2.3 | 0.32x | Passed (fidelity: 1.000000) | 189 |
| qulacs | 0.0708 ± 0.0054 | 1.5 | 66.68x | Passed (fidelity: 1.000000) | 40008 |

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
1. **qulacs** - 0.0708秒 (66.68x)
2. **qibojit (numba)** - 0.2241秒 (21.07x)
3. **qibotn (qutensornet)** - 1.3229秒 (3.57x)
4. **qiboml (jax)** - 3.0781秒 (1.53x)
5. **qiboml (pytorch)** - 3.2749秒 (1.44x)
6. **numpy** - 4.7218秒
7. **qiboml (tensorflow)** - 14.9809秒 (0.32x)

