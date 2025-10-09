# QASMBench电路基准测试报告: dnn_n16_transpiled

**生成时间**: 2025-10-09 15:55:19

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
| numpy | 4.4415 ± 0.0062 | 1.5 | N/A | Passed (no baseline) | 638 |
| qibojit (numba) | 0.2046 ± 0.0150 | 1.5 | 21.71x | Passed (fidelity: 1.000000) | 13850 |
| qibotn (qutensornet) | 1.2994 ± 0.1239 | 0.2 | 3.42x | Passed (fidelity: 1.000000) | 2180 |
| qiboml (jax) | 2.7953 ± 0.0339 | 3.8 | 1.59x | Passed (fidelity: 1.000000) | 1013 |
| qiboml (pytorch) | 2.1279 ± 0.1109 | 1597.3 | 2.09x | Passed (fidelity: 1.000000) | 1331 |
| qiboml (tensorflow) | 17.7101 ± 1.7088 | 2.3 | 0.25x | Passed (fidelity: 1.000000) | 160 |

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
1. **qibojit (numba)** - 0.2046秒 (21.71x)
2. **qibotn (qutensornet)** - 1.2994秒 (3.42x)
3. **qiboml (pytorch)** - 2.1279秒 (2.09x)
4. **qiboml (jax)** - 2.7953秒 (1.59x)
5. **numpy** - 4.4415秒
6. **qiboml (tensorflow)** - 17.7101秒 (0.25x)

