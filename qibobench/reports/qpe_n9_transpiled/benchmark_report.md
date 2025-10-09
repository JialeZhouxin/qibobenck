# QASMBench电路基准测试报告: qpe_n9_transpiled

**生成时间**: 2025-10-09 14:55:35

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | qpe_n9_transpiled | QASMBench电路 |
| 量子比特数 | 9 | 电路的宽度 |
| 电路深度 | 93 | 电路的层数 |
| 门数量 | 154 | 总门操作数 |
| 电路来源 | .\QASMBench\small\qpe_n9\qpe_n9_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 0.0074 ± 0.0010 | 0.0 | N/A | Passed (no baseline) | 20827 |
| qibojit (numba) | 0.0030 ± 0.0006 | 0.0 | 2.47x | Passed (fidelity: 1.000000) | 51466 |
| qibotn (qutensornet) | 0.1016 ± 0.0030 | 0.4 | 0.07x | Passed (fidelity: 1.000000) | 1516 |
| qiboml (jax) | 0.1023 ± 0.0054 | 0.1 | 0.07x | Passed (fidelity: 1.000000) | 1505 |
| qiboml (pytorch) | 0.0413 ± 0.0041 | 4.7 | 0.18x | Passed (fidelity: 1.000000) | 3731 |
| qiboml (tensorflow) | 0.1806 ± 0.0801 | 0.0 | 0.04x | Passed (fidelity: 1.000000) | 853 |

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
1. **qibojit (numba)** - 0.0030秒 (2.47x)
2. **numpy** - 0.0074秒
3. **qiboml (pytorch)** - 0.0413秒 (0.18x)
4. **qibotn (qutensornet)** - 0.1016秒 (0.07x)
5. **qiboml (jax)** - 0.1023秒 (0.07x)
6. **qiboml (tensorflow)** - 0.1806秒 (0.04x)

