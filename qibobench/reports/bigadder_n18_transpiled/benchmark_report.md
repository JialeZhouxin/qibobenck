# QASMBench电路基准测试报告: bigadder_n18_transpiled

**生成时间**: 2025-10-09 15:20:37

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | bigadder_n18_transpiled | QASMBench电路 |
| 量子比特数 | 18 | 电路的宽度 |
| 电路深度 | 181 | 电路的层数 |
| 门数量 | 332 | 总门操作数 |
| 电路来源 | .\QASMBench\medium\bigadder_n18\bigadder_n18_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 2.7124 ± 0.2232 | 4.0 | N/A | Passed (no baseline) | 122 |
| qibojit (numba) | 0.0816 ± 0.0109 | 4.0 | 33.23x | Passed (fidelity: 1.000000) | 4067 |
| qibotn (qutensornet) | 0.2133 ± 0.0027 | 4.6 | 12.72x | Passed (fidelity: 1.000000) | 1557 |
| qiboml (jax) | 1.2661 ± 0.0490 | 7.2 | 2.14x | Passed (fidelity: 1.000000) | 262 |
| qiboml (pytorch) | 0.8032 ± 0.1236 | 662.0 | 3.38x | Passed (fidelity: 1.000000) | 413 |
| qiboml (tensorflow) | 7.2239 ± 0.4072 | 4.0 | 0.38x | Passed (fidelity: 1.000000) | 46 |

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
1. **qibojit (numba)** - 0.0816秒 (33.23x)
2. **qibotn (qutensornet)** - 0.2133秒 (12.72x)
3. **qiboml (pytorch)** - 0.8032秒 (3.38x)
4. **qiboml (jax)** - 1.2661秒 (2.14x)
5. **numpy** - 2.7124秒
6. **qiboml (tensorflow)** - 7.2239秒 (0.38x)

