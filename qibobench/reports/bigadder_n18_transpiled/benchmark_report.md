# QASMBench电路基准测试报告: bigadder_n18_transpiled

**生成时间**: 2025-10-09 16:51:54

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
| numpy | 2.5380 ± 0.0201 | 4.0 | N/A | Passed (no baseline) | 131 |
| qibojit (numba) | 0.0841 ± 0.0210 | 4.0 | 30.18x | Passed (fidelity: 1.000000) | 3948 |
| qibotn (qutensornet) | 0.2132 ± 0.0065 | 4.3 | 11.91x | Passed (fidelity: 1.000000) | 1558 |
| qiboml (jax) | 1.2701 ± 0.0574 | 9.0 | 2.00x | Passed (fidelity: 1.000000) | 261 |
| qiboml (pytorch) | 0.8732 ± 0.1021 | 659.0 | 2.91x | Passed (fidelity: 1.000000) | 380 |
| qiboml (tensorflow) | 7.7780 ± 0.2386 | 4.1 | 0.33x | Passed (fidelity: 1.000000) | 43 |
| qulacs | 0.0721 ± 0.0009 | 4.0 | 35.19x | Passed (fidelity: 1.000000) | 4603 |

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
1. **qulacs** - 0.0721秒 (35.19x)
2. **qibojit (numba)** - 0.0841秒 (30.18x)
3. **qibotn (qutensornet)** - 0.2132秒 (11.91x)
4. **qiboml (pytorch)** - 0.8732秒 (2.91x)
5. **qiboml (jax)** - 1.2701秒 (2.00x)
6. **numpy** - 2.5380秒
7. **qiboml (tensorflow)** - 7.7780秒 (0.33x)

