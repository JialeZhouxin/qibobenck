# QASMBench电路基准测试报告: qft_n18_transpiled

**生成时间**: 2025-10-13 15:50:14

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | qft_n18_transpiled | QASMBench电路 |
| 量子比特数 | 18 | 电路的宽度 |
| 电路深度 | 138 | 电路的层数 |
| 门数量 | 820 | 总门操作数 |
| 电路来源 | .\QASMBench\medium\qft_n18\qft_n18_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 7.7751 ± 0.0482 | 6.0 | N/A | Passed (no baseline) | 105 |
| qibojit (numba) | 0.5626 ± 0.1962 | 6.0 | 13.82x | Passed (fidelity: 1.000000) | 1457 |
| qibotn (qutensornet) | 3.0006 ± 0.1137 | 4.1 | 2.59x | Passed (fidelity: 1.000000) | 273 |
| qiboml (jax) | 3.4307 ± 0.1828 | 11.0 | 2.27x | Passed (fidelity: 1.000000) | 239 |
| qiboml (pytorch) | 3.8889 ± 1.3424 | 550.3 | 2.00x | Passed (fidelity: 1.000000) | 211 |
| qiboml (tensorflow) | 17.4290 ± 0.9902 | 8.9 | 0.45x | Passed (fidelity: 1.000000) | 47 |
| qulacs | 0.1006 ± 0.0052 | 6.0 | 77.27x | Passed (fidelity: 1.000000) | 8150 |

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
1. **qulacs** - 0.1006秒 (77.27x)
2. **qibojit (numba)** - 0.5626秒 (13.82x)
3. **qibotn (qutensornet)** - 3.0006秒 (2.59x)
4. **qiboml (jax)** - 3.4307秒 (2.27x)
5. **qiboml (pytorch)** - 3.8889秒 (2.00x)
6. **numpy** - 7.7751秒
7. **qiboml (tensorflow)** - 17.4290秒 (0.45x)

