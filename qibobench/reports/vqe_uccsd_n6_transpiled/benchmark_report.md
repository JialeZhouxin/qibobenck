# QASMBench电路基准测试报告: vqe_uccsd_n6_transpiled

**生成时间**: 2025-10-09 15:11:57

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | vqe_uccsd_n6_transpiled | QASMBench电路 |
| 量子比特数 | 6 | 电路的宽度 |
| 电路深度 | 1619 | 电路的层数 |
| 门数量 | 2125 | 总门操作数 |
| 电路来源 | .\QASMBench\small\vqe_uccsd_n6\vqe_uccsd_n6_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 0.0556 ± 0.0014 | 0.0 | N/A | Passed (no baseline) | 38247 |
| qibojit (numba) | 0.0584 ± 0.0170 | 0.0 | 0.95x | Passed (fidelity: 1.000000) | 36393 |
| qibotn (qutensornet) | 1.9207 ± 0.1048 | 3.2 | 0.03x | Passed (fidelity: 1.000000) | 1106 |
| qiboml (jax) | 0.9594 ± 0.0355 | 0.1 | 0.06x | Passed (fidelity: 1.000000) | 2215 |
| qiboml (pytorch) | 0.3964 ± 0.0158 | 36.1 | 0.14x | Passed (fidelity: 1.000000) | 5360 |
| qiboml (tensorflow) | 1.6673 ± 0.1406 | 0.0 | 0.03x | Passed (fidelity: 1.000000) | 1274 |

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
1. **numpy** - 0.0556秒
2. **qibojit (numba)** - 0.0584秒 (0.95x)
3. **qiboml (pytorch)** - 0.3964秒 (0.14x)
4. **qiboml (jax)** - 0.9594秒 (0.06x)
5. **qiboml (tensorflow)** - 1.6673秒 (0.03x)
6. **qibotn (qutensornet)** - 1.9207秒 (0.03x)

