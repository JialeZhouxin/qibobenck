# QASMBench电路基准测试报告: adder_n10_transpiled

**生成时间**: 2025-10-09 09:45:05

## 测试电路参数

| 参数 | 值 | 描述 |
|------|----|------|
| 电路名称 | adder_n10_transpiled | QASMBench电路 |
| 量子比特数 | 10 | 电路的宽度 |
| 电路深度 | 120 | 电路的层数 |
| 门数量 | 167 | 总门操作数 |
| 电路来源 | ../QASMBench/small/adder_n10/adder_n10_transpiled.qasm | QASM文件路径 |

### 测试配置

- **运行次数**: 5次正式运行 + 1次预热运行
- **基准后端**: numpy (作为性能比较基准)
- **测试目标**: 比较不同后端在相同电路上的性能表现
- **输出格式**: CSV, Markdown, JSON

## 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 0.0112 ± 0.0004 | 0.0 | N/A | Passed | 14845 |
| qibojit (numba) | 0.0074 ± 0.0021 | 0.0 | 1.52x | Passed | 22628 |
| qibotn (qutensornet) | 0.1497 ± 0.0309 | 1.1 | 0.08x | Passed | 1116 |
| qiboml (jax) | 0.1178 ± 0.0120 | 0.0 | 0.10x | Passed | 1418 |
| qiboml (pytorch) | 0.0628 ± 0.0196 | 5.1 | 0.18x | Passed | 2658 |
| qiboml (tensorflow) | 0.1981 ± 0.0122 | 0.0 | 0.06x | Passed | 843 |

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
1. **qibojit (numba)** - 0.0074秒 (1.52x)
2. **numpy** - 0.0112秒
3. **qiboml (pytorch)** - 0.0628秒 (0.18x)
4. **qiboml (jax)** - 0.1178秒 (0.10x)
5. **qibotn (qutensornet)** - 0.1497秒 (0.08x)
6. **qiboml (tensorflow)** - 0.1981秒 (0.06x)

