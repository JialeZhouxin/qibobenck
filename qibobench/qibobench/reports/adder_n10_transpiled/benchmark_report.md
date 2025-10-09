# QASMBench电路基准测试报告: adder_n10_transpiled

**生成时间**: 2025-10-09 11:01:31

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
| numpy | 0.0117 ± 0.0009 | 0.0 | N/A | Passed | 14298 |
| qibojit (numba) | 0.0059 ± 0.0009 | 0.0 | 1.98x | Passed | 28246 |
| qibotn (qutensornet) | 0.1295 ± 0.0089 | 0.3 | 0.09x | Passed | 1290 |
| qiboml (jax) | 0.1089 ± 0.0156 | 0.0 | 0.11x | Passed | 1534 |
| qiboml (pytorch) | 0.0431 ± 0.0019 | 5.0 | 0.27x | Passed | 3875 |
| qiboml (tensorflow) | 0.1841 ± 0.0234 | 0.0 | 0.06x | Passed | 907 |

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
1. **qibojit (numba)** - 0.0059秒 (1.98x)
2. **numpy** - 0.0117秒
3. **qiboml (pytorch)** - 0.0431秒 (0.27x)
4. **qiboml (jax)** - 0.1089秒 (0.11x)
5. **qibotn (qutensornet)** - 0.1295秒 (0.09x)
6. **qiboml (tensorflow)** - 0.1841秒 (0.06x)

