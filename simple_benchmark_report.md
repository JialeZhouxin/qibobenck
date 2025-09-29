# Qibo后端基准测试报告 (简化版)

**生成时间**: 2025-09-29 16:16:34

## 核心性能指标

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |
|------|-------------|----------|--------|--------|---------------|
| numpy | 7.9004 ± 0.0166 | 6.0 | N/A | Passed | 104 |
| qibojit (numba) | 0.3982 ± 0.1074 | 6.0 | 19.84x | Passed | 2059 |
| qibotn (qutensornet) | 1.0147 ± 0.0888 | 4.6 | 7.79x | Passed | 808 |
| qiboml (jax) | 3.2678 ± 0.0297 | 4.7 | 2.42x | Passed | 251 |
| qiboml (pytorch) | 7.3221 ± 6.1278 | 2019.3 | 1.08x | Passed | 112 |
| qiboml (tensorflow) | 20.3841 ± 0.2961 | 8.1 | 0.39x | Passed | 40 |

## 测试环境

### numpy
- python_version: 3.12.0
- system: Windows
- backend: numpy
- platform: default

### qibojit (numba)
- python_version: 3.12.0
- system: Windows
- backend: qibojit
- platform: numba

### qibotn (qutensornet)
- python_version: 3.12.0
- system: Windows
- backend: qibotn
- platform: qutensornet

### qiboml (jax)
- python_version: 3.12.0
- system: Windows
- backend: qiboml
- platform: jax

### qiboml (pytorch)
- python_version: 3.12.0
- system: Windows
- backend: qiboml
- platform: pytorch

### qiboml (tensorflow)
- python_version: 3.12.0
- system: Windows
- backend: qiboml
- platform: tensorflow

