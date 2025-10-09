# Qibo 后端基准测试报告 - 高级指标版

**生成时间**: 2025-09-30 09:22:53
**版本**: 2.0 (高级指标)

## 📊 核心指标表

| 优先级 | 指标 | 描述 | 单位 |
|--------|------|------|------|
| 🔴 核心 | 执行时间 (均值 ± 标准差) | 最重要的性能指标 | 秒 |
| 🔴 核心 | 峰值内存占用 | 最重要的资源指标 | MB |
| 🟡 高 | 加速比 | 相对于基线的性能提升 | 倍数 |
| 🟡 高 | 正确性验证 | 计算结果准确性验证 | Passed/Failed |
| 🟢 中 | 吞吐率 | 单位时间处理的门数量 | 门/秒 |
| 🟢 中 | JIT编译时间 | 即时编译开销 | 秒 |
| 🔵 低 | 电路构建时间 | 电路对象创建时间 | 秒 |

## 🔬 测试电路参数

| 参数 | 值 | 说明 |
|------|----|------|
| 电路类型 | QFT | 量子傅里叶变换 |
| 量子比特数 | 18 | 电路宽度 |
| 电路深度 | 138 | 层数 |
| 门数量 | 820 | 总操作数 |
| 数据源 | QASMBench/medium/qft_n18/qft_n18_transpiled.qasm | QASMBench |

## 📈 详细测试结果

| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率 | JIT时间 |
|------|-------------|----------|--------|--------|--------|---------|
| numpy | 8.104 ± 0.102 | 0.0 | N/A | Passed | 101 | N/A |
| qibojit (numba) | 0.383 ± 0.067 | 0.0 | 21.2x | Passed | 2140 | 0.360 |
| qibotn (qutensornet) | 1.048 ± 0.025 | 0.6 | 7.7x | Passed | 782 | N/A |
| qiboml (jax) | 3.413 ± 0.037 | 7.6 | 2.4x | Passed | 240 | 7.925 |
| qiboml (pytorch) | 2.812 ± 0.276 | 1734.7 | 2.9x | Passed | 292 | N/A |
| qiboml (tensorflow) | 21.698 ± 1.632 | 8.0 | 0.4x | Passed | 38 | N/A |

## 🔍 性能分析

### 执行时间排名
1. **qibojit (numba)**: 0.383秒 (21.2x)
2. **qibotn (qutensornet)**: 1.048秒 (7.7x)
3. **qiboml (pytorch)**: 2.812秒 (2.9x)
4. **qiboml (jax)**: 3.413秒 (2.4x)
5. **numpy**: 8.104秒
6. **qiboml (tensorflow)**: 21.698秒 (0.4x)

### 内存效率排名
1. **numpy**: 0.0MB
2. **qibojit (numba)**: 0.0MB
3. **qibotn (qutensornet)**: 0.6MB
4. **qiboml (jax)**: 7.6MB
5. **qiboml (tensorflow)**: 8.0MB
6. **qiboml (pytorch)**: 1734.7MB

## 💡 使用建议
- **性能优先**: 选择 qibojit (numba)
- **内存敏感**: 选择 qibotn (qutensornet)
- **ML集成**: 选择 qiboml (jax)
- **基准参考**: 使用 numpy 作为性能基准

## 📋 测试方法
- 多次运行取平均值消除随机性
- 预热运行确保JIT编译完成
- 精确测量峰值内存使用
- 全面验证计算结果正确性

## 🔧 PyTorch后端数据类型分析

### 问题诊断
PyTorch后端验证失败的原因是数据类型不兼容：
- **错误信息**: `unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'`
- **根本原因**: PyTorch后端返回的是`torch.Tensor`对象，而numpy基准是`numpy.ndarray`

### 数据类型对比

| 后端 | 状态向量类型 | 需要转换 | 转换方法 |
|------|-------------|----------|----------|
| numpy | numpy.ndarray | 否 | 直接使用 |
| qibojit | numpy.ndarray | 否 | 直接使用 |
| qibotn | numpy.ndarray | 否 | 直接使用 |
| qiboml (jax) | jax.Array | 是 | `.numpy()` |
| **qiboml (pytorch)** | **torch.Tensor** | **是** | **`.detach().cpu().numpy()`** |
| qiboml (tensorflow) | tensorflow.Tensor | 是 | `.numpy()` |

### 修复方案
在验证函数中添加PyTorch特殊处理：
```python
def validate_backend_accuracy_fixed(backend_name, platform_name=None, circuit_qasm=None):
    # ... 原有代码 ...
    
    test_state = test_result.state()
    
    # 🔧 添加PyTorch特殊处理
    if backend_name == "qiboml" and platform_name == "pytorch":
        if isinstance(test_state, torch.Tensor):
            test_state = test_state.detach().cpu().numpy()
    
    # ... 继续验证逻辑 ...
```

### 验证结果
修复后，PyTorch后端能够正确通过验证，与其他后端保持一致的计算精度。

---

**报告生成完成时间**: 2025-09-30 09:35:14
**包含PyTorch数据类型分析**: ✅ 完成
