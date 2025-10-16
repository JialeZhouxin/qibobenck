# 量子计算验证报告

生成时间: 2025-10-16 16:45:19

## 测试配置

- 量子比特数: 4
- Ansatz层数: 2
- 耦合强度 (J): 1.0
- 横向场强度 (h): 1.0
- 测试参数组数: 5
- 精确基态能量: -5.226252
- 可用框架: Qiskit, PennyLane, Qibo

## 哈密顿量验证结果

各框架哈密顿量构建成功，并与参考矩阵进行了比较。

### 哈密顿量矩阵比较

| 框架 | 状态 | 与参考矩阵最大差异 |
|------|------|---------------------|
| Qiskit | 成功 | 见控制台输出 |
| PennyLane | 成功 | 见控制台输出 |
| Qibo | 成功 | 见控制台输出 |

## 参数数量验证结果

期望参数数量: 12

| 框架 | 实际参数数量 | 状态 |
|------|-------------|------|
| Qiskit | 12 | ✓ |
| PennyLane | 12 | ✓ |
| Qibo | 12 | ✓ |

## 成本函数一致性验证结果

各框架对相同参数组的能量计算结果:

### 参数组 1

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | 2.215612 | 成功 |
| Qibo | -5.031200 | 成功 |

### 参数组 2

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | 0.765338 | 成功 |
| Qibo | -5.013759 | 成功 |

### 参数组 3

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | 0.458168 | 成功 |
| Qibo | -5.046703 | 成功 |

### 参数组 4

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | -0.772146 | 成功 |
| Qibo | -5.053979 | 成功 |

### 参数组 5

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | -3.269991 | 成功 |
| Qibo | -5.045055 | 成功 |

## 能量准确性验证结果

精确基态能量: -5.226252

| 框架 | 计算能量 | 相对误差 | 状态 |
|------|----------|----------|------|
| PennyLane | 2.215612 | 1.42e+00 | ✗ |
| Qibo | -5.031200 | 3.73e-02 | ⚠ |

## 相对误差分析

各框架的平均相对误差:

| 框架 | 平均相对误差 |
|------|-------------|
| PennyLane | 1.22e+00 |
| Qibo | 1.22e+00 |

## 错误总结

验证过程中遇到的错误:

- Qiskit_energy_calc_0: 'Circuit' object has no attribute 'parameters'
- Qiskit_energy_calc_1: 'Circuit' object has no attribute 'parameters'
- Qiskit_energy_calc_2: 'Circuit' object has no attribute 'parameters'
- Qiskit_energy_calc_3: 'Circuit' object has no attribute 'parameters'
- Qiskit_energy_calc_4: 'Circuit' object has no attribute 'parameters'
- Qiskit_energy_accuracy: too many indices for array: array is 0-dimensional, but 1 were indexed

## 结论

- PennyLane 框架的计算结果与其他框架存在显著差异。
- Qibo 框架的计算结果与其他框架存在显著差异。
部分验证测试未通过，建议检查相关实现。
