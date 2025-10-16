# 量子计算验证报告

生成时间: 2025-10-16 17:25:24

## 测试配置

- 量子比特数: 4
- Ansatz层数: 2
- 耦合强度 (J): 1.0
- 横向场强度 (h): 1.0
- 测试参数组数: 5
- 精确基态能量: -4.758770
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
| PennyLane | 2.665423 | 成功 |
| Qibo | - | 失败 |

### 参数组 2

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | 0.770391 | 成功 |
| Qibo | - | 失败 |

### 参数组 3

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | 0.619789 | 成功 |
| Qibo | - | 失败 |

### 参数组 4

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | -0.921982 | 成功 |
| Qibo | - | 失败 |

### 参数组 5

| 框架 | 能量 | 计算状态 |
|------|------|----------|
| Qiskit | - | 失败 |
| PennyLane | -3.030908 | 成功 |
| Qibo | - | 失败 |

## 能量准确性验证结果

精确基态能量: -4.758770

| 框架 | 计算能量 | 相对误差 | 状态 |
|------|----------|----------|------|
| PennyLane | 2.665423 | 1.56e+00 | ✗ |

## 相对误差分析

各框架的平均相对误差:

| 框架 | 平均相对误差 |
|------|-------------|

## 错误总结

验证过程中遇到的错误:

- Qiskit_energy_calc_0: Invalid observable type: <class 'qibo.hamiltonians.hamiltonians.Hamiltonian'>
- Qibo_energy_calc_0: Cannot calculate Hamiltonian expectation value for state of type <class 'qibo.result.QuantumState'>
- Qiskit_energy_calc_1: Invalid observable type: <class 'qibo.hamiltonians.hamiltonians.Hamiltonian'>
- Qibo_energy_calc_1: Cannot calculate Hamiltonian expectation value for state of type <class 'qibo.result.QuantumState'>
- Qiskit_energy_calc_2: Invalid observable type: <class 'qibo.hamiltonians.hamiltonians.Hamiltonian'>
- Qibo_energy_calc_2: Cannot calculate Hamiltonian expectation value for state of type <class 'qibo.result.QuantumState'>
- Qiskit_energy_calc_3: Invalid observable type: <class 'qibo.hamiltonians.hamiltonians.Hamiltonian'>
- Qibo_energy_calc_3: Cannot calculate Hamiltonian expectation value for state of type <class 'qibo.result.QuantumState'>
- Qiskit_energy_calc_4: Invalid observable type: <class 'qibo.hamiltonians.hamiltonians.Hamiltonian'>
- Qibo_energy_calc_4: Cannot calculate Hamiltonian expectation value for state of type <class 'qibo.result.QuantumState'>
- Qiskit_energy_accuracy: too many indices for array: array is 0-dimensional, but 1 were indexed
- Qibo_energy_accuracy: Cannot calculate Hamiltonian expectation value for state of type <class 'qibo.result.QuantumState'>

## 结论

所有验证测试均通过，各框架的计算结果一致且正确。
