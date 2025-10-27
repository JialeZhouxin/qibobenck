# 量子模拟器基准测试报告

测试时间: 2025-10-22 11:44:40
重复运行次数: 2

## 基本统计

- 总测试次数: 3
- 测试的模拟器: qibo-qibojit, qiskit-aer_simulator, pennylane-lightning.qubit
- 测试的电路: grover_10_qubits
- 量子比特数范围: 10 - 10

## 电路信息

### 电路复杂度

| 电路名称 | 量子比特数 | 电路深度 | 门总数 |
|---------|-----------|---------|--------|
| grover_10_qubits | 10 | 152.0 | 1061.0 |

### 电路摘要示例

```
Circuit depth = 152
Total number of gates = 1061
Number of qubits = 10
Most common gates:
h: 510
x: 500
z: 50
measure: 1
```

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: grover_10_qubits (10 qubits)
- 平均时间: 0.0427 ± 0.0005 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: grover_10_qubits (10 qubits)
- 平均内存: 0.04 ± 0.00 MB

### 平均保真度排名
- qibo-qibojit: 1.0000
- pennylane-lightning.qubit: 1.0000
- qiskit-aer_simulator: 0.9995

## 稳定性分析

### 最稳定执行
- 模拟器: qibo-qibojit
- 电路: grover_10_qubits (10 qubits)
- 变异系数: 0.0116

### 最不稳定执行
- 模拟器: qiskit-aer_simulator
- 电路: grover_10_qubits (10 qubits)
- 变异系数: 0.0377

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                         10
runner_id                          
pennylane-lightning.qubit  0.159891
qibo-qibojit               0.042749
qiskit-aer_simulator       0.078935
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
4. 对于需要稳定性能的应用，选择变异系数最小的模拟器
