# 量子模拟器基准测试报告

测试时间: 2025-10-22 12:21:27
重复运行次数: 3

## 基本统计

- 总测试次数: 3
- 测试的模拟器: qibo-qibojit, qiskit-aer_simulator, pennylane-lightning.qubit
- 测试的电路: grover_20_qubits
- 量子比特数范围: 20 - 20

## 电路信息

### 电路复杂度

| 电路名称 | 量子比特数 | 电路深度 | 门总数 |
|---------|-----------|---------|--------|
| grover_20_qubits | 20 | 4826.0 | 65949.0 |

### 电路摘要示例

```
Circuit depth = 4826
Total number of gates = 65949
Number of qubits = 20
Most common gates:
h: 32180
x: 32160
z: 1608
measure: 1
```

## 性能指标

### 最快执行
- 模拟器: qiskit-aer_simulator
- 电路: grover_20_qubits (20 qubits)
- 平均时间: 36.5724 ± 0.8491 秒

### 内存使用最少
- 模拟器: qiskit-aer_simulator
- 电路: grover_20_qubits (20 qubits)
- 平均内存: 2.99 ± 0.00 MB

### 平均保真度排名
- qibo-qibojit: 1.0000
- pennylane-lightning.qubit: 1.0000
- qiskit-aer_simulator: 1.0000

## 稳定性分析

### 最稳定执行
- 模拟器: pennylane-lightning.qubit
- 电路: grover_20_qubits (20 qubits)
- 变异系数: 0.0046

### 最不稳定执行
- 模拟器: qiskit-aer_simulator
- 电路: grover_20_qubits (20 qubits)
- 变异系数: 0.0232

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                           20
runner_id                            
pennylane-lightning.qubit  121.450752
qibo-qibojit                96.358857
qiskit-aer_simulator        36.572439
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
4. 对于需要稳定性能的应用，选择变异系数最小的模拟器
