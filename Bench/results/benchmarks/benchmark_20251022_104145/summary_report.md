# 量子模拟器基准测试报告

测试时间: 2025-10-22 10:47:17
重复运行次数: 3

## 基本统计

- 总测试次数: 3
- 测试的模拟器: qibo-qibojit, qiskit-aer_simulator, pennylane-lightning.qubit
- 测试的电路: qft_26_qubits
- 量子比特数范围: 26 - 26

## 电路信息

### 电路复杂度

| 电路名称 | 量子比特数 | 电路深度 | 门总数 |
|---------|-----------|---------|--------|
| qft_26_qubits | 26 | 52.0 | 364.0 |

### 电路摘要示例

```
Circuit depth = 52
Total number of gates = 364
Number of qubits = 26
Most common gates:
cu1: 325
h: 26
swap: 13
```

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_26_qubits (26 qubits)
- 平均时间: 18.6395 ± 0.8083 秒

### 内存使用最少
- 模拟器: qiskit-aer_simulator
- 电路: qft_26_qubits (26 qubits)
- 平均内存: 0.02 ± 0.00 MB

### 平均保真度排名
- pennylane-lightning.qubit: 1.0000
- qibo-qibojit: 1.0000
- qiskit-aer_simulator: 1.0000

## 稳定性分析

### 最稳定执行
- 模拟器: qiskit-aer_simulator
- 电路: qft_26_qubits (26 qubits)
- 变异系数: 0.0376

### 最不稳定执行
- 模拟器: pennylane-lightning.qubit
- 电路: qft_26_qubits (26 qubits)
- 变异系数: 0.1015

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                          26
runner_id                           
pennylane-lightning.qubit  25.227334
qibo-qibojit               18.639539
qiskit-aer_simulator       19.144712
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
4. 对于需要稳定性能的应用，选择变异系数最小的模拟器
