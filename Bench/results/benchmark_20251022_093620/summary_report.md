# 量子模拟器基准测试报告

测试时间: 2025-10-22 09:38:14
重复运行次数: 3

## 基本统计

- 总测试次数: 2
- 测试的模拟器: qibo-qibojit, pennylane-lightning.qubit
- 测试的电路: qft_25_qubits
- 量子比特数范围: 25 - 25

## 电路信息

### 电路复杂度

| 电路名称 | 量子比特数 | 电路深度 | 门总数 |
|---------|-----------|---------|--------|
| qft_25_qubits | 25 | 50.0 | 337.0 |

### 电路摘要示例

```
Circuit depth = 50
Total number of gates = 337
Number of qubits = 25
Most common gates:
cu1: 300
h: 25
swap: 12
```

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_25_qubits (25 qubits)
- 平均时间: 8.2582 ± 0.0583 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_25_qubits (25 qubits)
- 平均内存: 512.00 ± 0.00 MB

### 平均保真度排名
- pennylane-lightning.qubit: 1.0000
- qibo-qibojit: 1.0000

## 稳定性分析

### 最稳定执行
- 模拟器: qibo-qibojit
- 电路: qft_25_qubits (25 qubits)
- 变异系数: 0.0071

### 最不稳定执行
- 模拟器: pennylane-lightning.qubit
- 电路: qft_25_qubits (25 qubits)
- 变异系数: 0.0383

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                          25
runner_id                           
pennylane-lightning.qubit  11.064476
qibo-qibojit                8.258248
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
4. 对于需要稳定性能的应用，选择变异系数最小的模拟器
