# 量子模拟器基准测试报告

测试时间: 2025-10-15 19:13:55

## 基本统计

- 总测试次数: 14
- 测试的模拟器: qibo-qibojit, pennylane-lightning.qubit
- 测试的电路: qft_4_qubits, qft_8_qubits, qft_12_qubits, qft_16_qubits, qft_18_qubits, qft_22_qubits, qft_24_qubits
- 量子比特数范围: 4 - 24

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_8_qubits (8 qubits)
- 时间: 0.0025 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_8_qubits (8 qubits)
- 内存: 0.01 MB

### 平均保真度排名
- qibo-qibojit: 1.0000
- pennylane-lightning.qubit: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                         4         8         12        16        18        22        24
runner_id                                                                                      
pennylane-lightning.qubit  0.026550  0.028715  0.042187  0.073488  0.113432  1.172727  4.925289
qibo-qibojit               0.119292  0.002458  0.004699  0.016483  0.045160  0.816528  3.402240
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
