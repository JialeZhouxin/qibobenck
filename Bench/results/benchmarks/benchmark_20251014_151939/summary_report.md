# 量子模拟器基准测试报告

测试时间: 2025-10-14 15:19:45

## 基本统计

- 总测试次数: 6
- 测试的模拟器: qibo-numpy, qiskit-aer_simulator
- 测试的电路: qft_2_qubits, qft_3_qubits, qft_4_qubits
- 量子比特数范围: 2 - 4

## 性能指标

### 最快执行
- 模拟器: qibo-numpy
- 电路: qft_3_qubits (3 qubits)
- 时间: 0.0021 秒

### 内存使用最少
- 模拟器: qibo-numpy
- 电路: qft_3_qubits (3 qubits)
- 内存: 0.01 MB

### 平均保真度排名
- qibo-numpy: 1.0000
- qiskit-aer_simulator: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                     2         3         4
runner_id                                         
qibo-numpy            0.103855  0.002082  0.002152
qiskit-aer_simulator  0.014586  0.004575  0.003814
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
