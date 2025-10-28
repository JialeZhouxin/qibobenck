# 量子模拟器基准测试报告

测试时间: 2025-10-14 16:19:08

## 基本统计

- 总测试次数: 15
- 测试的模拟器: qibo-numpy, qibo-qibojit, pennylane-lightning.qubit
- 测试的电路: qft_2_qubits, qft_3_qubits, qft_4_qubits, qft_5_qubits, qft_6_qubits
- 量子比特数范围: 2 - 6

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_3_qubits (3 qubits)
- 时间: 0.0005 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_2_qubits (2 qubits)
- 内存: 0.00 MB

### 平均保真度排名
- qibo-numpy: 1.0000
- pennylane-lightning.qubit: 1.0000
- qibo-qibojit: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                          2         3         4         5         6
runner_id                                                                  
pennylane-lightning.qubit  0.026169  0.014683  0.016061  0.017777  0.019481
qibo-numpy                 0.135061  0.000659  0.000869  0.002267  0.002163
qibo-qibojit               0.000475  0.000452  0.000934  0.001063  0.001981
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
