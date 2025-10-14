# 量子模拟器基准测试报告

测试时间: 2025-10-14 16:38:06

## 基本统计

- 总测试次数: 18
- 测试的模拟器: qibo-qibojit, pennylane-lightning.qubit
- 测试的电路: qft_10_qubits, qft_11_qubits, qft_12_qubits, qft_13_qubits, qft_14_qubits, qft_15_qubits, qft_16_qubits, qft_17_qubits, qft_18_qubits
- 量子比特数范围: 10 - 18

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_11_qubits (11 qubits)
- 时间: 0.0044 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_11_qubits (11 qubits)
- 内存: 0.03 MB

### 平均保真度排名
- qibo-qibojit: 1.0000
- pennylane-lightning.qubit: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits                         10        11        12        13        14        15        16        17        18
runner_id                                                                                                          
pennylane-lightning.qubit  0.032776  0.038744  0.042495  0.045850  0.051890  0.057591  0.068473  0.082454  0.113443
qibo-qibojit               0.079803  0.004378  0.005546  0.005885  0.007884  0.009468  0.014407  0.018139  0.030334
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
