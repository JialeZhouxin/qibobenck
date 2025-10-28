# 量子模拟器基准测试报告

测试时间: 2025-10-16 11:43:36
重复运行次数: 10

## 基本统计

- 总测试次数: 9
- 测试的模拟器: qibo-qibojit
- 测试的电路: qft_4_qubits, qft_8_qubits, qft_12_qubits, qft_16_qubits, qft_18_qubits, qft_20_qubits, qft_22_qubits, qft_24_qubits, qft_26_qubits
- 量子比特数范围: 4 - 26

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_8_qubits (8 qubits)
- 平均时间: 0.0028 ± 0.0007 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_8_qubits (8 qubits)
- 平均内存: 0.01 ± 0.00 MB

### 平均保真度排名
- qibo-qibojit: 1.0000

## 稳定性分析

### 最稳定执行
- 模拟器: qibo-qibojit
- 电路: qft_26_qubits (26 qubits)
- 变异系数: 0.0441

### 最不稳定执行
- 模拟器: qibo-qibojit
- 电路: qft_4_qubits (4 qubits)
- 变异系数: 2.8834

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits            4         8         12        16        18        20        22        24        26
runner_id                                                                                             
qibo-qibojit  0.008824  0.002782  0.005751  0.013505  0.029779  0.163179  0.708339  3.217464  15.00857
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
4. 对于需要稳定性能的应用，选择变异系数最小的模拟器
