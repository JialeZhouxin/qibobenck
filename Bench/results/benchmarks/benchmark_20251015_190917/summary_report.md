# 量子模拟器基准测试报告

测试时间: 2025-10-15 19:09:36

## 基本统计

- 总测试次数: 7
- 测试的模拟器: qibo-qibojit
- 测试的电路: qft_4_qubits, qft_8_qubits, qft_12_qubits, qft_16_qubits, qft_18_qubits, qft_22_qubits, qft_24_qubits
- 量子比特数范围: 4 - 24

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_4_qubits (4 qubits)
- 时间: 0.0008 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_4_qubits (4 qubits)
- 内存: 0.00 MB

### 平均保真度排名
- qibo-qibojit: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits            4         8         12        16        18        22        24
runner_id                                                                         
qibo-qibojit  0.000776  0.002487  0.006312  0.018846  0.052458  0.929162  3.905819
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
