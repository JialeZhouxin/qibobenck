# 量子模拟器基准测试报告

测试时间: 2025-10-20 13:41:45
重复运行次数: 1

## 基本统计

- 总测试次数: 3
- 测试的模拟器: qibo-qibojit
- 测试的电路: grover_2_qubits, grover_3_qubits, grover_4_qubits
- 量子比特数范围: 2 - 4

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: grover_2_qubits (2 qubits)
- 时间: 0.0008 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: grover_2_qubits (2 qubits)
- 内存: 0.00 MB

### 平均保真度排名
- qibo-qibojit: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits             2        3         4
runner_id                                
qibo-qibojit  0.000827  0.00234  0.002324
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
