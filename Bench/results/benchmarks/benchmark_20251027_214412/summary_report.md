# 量子模拟器基准测试报告

测试时间: 2025-10-27 21:44:19
重复运行次数: 1

## 基本统计

- 总测试次数: 3
- 测试的模拟器: qibo-qibojit
- 测试的电路: qft_2_qubits, qft_3_qubits, qft_4_qubits
- 量子比特数范围: 2 - 4

## 电路信息

### 电路复杂度

| 电路名称 | 量子比特数 | 电路深度 | 门总数 |
|---------|-----------|---------|--------|
| qft_2_qubits | 2 | 4 | 4 |
| qft_3_qubits | 3 | 6 | 7 |
| qft_4_qubits | 4 | 8 | 12 |

### 电路摘要示例

```
Circuit depth = 4
Total number of gates = 4
Number of qubits = 2
Most common gates:
h: 2
cu1: 1
swap: 1
```

## 性能指标

### 最快执行
- 模拟器: qibo-qibojit
- 电路: qft_2_qubits (2 qubits)
- 时间: 0.0003 秒

### 内存使用最少
- 模拟器: qibo-qibojit
- 电路: qft_2_qubits (2 qubits)
- 内存: 0.00 MB

### 平均保真度排名
- qibo-qibojit: 1.0000

## 扩展性分析

### 执行时间随量子比特数的变化
```
n_qubits             2         3        4
runner_id                                
qibo-qibojit  0.000347  0.000568  0.00079
```

## 建议

基于以上结果，建议:
1. 对于小型量子电路，选择执行时间最短的模拟器
2. 对于大型量子电路，优先考虑内存使用效率
3. 在需要高精度计算时，选择保真度最高的模拟器
