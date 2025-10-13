# QASMBench通用基准测试工具

这是一个基于Qibo的通用基准测试工具，可以加载QASMBench中的任意电路进行性能测试。

## 功能特性

- ✅ 支持加载QASMBench中所有规模的电路（small/medium/large）
- ✅ 测试Qibo所有后端性能（numpy, qibojit, qibotn, qiboml等）
- ✅ 生成详细的性能报告（CSV, Markdown, JSON格式）
- ✅ 测量执行时间、内存使用、吞吐率等关键指标
- ✅ 自动计算加速比和性能排名
- ✅ 支持自定义测试配置

## 快速开始

### 1. 列出可用电路

```bash
python qasmbench_runner.py --list
```

### 2. 测试特定电路

```bash
# 通过文件路径测试
python qasmbench_runner.py --circuit QASMBench/medium/qft_n18/qft_n18_transpiled.qasm
```

### 3. 使用示例代码

```python
# 查看使用示例
python example_usage.py
```

## 文件结构

```
qibobench/
├── qasmbench_runner.py     # 主工具文件
├── example_usage.py        # 使用示例
├── README.md              # 说明文档
└── qft/                   # 原有的QFT测试文件
    ├── run_qft_18.py
    └── ...
```

## 支持的QASMBench电路

工具支持QASMBench中的所有电路，包括：

- **Small规模**: 小规模量子电路（2-10量子比特）
- **Medium规模**: 中等规模量子电路（11-30量子比特）  
- **Large规模**: 大规模量子电路（30+量子比特）

具体电路包括：QFT、加法器、Grover搜索、量子相位估计、VQE等各类量子算法。

## 输出报告

每次测试会生成三种格式的报告：

1. **CSV报告**: 便于数据分析和处理
2. **Markdown报告**: 便于阅读和分享
3. **JSON报告**: 便于程序化处理

报告包含以下指标：
- 执行时间（均值和标准差）
- 峰值内存使用
- 加速比（相对于numpy后端）
- 正确性验证
- 电路参数（量子比特数、深度、门数量）
- 吞吐率（门操作/秒）

## 配置选项

可以通过修改`QASMBenchConfig`类来自定义测试：

```python
config = QASMBenchConfig()
config.num_runs = 5        # 运行次数
config.warmup_runs = 1     # 预热次数  
config.output_formats = ['csv', 'markdown', 'json']  # 输出格式
config.baseline_backend = "numpy"  # 基准后端
```

## 注意事项

1. **内存要求**: 大规模电路测试需要足够的内存
2. **时间要求**: 大规模电路测试可能需要较长时间
3. **依赖项**: 确保已安装所有Qibo后端（qibojit, qibotn, qiboml）
4. **文件路径**: 确保QASMBench目录结构正确

## 故障排除

如果遇到问题：

1. 检查QASMBench目录是否存在且结构正确
2. 确保已安装所有必要的Python包
3. 查看错误信息并相应调整配置
4. 对于大规模电路，考虑减少运行次数或使用更强大的硬件

## 许可证

本项目基于QASMBench和Qibo框架，遵循相应的开源许可证。