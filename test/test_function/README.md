
qibo_profiler.py是一个用于量子电路性能分析的工具，主要功能是测量和评估量子电路的执行效率、资源使用情况以及计算保真度。该工具采用模块化设计，包含多个功能类，可以全面分析量子电路的性能特征。
qprofun.py是qibo_profiler.py的简化版本，用于演示如何使用该工具进行量子电路性能分析。
主要使用示例在test_qibo_profiler.ipynb中，使用时看一下即可。

qibo_profiler.py的主要组件包括：
主要组件：
1. MetadataCollector：收集分析器版本和时间戳等元数据
2. InputAnalyzer：分析电路属性和环境配置
3. BenchmarkManager：管理基准状态计算和缓存
4. ExecutionEngine：执行电路并收集性能指标
5. ResultProcessor：处理原始数据并生成分析报告

使用方法：
1. 基本使用：
```python
from qibo.models import Circuit
import sys
import os
# 直接使用绝对路径
sys.path.append('E:/qiboenv/test/test_function')
from qibo_profiler import profile_circuit  # 导入用于分析电路性能的函数
from qibo_profiler import generate_markdown_report  # 导入用于生成Markdown格式报告的函数

# 创建或获取量子电路
circuit = Circuit(nqubits=10)

# 分析电路性能
report = profile_circuit(
    circuit,
    n_runs=3,              # 运行次数
    mode='basic',          # 分析模式 (这是一个可扩展的方向)
    calculate_fidelity=True # 是否计算保真度
)
```
生成markdown报告
```python
from qibo_profiler import generate_markdown_report

# 生成Markdown格式的报告
generate_markdown_report(report, output_path='report.md')
```

2. 高级使用：
```python
from qibo_profiler import profile_circuit

# 创建或获取量子电路
circuit = Circuit(nqubits=10)

# 分析电路性能
report = profile_circuit(
    circuit,
    n_runs=3,              # 运行次数
    mode='advanced',       # 分析模式 (这是一个可扩展的方向)
    calculate_fidelity=True # 是否计算保真度
)

2. 配置选项：
- n_runs：指定电路运行的次数，用于计算平均性能
- mode：分析模式，支持'basic'等模式
- calculate_fidelity：是否计算与基准状态的保真度

3. 输出报告结构：
```python
{
    "metadata": {
        "profiler_version": "1.0",
        "timestamp_utc": "2025-10-26T11:30:00Z"
    },
    "inputs": {
        "profiler_settings": {...},
        "circuit_properties": {...},
        "environment": {...}
    },
    "results": {
        "summary": {
            "runtime_avg": {...},
            "runtime_std_dev": {...},
            "cpu_utilization_avg": {...},
            "fidelity": {...}
        },
        "raw_metrics": {...}
    },
    "error": null
}
```

4. 性能指标：
- 运行时间统计（平均值、标准差）
- CPU利用率（平均值、标准差）
- 内存使用情况（平均使用量、峰值）
- 量子态保真度（可选）

5. 环境信息收集：
- Qibo后端配置
- Python版本
- CPU信息（型号、核心数）
- 系统内存信息

该工具支持多种后端配置，包括numpy、qibojit、qibotn等，可以灵活适应不同的计算环境。通过缓存机制优化基准状态的计算，提高分析效率。生成的报告采用JSON格式，便于后续处理和分析。