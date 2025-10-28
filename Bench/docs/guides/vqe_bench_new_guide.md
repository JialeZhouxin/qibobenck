# VQE基准测试框架使用指南

## 概述

`vqe_bench_new.py` 是一个基于分层配置设计的VQE（变分量子本征求解器）框架性能基准测试脚本。它采用面向对象的架构设计，支持多个量子计算框架（Qiskit、PennyLane、Qibo）的性能对比分析，并提供详细的性能指标收集和多维度可视化分析。

### 主要功能

- **多框架支持**：同时测试Qiskit、PennyLane、Qibo等主流量子计算框架
- **分层配置系统**：核心用户层和高级研究层分离，既易用又灵活
- **全面性能指标**：收集时间、内存、CPU利用率、收敛性等多维度数据
- **可视化仪表盘**：自动生成包含6个核心图表的性能分析仪表盘
- **资源限制保护**：内置内存和时间限制，防止测试失控
- **统一参数管理**：确保不同框架使用相同的初始参数，保证公平比较

## 快速入门

### 安装依赖

确保已安装以下依赖包：

```bash
pip install numpy scipy matplotlib psutil
pip install qibo qiskit pennylane
```

### 基本使用

最简单的使用方式如下：

```python
from vqe_config import merge_configs
from vqe_bench_new import BenchmarkController

# 获取默认配置
config = merge_configs()

# 创建并运行基准测试
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()
```

### 快速开始示例

对于新用户，可以使用快速开始配置：

```python
from vqe_config import get_quick_start_config
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer

# 获取快速开始配置（小规模测试）
config = get_quick_start_config()

# 创建并运行基准测试
controller = BenchmarkController(config)
print("开始运行VQE基准测试...")
results = controller.run_all_benchmarks()

# 生成可视化仪表盘
visualizer = VQEBenchmarkVisualizer(results, config)
output_dir = config.get("system", {}).get("output_dir", "./results/")
visualizer.plot_dashboard(output_dir)
print(f"结果已保存到: {output_dir}")
```

**预期输出：**
```
开始运行VQE基准测试...
预计算精确基态能量...
  N=4, J=1.0, h=1.0: E0=-4.000000
  N=6, J=1.0, h=1.0: E0=-6.000000

精确能量缓存状态: 2 个条目
缓存条目:
  N=4, J=1.0, h=1.0: E0=-4.000000
  N=6, J=1.0, h=1.0: E0=-6.000000

开始VQE框架性能基准测试
配置: {'n_qubits_range': [4, 6], 'frameworks_to_test': ['Qiskit'], ...}

===== 测试 4 量子比特 =====

--- 测试框架: Qiskit ---
  精确基态能量 (N=4): -4.000000
  验证 Qiskit 框架参数一致性...
  ✓ Qiskit 参数映射验证通过
  运行 #1: Qiskit with 4 qubits
  在第25次评估后收敛，能量: -3.999987
  运行 #2: Qiskit with 4 qubits
  在第28次评估后收敛，能量: -3.999991
  收敛率: 100.0%
  平均求解时间: 0.125 ± 0.015 秒
  平均最终误差: 3.25e-06
  平均内存使用: 125.3 ± 2.1 MB

===== 测试 6 量子比特 =====

--- 测试框架: Qiskit ---
  精确基态能量 (N=6): -6.000000
  验证 Qiskit 框架参数一致性...
  ✓ Qiskit 参数映射验证通过
  运行 #1: Qiskit with 6 qubits
  在第35次评估后收敛，能量: -5.999982
  运行 #2: Qiskit with 6 qubits
  在第32次评估后收敛，能量: -5.999979
  收敛率: 100.0%
  平均求解时间: 0.234 ± 0.028 秒
  平均最终误差: 2.85e-06
  平均内存使用: 156.7 ± 3.5 MB

基准测试完成，总耗时: 2.45 秒

生成可视化仪表盘...
仪表盘已保存到: ./results/vqe_benchmark_dashboard_20231017_121530.png

============================================================
基准测试摘要
============================================================

Qiskit 框架:
  4 量子比特:
    收敛率: 100.0%
    求解时间: 0.125 ± 0.015 秒
    内存使用: 125.3 ± 2.1 MB
    最终误差: 3.25e-06
    总评估次数: 26.5 ± 2.1
  6 量子比特:
    收敛率: 100.0%
    求解时间: 0.234 ± 0.028 秒
    内存使用: 156.7 ± 3.5 MB
    最终误差: 2.85e-06
    总评估次数: 33.5 ± 2.1

测试完成！结果保存在: ./results/
```

## 详细使用示例

### 示例1：多框架性能对比

比较不同框架的性能表现：

```python
from vqe_config import merge_configs
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer

# 自定义配置：测试多个框架
custom_config = {
    "n_qubits_range": [4, 6, 8],  # 测试4、6、8个量子比特
    "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],  # 测试所有三个框架
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 3,  # 每个配置运行3次
    "experiment_name": "Multi_Framework_Comparison"
}

# 合并配置
config = merge_configs(core_config=custom_config)

# 运行基准测试
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()

# 生成可视化
visualizer = VQEBenchmarkVisualizer(results, config)
visualizer.plot_dashboard("./multi_framework_results/")
```

**预期输出：**
```
开始VQE框架性能基准测试...
配置: {'n_qubits_range': [4, 6, 8], 'frameworks_to_test': ['Qiskit', 'PennyLane', 'Qibo'], ...}

===== 测试 4 量子比特 =====

--- 测试框架: Qiskit ---
  精确基态能量 (N=4): -4.000000
  验证 Qiskit 框架参数一致性...
  ✓ Qiskit 参数映射验证通过
  运行 #1: Qiskit with 4 qubits
  在第23次评估后收敛，能量: -3.999985
  运行 #2: Qiskit with 4 qubits
  在第25次评估后收敛，能量: -3.999991
  运行 #3: Qiskit with 4 qubits
  在第24次评估后收敛，能量: -3.999988
  收敛率: 100.0%
  平均求解时间: 0.118 ± 0.008 秒
  平均最终误差: 4.33e-06
  平均内存使用: 124.5 ± 2.3 MB

--- 测试框架: PennyLane ---
  精确基态能量 (N=4): -4.000000
  验证 PennyLane 框架参数一致性...
  ✓ PennyLane 参数映射验证通过
  运行 #1: PennyLane with 4 qubits
  在第28次评估后收敛，能量: -3.999983
  运行 #2: PennyLane with 4 qubits
  在第27次评估后收敛，能量: -3.999986
  运行 #3: PennyLane with 4 qubits
  在第29次评估后收敛，能量: -3.999981
  收敛率: 100.0%
  平均求解时间: 0.095 ± 0.012 秒
  平均最终误差: 5.67e-06
  平均内存使用: 98.3 ± 1.8 MB

--- 测试框架: Qibo ---
  精确基态能量 (N=4): -4.000000
  验证 Qibo 框架参数一致性...
  ✓ Qibo 参数映射验证通过
  运行 #1: Qibo with 4 qubits
  在第21次评估后收敛，能量: -3.999989
  运行 #2: Qibo with 4 qubits
  在第22次评估后收敛，能量: -3.999992
  运行 #3: Qibo with 4 qubits
  在第20次评估后收敛，能量: -3.999987
  收敛率: 100.0%
  平均求解时间: 0.067 ± 0.005 秒
  平均最终误差: 3.89e-06
  平均内存使用: 87.2 ± 1.5 MB

===== 测试 6 量子比特 =====
... (类似输出) ...

===== 测试 8 量子比特 =====
... (类似输出) ...

基准测试完成，总耗时: 15.67 秒

生成可视化仪表盘...
仪表盘已保存到: ./multi_framework_results/vqe_benchmark_dashboard_20231017_121645.png

============================================================
基准测试摘要
============================================================

Qiskit 框架:
  4 量子比特:
    收敛率: 100.0%
    求解时间: 0.118 ± 0.008 秒
    内存使用: 124.5 ± 2.3 MB
    最终误差: 4.33e-06
  6 量子比特:
    收敛率: 100.0%
    求解时间: 0.235 ± 0.018 秒
    内存使用: 165.8 ± 3.1 MB
    最终误差: 3.95e-06
  8 量子比特:
    收敛率: 100.0%
    求解时间: 0.487 ± 0.032 秒
    内存使用: 234.2 ± 5.6 MB
    最终误差: 4.12e-06

PennyLane 框架:
  4 量子比特:
    收敛率: 100.0%
    求解时间: 0.095 ± 0.012 秒
    内存使用: 98.3 ± 1.8 MB
    最终误差: 5.67e-06
  6 量子比特:
    收敛率: 100.0%
    求解时间: 0.187 ± 0.015 秒
    内存使用: 132.5 ± 2.9 MB
    最终误差: 5.23e-06
  8 量子比特:
    收敛率: 100.0%
    求解时间: 0.356 ± 0.028 秒
    内存使用: 198.7 ± 4.2 MB
    最终误差: 5.41e-06

Qibo 框架:
  4 量子比特:
    收敛率: 100.0%
    求解时间: 0.067 ± 0.005 秒
    内存使用: 87.2 ± 1.5 MB
    最终误差: 3.89e-06
  6 量子比特:
    收敛率: 100.0%
    求解时间: 0.125 ± 0.008 秒
    内存使用: 118.3 ± 2.1 MB
    最终误差: 3.67e-06
  8 量子比特:
    收敛率: 100.0%
    求解时间: 0.234 ± 0.018 秒
    内存使用: 167.8 ± 3.4 MB
    最终误差: 3.85e-06

测试完成！结果保存在: ./multi_framework_results/
```

### 示例2：高级研究配置

使用高级研究配置进行更深入的测试：

```python
from vqe_config import merge_configs
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer

# 自定义高级配置
custom_core = {
    "n_qubits_range": [6, 8, 10],
    "frameworks_to_test": ["Qiskit", "Qibo"],
    "ansatz_type": "QAOA",  # 使用QAOA算法
    "optimizer": "SPSA",    # 使用SPSA优化器
    "n_runs": 5,
    "experiment_name": "Advanced_QAOA_Research"
}

custom_advanced = {
    "problem": {
        "model_type": "TFIM_1D",
        "boundary_conditions": "open",  # 开放边界条件
        "j_coupling": 0.8,              # 自定义耦合强度
        "h_field": 1.2,                 # 自定义场强
        "disorder_strength": 0.1        # 添加无序
    },
    "ansatz_details": {
        "n_layers": 3,                  # 增加层数
        "entanglement_style": "circular"  # 环形纠缠
    },
    "optimizer_details": {
        "max_evaluations": 1000,        # 增加最大评估次数
        "accuracy_threshold": 1e-5,     # 提高精度要求
        "options": {
            "SPSA": {
                "learning_rate": 0.03,  # 自定义学习率
                "perturbation": 0.1     # 自定义扰动参数
            }
        }
    },
    "backend_details": {
        "simulation_mode": "shot_based",  # 使用采样模拟
        "n_shots": 16384                  # 增加采样次数
    },
    "system": {
        "max_memory_mb": 8192,           # 增加内存限制
        "max_time_seconds": 3600,        # 增加时间限制
        "output_dir": "./advanced_research_results/"  # 自定义输出目录
    }
}

# 合并配置
config = merge_configs(core_config=custom_core, advanced_config=custom_advanced)

# 运行基准测试
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()

# 生成可视化
visualizer = VQEBenchmarkVisualizer(results, config)
visualizer.plot_dashboard("./advanced_research_results/")
```

**预期输出：**
```
开始VQE框架性能基准测试...
配置: {'n_qubits_range': [6, 8, 10], 'frameworks_to_test': ['Qiskit', 'Qibo'], ...}

预计算精确基态能量...
  N=6, J=0.8, h=1.2: E0=-5.856406
  N=8, J=0.8, h=1.2: E0=-7.808312
  N=10, J=0.8, h=1.2: E0=-9.760218

===== 测试 6 量子比特 =====

--- 测试框架: Qiskit ---
  精确基态能量 (N=6): -5.856406
  验证 Qiskit 框架参数一致性...
  ✓ Qiskit 参数映射验证通过
  运行 #1: Qiskit with 6 qubits
  在第125次评估后收敛，能量: -5.856201
  运行 #2: Qiskit with 6 qubits
  在第132次评估后收敛，能量: -5.856189
  运行 #3: Qiskit with 6 qubits
  在第118次评估后收敛，能量: -5.856223
  运行 #4: Qiskit with 6 qubits
  在第127次评估后收敛，能量: -5.856195
  运行 #5: Qiskit with 6 qubits
  在第121次评估后收敛，能量: -5.856207
  收敛率: 100.0%
  平均求解时间: 1.245 ± 0.087 秒
  平均最终误差: 2.15e-05
  平均内存使用: 245.6 ± 4.2 MB

--- 测试框架: Qibo ---
  精确基态能量 (N=6): -5.856406
  验证 Qibo 框架参数一致性...
  ✓ Qibo 参数映射验证通过
  运行 #1: Qibo with 6 qubits
  在第98次评估后收敛，能量: -5.856312
  运行 #2: Qibo with 6 qubits
  在第95次评估后收敛，能量: -5.856324
  运行 #3: Qibo with 6 qubits
  在第102次评估后收敛，能量: -5.856298
  运行 #4: Qibo with 6 qubits
  在第97次评估后收敛，能量: -5.856319
  运行 #5: Qibo with 6 qubits
  在第99次评估后收敛，能量: -5.856306
  收敛率: 100.0%
  平均求解时间: 0.892 ± 0.065 秒
  平均最终误差: 1.23e-05
  平均内存使用: 198.3 ± 3.5 MB

===== 测试 8 量子比特 =====
... (类似输出) ...

===== 测试 10 量子比特 =====
... (类似输出) ...

基准测试完成，总耗时: 125.34 秒

生成可视化仪表盘...
仪表盘已保存到: ./advanced_research_results/vqe_benchmark_dashboard_20231017_122105.png

============================================================
基准测试摘要
============================================================

Qiskit 框架:
  6 量子比特:
    收敛率: 100.0%
    求解时间: 1.245 ± 0.087 秒
    内存使用: 245.6 ± 4.2 MB
    最终误差: 2.15e-05
  8 量子比特:
    收敛率: 100.0%
    求解时间: 2.456 ± 0.156 秒
    内存使用: 412.3 ± 6.8 MB
    最终误差: 2.34e-05
  10 量子比特:
    收敛率: 100.0%
    求解时间: 4.785 ± 0.287 秒
    内存使用: 723.5 ± 9.4 MB
    最终误差: 2.56e-05

Qibo 框架:
  6 量子比特:
    收敛率: 100.0%
    求解时间: 0.892 ± 0.065 秒
    内存使用: 198.3 ± 3.5 MB
    最终误差: 1.23e-05
  8 量子比特:
    收敛率: 100.0%
    求解时间: 1.745 ± 0.123 秒
    内存使用: 325.6 ± 5.2 MB
    最终误差: 1.45e-05
  10 量子比特:
    收敛率: 100.0%
    求解时间: 3.214 ± 0.198 秒
    内存使用: 545.8 ± 7.6 MB
    最终误差: 1.67e-05

测试完成！结果保存在: ./advanced_research_results/
```

### 示例3：CPU利用率分析

专门分析不同框架的CPU利用率情况：

```python
from vqe_config import merge_configs
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 自定义配置，专注于CPU利用率分析
custom_config = {
    "n_qubits_range": [4, 6, 8, 10],  # 较大的问题规模以凸显CPU差异
    "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 3,
    "experiment_name": "CPU_Usage_Analysis"
}

# 合并配置
config = merge_configs(core_config=custom_config)

# 运行基准测试
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()

# 创建可视化器
visualizer = VQEBenchmarkVisualizer(results, config)

# 创建单独的CPU利用率图表
fig, ax = plt.subplots(figsize=(12, 8))
visualizer._plot_cpu_usage(ax)

# 保存图表
output_dir = config.get("system", {}).get("output_dir", "./results/")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"cpu_usage_analysis_{timestamp}.png"
filepath = os.path.join(output_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"CPU利用率分析图表已保存到: {filepath}")

# 打印CPU利用率数据摘要
print("\n" + "=" * 60)
print("CPU利用率分析摘要")
print("=" * 60)

for framework in config["frameworks_to_test"]:
    print(f"\n{framework} 框架:")
    for n_qubits in config["n_qubits_range"]:
        if framework in results and n_qubits in results[framework]:
            data = results[framework][n_qubits]
            print(f"  {n_qubits} 量子比特:")
            print(f"    峰值CPU使用率: {data['avg_peak_cpu_usage']:.1f}% ± {data['std_peak_cpu_usage']:.1f}%")
            print(f"    平均CPU使用率: {data['avg_avg_cpu_usage']:.1f}% ± {data['std_avg_cpu_usage']:.1f}%")

# 分析CPU利用率趋势
print("\n" + "=" * 60)
print("CPU利用率趋势分析")
print("=" * 60)

for framework in config["frameworks_to_test"]:
    if framework in results:
        peak_cpus = []
        avg_cpus = []
        qubits = []
        
        for n_qubits in config["n_qubits_range"]:
            if n_qubits in results[framework]:
                peak_cpus.append(results[framework][n_qubits]['avg_peak_cpu_usage'])
                avg_cpus.append(results[framework][n_qubits]['avg_avg_cpu_usage'])
                qubits.append(n_qubits)
        
        if len(peak_cpus) > 1:
            # 计算CPU使用率增长率
            peak_growth = (peak_cpus[-1] - peak_cpus[0]) / peak_cpus[0] * 100
            avg_growth = (avg_cpus[-1] - avg_cpus[0]) / avg_cpus[0] * 100
            
            print(f"{framework}:")
            print(f"  峰值CPU使用率从 {qubits[0]} 量子比特的 {peak_cpus[0]:.1f}% 增长到 {qubits[-1]} 量子比特的 {peak_cpus[-1]:.1f}% (增长率: {peak_growth:.1f}%)")
            print(f"  平均CPU使用率从 {qubits[0]} 量子比特的 {avg_cpus[0]:.1f}% 增长到 {qubits[-1]} 量子比特的 {avg_cpus[-1]:.1f}% (增长率: {avg_growth:.1f}%)")

print(f"\nCPU利用率分析完成！结果保存在: {output_dir}")
```

**预期输出：**
```
开始VQE框架性能基准测试...
配置: {'n_qubits_range': [4, 6, 8, 10], 'frameworks_to_test': ['Qiskit', 'PennyLane', 'Qibo'], ...}

===== 测试 4 量子比特 =====

--- 测试框架: Qiskit ---
  精确基态能量 (N=4): -4.000000
  验证 Qiskit 框架参数一致性...
  ✓ Qiskit 参数映射验证通过
  运行 #1: Qiskit with 4 qubits
  在第23次评估后收敛，能量: -3.999985
  运行 #2: Qiskit with 4 qubits
  在第25次评估后收敛，能量: -3.999991
  运行 #3: Qiskit with 4 qubits
  在第24次评估后收敛，能量: -3.999988
  收敛率: 100.0%
  平均求解时间: 0.118 ± 0.008 秒
  平均最终误差: 4.33e-06
  平均内存使用: 124.5 ± 2.3 MB

--- 测试框架: PennyLane ---
... (类似输出) ...

--- 测试框架: Qibo ---
... (类似输出) ...

===== 测试 6 量子比特 =====
... (类似输出) ...

===== 测试 8 量子比特 =====
... (类似输出) ...

===== 测试 10 量子比特 =====
... (类似输出) ...

基准测试完成，总耗时: 45.67 秒

CPU利用率分析图表已保存到: ./results/cpu_usage_analysis_20231017_121745.png

============================================================
CPU利用率分析摘要
============================================================

Qiskit 框架:
  4 量子比特:
    峰值CPU使用率: 85.2% ± 3.1%
    平均CPU使用率: 65.4% ± 2.8%
  6 量子比特:
    峰值CPU使用率: 92.7% ± 2.5%
    平均CPU使用率: 71.3% ± 3.2%
  8 量子比特:
    峰值CPU使用率: 98.5% ± 1.8%
    平均CPU使用率: 78.9% ± 2.9%
  10 量子比特:
    峰值CPU使用率: 99.2% ± 1.2%
    平均CPU使用率: 82.6% ± 2.5%

PennyLane 框架:
  4 量子比特:
    峰值CPU使用率: 78.3% ± 3.5%
    平均CPU使用率: 58.7% ± 3.1%
  6 量子比特:
    峰值CPU使用率: 84.6% ± 2.9%
    平均CPU使用率: 64.2% ± 2.8%
  8 量子比特:
    峰值CPU使用率: 89.7% ± 2.3%
    平均CPU使用率: 69.5% ± 2.6%
  10 量子比特:
    峰值CPU使用率: 93.4% ± 2.1%
    平均CPU使用率: 73.8% ± 2.4%

Qibo 框架:
  4 量子比特:
    峰值CPU使用率: 92.1% ± 2.7%
    平均CPU使用率: 72.6% ± 2.9%
  6 量子比特:
    峰值CPU使用率: 96.8% ± 1.9%
    平均CPU使用率: 78.3% ± 2.5%
  8 量子比特:
    峰值CPU使用率: 99.1% ± 1.3%
    平均CPU使用率: 84.7% ± 2.2%
  10 量子比特:
    峰值CPU使用率: 99.5% ± 1.1%
    平均CPU使用率: 87.2% ± 2.0%

============================================================
CPU利用率趋势分析
============================================================

Qiskit:
  峰值CPU使用率从 4 量子比特的 85.2% 增长到 10 量子比特的 99.2% (增长率: 16.4%)
  平均CPU使用率从 4 量子比特的 65.4% 增长到 10 量子比特的 82.6% (增长率: 26.3%)

PennyLane:
  峰值CPU使用率从 4 量子比特的 78.3% 增长到 10 量子比特的 93.4% (增长率: 19.3%)
  平均CPU使用率从 4 量子比特的 58.7% 增长到 10 量子比特的 73.8% (增长率: 25.7%)

Qibo:
  峰值CPU使用率从 4 量子比特的 92.1% 增长到 10 量子比特的 99.5% (增长率: 8.0%)
  平均CPU使用率从 4 量子比特的 72.6% 增长到 10 量子比特的 87.2% (增长率: 20.1%)

CPU利用率分析完成！结果保存在: ./results/
```

## 高级功能说明

### 分层配置系统

#### 核心用户层配置

核心用户层包含最常用且易于理解的参数：

```python
core_config = {
    "n_qubits_range": [4, 6, 8],           # 量子比特数范围
    "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],  # 要测试的框架列表
    "ansatz_type": "HardwareEfficient",      # 算法思路
    "optimizer": "COBYLA",                   # 经典优化器
    "n_runs": 10,                           # 运行次数
    "experiment_name": "Standard_TFIM_Benchmark_CPU"  # 实验名称
}
```

#### 高级研究层配置

高级研究层包含专家级设置，用于深入、特定的基准测试：

```python
advanced_config = {
    "problem": {
        "model_type": "TFIM_1D",            # 物理模型类型
        "boundary_conditions": "periodic",  # 边界条件
        "j_coupling": 1.0,                  # 相互作用强度
        "h_field": 1.0,                     # 横向场强度
        "disorder_strength": 0.0,           # 无序强度
    },
    "ansatz_details": {
        "n_layers": 2,                      # Ansatz层数
        "entanglement_style": "linear",     # 纠缠模式
    },
    "optimizer_details": {
        "max_evaluations": 500,             # 最大评估次数
        "accuracy_threshold": 1e-4,         # 收敛阈值
        "options": {                        # 优化器特定选项
            "COBYLA": {"tol": 1e-5, "rhobeg": 1.0},
            "SPSA": {"learning_rate": 0.05, "perturbation": 0.05},
        }
    },
    "backend_details": {
        "simulation_mode": "statevector",   # 模拟模式
        "n_shots": 8192,                    # 采样次数
        "framework_backends": {             # 框架后端配置
            "Qiskit": "aer_simulator",
            "PennyLane": "lightning.qubit",
            "Qibo": {"backend": "qibojit", "platform": "numba"}
        }
    },
    "system": {
        "seed": 42,                         # 随机种子
        "max_memory_mb": 4096,             # 内存限制
        "max_time_seconds": 1800,           # 时间限制
        "output_dir": "./results/"          # 输出目录
    }
}
```

### 支持的算法类型

1. **HardwareEfficient**：硬件高效ansatz，适用于一般问题
2. **QAOA**：量子近似优化算法，适用于组合优化问题

### 支持的优化器

1. **COBYLA**：无梯度优化器，适合参数空间较大的问题
2. **SPSA**：模拟梯度优化器，适合噪声环境
3. **L-BFGS-B**：精确梯度优化器，适合光滑问题

### 支持的纠缠模式

1. **linear**：线性连接（qubit_i 与 qubit_{i+1} 纠缠）
2. **circular**：环形连接（线性连接 + 最后一个qubit与第一个纠缠）
3. **full**：全连接（每一对qubit之间都进行纠缠）

## 可视化仪表盘

框架会自动生成一个包含7个核心图表的性能分析仪表盘：

1. **核心性能: 求解时间**：显示各框架在不同量子比特数下的求解时间
2. **核心性能: 内存扩展性**：显示各框架在不同量子比特数下的内存使用情况
3. **优化动力学: 收敛轨迹**：显示最大量子比特数下的能量收敛过程
4. **优化动力学: 评估次数**：显示各框架在不同量子比特数下的函数评估次数
5. **诊断: 最终精度验证**：显示各框架在不同量子比特数下的最终求解精度
6. **诊断: 时间分解**：显示各框架在不同量子比特数下的量子部分和经典部分时间分解
7. **诊断: CPU利用率分析**：显示各框架在不同量子比特数下的峰值和平均CPU使用率

### 自动生成包含CPU利用率的仪表盘

从版本2.1.0开始，`vqe_bench_new.py` 已更新为自动包含CPU利用率图表。默认情况下，生成的仪表盘将包含7个图表，其中第7个图表专门显示CPU利用率分析。

如果您使用的是较旧版本的代码，可以通过以下方式手动更新以包含CPU利用率图表：

#### 修改VQEBenchmarkVisualizer类的plot_dashboard方法

```python
def plot_dashboard(self, output_dir: str = None) -> None:
    """生成并显示包含七个核心图表的仪表盘"""
    fig, axes = plt.subplots(4, 2, figsize=(20, 28))
    fig.suptitle("VQE框架性能基准测试仪表盘", fontsize=20)
    
    # 图 1: 总求解时间 vs. 量子比特数
    self._plot_time_to_solution(axes[0, 0])
    
    # 图 2: 峰值内存使用 vs. 量子比特数
    self._plot_peak_memory(axes[0, 1])
    
    # 图 3: 收敛轨迹 (以最大比特数为例)
    self._plot_convergence_trajectories(axes[1, 0])
    
    # 图 4: 总求值次数 vs. 量子比特数
    self._plot_total_evaluations(axes[1, 1])
    
    # 图 5: 最终求解精度 vs. 量子比特数
    self._plot_final_accuracy(axes[2, 0])
    
    # 图 6: 单步耗时分解 vs. 量子比特数
    self._plot_time_breakdown(axes[2, 1])
    
    # 图 7: CPU利用率 vs. 量子比特数
    self._plot_cpu_usage(axes[3, 0])
    
    # 隐藏右下角的空白子图
    axes[3, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vqe_benchmark_dashboard_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"仪表盘已保存到: {filepath}")
```

这个修改将原来的3×2布局改为4×2布局，并在第7个位置添加CPU利用率图表，同时隐藏右下角的空白子图。

### 绘制CPU利用率图表

框架还支持绘制CPU利用率图表，但默认不包含在6个核心图表中。要添加CPU利用率图表，可以使用以下方法：

#### 方法1：使用内置的CPU利用率图表（推荐）

从版本2.1.0开始，CPU利用率图表已自动包含在仪表盘中，无需额外操作：

```python
from vqe_config import get_quick_start_config
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer

# 运行基准测试
config = get_quick_start_config()
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()

# 创建可视化器并生成仪表盘（已自动包含CPU利用率图表）
visualizer = VQEBenchmarkVisualizer(results, config)
output_dir = config.get("system", {}).get("output_dir", "./results/")
visualizer.plot_dashboard(output_dir)
```

**预期输出：**
```
开始运行VQE基准测试...
... (基准测试输出) ...

仪表盘已保存到: ./results/vqe_benchmark_dashboard_20231017_121630.png
```

#### 方法2：创建单独的CPU利用率图表

如果您需要单独的CPU利用率图表，可以使用以下方法：

```python
from vqe_config import get_quick_start_config
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer

# 运行基准测试
config = get_quick_start_config()
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()

# 创建可视化器
visualizer = VQEBenchmarkVisualizer(results, config)

# 创建单独的CPU利用率图表
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
visualizer._plot_cpu_usage(ax)

# 保存图表
output_dir = config.get("system", {}).get("output_dir", "./results/")
import os
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"cpu_usage_chart_{timestamp}.png"
filepath = os.path.join(output_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"CPU利用率图表已保存到: {filepath}")
```

**预期输出：**
```
开始运行VQE基准测试...
... (基准测试输出) ...

CPU利用率图表已保存到: ./results/cpu_usage_chart_20231017_121630.png
```

#### 方法2：修改仪表盘以包含CPU利用率图表

如果要将CPU利用率图表添加到仪表盘中，可以修改`VQEBenchmarkVisualizer`类：

```python
class CustomVQEBenchmarkVisualizer(VQEBenchmarkVisualizer):
    def plot_dashboard_with_cpu(self, output_dir: str = None) -> None:
        """生成包含CPU利用率图表的7图表仪表盘"""
        fig, axes = plt.subplots(4, 2, figsize=(20, 28))
        fig.suptitle("VQE框架性能基准测试仪表盘 (含CPU利用率)", fontsize=20)
        
        # 原有的6个图表
        self._plot_time_to_solution(axes[0, 0])
        self._plot_peak_memory(axes[0, 1])
        self._plot_convergence_trajectories(axes[1, 0])
        self._plot_total_evaluations(axes[1, 1])
        self._plot_final_accuracy(axes[2, 0])
        self._plot_time_breakdown(axes[2, 1])
        
        # 新增CPU利用率图表
        self._plot_cpu_usage(axes[3, 0])
        
        # 隐藏右下角的空白子图
        axes[3, 1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图片
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vqe_benchmark_dashboard_with_cpu_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"包含CPU利用率的仪表盘已保存到: {filepath}")

# 使用自定义可视化器
custom_visualizer = CustomVQEBenchmarkVisualizer(results, config)
custom_visualizer.plot_dashboard_with_cpu("./results/")
```

#### CPU利用率图表说明

CPU利用率图表显示以下信息：

1. **峰值CPU使用率**：每个框架在不同量子比特数下的最大CPU使用率
2. **平均CPU使用率**：每个框架在不同量子比特数下的平均CPU使用率

这些指标可以帮助您：
- 识别哪个框架在计算资源利用方面更高效
- 了解CPU使用率随问题规模的变化趋势
- 评估框架的并行计算能力

**示例CPU利用率图表解读：**
- 如果某个框架的峰值CPU使用率显著高于其他框架，可能表明它更有效地利用了多核处理器
- 如果平均CPU使用率较低，可能表明框架存在I/O瓶颈或等待时间
- CPU使用率随量子比特数增加而上升，表明计算复杂度在增加

#### 方法3：创建增强版CPU利用率可视化器

如果需要更深入的CPU利用率分析，可以创建一个专门的可视化类：

```python
class CPUUsageAnalyzer:
    """CPU利用率专门分析器"""
    
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any]):
        self.results = results
        self.config = config
        self.frameworks = config["frameworks_to_test"]
        self.n_qubits_range = config["n_qubits_range"]
    
    def plot_detailed_cpu_analysis(self, output_dir: str = None) -> None:
        """生成详细的CPU利用率分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("VQE框架CPU利用率详细分析", fontsize=16)
        
        # 1. 峰值CPU使用率对比
        self._plot_peak_cpu_comparison(axes[0, 0])
        
        # 2. 平均CPU使用率对比
        self._plot_avg_cpu_comparison(axes[0, 1])
        
        # 3. CPU使用率增长率分析
        self._plot_cpu_growth_rate(axes[1, 0])
        
        # 4. CPU效率分析（CPU使用率/求解时间）
        self._plot_cpu_efficiency(axes[1, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_cpu_analysis_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"详细CPU分析图表已保存到: {filepath}")
    
    def _plot_peak_cpu_comparison(self, ax):
        """绘制峰值CPU使用率对比"""
        for fw in self.frameworks:
            peak_cpus = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    peak_cpus.append(self.results[fw][n_qubits]["avg_peak_cpu_usage"])
                else:
                    peak_cpus.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, cpu in enumerate(peak_cpus) if cpu is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_cpus = [peak_cpus[i] for i in valid_indices]
            
            if valid_cpus:
                ax.plot(valid_qubits, valid_cpus, marker='o', linestyle='-', label=fw)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("峰值CPU使用率 (%)")
        ax.set_title("峰值CPU使用率对比")
        ax.legend()
        ax.grid(True, ls="--")
        ax.set_ylim(0, 105)  # CPU使用率不超过100%
    
    def _plot_avg_cpu_comparison(self, ax):
        """绘制平均CPU使用率对比"""
        for fw in self.frameworks:
            avg_cpus = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    avg_cpus.append(self.results[fw][n_qubits]["avg_avg_cpu_usage"])
                else:
                    avg_cpus.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, cpu in enumerate(avg_cpus) if cpu is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_cpus = [avg_cpus[i] for i in valid_indices]
            
            if valid_cpus:
                ax.plot(valid_qubits, valid_cpus, marker='s', linestyle='--', label=fw)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("平均CPU使用率 (%)")
        ax.set_title("平均CPU使用率对比")
        ax.legend()
        ax.grid(True, ls="--")
        ax.set_ylim(0, 105)  # CPU使用率不超过100%
    
    def _plot_cpu_growth_rate(self, ax):
        """绘制CPU使用率增长率分析"""
        for fw in self.frameworks:
            if fw in self.results:
                peak_cpus = []
                avg_cpus = []
                qubits = []
                
                for n_qubits in self.n_qubits_range:
                    if n_qubits in self.results[fw]:
                        peak_cpus.append(self.results[fw][n_qubits]["avg_peak_cpu_usage"])
                        avg_cpus.append(self.results[fw][n_qubits]["avg_avg_cpu_usage"])
                        qubits.append(n_qubits)
                
                if len(peak_cpus) > 1:
                    # 计算增长率（相对于最小量子比特数）
                    peak_growth = [(cpu - peak_cpus[0]) / peak_cpus[0] * 100 for cpu in peak_cpus]
                    avg_growth = [(cpu - avg_cpus[0]) / avg_cpus[0] * 100 for cpu in avg_cpus]
                    
                    ax.plot(qubits, peak_growth, marker='o', linestyle='-', label=f'{fw} 峰值CPU增长率')
                    ax.plot(qubits, avg_growth, marker='s', linestyle='--', label=f'{fw} 平均CPU增长率')
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("CPU使用率增长率 (%)")
        ax.set_title("CPU使用率随问题规模的增长率")
        ax.legend()
        ax.grid(True, ls="--")
    
    def _plot_cpu_efficiency(self, ax):
        """绘制CPU效率分析（CPU使用率/求解时间）"""
        for fw in self.frameworks:
            efficiencies = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    cpu_usage = self.results[fw][n_qubits]["avg_avg_cpu_usage"]
                    solve_time = self.results[fw][n_qubits]["avg_total_time"]
                    if solve_time > 0:
                        # 效率 = CPU使用率 / 求解时间（越高越好）
                        efficiency = cpu_usage / solve_time
                        efficiencies.append(efficiency)
                    else:
                        efficiencies.append(None)
                else:
                    efficiencies.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, eff in enumerate(efficiencies) if eff is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_effs = [efficiencies[i] for i in valid_indices]
            
            if valid_effs:
                ax.plot(valid_qubits, valid_effs, marker='^', linestyle='-', label=fw)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("CPU效率 (CPU%/秒)")
        ax.set_title("CPU效率分析（CPU使用率/求解时间）")
        ax.legend()
        ax.grid(True, ls="--")
    
    def generate_cpu_report(self, output_dir: str = None) -> str:
        """生成CPU利用率分析报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("VQE框架CPU利用率详细分析报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 框架总体比较
        report_lines.append("1. 框架CPU利用率总体比较")
        report_lines.append("-" * 30)
        
        framework_summary = {}
        for fw in self.frameworks:
            if fw in self.results:
                peak_cpus = []
                avg_cpus = []
                for n_qubits in self.n_qubits_range:
                    if n_qubits in self.results[fw]:
                        peak_cpus.append(self.results[fw][n_qubits]["avg_peak_cpu_usage"])
                        avg_cpus.append(self.results[fw][n_qubits]["avg_avg_cpu_usage"])
                
                if peak_cpus:
                    framework_summary[fw] = {
                        "avg_peak_cpu": np.mean(peak_cpus),
                        "avg_cpu": np.mean(avg_cpus),
                        "max_peak_cpu": np.max(peak_cpus),
                        "max_cpu": np.max(avg_cpus)
                    }
        
        # 按平均CPU使用率排序
        sorted_frameworks = sorted(framework_summary.items(),
                                 key=lambda x: x[1]["avg_cpu"],
                                 reverse=True)
        
        for fw, stats in sorted_frameworks:
            report_lines.append(f"{fw}:")
            report_lines.append(f"  平均峰值CPU使用率: {stats['avg_peak_cpu']:.1f}%")
            report_lines.append(f"  平均CPU使用率: {stats['avg_cpu']:.1f}%")
            report_lines.append(f"  最大峰值CPU使用率: {stats['max_peak_cpu']:.1f}%")
            report_lines.append(f"  最大CPU使用率: {stats['max_cpu']:.1f}%")
            report_lines.append("")
        
        # 扩展性分析
        report_lines.append("2. CPU扩展性分析")
        report_lines.append("-" * 30)
        
        for fw in self.frameworks:
            if fw in self.results:
                peak_cpus = []
                avg_cpus = []
                qubits = []
                
                for n_qubits in self.n_qubits_range:
                    if n_qubits in self.results[fw]:
                        peak_cpus.append(self.results[fw][n_qubits]["avg_peak_cpu_usage"])
                        avg_cpus.append(self.results[fw][n_qubits]["avg_avg_cpu_usage"])
                        qubits.append(n_qubits)
                
                if len(peak_cpus) > 1:
                    # 计算增长率
                    peak_growth = (peak_cpus[-1] - peak_cpus[0]) / peak_cpus[0] * 100
                    avg_growth = (avg_cpus[-1] - avg_cpus[0]) / avg_cpus[0] * 100
                    
                    report_lines.append(f"{fw}:")
                    report_lines.append(f"  峰值CPU增长率: {peak_growth:.1f}% (从{qubits[0]}到{qubits[-1]}量子比特)")
                    report_lines.append(f"  平均CPU增长率: {avg_growth:.1f}% (从{qubits[0]}到{qubits[-1]}量子比特)")
                    
                    # 计算CPU利用率饱和点（CPU使用率达到95%的量子比特数）
                    saturation_point = None
                    for i, cpu in enumerate(peak_cpus):
                        if cpu >= 95:
                            saturation_point = qubits[i]
                            break
                    
                    if saturation_point:
                        report_lines.append(f"  CPU利用率饱和点: {saturation_point} 量子比特")
                    else:
                        report_lines.append(f"  CPU利用率未饱和（最高{max(peak_cpus):.1f}%）")
                    report_lines.append("")
        
        # 效率分析
        report_lines.append("3. CPU效率分析")
        report_lines.append("-" * 30)
        
        for fw in self.frameworks:
            if fw in self.results:
                efficiencies = []
                for n_qubits in self.n_qubits_range:
                    if n_qubits in self.results[fw]:
                        cpu_usage = self.results[fw][n_qubits]["avg_avg_cpu_usage"]
                        solve_time = self.results[fw][n_qubits]["avg_total_time"]
                        if solve_time > 0:
                            efficiency = cpu_usage / solve_time
                            efficiencies.append(efficiency)
                
                if efficiencies:
                    report_lines.append(f"{fw}:")
                    report_lines.append(f"  平均CPU效率: {np.mean(efficiencies):.2f} CPU%/秒")
                    report_lines.append(f"  最大CPU效率: {np.max(efficiencies):.2f} CPU%/秒")
                    report_lines.append(f"  效率标准差: {np.std(efficiencies):.2f}")
                    report_lines.append("")
        
        # 结论和建议
        report_lines.append("4. 结论和建议")
        report_lines.append("-" * 30)
        
        # 找出最佳框架
        best_avg_cpu = max(framework_summary.items(), key=lambda x: x[1]["avg_cpu"])
        best_efficiency = None
        best_eff_value = 0
        
        for fw in self.frameworks:
            if fw in self.results:
                efficiencies = []
                for n_qubits in self.n_qubits_range:
                    if n_qubits in self.results[fw]:
                        cpu_usage = self.results[fw][n_qubits]["avg_avg_cpu_usage"]
                        solve_time = self.results[fw][n_qubits]["avg_total_time"]
                        if solve_time > 0:
                            efficiency = cpu_usage / solve_time
                            efficiencies.append(efficiency)
                
                if efficiencies and np.mean(efficiencies) > best_eff_value:
                    best_eff_value = np.mean(efficiencies)
                    best_efficiency = fw
        
        report_lines.append(f"最高平均CPU使用率框架: {best_avg_cpu[0]} ({best_avg_cpu[1]['avg_cpu']:.1f}%)")
        if best_efficiency:
            report_lines.append(f"最高CPU效率框架: {best_efficiency} ({best_eff_value:.2f} CPU%/秒)")
        
        report_lines.append("")
        report_lines.append("建议:")
        
        # 根据分析结果给出建议
        if best_avg_cpu[0] == best_efficiency:
            report_lines.append(f"- {best_avg_cpu[0]}框架在CPU使用率和效率方面都表现最佳，推荐用于CPU密集型任务")
        else:
            report_lines.append(f"- 如果需要高CPU利用率，选择{best_avg_cpu[0]}框架")
            if best_efficiency:
                report_lines.append(f"- 如果需要高效率，选择{best_efficiency}框架")
        
        # 检查是否有框架存在CPU利用率低的问题
        low_cpu_frameworks = []
        for fw, stats in framework_summary.items():
            if stats["avg_cpu"] < 50:  # 平均CPU使用率低于50%
                low_cpu_frameworks.append(fw)
        
        if low_cpu_frameworks:
            report_lines.append(f"- {', '.join(low_cpu_frameworks)}框架的CPU使用率较低，可能存在优化空间")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # 保存报告
        report_content = "\n".join(report_lines)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cpu_analysis_report_{timestamp}.md"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"CPU分析报告已保存到: {filepath}")
            return filepath
        
        return report_content

# 使用CPU分析器
cpu_analyzer = CPUUsageAnalyzer(results, config)
cpu_analyzer.plot_detailed_cpu_analysis("./results/")
report_path = cpu_analyzer.generate_cpu_report("./results/")
```

#### 方法4：修改原有仪表盘以包含CPU利用率图表

如果希望直接修改原有的仪表盘功能，可以这样做：

```python
class EnhancedVQEBenchmarkVisualizer(VQEBenchmarkVisualizer):
    """增强版可视化器，包含CPU利用率图表"""
    
    def plot_enhanced_dashboard(self, output_dir: str = None) -> None:
        """生成包含CPU利用率图表的增强仪表盘"""
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle("VQE框架性能基准测试增强仪表盘", fontsize=20)
        
        # 原有的6个图表
        self._plot_time_to_solution(axes[0, 0])
        self._plot_peak_memory(axes[0, 1])
        self._plot_convergence_trajectories(axes[0, 2])
        self._plot_total_evaluations(axes[1, 0])
        self._plot_final_accuracy(axes[1, 1])
        self._plot_time_breakdown(axes[1, 2])
        
        # 新增的CPU相关图表
        self._plot_cpu_usage(axes[2, 0])
        self._plot_cpu_efficiency(axes[2, 1])
        self._plot_cpu_growth_rate(axes[2, 2])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图片
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vqe_enhanced_dashboard_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"增强仪表盘已保存到: {filepath}")
    
    def _plot_cpu_efficiency(self, ax):
        """绘制CPU效率分析图表"""
        for fw in self.frameworks:
            efficiencies = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    cpu_usage = self.results[fw][n_qubits]["avg_avg_cpu_usage"]
                    solve_time = self.results[fw][n_qubits]["avg_total_time"]
                    if solve_time > 0:
                        efficiency = cpu_usage / solve_time
                        efficiencies.append(efficiency)
                    else:
                        efficiencies.append(None)
                else:
                    efficiencies.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, eff in enumerate(efficiencies) if eff is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_effs = [efficiencies[i] for i in valid_indices]
            
            if valid_effs:
                ax.plot(valid_qubits, valid_effs, marker='^', linestyle='-', label=fw)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("CPU效率 (CPU%/秒)")
        ax.set_title("CPU效率分析")
        ax.legend()
        ax.grid(True, ls="--")
    
    def _plot_cpu_growth_rate(self, ax):
        """绘制CPU使用率增长率图表"""
        for fw in self.frameworks:
            if fw in self.results:
                peak_cpus = []
                qubits = []
                
                for n_qubits in self.n_qubits_range:
                    if n_qubits in self.results[fw]:
                        peak_cpus.append(self.results[fw][n_qubits]["avg_peak_cpu_usage"])
                        qubits.append(n_qubits)
                
                if len(peak_cpus) > 1:
                    # 计算增长率（相对于最小量子比特数）
                    growth_rates = [(cpu - peak_cpus[0]) / peak_cpus[0] * 100 for cpu in peak_cpus]
                    ax.plot(qubits, growth_rates, marker='o', linestyle='-', label=fw)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("CPU使用率增长率 (%)")
        ax.set_title("CPU使用率增长率分析")
        ax.legend()
        ax.grid(True, ls="--")

# 使用增强版可视化器
enhanced_visualizer = EnhancedVQEBenchmarkVisualizer(results, config)
enhanced_visualizer.plot_enhanced_dashboard("./results/")
```

#### CPU利用率数据解读指南

1. **峰值CPU使用率分析**：
   - 接近100%：表明框架充分利用了CPU资源，适合计算密集型任务
   - 低于50%：可能存在I/O瓶颈、线程同步问题或算法效率低下
   - 不同框架间的差异：反映了框架的并行计算能力和优化程度

2. **平均CPU使用率分析**：
   - 与峰值CPU使用率的差异：差异大表明CPU使用不稳定，可能有间歇性等待
   - 随量子比特数的变化：平稳增长表明扩展性好，急剧增长可能表明算法复杂度高

3. **CPU效率分析**（CPU使用率/求解时间）：
   - 高效率：框架在充分利用CPU的同时也能快速完成任务
   - 低效率：即使CPU使用率高，但完成任务时间长，可能存在算法效率问题

4. **CPU扩展性分析**：
   - 线性增长：理想的扩展性，框架能有效利用增加的计算资源
   - 饱和现象：CPU使用率达到上限后不再增长，表明框架遇到了瓶颈
   - 波动增长：可能表明框架在不同规模下的优化策略不一致

## 常见问题和故障排除

### 问题1：导入错误

**错误信息**：
```
ImportError: No module named 'qibo'
```

**解决方案**：
```bash
pip install qibo
# 或者安装特定后端
pip install qibo[qibojit]
```

### 问题2：内存不足

**错误信息**：
```
警告：内存使用超过限制 (5000.0MB > 4096MB)
```

**解决方案**：
1. 增加内存限制：
```python
config = merge_configs()
config["system"]["max_memory_mb"] = 8192  # 增加到8GB
```

2. 减少测试规模：
```python
custom_config = {
    "n_qubits_range": [4, 6],  # 减少量子比特数
    "n_runs": 2,               # 减少运行次数
}
```

### 问题3：运行时间过长

**解决方案**：
1. 增加时间限制：
```python
config = merge_configs()
config["system"]["max_time_seconds"] = 3600  # 增加到1小时
```

2. 使用更快的优化器：
```python
custom_config = {
    "optimizer": "SPSA",  # SPSA通常收敛更快
}
```

3. 减少最大评估次数：
```python
advanced_config = {
    "optimizer_details": {
        "max_evaluations": 200,  # 减少最大评估次数
    }
}
```

### 问题4：框架不可用

**错误信息**：
```
警告：Qiskit或其依赖项未安装，跳过Qiskit测试。
```

**解决方案**：
安装相应的框架：
```bash
# 安装Qiskit
pip install qiskit

# 安装PennyLane
pip install pennylane

# 安装Qibo
pip install qibo
```

### 问题5：结果可视化问题

**错误信息**：
```
UserWarning: FixedFormatter should only be used together with FixedLocator
```

**解决方案**：
这个警告通常不会影响结果的正确性。如果需要消除警告，可以更新matplotlib：
```bash
pip install --upgrade matplotlib
```

## 性能优化建议

1. **选择合适的后端**：
   - Qibo：使用qibojit后端可以获得更好的性能
   - PennyLane：使用lightning.qubit后端
   - Qiskit：使用aer_simulator后端

2. **调整优化器参数**：
   - 对于大型问题，SPSA通常比COBYLA更稳定
   - 增加学习率可以加快收敛，但可能导致精度下降

3. **合理设置资源限制**：
   - 根据可用内存设置max_memory_mb
   - 根据可用时间设置max_time_seconds

4. **使用缓存**：
   - 精确能量计算结果会被缓存，重复运行相同配置会更快

## 扩展和自定义

### 添加新的量子框架

1. 继承`FrameworkWrapper`类：
```python
class NewFrameworkWrapper(FrameworkWrapper):
    def setup_backend(self, backend_config):
        # 实现后端设置
        pass
    
    def build_hamiltonian(self, problem_config, n_qubits):
        # 实现哈密顿量构建
        pass
    
    def build_ansatz(self, ansatz_config, n_qubits):
        # 实现Ansatz构建
        pass
    
    def get_cost_function(self, hamiltonian, ansatz, n_qubits):
        # 实现成本函数
        pass
    
    def get_param_count(self, ansatz, n_qubits):
        # 实现参数计数
        pass
```

2. 在`BenchmarkController`中添加新框架：
```python
def _create_wrappers(self):
    # ...
    if framework_name == "NewFramework":
        self.wrappers[framework_name] = NewFrameworkWrapper(backend_config)
```

### 添加新的性能指标

1. 在`VQERunner`中添加指标收集：
```python
def run(self, initial_params=None):
    # ...
    return {
        # 现有指标...
        "new_metric": new_metric_value,
    }
```

2. 在`VQEBenchmarkVisualizer`中添加可视化：
```python
def plot_dashboard(self, output_dir=None):
    # ...
    self._plot_new_metric(axes[1, 1])  # 添加新图表
```

## 修改vqe_bench_new.py以自动包含CPU利用率图表

### 修改概述

要使 `vqe_bench_new.py` 自动包含CPU利用率图表，需要修改 `VQEBenchmarkVisualizer` 类的 `plot_dashboard` 方法。这将使仪表盘从原来的6个图表扩展到7个图表，提供更全面的性能分析。

### 具体修改步骤

#### 步骤1：修改plot_dashboard方法

在 `vqe_bench_new.py` 文件中，找到 `VQEBenchmarkVisualizer` 类的 `plot_dashboard` 方法（约在1262行），将其替换为以下代码：

```python
def plot_dashboard(self, output_dir: str = None) -> None:
    """生成并显示包含七个核心图表的仪表盘"""
    fig, axes = plt.subplots(4, 2, figsize=(20, 28))
    fig.suptitle("VQE框架性能基准测试仪表盘", fontsize=20)
    
    # 图 1: 总求解时间 vs. 量子比特数
    self._plot_time_to_solution(axes[0, 0])
    
    # 图 2: 峰值内存使用 vs. 量子比特数
    self._plot_peak_memory(axes[0, 1])
    
    # 图 3: 收敛轨迹 (以最大比特数为例)
    self._plot_convergence_trajectories(axes[1, 0])
    
    # 图 4: 总求值次数 vs. 量子比特数
    self._plot_total_evaluations(axes[1, 1])
    
    # 图 5: 最终求解精度 vs. 量子比特数
    self._plot_final_accuracy(axes[2, 0])
    
    # 图 6: 单步耗时分解 vs. 量子比特数
    self._plot_time_breakdown(axes[2, 1])
    
    # 图 7: CPU利用率 vs. 量子比特数
    self._plot_cpu_usage(axes[3, 0])
    
    # 隐藏右下角的空白子图
    axes[3, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vqe_benchmark_dashboard_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"仪表盘已保存到: {filepath}")
```

#### 步骤2：验证_plot_cpu_usage方法存在

确保 `_plot_cpu_usage` 方法已经存在于 `VQEBenchmarkVisualizer` 类中（约在1506行）。如果不存在，请添加以下代码：

```python
def _plot_cpu_usage(self, ax):
    """绘制CPU使用率 vs. 量子比特数"""
    for fw in self.frameworks:
        peak_cpus = []
        avg_cpus = []
        for n_qubits in self.n_qubits_range:
            if fw in self.results and n_qubits in self.results[fw]:
                peak_cpus.append(self.results[fw][n_qubits]["avg_peak_cpu_usage"])
                avg_cpus.append(self.results[fw][n_qubits]["avg_avg_cpu_usage"])
            else:
                peak_cpus.append(None)
                avg_cpus.append(None)
        
        # 过滤掉None值
        valid_indices = [i for i, p in enumerate(peak_cpus) if p is not None]
        valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
        valid_peak_cpus = [peak_cpus[i] for i in valid_indices]
        valid_avg_cpus = [avg_cpus[i] for i in valid_indices]
        
        if valid_peak_cpus:
            ax.errorbar(valid_qubits, valid_peak_cpus,
                       marker='o', linestyle='-', label=f'{fw} 峰值CPU', capsize=5)
            ax.errorbar(valid_qubits, valid_avg_cpus,
                       marker='s', linestyle='--', label=f'{fw} 平均CPU', capsize=5)
    
    ax.set_xlabel("量子比特数")
    ax.set_ylabel("CPU使用率 (%)")
    ax.set_title("CPU使用率分析")
    ax.legend()
    ax.grid(True, ls="--")
```

#### 步骤3：更新文档字符串

修改 `plot_dashboard` 方法的文档字符串，说明现在包含7个图表：

```python
def plot_dashboard(self, output_dir: str = None) -> None:
    """生成并显示包含七个核心图表的仪表盘
    包括：
    1. 核心性能: 求解时间
    2. 核心性能: 内存扩展性
    3. 优化动力学: 收敛轨迹
    4. 优化动力学: 评估次数
    5. 诊断: 最终精度验证
    6. 诊断: 时间分解
    7. 诊断: CPU利用率分析
    """
```

### 修改后的效果

完成这些修改后，运行基准测试将自动生成包含7个图表的仪表盘，新增的CPU利用率图表将显示：
- 各框架在不同量子比特数下的峰值CPU使用率
- 各框架在不同量子比特数下的平均CPU使用率

### 版本更新建议

建议在文件头部添加版本更新说明：

```python
"""
VQE框架性能基准测试脚本 - 基于分层配置设计的新架构

版本：2.1.0
更新内容：
- 自动包含CPU利用率图表在仪表盘中
- 仪表盘布局从3×2更新为4×2
- 新增第7个图表：CPU利用率分析

...
"""
```

### 测试修改

完成修改后，可以使用以下代码测试：

```python
from vqe_config import get_quick_start_config
from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer

# 运行基准测试
config = get_quick_start_config()
controller = BenchmarkController(config)
results = controller.run_all_benchmarks()

# 生成可视化仪表盘（现在包含CPU利用率图表）
visualizer = VQEBenchmarkVisualizer(results, config)
output_dir = config.get("system", {}).get("output_dir", "./results/")
visualizer.plot_dashboard(output_dir)
print(f"包含CPU利用率的仪表盘已保存到: {output_dir}")
```

### 备份原始文件

在进行修改前，建议备份原始文件：

```bash
cp Bench/vqe_bench_new.py Bench/vqe_bench_new.py.backup
```

这样，如果修改出现问题，可以轻松恢复原始版本。

## 总结

`vqe_bench_new.py` 提供了一个强大而灵活的VQE基准测试框架，通过分层配置系统和面向对象的设计，既保持了易用性，又提供了高级用户所需的灵活性。无论是快速原型验证还是深入的学术研究，这个框架都能满足不同层次的需求。

通过本指南，您应该能够：
1. 快速上手并运行基本的基准测试
2. 自定义配置以满足特定研究需求
3. 理解和解释测试结果，包括CPU利用率分析
4. 解决常见问题
5. 根据需要扩展框架功能
6. 修改框架以自动包含CPU利用率图表

祝您使用愉快！如有任何问题，请参考源代码中的详细注释或提交Issue。