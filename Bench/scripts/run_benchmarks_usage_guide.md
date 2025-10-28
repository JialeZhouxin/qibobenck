# run_benchmarks.py 使用指南

## 概述

**脚本名称：** run_benchmarks.py

**一句话简介：** 一个专业的量子模拟器基准测试运行器，支持多种量子计算框架的性能比较，包括执行时间、内存使用、CPU利用率和状态保真度等关键指标的测量和分析。

**核心功能列表：**
- **多框架支持**: 同时测试 Qibo、Qiskit、PennyLane 等量子计算框架
- **多电路类型**: 支持 QFT、Grover 等基准量子电路
- **扩展性测试**: 支持多个量子比特数的性能扩展性分析
- **参考态生成**: 自动生成参考态用于保真度计算
- **智能缓存系统**: 支持内存、磁盘和混合缓存以提升测试效率
- **统计分析**: 支持多次运行和统计显著性分析
- **自动化报告**: 自动生成详细的性能分析报告和可视化图表

## 安装和环境要求

### 必需依赖

```bash
# 核心依赖
pip install pandas numpy matplotlib psutil

# 量子计算框架（根据需要选择安装）
pip install qibo              # Qibo框架
pip install qiskit            # Qiskit框架
pip install pennylane         # PennyLane框架

# 可选高性能后端
pip install qibo[qibojit]     # Qibo JIT编译后端
pip install lightning-qubit   # PennyLane高性能后端
```

### 系统要求

- **Python版本**: 3.7+
- **操作系统**: Windows、Linux、macOS
- **内存**: 建议 4GB+ RAM
- **存储**: 至少 1GB 可用磁盘空间（用于缓存和结果）

### 目录结构

脚本需要以下目录结构：

```
Bench/
├── scripts/
│   └── run_benchmarks.py          # 主脚本
├── src/
│   ├── abstractions.py            # 抽象基类
│   ├── caching/                   # 缓存系统
│   ├── circuits/                  # 量子电路定义
│   ├── post_processing.py         # 结果处理
│   ├── simulators/                # 模拟器包装器
│   └── path_config.py             # 路径配置
└── results/                        # 默认输出目录
```

## 命令行参数详解

### 核心参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--circuits` | 无 | List[str] | `["qft"]` | 要运行的基准测试电路列表 |
| `--qubits` | 无 | List[int] | `[2, 3, 4]` | 要测试的量子比特数列表 |
| `--simulators` | 无 | List[str] | `["qibo-qibojit"]` | 要测试的模拟器列表 |
| `--golden-standard` | 无 | String | `"qibo-qibojit"` | 用于生成参考态的模拟器 |

### 缓存控制参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--enable-cache` | 无 | bool | `True` | 启用参考态缓存 |
| `--no-cache` | 无 | bool | `False` | 禁用缓存（覆盖--enable-cache） |
| `--cache-type` | 无 | String | `"hybrid"` | 缓存类型：memory/disk/hybrid |
| `--cache-dir` | 无 | String | `".benchmark_cache"` | 磁盘缓存目录 |
| `--memory-cache-size` | 无 | int | `64` | 内存缓存最大条目数 |
| `--clear-cache` | 无 | bool | `False` | 开始前清空缓存 |
| `--cache-stats` | 无 | bool | `False` | 显示缓存统计信息 |

### 测试控制参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--repeat` | 无 | int | `1` | 每个电路重复运行的次数 |
| `--warmup-runs` | 无 | int | `0` | 正式测量前的预热运行次数 |
| `--statistical-analysis` | 无 | bool | `False` | 启用统计分析 |

### 输出控制参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--output-dir` | 无 | String | `"results"` | 结果输出目录 |
| `--verbose` | 无 | bool | `False` | 启用详细输出模式 |

## 基本使用方法

### 1. 快速开始

```bash
# 进入脚本目录
cd E:\qiboenv\Bench\scripts

# 基本用法 - 使用默认设置
python run_benchmarks.py

# 指定电路和量子比特数
python run_benchmarks.py --circuits qft --qubits 2 3 4 --verbose

# 测试多个模拟器
python run_benchmarks.py --simulators qibo-numpy qibo-qibojit --repeat 3

# 启用统计分析
python run_benchmarks.py --qubits 2 3 4 5 6 --repeat 5 --statistical-analysis --verbose
```

### 2. 高级用法示例

#### 示例1: 完整的基准测试

```bash
# 运行完整的基准测试
python run_benchmarks.py \
    --circuits qft grover \
    --qubits 2 3 4 5 6 \
    --simulators qibo-numpy qibo-qibojit \
    --repeat 3 \
    --warmup-runs 1 \
    --statistical-analysis \
    --verbose \
    --output-dir my_benchmark_results
```

#### 示例2: 缓存优化测试

```bash
# 使用混合缓存进行大规模测试
python run_benchmarks.py \
    --qubits 4 6 8 10 12 \
    --simulators qibo-qibojit \
    --repeat 5 \
    --cache-type hybrid \
    --memory-cache-size 128 \
    --enable-cache \
    --verbose
```

#### 示例3: 统计分析测试

```bash
# 进行多次运行的统计分析
python run_benchmarks.py \
    --circuits qft \
    --qubits 3 4 5 6 \
    --simulators qibo-qibojit \
    --repeat 10 \
    --warmup-runs 2 \
    --statistical-analysis \
    --cache-stats \
    --verbose
```

## 支持的电路和模拟器

### 支持的量子电路

| 电路名称 | 描述 | 特点 |
|----------|------|------|
| `qft` | 量子傅里叶变换 | 经典量子算法，广泛用于量子计算 |
| `grover` | Grover搜索算法 | 量子搜索算法，展示量子加速 |

### 支持的模拟器配置

| 配置格式 | 平台 | 后端 | 说明 |
|----------|------|------|------|
| `qibo-numpy` | Qibo | numpy | 基础 NumPy 后端 |
| `qibo-qibojit` | Qibo | qibojit | 高性能 JIT 编译后端 |
| `qibo-custom` | Qibo | custom | 自定义后端 |

## 使用示例详解

### 示例1: 基础性能比较

```python
#!/usr/bin/env python3
"""
基础性能比较示例
"""

import subprocess
import os
from pathlib import Path

def basic_performance_comparison():
    """运行基础的三框架性能比较"""

    # 基础配置
    cmd = [
        "python", "run_benchmarks.py",
        "--circuits", "qft",
        "--qubits", "2", "3", "4",
        "--simulators", "qibo-numpy", "qibo-qibojit",
        "--repeat", "2",
        "--verbose"
    ]

    print("开始基础性能比较...")
    print(f"命令: {' '.join(cmd)}")

    # 运行测试
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

    if result.returncode == 0:
        print("测试完成!")
        print("\n输出:")
        print(result.stdout)
    else:
        print("测试失败!")
        print("\n错误:")
        print(result.stderr)

if __name__ == "__main__":
    basic_performance_comparison()
```

### 示例2: 扩展性测试

```python
#!/usr/bin/env python3
"""
扩展性测试示例
"""

import subprocess
import time
from pathlib import Path

def scalability_test():
    """运行扩展性测试，测试不同量子比特数的性能"""

    # 扩展性测试配置
    test_configs = [
        {
            "qubits": [2, 3, 4, 5, 6],
            "repeat": 2,
            "name": "small_scale"
        },
        {
            "qubits": [6, 8, 10],
            "repeat": 1,
            "name": "medium_scale"
        },
        {
            "qubits": [4, 6, 8, 10, 12, 14],
            "repeat": 1,
            "name": "large_scale"
        }
    ]

    base_cmd = [
        "python", "run_benchmarks.py",
        "--circuits", "qft",
        "--simulators", "qibo-qibojit",
        "--warmup-runs", "1",
        "--enable-cache",
        "--verbose"
    ]

    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"运行 {config['name']} 扩展性测试")
        print(f"量子比特数: {config['qubits']}")
        print(f"重复次数: {config['repeat']}")
        print('='*60)

        # 构建完整命令
        cmd = base_cmd + [
            "--qubits"] + [str(q) for q in config["qubits"]] + [
            "--repeat", str(config["repeat"])
        ]

        start_time = time.time()

        # 运行测试
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        end_time = time.time()

        print(f"测试完成，耗时: {end_time - start_time:.2f}秒")

        if result.returncode == 0:
            # 提取关键结果
            lines = result.stdout.split('\n')
            for line in lines:
                if "Completed" in line and "benchmark runs" in line:
                    print(f"结果: {line.strip()}")
                elif "avg" in line and "s" in line:
                    print(f"性能: {line.strip()}")
        else:
            print(f"测试失败: {result.stderr}")

if __name__ == "__main__":
    scalability_test()
```

### 示例3: 缓存性能测试

```python
#!/usr/bin/env python3
"""
缓存性能测试示例
"""

import subprocess
import time
from pathlib import Path

def cache_performance_test():
    """测试不同缓存配置的性能影响"""

    cache_configs = [
        {
            "cache_type": "memory",
            "memory_size": 32,
            "name": "memory_cache"
        },
        {
            "cache_type": "disk",
            "cache_dir": ".disk_cache_test",
            "name": "disk_cache"
        },
        {
            "cache_type": "hybrid",
            "memory_size": 64,
            "cache_dir": ".hybrid_cache_test",
            "name": "hybrid_cache"
        },
        {
            "name": "no_cache"
        }
    ]

    base_cmd = [
        "python", "run_benchmarks.py",
        "--circuits", "qft",
        "--qubits", "4", "6", "8",
        "--simulators", "qibo-qibojit",
        "--repeat", "3",
        "--verbose"
    ]

    results = {}

    for config in cache_configs:
        print(f"\n{'='*60}")
        print(f"测试缓存配置: {config['name']}")
        print('='*60)

        # 构建命令
        cmd = base_cmd.copy()

        if config["name"] != "no_cache":
            cmd.extend(["--cache-type", config["cache_type"]])

            if "memory_size" in config:
                cmd.extend(["--memory-cache-size", str(config["memory_size"])])

            if "cache_dir" in config:
                cmd.extend(["--cache-dir", config["cache_dir"]])

            cmd.extend(["--enable-cache"])
        else:
            cmd.extend(["--no-cache"])

        # 清理缓存（如果有）
        clear_cmd = cmd.copy()
        clear_cmd.append("--clear-cache")
        subprocess.run(clear_cmd, capture_output=True, cwd=Path(__file__).parent)

        # 运行测试
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        end_time = time.time()

        execution_time = end_time - start_time
        results[config["name"]] = execution_time

        print(f"执行时间: {execution_time:.2f}秒")

        if result.returncode == 0:
            # 提取平均执行时间
            lines = result.stdout.split('\n')
            for line in lines:
                if "Completed" in line and "benchmark runs" in line:
                    print(f"结果: {line.strip()}")
                    break
        else:
            print(f"测试失败: {result.stderr}")

    # 性能比较
    print(f"\n{'='*60}")
    print("缓存性能比较")
    print('='*60)

    for name, time_taken in sorted(results.items(), key=lambda x: x[1]):
        improvement = ((results["no_cache"] - time_taken) / results["no_cache"]) * 100
        print(f"{name:15}: {time_taken:6.2f}s (改进: {improvement:+5.1f}%)")

if __name__ == "__main__":
    cache_performance_test()
```

### 示例4: 统计分析测试

```python
#!/usr/bin/env python3
"""
统计分析测试示例
"""

import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def statistical_analysis_test():
    """运行统计分析测试并生成可视化报告"""

    # 统计分析配置
    cmd = [
        "python", "run_benchmarks.py",
        "--circuits", "qft",
        "--qubits", "3", "4", "5",
        "--simulators", "qibo-qibojit",
        "--repeat", "10",        # 多次运行以获得统计显著性
        "--warmup-runs", "2",   # 预热运行以减少噪声
        "--statistical-analysis",
        "--verbose",
        "--output-dir", "statistical_results"
    ]

    print("开始统计分析测试...")
    print(f"重复运行次数: 10")
    print(f"预热运行次数: 2")
    print("这将生成具有统计意义的结果...")

    # 运行测试
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"测试失败: {result.stderr}")
        return

    print("测试完成!")

    # 尝试读取结果数据
    results_dir = Path("statistical_results")
    if results_dir.exists():
        # 查找CSV文件
        csv_files = list(results_dir.glob("**/*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"\n分析结果文件: {latest_csv}")

            try:
                df = pd.read_csv(latest_csv)

                # 基本统计
                print(f"\n基本统计信息:")
                print(f"总测试次数: {len(df)}")
                print(f"量子比特数范围: {df['n_qubits'].min()} - {df['n_qubits'].max()}")
                print(f"平均执行时间: {df['wall_time_sec'].mean():.4f}s")
                print(f"执行时间标准差: {df['wall_time_sec'].std():.4f}s")

                # 按量子比特数分组统计
                print(f"\n按量子比特数分组统计:")
                for n_qubits in sorted(df['n_qubits'].unique()):
                    subset = df[df['n_qubits'] == n_qubits]
                    print(f"N={n_qubits}: 平均时间={subset['wall_time_sec'].mean():.4f}s, "
                          f"标准差={subset['wall_time_sec'].std():.4f}s, "
                          f"样本数={len(subset)}")

                # 生成简单的可视化
                plt.figure(figsize=(10, 6))
                for n_qubits in sorted(df['n_qubits'].unique()):
                    subset = df[df['n_qubits'] == n_qubits]
                    plt.hist(subset['wall_time_sec'], alpha=0.7, label=f'N={n_qubits}', bins=10)

                plt.xlabel('执行时间 (秒)')
                plt.ylabel('频次')
                plt.title('执行时间分布')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plot_path = results_dir / "execution_time_distribution.png"
                plt.savefig(plot_path)
                print(f"\n可视化图表已保存到: {plot_path}")

            except Exception as e:
                print(f"分析结果失败: {e}")
        else:
            print("未找到CSV结果文件")
    else:
        print("未找到结果目录")

if __name__ == "__main__":
    statistical_analysis_test()
```

## 高级功能详解

### 1. 缓存系统

脚本支持三种缓存类型：

#### 内存缓存 (Memory Cache)
- **特点**: 最快的缓存类型，数据存储在内存中
- **用途**: 适合短期测试和快速原型开发
- **配置**: `--cache-type memory --memory-cache-size 64`

#### 磁盘缓存 (Disk Cache)
- **特点**: 持久化存储，重启后仍可用
- **用途**: 适合大规模测试和结果复用
- **配置**: `--cache-type disk --cache-dir ./my_cache`

#### 混合缓存 (Hybrid Cache)
- **特点**: 结合内存和磁盘缓存的优势
- **用途**: 平衡性能和持久性的最佳选择
- **配置**: `--cache-type hybrid --memory-cache-size 64`

### 2. 统计分析

启用统计分析可以提供更可靠的结果：

```bash
# 启用统计分析
python run_benchmarks.py --statistical-analysis --repeat 10

# 这将生成：
# - 均值和标准差
# - 置信区间
# - 异常值检测
# - 统计显著性测试
```

### 3. 预热运行

预热运行可以减少JIT编译等初始开销：

```bash
# 设置预热运行
python run_benchmarks.py --warmup-runs 2 --repeat 5

# 执行流程：
# 1. 预热运行2次（不记录结果）
# 2. 正式运行5次（记录结果）
# 3. 计算统计指标
```

## 结果分析

### 输出文件结构

测试完成后，会在输出目录中生成以下文件：

```
results/benchmark_YYYYMMDD_HHMMSS/
├── raw_results.csv              # 原始测试数据
├── detailed_runs.csv           # 详细运行数据
├── summary_report.md           # Markdown格式摘要报告
├── cpu_time_scaling.png        # CPU时间扩展性图表
├── wall_time_scaling.png       # 实际时间扩展性图表
├── memory_scaling.png          # 内存使用扩展性图表
├── cpu_utilization.png         # CPU利用率图表
├── fidelity.png                # 状态保真度图表
├── confidence_intervals.png    # 置信区间图表
└── execution_stability.png     # 执行稳定性图表
```

### 关键性能指标

#### 1. 执行时间指标
- **wall_time_sec**: 实际运行时间
- **cpu_time_sec**: CPU时间
- **wall_time_std**: 执行时间标准差

#### 2. 内存使用指标
- **peak_memory_mb**: 峰值内存使用量
- **memory_std**: 内存使用标准差

#### 3. 计算准确性指标
- **state_fidelity**: 状态保真度
- **fidelity_std**: 保真度标准差

#### 4. 资源利用率指标
- **cpu_utilization_percent**: CPU利用率

### 结果解读指南

#### 时间复杂度分析
```python
import pandas as pd
import numpy as np

# 读取结果
df = pd.read_csv('results/benchmark_20251027_143645/raw_results.csv')

# 分析时间复杂度
for simulator in df['simulator'].unique():
    subset = df[df['simulator'] == simulator]

    # 对数拟合
    log_qubits = np.log(subset['n_qubits'])
    log_times = np.log(subset['wall_time_sec'])

    coeffs = np.polyfit(log_qubits, log_times, 1)
    complexity = coeffs[0]

    print(f"{simulator} 时间复杂度指数: {complexity:.2f}")
```

#### 性能比较
```python
# 比较不同模拟器的性能
comparison = df.groupby(['n_qubits', 'simulator'])['wall_time_sec'].mean().unstack()

# 计算加速比
baseline = comparison['qibo-numpy']
for col in comparison.columns:
    if col != 'qibo-numpy':
        speedup = baseline / comparison[col]
        print(f"{col} 相对于 numpy 的平均加速比: {speedup.mean():.2f}x")
```

## 故障排除

### 常见问题及解决方案

#### 1. 导入错误

**问题**: `ModuleNotFoundError: No module named 'src.abstractions'`

**解决方案**:
```bash
# 确保在正确的目录中运行
cd E:\qiboenv\Bench\scripts

# 检查目录结构
ls -la ../src/
```

#### 2. 模拟器创建失败

**问题**: `Warning: Failed to create qibo simulator with backend qibojit`

**解决方案**:
```bash
# 安装相应的依赖
pip install qibo[qibojit]

# 或使用可用的后端
python run_benchmarks.py --simulators qibo-numpy
```

#### 3. 缓存权限错误

**问题**: `Permission denied: '.benchmark_cache'`

**解决方案**:
```bash
# 使用自定义缓存目录
python run_benchmarks.py --cache-dir ./temp_cache

# 或禁用缓存
python run_benchmarks.py --no-cache
```

#### 4. 内存不足

**问题**: 测试大量子比特数时内存不足

**解决方案**:
```bash
# 减少量子比特数
python run_benchmarks.py --qubits 2 3 4

# 或减少重复次数
python run_benchmarks.py --repeat 1

# 使用内存缓存限制
python run_benchmarks.py --memory-cache-size 16
```

#### 5. 结果保存失败

**问题**: 无法保存结果到输出目录

**解决方案**:
```bash
# 使用自定义输出目录
python run_benchmarks.py --output-dir ./my_results

# 确保目录有写入权限
mkdir -p ./my_results
chmod 755 ./my_results
```

## 最佳实践

### 1. 测试设计

#### 从小规模开始
```bash
# 先测试小规模，确保配置正确
python run_benchmarks.py --qubits 2 3 --verbose
```

#### 逐步增加复杂度
```bash
# 逐步增加量子比特数
python run_benchmarks.py --qubits 2 3 4 5 --verbose
python run_benchmarks.py --qubits 4 6 8 --verbose
```

#### 使用缓存提高效率
```bash
# 启用缓存以避免重复计算
python run_benchmarks.py --enable-cache --verbose
```

### 2. 性能优化

#### 选择合适的后端
- **小规模测试**: 使用 `qibo-numpy`
- **性能测试**: 使用 `qibo-qibojit`
- **大规模测试**: 根据可用内存选择

#### 合理设置重复次数
- **快速测试**: `--repeat 1`
- **统计分析**: `--repeat 10`
- **生产测试**: `--repeat 3-5`

#### 使用预热运行
```bash
# 启用预热以减少JIT编译影响
python run_benchmarks.py --warmup-runs 2 --repeat 5
```

### 3. 结果分析

#### 关注关键指标
- **执行时间**: 评估算法效率
- **内存使用**: 评估资源需求
- **保真度**: 评估计算准确性
- **稳定性**: 评估结果可靠性

#### 进行统计验证
```bash
# 启用统计分析以获得可靠结果
python run_benchmarks.py --statistical-analysis --repeat 10
```

#### 比较不同配置
```bash
# 比较不同后端
python run_benchmarks.py --simulators qibo-numpy qibo-qibojit

# 比较不同缓存策略
python run_benchmarks.py --cache-type memory
python run_benchmarks.py --cache-type hybrid
```

## 扩展和自定义

### 1. 添加新电路

要添加新的量子电路类型：

1. 在 `src/circuits/` 目录下创建新的电路模块
2. 实现继承自 `BenchmarkCircuit` 的电路类
3. 在 `create_circuit_instances()` 函数中添加支持

```python
# src/circuits/my_circuit.py
from src.abstractions import BenchmarkCircuit

class MyCircuit(BenchmarkCircuit):
    def __init__(self):
        self.name = "My Custom Circuit"
        # 初始化电路参数

    def build(self, platform, n_qubits):
        # 构建电路逻辑
        pass

    def execute(self, n_qubits, **kwargs):
        # 执行电路逻辑
        pass
```

### 2. 添加新模拟器

要添加新的量子计算框架：

1. 在 `src/simulators/` 目录下创建包装器模块
2. 实现继承自 `SimulatorInterface` 的包装器类
3. 在 `create_simulator_instances()` 函数中添加支持

```python
# src/simulators/my_framework_wrapper.py
from src.abstractions import SimulatorInterface

class MyFrameworkWrapper(SimulatorInterface):
    def __init__(self, backend):
        # 初始化模拟器
        pass

    def execute(self, circuit, n_qubits, **kwargs):
        # 执行模拟
        pass
```

### 3. 自定义缓存策略

可以通过修改缓存配置来优化性能：

```python
# src/caching/custom_cache.py
from src.caching import CacheConfig

class CustomCacheConfig(CacheConfig):
    def __init__(self):
        # 自定义缓存配置
        self.enable_cache = True
        self.cache_type = "custom"
        # 添加自定义配置项
```

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**作者**: 量子计算基准测试团队

这份使用指南提供了 run_benchmarks.py 的完整使用说明，从基本操作到高级功能，涵盖了安装、配置、运行、分析和故障排除的各个方面。通过遵循本指南，用户可以有效地进行量子计算框架的性能基准测试和分析。