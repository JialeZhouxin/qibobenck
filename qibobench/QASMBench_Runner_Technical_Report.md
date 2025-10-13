# QASMBench Runner 技术报告 - 后端选择版本

## 目录

1. [概述](#概述)
2. [使用指南](#使用指南)
   - [2.1 命令行调用方式](#21-命令行调用方式)
   - [2.2 Python模块导入使用](#22-python模块导入使用)
   - [2.3 配置选项说明](#23-配置选项说明)
   - [2.4 实际使用示例](#24-实际使用示例)
3. [技术架构](#技术架构)
4. [API参考](#api参考)
5. [扩展开发](#扩展开发)
6. [故障排除](#故障排除)

---

## 概述

### 项目介绍

`qasmbench_runner_backend_selection.py` 是 QASMBench 通用基准测试工具的增强版本，专门用于测试不同 Qibo 后端在量子电路上的性能表现。该工具基于原版 `qasmbench_runner.py` 进行了重大升级，新增了灵活的后端选择功能，让用户可以精确控制要测试的后端组合。

### 主要功能

- **🔧 后端选择**: 支持选择性运行指定的 Qibo 后端
- **📊 性能基准**: 全面的性能测试，包括执行时间、内存使用、吞吐率等指标
- **✅ 正确性验证**: 自动验证不同后端的计算结果一致性
- **📈 多格式报告**: 支持 CSV、Markdown、JSON 三种格式的详细报告
- **🔍 状态监控**: 实时显示后端可用性和依赖状态
- **🎯 灵活配置**: 可自定义运行次数、预热次数等测试参数

### 新特性说明

**版本 v2.0 主要变更**：
- ✨ 新增 `BackendConfig` 数据类，提供结构化的后端配置管理
- ✨ 新增 `BackendRegistry` 注册器，实现后端的动态管理
- ✨ 支持 `--backends` 参数，允许用户精确选择测试后端
- ✨ 新增 `--list-backends` 和 `--backend-status` 命令
- ✨ 改进错误处理和用户友好的提示信息
- ✨ 增强的可扩展性，便于添加新后端支持

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    QASMBench Runner v2.0                    │
├─────────────────────────────────────────────────────────────┤
│  命令行接口 (argparse)                                      │
├─────────────────────────────────────────────────────────────┤
│  后端管理层                                                │
│  ├── BackendRegistry (全局注册器)                           │
│  ├── BackendConfig (配置类)                                │
│  └── 依赖验证机制                                           │
├─────────────────────────────────────────────────────────────┤
│  核心测试引擎                                              │
│  ├── QASMBenchRunner (主控制器)                            │
│  ├── QASMBenchMetrics (指标收集)                           │
│  └── 正确性验证                                             │
├─────────────────────────────────────────────────────────────┤
│  报告生成系统                                              │
│  ├── CSV 报告                                             │
│  ├── Markdown 报告                                        │
│  └── JSON 报告                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 使用指南

### 2.1 命令行调用方式

#### 基本语法

```bash
python qasmbench_runner_backend_selection.py [选项] [参数]
```

#### 命令行参数详解

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--list` | 标志 | 否 | 列出所有可用的 QASMBench 电路 |
| `--list-backends` | 标志 | 否 | 列出所有可用的 Qibo 后端 |
| `--backend-status` | 标志 | 否 | 显示后端的详细状态信息 |
| `--circuit` | 字符串 | 条件必需 | 指定 QASM 电路文件的完整路径 |
| `--backends` | 字符串 | 否 | 指定要测试的后端，用逗号分隔 |

#### 基础使用示例

```bash
# 1. 查看所有可用电路
python qasmbench_runner_backend_selection.py --list

# 2. 查看所有可用后端
python qasmbench_runner_backend_selection.py --list-backends

# 3. 查看后端详细状态
python qasmbench_runner_backend_selection.py --backend-status

# 4. 测试单个电路的所有后端（默认行为）
python qasmbench_runner_backend_selection.py --circuit "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"

# 5. 测试指定后端
python qasmbench_runner_backend_selection.py --circuit "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm" --backends "qibojit(numba)"

# 6. 测试多个后端
python qasmbench_runner_backend_selection.py --circuit "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm" --backends "numpy,qibojit(numba),qiboml(jax)"
```

#### 高级使用示例

```bash
# 测试所有 ML 后端
python qasmbench_runner_backend_selection.py --circuit "QASMBench/small/qft_n4/qft_n4_transpiled.qasm" --backends "qiboml(jax),qiboml(pytorch),qiboml(tensorflow)"

# 比较 JIT 编译器性能
python qasmbench_runner_backend_selection.py --circuit "QASMBench/medium/bv_n14/bv_n14_transpiled.qasm" --backends "numpy,qibojit(numba)"

# 测试张量网络后端
python qasmbench_runner_backend_selection.py --circuit "QASMBench/large/ghz_n40/ghz_n40_transpiled.qasm" --backends "qibotn(qutensornet)"
```

#### 后端名称规范

支持的后端名称格式：

| 显示名称 | 实际后端 | 平台 | 说明 |
|----------|----------|------|------|
| `numpy` | `numpy` | None | NumPy 后端（默认基准） |
| `qibojit(numba)` | `qibojit` | `numba` | QiboJIT with Numba |
| `qibotn(qutensornet)` | `qibotn` | `qutensornet` | QiboTensorNetwork |
| `qiboml(jax)` | `qiboml` | `jax` | QiboML with JAX |
| `qiboml(pytorch)` | `qiboml` | `pytorch` | QiboML with PyTorch |
| `qiboml(tensorflow)` | `qiboml` | `tensorflow` | QiboML with TensorFlow |
| `qulacs` | `qulacs` | None | Qulacs 后端 |

### 2.2 Python 模块导入使用

#### 基础导入方式

```python
# 导入主要功能函数
from qasmbench_runner_backend_selection import (
    run_benchmark_for_circuit,
    list_available_backends,
    list_backend_status,
    parse_backend_string
)

# 导入核心类
from qasmbench_runner_backend_selection import (
    QASMBenchRunner,
    QASMBenchConfig,
    BackendRegistry,
    BackendConfig
)
```

#### 简单使用示例

```python
# 示例 1: 基础基准测试
from qasmbench_runner_backend_selection import run_benchmark_for_circuit

# 测试所有后端
results = run_benchmark_for_circuit("QASMBench/medium/qft_n18/qft_n18_transpiled.qasm")

# 测试指定后端
selected_backends = ["numpy", "qibojit(numba)"]
results = run_benchmark_for_circuit(
    "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm", 
    selected_backends
)
```

```python
# 示例 2: 使用 Runner 类进行精细控制
from qasmbench_runner_backend_selection import QASMBenchRunner, QASMBenchConfig

# 创建配置
config = QASMBenchConfig()
config.num_runs = 10  # 增加运行次数
config.warmup_runs = 2  # 增加预热次数

# 创建 Runner
runner = QASMBenchRunner(config)

# 运行基准测试
circuit_path = "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"
selected_backends = ["numpy", "qibojit(numba)", "qiboml(jax)"]
results = runner.run_benchmark_for_circuit("qft_n18", circuit_path, selected_backends)

# 生成自定义报告
runner.generate_reports(results, "qft_n18_custom")
```

```python
# 示例 3: 后端管理和状态检查
from qasmbench_runner_backend_selection import backend_registry, list_available_backends

# 查看可用后端
list_available_backends()

# 获取特定后端配置
numpy_config = backend_registry.get_backend("numpy")
print(f"NumPy 后端描述: {numpy_config.description}")

# 获取基准后端
baseline = backend_registry.get_baseline_backend()
print(f"基准后端: {baseline.display_name}")

# 检查后端可用性
available_backends = backend_registry.get_available_backends()
print(f"可用后端数量: {len(available_backends)}")
```

```python
# 示例 4: 批量测试多个电路
from qasmbench_runner_backend_selection import QASMBenchRunner, QASMBenchConfig
import glob

def batch_test_circuits(circuit_patterns, backends):
    """批量测试多个电路"""
    config = QASMBenchConfig()
    runner = QASMBenchRunner(config)
    
    all_results = {}
    
    for pattern in circuit_patterns:
        circuits = glob.glob(pattern)
        for circuit_path in circuits:
            circuit_name = circuit_path.split('/')[-1].replace('.qasm', '')
            print(f"测试电路: {circuit_name}")
            
            results = runner.run_benchmark_for_circuit(
                circuit_name, circuit_path, backends
            )
            all_results[circuit_name] = results
    
    return all_results

# 使用示例
circuits_to_test = [
    "QASMBench/small/*_transpiled.qasm",
    "QASMBench/medium/qft_*_transpiled.qasm"
]
backends_to_test = ["numpy", "qibojit(numba)"]

batch_results = batch_test_circuits(circuits_to_test, backends_to_test)
```

```python
# 示例 5: 自定义后端配置
from qasmbench_runner_backend_selection import BackendConfig, backend_registry

# 创建自定义后端配置
custom_backend = BackendConfig(
    display_name="custom_gpu",
    backend_name="qibojit",
    platform_name="cuda",
    description="自定义 GPU 后端",
    dependencies=["cupy", "qibo"],
    priority=10
)

# 注册自定义后端
backend_registry.register(custom_backend)

# 使用自定义后端
results = run_benchmark_for_circuit(
    "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm",
    ["custom_gpu", "numpy"]
)
```

#### 高级使用模式

```python
# 示例 6: 结果分析和比较
from qasmbench_runner_backend_selection import QASMBenchRunner, QASMBenchConfig
import pandas as pd

def analyze_results(results):
    """分析基准测试结果"""
    data = []
    
    for backend, metrics in results.items():
        if metrics.execution_time_mean is not None:
            data.append({
                'backend': backend,
                'execution_time': metrics.execution_time_mean,
                'memory_mb': metrics.peak_memory_mb,
                'speedup': metrics.speedup,
                'throughput': metrics.throughput_gates_per_sec,
                'correctness': metrics.correctness
            })
    
    df = pd.DataFrame(data)
    
    # 性能排名
    df_sorted = df.sort_values('execution_time')
    print("性能排名:")
    for i, row in df_sorted.iterrows():
        print(f"{i+1}. {row['backend']}: {row['execution_time']:.4f}s")
    
    return df

# 使用示例
config = QASMBenchConfig()
runner = QASMBenchRunner(config)
results = runner.run_benchmark_for_circuit(
    "test_circuit", "path/to/circuit.qasm", 
    ["numpy", "qibojit(numba)", "qiboml(jax)"]
)

analysis_df = analyze_results(results)
```

```python
# 示例 7: 集成到 Jupyter Notebook
from qasmbench_runner_backend_selection import *
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_performance(results):
    """可视化性能结果"""
    backends = []
    times = []
    memories = []
    
    for backend, metrics in results.items():
        if metrics.execution_time_mean is not None:
            backends.append(backend)
            times.append(metrics.execution_time_mean)
            memories.append(metrics.peak_memory_mb)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 执行时间对比
    ax1.bar(backends, times)
    ax1.set_title('执行时间对比')
    ax1.set_ylabel('时间 (秒)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 内存使用对比
    ax2.bar(backends, memories)
    ax2.set_title('内存使用对比')
    ax2.set_ylabel('内存 (MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# 在 Jupyter 中使用
results = run_benchmark_for_circuit(
    "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm",
    ["numpy", "qibojit(numba)", "qiboml(jax)"]
)
visualize_performance(results)
```

### 2.3 配置选项说明

#### QASMBenchConfig 类参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_runs` | int | 5 | 每个后端的正式运行次数 |
| `warmup_runs` | int | 1 | 预热运行次数（用于 JIT 编译） |
| `output_formats` | list | `['csv', 'markdown', 'json']` | 输出报告格式 |
| `baseline_backend` | str | `"numpy"` | 基准后端名称 |
| `qasm_directory` | str | `"../QASMBench"` | QASMBench 电路根目录 |

#### BackendConfig 类参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `display_name` | str | 是 | 显示名称（如 "qibojit(numba)"） |
| `backend_name` | str | 是 | Qibo 后端名称 |
| `platform_name` | Optional[str] | 否 | 平台名称 |
| `description` | str | 是 | 后端描述 |
| `dependencies` | List[str] | 是 | 依赖包列表 |
| `priority` | int | 否 | 优先级（用于排序） |
| `is_baseline` | bool | 否 | 是否为基准后端 |

### 2.4 实际使用示例

#### 场景 1: 性能评估研究

```python
"""
研究场景：比较不同后端在量子傅里叶变换电路上的性能
"""

from qasmbench_runner_backend_selection import QASMBenchRunner, QASMBenchConfig
import pandas as pd
import time

def performance_study():
    # 配置测试参数
    config = QASMBenchConfig()
    config.num_runs = 10  # 增加运行次数以获得更准确的结果
    config.warmup_runs = 3  # 增加预热次数
    
    runner = QASMBenchRunner(config)
    
    # 测试不同规模的 QFT 电路
    qft_circuits = [
        "QASMBench/small/qft_n4/qft_n4_transpiled.qasm",
        "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm",
        "QASMBench/large/qft_n29/qft_n29_transpiled.qasm"
    ]
    
    # 测试不同类型的后端
    backend_groups = {
        "traditional": ["numpy"],
        "jit_compiled": ["qibojit(numba)"],
        "ml_backends": ["qiboml(jax)", "qiboml(pytorch)"],
        "tensor_network": ["qibotn(qutensornet)"]
    }
    
    all_results = {}
    
    for circuit_path in qft_circuits:
        circuit_name = circuit_path.split('/')[-2]
        print(f"\n{'='*60}")
        print(f"测试电路: {circuit_name}")
        print(f"{'='*60}")
        
        circuit_results = {}
        
        for group_name, backends in backend_groups.items():
            print(f"\n测试 {group_name} 组后端...")
            try:
                results = runner.run_benchmark_for_circuit(
                    circuit_name, circuit_path, backends
                )
                circuit_results[group_name] = results
                
                # 打印简要结果
                print(f"{group_name} 结果:")
                for backend, metrics in results.items():
                    if metrics.execution_time_mean:
                        speedup_str = f" ({metrics.speedup:.2f}x)" if metrics.speedup else ""
                        print(f"  {backend}: {metrics.execution_time_mean:.4f}s{speedup_str}")
                        
            except Exception as e:
                print(f"  错误: {e}")
        
        all_results[circuit_name] = circuit_results
    
    return all_results

# 执行性能研究
results = performance_study()
```

#### 场景 2: 自动化测试管道

```python
"""
自动化测试：定期检查后端性能和正确性
"""

from qasmbench_runner_backend_selection import *
import json
import datetime
import os

class AutomatedTestPipeline:
    def __init__(self, results_dir="test_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def run_daily_tests(self):
        """运行每日测试"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 测试配置
        test_circuits = [
            "QASMBench/small/qft_n4/qft_n4_transpiled.qasm",
            "QASMBench/medium/ghz_state_n23/ghz_state_n23_transpiled.qasm",
            "QASMBench/medium/bv_n14/bv_n14_transpiled.qasm"
        ]
        
        critical_backends = ["numpy", "qibojit(numba)", "qiboml(jax)"]
        
        daily_results = {
            "timestamp": timestamp,
            "test_circuits": test_circuits,
            "backends": critical_backends,
            "results": {}
        }
        
        for circuit_path in test_circuits:
            circuit_name = os.path.basename(circuit_path).replace('_transpiled.qasm', '')
            print(f"测试电路: {circuit_name}")
            
            try:
                results = run_benchmark_for_circuit(circuit_path, critical_backends)
                daily_results["results"][circuit_name] = self._extract_key_metrics(results)
                
                # 检查正确性
                correctness_issues = []
                for backend, metrics in results.items():
                    if "Failed" in metrics.correctness:
                        correctness_issues.append(f"{backend}: {metrics.correctness}")
                
                if correctness_issues:
                    print(f"⚠️ 正确性问题: {correctness_issues}")
                    self._send_alert(f"正确性问题检测到: {circuit_name}", correctness_issues)
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                daily_results["results"][circuit_name] = {"error": str(e)}
        
        # 保存结果
        results_file = os.path.join(self.results_dir, f"daily_test_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(daily_results, f, indent=2)
        
        print(f"✅ 每日测试完成，结果保存到: {results_file}")
        return daily_results
    
    def _extract_key_metrics(self, results):
        """提取关键指标"""
        key_metrics = {}
        for backend, metrics in results.items():
            if metrics.execution_time_mean:
                key_metrics[backend] = {
                    "execution_time": metrics.execution_time_mean,
                    "memory_mb": metrics.peak_memory_mb,
                    "speedup": metrics.speedup,
                    "correctness": metrics.correctness
                }
        return key_metrics
    
    def _send_alert(self, subject, message):
        """发送警报（示例实现）"""
        print(f"🚨 警报: {subject}")
        print(f"详情: {message}")
        # 在实际应用中，这里可以发送邮件、Slack 消息等

# 使用自动化测试管道
pipeline = AutomatedTestPipeline()
daily_results = pipeline.run_daily_tests()
```

#### 场景 3: 交互式后端选择工具

```python
"""
交互式工具：让用户通过菜单选择后端和电路
"""

from qasmbench_runner_backend_selection import *
import sys

class InteractiveBenchmarkTool:
    def __init__(self):
        self.config = QASMBenchConfig()
        self.runner = QASMBenchRunner(self.config)
    
    def show_main_menu(self):
        """显示主菜单"""
        while True:
            print("\n" + "="*60)
            print("🚀 QASMBench 基准测试工具 - 交互式模式")
            print("="*60)
            print("1. 查看可用电路")
            print("2. 查看可用后端")
            print("3. 运行基准测试")
            print("4. 查看后端状态")
            print("5. 自定义配置")
            print("0. 退出")
            print("-"*60)
            
            choice = input("请选择操作 (0-5): ").strip()
            
            if choice == "1":
                self.show_circuits()
            elif choice == "2":
                self.show_backends()
            elif choice == "3":
                self.run_interactive_test()
            elif choice == "4":
                self.show_backend_status()
            elif choice == "5":
                self.customize_config()
            elif choice == "0":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请重试。")
    
    def show_circuits(self):
        """显示可用电路"""
        print("\n📋 可用电路列表:")
        circuits = list_available_circuits()
        
        # 按规模分组显示
        by_size = {}
        for name, info in circuits.items():
            size = info['size']
            if size not in by_size:
                by_size[size] = []
            by_size[size].append((name, info))
        
        for size in ['small', 'medium', 'large']:
            if size in by_size:
                print(f"\n📁 {size.upper()} 规模:")
                for i, (name, info) in enumerate(by_size[size], 1):
                    print(f"  {i}. {name}")
    
    def show_backends(self):
        """显示可用后端"""
        print("\n🔧 可用后端列表:")
        list_available_backends()
    
    def show_backend_status(self):
        """显示后端状态"""
        print("\n🔍 后端状态详情:")
        list_backend_status()
    
    def run_interactive_test(self):
        """交互式运行测试"""
        print("\n🧪 配置基准测试")
        
        # 选择电路
        circuit_path = self._select_circuit()
        if not circuit_path:
            return
        
        # 选择后端
        selected_backends = self._select_backends()
        if not selected_backends:
            return
        
        # 确认配置
        print(f"\n📋 测试配置确认:")
        print(f"电路: {circuit_path}")
        print(f"后端: {', '.join(selected_backends)}")
        print(f"运行次数: {self.config.num_runs}")
        print(f"预热次数: {self.config.warmup_runs}")
        
        confirm = input("\n确认开始测试? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 测试已取消")
            return
        
        # 运行测试
        print("\n🚀 开始基准测试...")
        results = run_benchmark_for_circuit(circuit_path, selected_backends)
        
        # 显示结果摘要
        self._show_results_summary(results)
    
    def _select_circuit(self):
        """选择电路"""
        circuits = list_available_circuits()
        circuit_list = list(circuits.keys())
        
        print("\n选择电路:")
        for i, circuit in enumerate(circuit_list, 1):
            print(f"{i:2d}. {circuit}")
        
        try:
            choice = int(input(f"请输入电路编号 (1-{len(circuit_list)}): ").strip())
            if 1 <= choice <= len(circuit_list):
                circuit_name = circuit_list[choice-1]
                return circuits[circuit_name]['path']
            else:
                print("❌ 无效编号")
                return None
        except ValueError:
            print("❌ 请输入有效数字")
            return None
    
    def _select_backends(self):
        """选择后端"""
        available_backends = backend_registry.get_available_backends()
        backend_list = list(available_backends.keys())
        
        print("\n选择后端 (可多选，用逗号分隔):")
        for i, backend in enumerate(backend_list, 1):
            config = available_backends[backend]
            marker = " (基准)" if config.is_baseline else ""
            print(f"{i:2d}. {backend}{marker}")
        
        print(f"{len(backend_list)+1:2d}. 测试所有后端")
        
        try:
            choice = input(f"请输入后端编号 (1-{len(backend_list)+1}): ").strip()
            
            if choice == str(len(backend_list)+1):
                return None  # 测试所有后端
            
            choices = [int(x.strip()) for x in choice.split(',')]
            selected = []
            
            for choice_num in choices:
                if 1 <= choice_num <= len(backend_list):
                    selected.append(backend_list[choice_num-1])
                else:
                    print(f"❌ 无效编号: {choice_num}")
                    return None
            
            return selected if selected else None
            
        except ValueError:
            print("❌ 请输入有效数字")
            return None
    
    def _show_results_summary(self, results):
        """显示结果摘要"""
        print("\n📊 测试结果摘要:")
        print("-"*60)
        
        successful = {}
        for backend, metrics in results.items():
            if metrics.execution_time_mean:
                successful[backend] = metrics
        
        if not successful:
            print("❌ 没有成功的测试结果")
            return
        
        # 按性能排序
        sorted_results = sorted(successful.items(), 
                              key=lambda x: x[1].execution_time_mean)
        
        print("性能排名:")
        for i, (backend, metrics) in enumerate(sorted_results, 1):
            time_str = f"{metrics.execution_time_mean:.4f}s"
            memory_str = f"{metrics.peak_memory_mb:.1f}MB"
            speedup_str = f" ({metrics.speedup:.2f}x)" if metrics.speedup else ""
            correctness_str = f" [{metrics.correctness}]" if metrics.correctness != "Passed (no baseline)" else ""
            
            print(f"{i}. {backend}: {time_str}, {memory_str}{speedup_str}{correctness_str}")
    
    def customize_config(self):
        """自定义配置"""
        print("\n⚙️ 自定义测试配置")
        print(f"当前配置:")
        print(f"  运行次数: {self.config.num_runs}")
        print(f"  预热次数: {self.config.warmup_runs}")
        print(f"  输出格式: {', '.join(self.config.output_formats)}")
        
        try:
            num_runs = int(input(f"运行次数 (当前: {self.config.num_runs}): ").strip() or str(self.config.num_runs))
            warmup_runs = int(input(f"预热次数 (当前: {self.config.warmup_runs}): ").strip() or str(self.config.warmup_runs))
            
            self.config.num_runs = max(1, num_runs)
            self.config.warmup_runs = max(0, warmup_runs)
            
            print(f"✅ 配置已更新")
            print(f"  运行次数: {self.config.num_runs}")
            print(f"  预热次数: {self.config.warmup_runs}")
            
        except ValueError:
            print("❌ 请输入有效数字")

# 启动交互式工具
if __name__ == "__main__":
    tool = InteractiveBenchmarkTool()
    tool.show_main_menu()
```

---

## 技术架构

### 核心类设计

#### BackendConfig 类
`BackendConfig` 是后端配置的核心数据类，使用 Python 的 `@dataclass` 装饰器实现：

```python
@dataclass
class BackendConfig:
    display_name: str           # 用户友好的显示名称
    backend_name: str          # Qibo 框架中的实际后端名称
    platform_name: Optional[str]  # 平台特定名称（如 numba, jax 等）
    description: str           # 后端描述信息
    dependencies: List[str]    # 依赖包列表
    priority: int = 0          # 优先级，用于排序显示
    is_baseline: bool = False  # 是否为基准后端
    
    def validate(self) -> bool:
        """验证后端依赖是否满足"""
        try:
            for dep in self.dependencies:
                importlib.import_module(dep)
            return True
        except ImportError:
            return False
```

#### BackendRegistry 类
`BackendRegistry` 实现了后端的注册和管理：

```python
class BackendRegistry:
    def __init__(self):
        self._backends: Dict[str, BackendConfig] = {}
    
    def register(self, config: BackendConfig):
        """注册新后端"""
        self._backends[config.display_name] = config
    
    def get_available_backends(self) -> Dict[str, BackendConfig]:
        """获取所有可用（依赖满足）的后端"""
        return {name: config for name, config in self._backends.items() 
                if config.validate()}
```

### 后端管理机制

系统采用注册器模式管理后端：

1. **注册阶段**: 在模块加载时自动注册所有默认后端
2. **验证阶段**: 在使用时动态验证后端可用性
3. **选择阶段**: 根据用户输入过滤要测试的后端
4. **执行阶段**: 按优先级顺序执行基准测试

### 基准测试流程

```
┌─────────────────┐
│   用户输入       │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   后端选择       │
│ - 解析后端字符串  │
│ - 验证可用性     │
│ - 过滤选择       │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   电路加载       │
│ - 读取 QASM     │
│ - 清理代码       │
│ - 构建电路对象   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   基准测试       │
│ - 预热运行       │
│ - 正式测试       │
│ - 性能测量       │
│ - 结果验证       │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   报告生成       │
│ - CSV 格式      │
│ - Markdown 格式 │
│ - JSON 格式     │
└─────────────────┘
```

---

## API 参考

### 核心函数

#### run_benchmark_for_circuit()

```python
def run_benchmark_for_circuit(circuit_path: str, selected_backends: Optional[List[str]] = None) -> Optional[Dict]:
    """
    为指定电路路径运行基准测试
    
    Args:
        circuit_path: QASM 文件路径
        selected_backends: 要测试的后端列表，None 表示测试所有后端
    
    Returns:
        测试结果字典，键为后端名称，值为 QASMBenchMetrics 对象
    """
```

#### list_available_backends()

```python
def list_available_backends() -> None:
    """
    列出所有可用的后端信息
    
    显示每个后端的名称、描述、状态和依赖信息
    """
```

#### parse_backend_string()

```python
def parse_backend_string(backend_string: str) -> Optional[List[str]]:
    """
    解析后端字符串为列表
    
    Args:
        backend_string: 后端字符串，如 "qibojit(numba)" 或 "numpy,qibojit(numba)"
    
    Returns:
        后端名称列表，None 表示全部
    """
```

### 核心类

#### QASMBenchRunner

```python
class QASMBenchRunner:
    def __init__(self, config: QASMBenchConfig):
        """初始化基准测试运行器"""
    
    def run_benchmark_for_circuit(self, circuit_name: str, qasm_file_path: str, 
                                selected_backends: Optional[List[str]] = None) -> Dict:
        """为特定电路运行基准测试"""
    
    def generate_reports(self, results: Dict, circuit_name: str, circuit: Optional[Circuit] = None):
        """生成所有格式的报告"""
```

#### QASMBenchConfig

```python
class QASMBenchConfig:
    def __init__(self):
        self.num_runs = 5                    # 正式运行次数
        self.warmup_runs = 1                 # 预热运行次数
        self.output_formats = ['csv', 'markdown', 'json']
        self.baseline_backend = "numpy"
        self.qasm_directory = "../QASMBench"
```

#### QASMBenchMetrics

```python
class QASMBenchMetrics:
    def __init__(self):
        # 核心指标
        self.execution_time_mean = None      # 平均执行时间
        self.execution_time_std = None       # 执行时间标准差
        self.peak_memory_mb = None          # 峰值内存使用
        self.speedup = None                 # 加速比
        self.correctness = "Unknown"        # 正确性验证结果
        
        # 电路信息
        self.circuit_parameters = {}         # 电路参数
        self.backend_info = {}              # 后端信息
        
        # 性能指标
        self.throughput_gates_per_sec = None
        self.jit_compilation_time = None
        self.environment_info = {}
        
        # 元数据
        self.circuit_build_time = None
        self.report_metadata = {}
```

---

## 扩展开发

### 添加新后端

#### 方法 1: 修改默认配置

```python
# 在 register_default_backends() 函数中添加
def register_default_backends():
    default_configs = [
        # ... 现有配置 ...
        
        # 添加新后端
        BackendConfig(
            display_name="my_backend",
            backend_name="custom_backend",
            platform_name="custom_platform",
            description="我的自定义后端",
            dependencies=["my_backend_lib"],
            priority=10
        )
    ]
```

#### 方法 2: 动态注册

```python
from qasmbench_runner_backend_selection import BackendConfig, backend_registry

# 创建自定义后端配置
custom_backend = BackendConfig(
    display_name="experimental_gpu",
    backend_name="qibojit",
    platform_name="cuda",
    description="实验性 GPU 后端",
    dependencies=["cupy", "qibo"],
    priority=15
)

# 注册到全局注册器
backend_registry.register(custom_backend)
```

### 自定义报告格式

```python
from qasmbench_runner_backend_selection import QASMBenchReporter

class CustomReporter(QASMBenchReporter):
    @staticmethod
    def generate_html_report(results, circuit_name, filename=None):
        """生成 HTML 格式报告"""
        if filename is None:
            clean_circuit_name = circuit_name.replace('/', '_').replace('\\', '_')
            filename = f"qibobench/reports/{clean_circuit_name}/benchmark_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QASMBench 报告: {circuit_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>QASMBench 基准测试报告: {circuit_name}</h1>
            <table>
                <tr><th>后端</th><th>执行时间</th><th>内存</th><th>加速比</th></tr>
        """
        
        for backend, metrics in results.items():
            if metrics.execution_time_mean:
                html_content += f"""
                <tr>
                    <td>{backend}</td>
                    <td>{metrics.execution_time_mean:.4f}s</td>
                    <td>{metrics.peak_memory_mb:.1f}MB</td>
                    <td>{metrics.speedup:.2f}x</td>
                </tr>
                """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {filename}")

# 使用自定义报告器
custom_reporter = CustomReporter()
custom_reporter.generate_html_report(results, "test_circuit")
```

---

## 故障排除

### 常见问题

#### 1. 后端不可用

**问题**: `⚠️ 警告: 未知后端 'xxx'，已跳过`

**解决方案**:
```bash
# 检查可用后端
python qasmbench_runner_backend_selection.py --list-backends

# 检查后端状态
python qasmbench_runner_backend_selection.py --backend-status
```

#### 2. 依赖缺失

**问题**: `❌ 状态: 不可用 (缺少依赖)`

**解决方案**:
```bash
# 安装缺失的依赖
pip install numba jax torch tensorflow qutensornet qulacs

# 或者安装完整环境
pip install qibo[jit,ml,tn,qulacs]
```

#### 3. 电路文件找不到

**问题**: `错误: 电路文件不存在: xxx`

**解决方案**:
```bash
# 检查文件路径
ls -la QASMBench/medium/qft_n18/

# 使用绝对路径
python qasmbench_runner_backend_selection.py --circuit "/full/path/to/circuit.qasm"
```

#### 4. 内存不足

**问题**: 大规模电路测试时内存溢出

**解决方案**:
```python
# 减少运行次数
config = QASMBenchConfig()
config.num_runs = 1  # 减少到单次运行
config.warmup_runs = 0  # 跳过预热

# 或者在命令行中（如果支持）
# --num-runs 1 --warmup-runs 0
```

#### 5. JIT 编译时间过长

**问题**: 首次运行某个后端时时间很长

**解决方案**:
```python
# 增加预热次数
config = QASMBenchConfig()
config.warmup_runs = 3  # 增加预热次数

# 或者预热后单独测试
# 先运行一次预热
run_benchmark_for_circuit("circuit.qasm", ["qibojit(numba)"])
# 再进行正式测试
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 然后运行测试
results = run_benchmark_for_circuit("circuit.qasm", ["numpy"])
```

#### 2. 单步调试

```python
# 分步执行以便调试
from qasmbench_runner_backend_selection import QASMBenchRunner, QASMBenchConfig

config = QASMBenchConfig()
runner = QASMBenchRunner(config)

# 1. 先测试电路加载
circuit = runner.load_qasm_circuit("circuit.qasm")
print(f"电路加载成功: {circuit is not None}")

# 2. 测试单个后端
result, metrics = runner._run_single_backend_benchmark(
    "numpy", "numpy", None, "circuit.qasm"
)
print(f"单后端测试: {metrics.execution_time_mean}")
```

#### 3. 性能分析

```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

results = run_benchmark_for_circuit("circuit.qasm", ["numpy"])

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 显示前10个最耗时的函数
```

### 错误代码参考

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `未知后端` | 后端名称拼写错误或未注册 | 检查 `--list-backends` 输出 |
| `缺少依赖` | 相关 Python 包未安装 | 使用 pip 安装缺失依赖 |
| `电路文件不存在` | 文件路径错误 | 检查文件路径和权限 |
| `基准测试失败` | 后端运行时错误 | 查看详细错误信息，检查后端配置 |
| `Failed - Shape mismatch` | 不同后端结果形状不一致 | 检查电路兼容性 |

---

## 总结

`qasmbench_runner_backend_selection.py` 是一个功能强大、灵活可扩展的量子电路基准测试工具。通过模块化的设计和丰富的配置选项，它能够满足从简单性能比较到复杂研究项目的各种需求。

### 主要优势

- ✅ **灵活的后端选择**: 精确控制要测试的后端组合
- ✅ **全面的性能指标**: 执行时间、内存使用、吞吐率等多维度分析
- ✅ **自动正确性验证**: 确保不同后端计算结果的一致性
- ✅ **多格式报告**: CSV、Markdown、JSON 满足不同使用场景
- ✅ **良好的可扩展性**: 易于添加新后端和自定义功能
- ✅ **友好的用户界面**: 详细的帮助信息和错误提示

### 适用场景

- 📊 **性能评估研究**: 比较不同后端的性能表现
- 🔬 **算法开发**: 测试新算法在不同后端上的表现
- 🚀 **系统优化**: 找到特定场景下的最优后端配置
- 📚 **教学演示**: 展示量子计算后端的差异和特点
- 🏭 **生产部署**: 为实际应用选择最适合的后端

通过本报告的详细说明，用户可以充分利用该工具的强大功能，进行高效的量子电路基准测试和性能分析。

---

*报告版本: v1.0*  
*更新时间: 2025-10-13*  
*作者: QASMBench 开发团队*
