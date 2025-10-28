# run_benchmarks.py API 文档

## 概述

`run_benchmarks.py` 是量子模拟器基准测试运行器的核心模块，提供了完整的命令行接口和API接口，用于运行不同量子计算框架的性能基准测试。该模块支持多种量子电路类型和模拟器框架，能够自动生成详细的性能报告和可视化图表。

### 主要功能

- **多框架支持**: 支持 Qibo、Qiskit 等多种量子计算框架的性能比较
- **多种电路**: 支持量子傅里叶变换(QFT)、Grover算法等基准电路
- **性能指标**: 测量执行时间、内存使用、CPU利用率和状态保真度
- **统计分析**: 支持重复运行、预热运行和统计分析
- **缓存系统**: 智能参考态缓存，提高测试效率
- **自动化报告**: 自动生成CSV数据、可视化图表和Markdown摘要报告

## 核心函数和类

### 1. parse_arguments()

解析命令行参数并配置基准测试运行器。

```python
def parse_arguments() -> argparse.Namespace:
```

**返回值**: `argparse.Namespace` - 包含所有解析后的命令行参数的对象

**主要参数**:
- `--circuits`: 电路类型列表 (默认: ["qft"])
- `--qubits`: 量子比特数列表 (默认: [2, 3, 4])
- `--simulators`: 模拟器配置列表 (默认: ["qibo-qibojit"])
- `--golden-standard`: 参考态生成模拟器 (默认: "qibo-qibojit")
- `--output-dir`: 输出目录 (默认: "results")
- `--repeat`: 重复运行次数 (默认: 1)
- `--statistical-analysis`: 启用统计分析
- `--enable-cache`: 启用缓存 (默认: True)

### 2. create_simulator_instances()

根据配置列表创建模拟器实例字典。

```python
def create_simulator_instances(simulator_configs: List[str]) -> Dict[str, SimulatorInterface]:
```

**参数**:
- `simulator_configs` (List[str]): 模拟器配置列表，格式为"platform-backend"

**返回值**: `Dict[str, SimulatorInterface]` - 成功创建的模拟器实例字典

**示例**:
```python
configs = ["qibo-numpy", "qibo-qibojit"]
simulators = create_simulator_instances(configs)
# 返回: {"qibo-numpy": QiboWrapper(...), "qibo-qibojit": QiboWrapper(...)}
```

### 3. create_circuit_instances()

根据电路名称列表创建电路实例。

```python
def create_circuit_instances(circuit_names: List[str]) -> List[BenchmarkCircuit]:
```

**参数**:
- `circuit_names` (List[str]): 电路名称列表，如["qft", "grover"]

**返回值**: `List[BenchmarkCircuit]` - 成功创建的电路实例列表

**支持的电路**:
- `qft`: 量子傅里叶变换电路
- `grover`: Grover搜索算法电路

### 4. run_benchmarks()

运行量子模拟器基准测试的核心函数。

```python
def run_benchmarks(
    circuits: List[BenchmarkCircuit],
    qubit_ranges: List[int],
    simulators: Dict[str, SimulatorInterface],
    golden_standard_key: str,
    cache_config: Optional[CacheConfig] = None,
) -> List[Any]:
```

**参数**:
- `circuits`: 要测试的电路实例列表
- `qubit_ranges`: 要测试的量子比特数列表
- `simulators`: 模拟器实例字典
- `golden_standard_key`: 黄金标准模拟器的键名
- `cache_config`: 缓存配置对象 (可选)

**返回值**: `List[Any]` - 所有基准测试结果的列表

**执行流程**:
1. 使用黄金标准模拟器生成参考态
2. 在所有模拟器上运行电路并收集性能指标
3. 计算状态保真度和其他统计信息

### 5. setup_output_directory()

设置输出目录并确保必要的目录结构存在。

```python
def setup_output_directory(args) -> Path:
```

**参数**: `args` - 命令行参数对象

**返回值**: `Path` - 输出目录路径

### 6. initialize_components()

初始化模拟器和电路组件。

```python
def initialize_components(args) -> tuple:
```

**参数**: `args` - 命令行参数对象

**返回值**: `tuple` - (simulators, circuits) 模拟器和电路组件

### 7. setup_cache_configuration()

设置缓存配置。

```python
def setup_cache_configuration(args) -> Optional[CacheConfig]:
```

**参数**: `args` - 命令行参数对象

**返回值**: `Optional[CacheConfig]` - 缓存配置对象

### 8. main()

主入口函数，协调整个基准测试流程。

```python
def main() -> int:
```

**返回值**: `int` - 程序退出码 (0表示成功，1表示失败)

**执行步骤**:
1. 解析命令行参数
2. 设置输出目录
3. 初始化组件
4. 设置缓存配置
5. 执行基准测试
6. 显示缓存统计信息
7. 处理结果并生成报告

## 使用示例

### 示例 1: 基本命令行使用

```bash
# 基本用法 - 使用默认设置
python scripts/run_benchmarks.py

# 指定电路和量子比特数
python scripts/run_benchmarks.py --circuits qft --qubits 2 3 4 5 --verbose

# 测试多个模拟器
python scripts/run_benchmarks.py --simulators qibo-numpy qibo-qibojit --repeat 3

# 启用统计分析
python scripts/run_benchmarks.py --qubits 2 3 4 5 6 --repeat 5 --statistical-analysis --verbose
```

### 示例 2: 程序化调用

```python
import sys
from pathlib import Path

# 添加Bench目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_benchmarks import (
    parse_arguments,
    create_simulator_instances,
    create_circuit_instances,
    run_benchmarks,
    setup_output_directory,
    initialize_components,
    setup_cache_configuration,
    process_results
)

class BenchmarkConfig:
    def __init__(self):
        self.circuits = ["qft"]
        self.qubits = [2, 3, 4]
        self.simulators = ["qibo-qibojit"]
        self.golden_standard = "qibo-qibojit"
        self.output_dir = "results"
        self.verbose = True
        self.enable_cache = True
        self.no_cache = False
        self.repeat = 3
        self.warmup_runs = 1
        self.statistical_analysis = True
        self.cache_type = "hybrid"
        self.cache_dir = ".benchmark_cache"
        self.memory_cache_size = 64
        self.clear_cache = False
        self.cache_stats = False

def run_custom_benchmark():
    """运行自定义基准测试"""

    # 创建配置
    config = BenchmarkConfig()

    # 设置输出目录
    output_dir = setup_output_directory(config)
    print(f"输出目录: {output_dir}")

    # 初始化组件
    simulators, circuits = initialize_components(config)

    # 设置缓存
    cache_config = setup_cache_configuration(config)

    # 执行基准测试
    results = run_benchmarks(
        circuits=circuits,
        qubit_ranges=config.qubits,
        simulators=simulators,
        golden_standard_key=config.golden_standard,
        cache_config=cache_config
    )

    # 处理结果
    success = process_results(results, output_dir, config)

    if success:
        print(f"基准测试完成! 结果保存在: {output_dir}")
        return 0
    else:
        print("基准测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(run_custom_benchmark())
```

### 示例 3: 高级配置和批量测试

```python
import argparse
from scripts.run_benchmarks import *

def run_comparative_benchmark():
    """运行比较性基准测试"""

    # 创建自定义参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuits", nargs="+", default=["qft", "grover"])
    parser.add_argument("--qubits", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--simulators", nargs="+",
                       default=["qibo-numpy", "qibo-qibojit"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--statistical-analysis", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enable-cache", action="store_true", default=True)

    args = parser.parse_args([])

    # 设置输出目录
    output_dir = setup_output_directory(args)

    # 初始化所有组件
    simulators, circuits = initialize_components(args)
    cache_config = setup_cache_configuration(args)

    print(f"开始基准测试:")
    print(f"  电路: {[c.name for c in circuits]}")
    print(f"  量子比特数: {args.qubits}")
    print(f"  模拟器: {list(simulators.keys())}")
    print(f"  重复次数: {args.repeat}")

    # 执行基准测试
    results = run_benchmarks(
        circuits=circuits,
        qubit_ranges=args.qubits,
        simulators=simulators,
        golden_standard_key="qibo-qibojit",
        cache_config=cache_config
    )

    # 处理和展示结果
    success = process_results(results, output_dir, args)

    # 输出性能摘要
    if success and results:
        print("\n=== 性能摘要 ===")
        for result in results[:5]:  # 显示前5个结果
            print(f"{result.simulator}-{result.backend} "
                  f"| {result.circuit_name} ({result.n_qubits} qubits) "
                  f"| 时间: {result.wall_time_sec:.4f}s "
                  f"| 保真度: {result.state_fidelity:.4f}")

    return success

# 运行比较测试
if __name__ == "__main__":
    run_comparative_benchmark()
```

## 依赖和环境要求

### 核心依赖

- **Python 3.7+**: 基础运行环境
- **pandas**: 数据处理和分析 (`pip install pandas`)
- **argparse**: 命令行参数解析 (Python标准库)
- **pathlib**: 路径操作 (Python标准库)
- **datetime**: 时间处理 (Python标准库)

### 项目内部依赖

- **src.abstractions**: 基准测试抽象类
  - `BenchmarkCircuit`: 电路基准测试抽象基类
  - `SimulatorInterface`: 模拟器接口抽象基类

- **src.simulators**: 模拟器包装器
  - `QiboWrapper`: Qibo框架包装器

- **src.circuits**: 基准电路实现
  - `QFTCircuit`: 量子傅里叶变换电路
  - `GroverCircuit`: Grover搜索算法电路

- **src.caching**: 缓存系统
  - `CacheConfig`: 缓存配置类
  - `create_cache_instance()`: 创建缓存实例函数

- **src.post_processing**: 后处理模块
  - `analyze_results()`: 结果分析函数
  - `generate_summary_report()`: 生成摘要报告函数

- **src.path_config**: 路径配置
  - `get_benchmark_path()`: 获取基准测试路径
  - `ensure_directories()`: 确保目录存在

### 量子计算框架依赖

根据使用的模拟器不同，需要安装相应的量子计算框架：

- **Qibo**: `pip install qibo[qibojit]`
- **Qiskit**: `pip install qiskit` (如果使用Qiskit模拟器)

### 可选依赖

- **matplotlib**: 可视化图表生成 (通常由pandas或其他依赖自动安装)
- **numpy**: 数值计算 (量子框架的依赖)

## 输出结果结构

基准测试完成后，会在指定的输出目录中生成以下文件：

```
benchmark_YYYYMMDD_HHMMSS/
├── raw_results.csv              # 原始测试数据
├── detailed_runs.csv           # 详细运行数据 (重复运行时)
├── summary_report.md           # Markdown格式摘要报告
├── cpu_time_scaling.png        # CPU时间扩展性图表
├── wall_time_scaling.png       # 实际时间扩展性图表
├── memory_scaling.png          # 内存使用扩展性图表
├── cpu_utilization.png         # CPU利用率图表
├── fidelity.png                # 状态保真度图表
├── confidence_intervals.png    # 置信区间图表 (统计分析时)
└── execution_stability.png     # 执行稳定性图表 (统计分析时)
```

## 错误处理

模块提供了完善的错误处理机制：

- **模拟器初始化失败**: 打印警告并跳过该模拟器
- **电路创建失败**: 打印警告并跳过该电路
- **参考态生成失败**: 跳过当前电路和量子比特数的测试
- **单个测试执行失败**: 跳过当前模拟器，继续其他测试
- **缓存系统失败**: 降级为直接计算，不影响主要功能

## 性能优化建议

1. **启用缓存**: 对于重复测试，启用参考态缓存可显著提高性能
2. **合理设置重复次数**: 根据需要设置合适的重复运行次数
3. **预热运行**: 对于精确测量，建议设置预热运行次数
4. **批量测试**: 一次性测试多个量子比特数，减少初始化开销
5. **选择合适的黄金标准**: 使用性能较好但精度高的模拟器作为黄金标准

---

*文档版本: 1.0.0*
*最后更新: 2025-10-27*