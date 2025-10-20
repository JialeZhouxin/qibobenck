#!/usr/bin/env python3
"""
量子模拟器基准测试运行器

这个脚本提供了一个命令行接口，用于运行不同量子计算框架的基准测试。
它支持多种量子电路和模拟器的性能比较，包括执行时间、内存使用、
CPU利用率和状态保真度等指标的测量。

主要功能：
- 支持多种量子电路（如QFT）的基准测试
- 支持多个量子计算框架的性能比较
- 自动生成参考态用于保真度计算
- 详细的性能指标收集和分析
- 自动生成可视化报告和摘要

使用示例：
    # 基本用法
    python run_benchmarks.py --circuits qft --qubits 2 3 4 --simulators qibo-numpy
    
    # 完整测试
    python run_benchmarks.py --circuits qft --qubits 2 3 4 5 --simulators qibo-numpy qibo-qibojit --verbose
    
    # 自定义输出目录
    python run_benchmarks.py --output-dir my_results --verbose

依赖：
- pandas: 数据处理和分析
- benchmark_harness: 基准测试框架（包含抽象类、电路定义、模拟器包装器等）

作者：量子计算研究团队
版本：1.0.0
"""

import argparse
import importlib
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# 添加当前目录到Python路径，确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from benchmark_harness.abstractions import BenchmarkCircuit, SimulatorInterface
from benchmark_harness.caching import CacheConfig, create_cache_instance
from benchmark_harness.post_processing import (analyze_results,
                                               generate_summary_report)
from benchmark_harness.simulators import QiboWrapper

# 全局变量存储命令行参数，用于在函数中访问verbose选项
# 注意：这个变量在main()函数中被初始化，在其他函数中使用时需要确保已初始化
args: argparse.Namespace


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数并配置基准测试运行器。

    这个函数设置并解析所有必要的命令行参数，包括电路选择、
    量子比特数范围、模拟器配置等。使用argparse库提供
    自动生成的帮助信息和默认值显示。

    Returns:
        argparse.Namespace: 包含所有解析后的命令行参数的对象
        
    Note:
        主要参数说明：
        - circuits: 支持的量子电路类型（目前仅支持qft）
        - qubits: 测试的量子比特数列表，用于扩展性测试
        - simulators: 模拟器配置，格式为"platform-backend"
        - golden-standard: 用于生成参考态的黄金标准模拟器
        - output-dir: 结果保存目录，会自动添加时间戳子目录
        - verbose: 详细输出模式，显示更多调试信息

    Examples:
        >>> args = parse_arguments()
        >>> args.circuits
        ['qft']
        >>> args.qubits
        [2, 3, 4]
        >>> args.simulators
        ['qibo-numpy']
    """
    # 创建参数解析器，设置描述和帮助格式
    parser = argparse.ArgumentParser(
        description="运行量子模拟器基准测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 电路选择参数
    parser.add_argument(
        "--circuits",
        nargs="+",
        default=["qft"],
        choices=["qft","grover"],
        help="要运行的基准测试电路列表",
    )

    # 量子比特数范围参数
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="要测试的量子比特数列表",
    )

    # 模拟器选择参数
    parser.add_argument(
        "--simulators",
        nargs="+",
        default=["qibo-qibojit"],
        help="要测试的模拟器列表，格式为platform-backend",
    )

    # 黄金标准参考态参数
    parser.add_argument(
        "--golden-standard", 
        default="qibo-qibojit", 
        help="用于生成参考态的模拟器"
    )

    # 输出目录参数
    parser.add_argument(
        "--output-dir", 
        default="results", 
        help="结果输出目录"
    )

    # 详细输出参数
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细输出"
    )

    # 缓存相关参数
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=True,
        help="启用参考态缓存"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用缓存（覆盖--enable-cache）"
    )

    parser.add_argument(
        "--cache-type",
        choices=["memory", "disk", "hybrid"],
        default="hybrid",
        help="缓存类型选择"
    )

    parser.add_argument(
        "--cache-dir",
        default=".benchmark_cache",
        help="磁盘缓存目录"
    )

    parser.add_argument(
        "--memory-cache-size",
        type=int,
        default=64,
        help="内存缓存最大条目数"
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="开始前清空缓存"
    )

    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="显示缓存统计信息"
    )

    # 重复运行参数
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每个电路重复运行的次数（默认：1）"
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="正式测量前的预热运行次数（默认：0）"
    )

    parser.add_argument(
        "--statistical-analysis",
        action="store_true",
        help="启用统计分析，计算标准差、置信区间等"
    )

    return parser.parse_args()


def create_simulator_instances(
    simulator_configs: List[str],
) -> Dict[str, SimulatorInterface]:
    """根据配置列表创建模拟器实例字典。

    这个函数解析模拟器配置字符串，动态导入相应的模拟器包装器类，
    并创建模拟器实例。支持多种量子计算框架的插件式扩展。

    Args:
        simulator_configs (List[str]): 模拟器配置列表，每个配置格式为"platform-backend"
            例如：["qibo-numpy", "qibo-qibojit", "qiskit-aer"]

    Returns:
        Dict[str, SimulatorInterface]: 成功创建的模拟器实例字典，
            键为配置字符串，值为对应的模拟器实例

    Raises:
        ValueError: 当配置字符串格式不正确时抛出

    Note:
        动态导入机制：
        - 对于"qibo"平台，直接使用QiboWrapper类
        - 对于其他平台，尝试从"benchmark_harness.simulators.{platform}_wrapper"模块导入
        - 类名预期格式为"{Platform}Wrapper"（如QiskitWrapper）
        
        错误处理策略：
        - 配置格式错误：抛出ValueError异常
        - 导入失败：打印警告并跳过该配置
        - 实例化失败：打印警告并跳过该配置

    Examples:
        >>> configs = ["qibo-numpy", "qibo-qibojit"]
        >>> simulators = create_simulator_instances(configs)
        >>> list(simulators.keys())
        ['qibo-numpy', 'qibo-qibojit']
    """
    simulators = {}

    # 遍历每个模拟器配置
    for config in simulator_configs:
        # 解析配置字符串，分离平台和后端
        try:
            platform, backend = config.split("-", 1)
        except ValueError:
            raise ValueError(
                f"Invalid simulator configuration: {config}. Expected format: platform-backend"
            )

        # 动态导入模拟器包装器类
        try:
            if platform == "qibo":
                # Qibo平台使用预导入的QiboWrapper类
                simulator_class = QiboWrapper
            else:
                # 尝试动态导入其他平台的包装器类
                module_name = f"benchmark_harness.simulators.{platform}_wrapper"
                module = importlib.import_module(module_name)
                simulator_class = getattr(module, f"{platform.title()}Wrapper")
        except (ImportError, AttributeError) as e:
            print(f"Warning: Failed to import {platform} wrapper: {e}")
            continue

        # 创建模拟器实例
        try:
            simulator_instance = simulator_class(backend)
            simulators[config] = simulator_instance
            if args.verbose:
                print(
                    f"Successfully created {platform} simulator with backend {backend}"
                )
        except Exception as e:
            print(
                f"Warning: Failed to create {platform} simulator with backend {backend}: {e}"
            )
            continue

    return simulators


def create_circuit_instances(circuit_names: List[str]) -> List[BenchmarkCircuit]:
    """根据电路名称列表创建电路实例。

    这个函数通过动态导入机制加载指定的量子电路类，
    并创建相应的电路实例。支持多种量子电路类型的
    插件式扩展。

    Args:
        circuit_names (List[str]): 电路名称列表，例如["qft", "ghz", "bv"]
            这些名称对应benchmark_harness.circuits包中的模块

    Returns:
        List[BenchmarkCircuit]: 成功创建的电路实例列表

    Note:
        动态导入规则：
        - 模块路径：benchmark_harness.circuits.{circuit_name}
        - 类名规则：
          * 对于"qft"：使用"QFTCircuit"类
          * 对于其他电路：使用"{CircuitName}Circuit"类（如GHZCircuit）
        
        错误处理策略：
        - 导入失败：打印警告并跳过该电路
        - 类获取失败：打印警告并跳过该电路
        - 实例化失败：打印警告并跳过该电路

    Examples:
        >>> circuit_names = ["qft", "ghz"]
        >>> circuits = create_circuit_instances(circuit_names)
        >>> len(circuits)
        2
        >>> [c.name for c in circuits]
        ['Quantum Fourier Transform', 'Greenberger-Horne-Zeilinger State']
    """
    circuits = []

    # 遍历每个电路名称
    for circuit_name in circuit_names:
        try:
            # 动态导入电路模块
            module_name = f"benchmark_harness.circuits.{circuit_name}"
            module = importlib.import_module(module_name)
            
            # 根据电路名称获取对应的电路类
            if circuit_name.lower() == "qft":
                # 特殊处理QFT电路，使用QFTCircuit类名
                circuit_class = getattr(module, "QFTCircuit")
            elif circuit_name.lower() == "grover":
                # 特殊处理Grover电路，使用GroverCircuit类名
                circuit_class = getattr(module, "GroverCircuit")
            else:
                # 其他电路使用标准命名规则：{CircuitName}Circuit
                circuit_class = getattr(module, f"{circuit_name.title()}Circuit")

            # 创建电路实例
            circuit_instance = circuit_class()
            circuits.append(circuit_instance)

            # 详细模式下输出成功信息
            if args.verbose:
                print(f"Successfully created {circuit_name} circuit")
                
        except (ImportError, AttributeError) as e:
            print(f"Warning: Failed to import {circuit_name} circuit: {e}")
            continue

    return circuits


def run_benchmarks(
    circuits: List[BenchmarkCircuit],
    qubit_ranges: List[int],
    simulators: Dict[str, SimulatorInterface],
    golden_standard_key: str,
    cache_config: Optional[CacheConfig] = None,
) -> List[Any]:
    """运行量子模拟器基准测试的核心函数。

    这个函数执行两阶段的基准测试流程：
    1. 使用黄金标准模拟器生成参考态
    2. 在所有模拟器上运行相同的电路并比较结果

    Args:
        circuits (List[BenchmarkCircuit]): 要测试的电路实例列表
        qubit_ranges (List[int]): 要测试的量子比特数列表
        simulators (Dict[str, SimulatorInterface]): 模拟器实例字典
        golden_standard_key (str): 黄金标准模拟器的键名

    Returns:
        List[Any]: 所有基准测试结果的列表，每个结果包含性能指标

    Raises:
        ValueError: 当黄金标准模拟器不可用时抛出

    Note:
        测试流程说明：
        阶段A - 参考态生成：
        - 使用黄金标准模拟器执行电路
        - 获得精确的量子态作为参考基准
        - 用于计算其他模拟器的状态保真度

        阶段B - 性能测试：
        - 在每个模拟器上执行相同的电路
        - 测量执行时间、内存使用、CPU利用率等指标
        - 计算与参考态的保真度
        - 黄金标准模拟器的保真度设为1.0

        错误处理策略：
        - 参考态生成失败：跳过当前电路和量子比特数的测试
        - 单个模拟器执行失败：跳过该模拟器，继续其他测试
        - 保证部分失败不影响其他测试的执行

    Examples:
        >>> circuits = [qft_circuit]
        >>> qubits = [2, 3]
        >>> simulators = {"qibo-numpy": qibo_wrapper}
        >>> results = run_benchmarks(circuits, qubits, simulators, "qibo-numpy")
        >>> len(results)
        2  # 2个量子比特数 × 1个电路 × 1个模拟器
    """
    all_results = []

    # 验证黄金标准模拟器是否可用
    if golden_standard_key not in simulators:
        raise ValueError(
            f"Golden standard simulator '{golden_standard_key}' not available"
        )

    golden_wrapper = simulators[golden_standard_key]
    
    # 初始化缓存
    cache = None
    if cache_config and cache_config.enable_cache:
        try:
            cache = create_cache_instance(cache_config)
            if args.clear_cache:
                cache.clear_cache()
            if args.verbose:
                print(f"Initialized {cache_config.cache_type} cache")
        except Exception as e:
            print(f"Warning: Failed to initialize cache: {e}")
            cache = None

    # 遍历所有电路和量子比特数组合
    for circuit_instance in circuits:
        for n_qubits in qubit_ranges:
            print(f"\nRunning {circuit_instance.name} with {n_qubits} qubits...")

            # 阶段A: 获取参考态（使用缓存）
            reference_state = None
            circuit_name_key = circuit_instance.__class__.__name__.lower().replace('circuit', '')
            
            if cache:
                try:
                    print(f"  Getting reference state using cache...")
                    reference_state = cache.get_reference_state(
                        circuit_name=circuit_name_key,
                        n_qubits=n_qubits,
                        backend=golden_wrapper.backend_name,
                        circuit_instance=circuit_instance
                    )
                    print(f"  Reference state obtained from cache")
                except Exception as e:
                    if args.verbose:
                        print(f"  Cache failed, computing reference state: {e}")
                    reference_state = None
            
            # 如果缓存失败或未启用，直接计算参考态
            golden_result = None  # 初始化变量
            if reference_state is None:
                print(f"  Generating reference state using {golden_standard_key}...")
                circuit_for_golden = circuit_instance.build(
                    platform=golden_wrapper.platform_name, n_qubits=n_qubits
                )

                try:
                    # 执行黄金标准模拟器获得参考态
                    golden_results = golden_wrapper.execute(
                        circuit_for_golden,
                        n_qubits,
                        repeat=1,
                        warmup_runs=0
                    )
                    # 使用第一次运行的结果作为参考态
                    reference_state = golden_results[0].final_state
                     
                    # 如果有缓存，保存计算结果
                    if cache:
                        try:
                            cache.get_reference_state(
                                circuit_name=circuit_name_key,
                                n_qubits=n_qubits,
                                backend=golden_wrapper.backend_name,
                                circuit_instance=circuit_instance
                            )
                        except Exception as e:
                            if args.verbose:
                                print(f"  Warning: Failed to cache reference state: {e}")
                    
                    print(f"  Reference state generated successfully")
                except Exception as e:
                    print(f"  Error generating reference state: {e}")
                    continue  # 跳过当前测试组合

            # 阶段B: 在所有模拟器上运行基准测试
            for runner_id, wrapper_instance in simulators.items():
                print(f"  Running on {runner_id}...")

                try:
                    # 为当前模拟器构建电路
                    circuit_for_current = circuit_instance.build(
                        platform=wrapper_instance.platform_name, n_qubits=n_qubits
                    )
                    # 收集电路信息
                    circuit_info = {}
                    if wrapper_instance.platform_name == "qibo":
                        try:
                            circuit_summary = circuit_for_current.summary()
                            circuit_info["circuit_summary"] = str(circuit_summary)
                            
                            # 解析summary获取关键指标
                            summary_lines = str(circuit_summary).split('\n')
                            for line in summary_lines:
                                if "depth" in line.lower() and "=" in line:
                                    # 处理 "Circuit depth = 14" 格式
                                    circuit_info["circuit_depth"] = int(line.split("=")[-1].strip())
                                elif "qubits" in line.lower() and "=" in line:
                                    # 处理 "Number of qubits = 4" 格式
                                    circuit_info["n_qubits_in_circuit"] = int(line.split("=")[-1].strip())
                                elif "gates" in line.lower() and "total" in line.lower() and "=" in line:
                                    # 处理 "Total number of gates = 20" 格式
                                    circuit_info["total_gates"] = int(line.split("=")[-1].strip())
                            
                            if args.verbose:
                                print(f"    Circuit summary:\n{circuit_summary}")
                        except Exception as e:
                            if args.verbose:
                                print(f"    Warning: Failed to get circuit summary: {e}")
                    # 执行基准测试
                    results = wrapper_instance.execute(
                        circuit=circuit_for_current,
                        n_qubits=n_qubits,
                        reference_state=reference_state,
                        repeat=args.repeat,
                        warmup_runs=args.warmup_runs
                    )

                    # 将电路信息添加到结果对象中（使用setattr避免属性错误）
                    for result in results:
                        setattr(result, 'circuit_info', circuit_info)

                    # 如果是黄金标准模拟器，设置保真度为1.0
                    if runner_id == golden_standard_key:
                        for result in results:
                            result.state_fidelity = 1.0

                    # 收集所有测试结果
                    all_results.extend(results)
                    
                    # 显示汇总信息
                    if len(results) > 1:
                        avg_time = results[0].wall_time_mean if results[0].wall_time_mean else sum(r.wall_time_sec for r in results) / len(results)
                        std_time = results[0].wall_time_std if results[0].wall_time_std else 0
                        avg_fidelity = results[0].fidelity_mean if results[0].fidelity_mean else sum(r.state_fidelity for r in results) / len(results)
                        print(
                            f"    Completed {len(results)} runs: avg {avg_time:.4f}s ± {std_time:.4f}s, avg fidelity {avg_fidelity:.4f}"
                        )
                    else:
                        print(
                            f"    Completed in {results[0].wall_time_sec:.4f}s, fidelity: {results[0].state_fidelity:.4f}"
                        )

                except Exception as e:
                    print(f"    Error: {e}")
                    continue  # 跳过当前模拟器，继续其他测试

    return all_results


def main() -> int:
    """量子模拟器基准测试运行器的主入口函数。

    这个函数协调整个基准测试流程，包括：
    1. 解析命令行参数
    2. 初始化模拟器和电路实例
    3. 执行基准测试
    4. 处理和保存结果

    Returns:
        int: 程序退出码，0表示成功，1表示失败

    Note:
        执行流程：
        1. 参数解析和验证
        2. 创建带时间戳的输出目录
        3. 初始化所有配置的模拟器实例
        4. 初始化所有配置的电路实例
        5. 执行基准测试循环
        6. 结果后处理和报告生成

        错误处理策略：
        - 模拟器初始化失败：程序退出
        - 电路初始化失败：程序退出
        - 基准测试执行失败：记录错误但继续处理
        - 后处理失败：记录错误但保留原始结果

    Examples:
        >>> # 直接运行脚本
        >>> # python run_benchmarks.py --verbose
        >>> exit_code = main()
        >>> exit_code
        0
    """
    global args
    
    # 步骤1: 解析命令行参数
    args = parse_arguments()

    # 步骤2: 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 显示配置信息
    print(f"Quantum Simulator Benchmark Runner")
    print(f"Output directory: {output_dir}")
    print(f"Verbose mode: {args.verbose}")

    # 步骤3: 初始化模拟器实例
    print("\nInitializing simulators...")
    simulators = create_simulator_instances(args.simulators)

    if not simulators:
        print("Error: No simulators available. Exiting.")
        return 1

    print(f"Available simulators: {list(simulators.keys())}")

    # 步骤4: 初始化电路实例
    print("\nInitializing circuits...")
    circuits = create_circuit_instances(args.circuits)

    if not circuits:
        print("Error: No circuits available. Exiting.")
        return 1

    print(f"Available circuits: {[c.name for c in circuits]}")

    # 步骤5: 创建缓存配置
    cache_config = None
    if not args.no_cache and args.enable_cache:
        cache_config = CacheConfig.from_args(args)
        if args.verbose:
            print(f"Cache configuration: {cache_config.to_dict()}")
    
    # 步骤6: 执行基准测试
    print("\nRunning benchmarks...")
    try:
        results = run_benchmarks(
            circuits=circuits,
            qubit_ranges=args.qubits,
            simulators=simulators,
            golden_standard_key=args.golden_standard,
            cache_config=cache_config,
        )

        print(f"\nCompleted {len(results)} benchmark runs")
        
        # 显示缓存统计信息
        if cache_config and args.cache_stats and cache_config.enable_cache:
            print("\n" + "="*50)
            print("Cache Statistics:")
            try:
                cache = create_cache_instance(cache_config)
                stats = cache.get_cache_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error getting cache stats: {e}")
            print("="*50)

        # 步骤7: 结果后处理和报告生成
        if results:
            print(f"\nProcessing results...")
            try:
                # 分析结果并生成可视化图表
                analyze_results(results, output_dir, repeat=args.repeat)

                # 准备数据用于生成摘要报告
                data = []
                for result in results:
                    # 对于多次运行，只使用汇总结果（第一个结果）
                    if args.repeat > 1 and result.run_id > 1:
                        continue
                        
                    data.append(
                        {
                            "simulator": result.simulator,
                            "backend": result.backend,
                            "circuit_name": result.circuit_name,
                            "n_qubits": result.n_qubits,
                            "wall_time_sec": result.wall_time_mean if result.wall_time_mean else result.wall_time_sec,
                            "wall_time_std": result.wall_time_std if result.wall_time_std else 0.0,
                            "cpu_time_sec": result.cpu_time_mean if result.cpu_time_mean else result.cpu_time_sec,
                            "cpu_time_std": result.cpu_time_std if result.cpu_time_std else 0.0,
                            "peak_memory_mb": result.memory_mean if result.memory_mean else result.peak_memory_mb,
                            "memory_std": result.memory_std if result.memory_std else 0.0,
                            "cpu_utilization_percent": result.cpu_utilization_percent,
                            "state_fidelity": result.fidelity_mean if result.fidelity_mean else result.state_fidelity,
                            "fidelity_std": result.fidelity_std if result.fidelity_std else 0.0,
                            # 添加电路信息字段
                            "circuit_depth": getattr(result, 'circuit_info', {}).get("circuit_depth", None),
                            "total_gates": getattr(result, 'circuit_info', {}).get("total_gates", None),
                            "circuit_summary": getattr(result, 'circuit_info', {}).get("circuit_summary", ""),
                        }
                    )
                
                # 创建DataFrame并生成报告
                df = pd.DataFrame(data)
                df["runner_id"] = df["simulator"] + "-" + df["backend"]
                generate_summary_report(df, output_dir, repeat=args.repeat)

                print(f"All results processed and saved to {output_dir}")
            except Exception as e:
                print(f"Error during post-processing: {e}")
                print(f"Raw results are available in {output_dir}")
        else:
            print("No results to save")

    except Exception as e:
        print(f"Error during benchmark execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # 脚本入口点：当直接运行此文件时执行main()函数
    # 使用sys.exit()确保正确的退出码传递给操作系统
    sys.exit(main())
