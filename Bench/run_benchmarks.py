#!/usr/bin/env python3
"""
量子模拟器基准测试运行器

这个脚本提供了一个命令行接口，用于运行不同量子计算框架的基准测试。
"""

import argparse
import importlib
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from benchmark_harness.abstractions import BenchmarkCircuit, SimulatorInterface
from benchmark_harness.post_processing import (analyze_results,
                                               generate_summary_report)
from benchmark_harness.simulators import QiboWrapper


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行量子模拟器基准测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 电路选择
    parser.add_argument(
        "--circuits",
        nargs="+",
        default=["qft"],
        choices=["qft"],
        help="要运行的基准测试电路列表",
    )

    # 量子比特数范围
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="要测试的量子比特数列表",
    )

    # 模拟器选择
    parser.add_argument(
        "--simulators",
        nargs="+",
        default=["qibo-numpy"],
        help="要测试的模拟器列表，格式为platform-backend",
    )

    # 黄金标准参考态
    parser.add_argument(
        "--golden-standard", default="qibo-numpy", help="用于生成参考态的模拟器"
    )

    # 输出目录
    parser.add_argument("--output-dir", default="results", help="结果输出目录")

    # 详细输出
    parser.add_argument("--verbose", action="store_true", help="启用详细输出")

    return parser.parse_args()


def create_simulator_instances(
    simulator_configs: List[str],
) -> Dict[str, SimulatorInterface]:
    """创建模拟器实例"""
    simulators = {}

    for config in simulator_configs:
        try:
            platform, backend = config.split("-", 1)
        except ValueError:
            raise ValueError(
                f"Invalid simulator configuration: {config}. Expected format: platform-backend"
            )

        # 动态导入模拟器类
        try:
            if platform == "qibo":
                simulator_class = QiboWrapper
            else:
                # 尝试动态导入其他模拟器
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
    """创建电路实例"""
    circuits = []

    for circuit_name in circuit_names:
        try:
            # 动态导入电路类
            module_name = f"benchmark_harness.circuits.{circuit_name}"
            module = importlib.import_module(module_name)
            # 特殊处理QFT电路名称
            if circuit_name.lower() == "qft":
                circuit_class = getattr(module, "QFTCircuit")
            else:
                circuit_class = getattr(module, f"{circuit_name.title()}Circuit")

            circuit_instance = circuit_class()
            circuits.append(circuit_instance)

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
) -> List[Any]:
    """运行基准测试"""
    all_results = []

    # 检查黄金标准是否可用
    if golden_standard_key not in simulators:
        raise ValueError(
            f"Golden standard simulator '{golden_standard_key}' not available"
        )

    golden_wrapper = simulators[golden_standard_key]

    # 主循环
    for circuit_instance in circuits:
        for n_qubits in qubit_ranges:
            print(f"\nRunning {circuit_instance.name} with {n_qubits} qubits...")

            # 阶段A: 生成参考态
            print(f"  Generating reference state using {golden_standard_key}...")
            circuit_for_golden = circuit_instance.build(
                platform=golden_wrapper.platform_name, n_qubits=n_qubits
            )

            try:
                golden_result = golden_wrapper.execute(circuit_for_golden, n_qubits)
                reference_state = golden_result.final_state
                print(f"  Reference state generated successfully")
            except Exception as e:
                print(f"  Error generating reference state: {e}")
                continue

            # 阶段B: 在所有模拟器上运行基准测试
            for runner_id, wrapper_instance in simulators.items():
                print(f"  Running on {runner_id}...")

                try:
                    circuit_for_current = circuit_instance.build(
                        platform=wrapper_instance.platform_name, n_qubits=n_qubits
                    )

                    # 如果是黄金标准，重用结果
                    if runner_id == golden_standard_key:
                        golden_result.state_fidelity = 1.0
                        result = golden_result
                    else:
                        result = wrapper_instance.execute(
                            circuit=circuit_for_current,
                            n_qubits=n_qubits,
                            reference_state=reference_state,
                        )

                    all_results.append(result)
                    print(
                        f"    Completed in {result.wall_time_sec:.4f}s, fidelity: {result.state_fidelity:.4f}"
                    )

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

    return all_results


def main():
    """主函数"""
    global args
    args = parse_arguments()

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Quantum Simulator Benchmark Runner")
    print(f"Output directory: {output_dir}")
    print(f"Verbose mode: {args.verbose}")

    # 创建模拟器实例
    print("\nInitializing simulators...")
    simulators = create_simulator_instances(args.simulators)

    if not simulators:
        print("Error: No simulators available. Exiting.")
        return 1

    print(f"Available simulators: {list(simulators.keys())}")

    # 创建电路实例
    print("\nInitializing circuits...")
    circuits = create_circuit_instances(args.circuits)

    if not circuits:
        print("Error: No circuits available. Exiting.")
        return 1

    print(f"Available circuits: {[c.name for c in circuits]}")

    # 运行基准测试
    print("\nRunning benchmarks...")
    try:
        results = run_benchmarks(
            circuits=circuits,
            qubit_ranges=args.qubits,
            simulators=simulators,
            golden_standard_key=args.golden_standard,
        )

        print(f"\nCompleted {len(results)} benchmark runs")

        # 保存结果并执行后处理
        if results:
            print(f"\nProcessing results...")
            try:
                # 分析结果并生成可视化图表
                analyze_results(results, output_dir)

                # 生成摘要报告

                data = []
                for result in results:
                    data.append(
                        {
                            "simulator": result.simulator,
                            "backend": result.backend,
                            "circuit_name": result.circuit_name,
                            "n_qubits": result.n_qubits,
                            "wall_time_sec": result.wall_time_sec,
                            "cpu_time_sec": result.cpu_time_sec,
                            "peak_memory_mb": result.peak_memory_mb,
                            "cpu_utilization_percent": result.cpu_utilization_percent,
                            "state_fidelity": result.state_fidelity,
                        }
                    )
                df = pd.DataFrame(data)
                df["runner_id"] = df["simulator"] + "-" + df["backend"]
                generate_summary_report(df, output_dir)

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
    sys.exit(main())
