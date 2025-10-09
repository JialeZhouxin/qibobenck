#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QASMBench通用基准测试工具使用示例
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qasmbench_runner import list_available_circuits, run_benchmark_for_circuit

def example_1_list_circuits():
    """示例1: 列出所有可用电路"""
    print("示例1: 列出所有可用QASMBench电路")
    print("="*50)
    circuits = list_available_circuits()
    return circuits

def example_2_test_specific_circuit():
    """示例2: 测试特定电路"""
    print("\n示例2: 测试特定QASMBench电路")
    print("="*50)
    
    # 测试一个中等规模的电路
    circuit_path = "QASMBench/medium/qft_n18/qft_n18.qasm"
    
    if os.path.exists(circuit_path):
        results = run_benchmark_for_circuit(circuit_path)
        return results
    else:
        print(f"电路文件不存在: {circuit_path}")
        print("请确保QASMBench目录结构正确")
        return None

def example_3_test_by_name():
    """示例3: 通过电路名称测试"""
    print("\n示例3: 通过电路名称测试")
    print("="*50)
    
    # 导入必要的函数
    from qasmbench_runner import QASMBenchConfig, QASMBenchRunner
    
    config = QASMBenchConfig()
    runner = QASMBenchRunner(config)
    
    # 发现所有电路
    circuits = runner.discover_qasm_circuits()
    
    # 选择一个电路进行测试
    if "medium/qft_n18" in circuits:
        circuit_info = circuits["medium/qft_n18"]
        circuit_path = circuit_info['path']
        
        print(f"测试电路: medium/qft_n18")
        print(f"电路文件: {circuit_path}")
        
        results = runner.run_benchmark_for_circuit("medium/qft_n18", circuit_path)
        runner.generate_reports(results, "medium_qft_n18")
        
        return results
    else:
        print("电路 medium/qft_n18 不存在")
        return None

def example_4_custom_config():
    """示例4: 自定义配置测试"""
    print("\n示例4: 自定义配置测试")
    print("="*50)
    
    from qasmbench_runner import QASMBenchConfig, QASMBenchRunner
    
    # 自定义配置
    config = QASMBenchConfig()
    config.num_runs = 3  # 减少运行次数以加快测试
    config.warmup_runs = 1
    config.output_formats = ['csv', 'markdown']  # 只生成CSV和Markdown报告
    
    runner = QASMBenchRunner(config)
    
    # 测试一个小规模电路
    circuit_path = "QASMBench/small/qft_n4/qft_n4.qasm"
    
    if os.path.exists(circuit_path):
        results = runner.run_benchmark_for_circuit("small_qft_n4", circuit_path)
        runner.generate_reports(results, "small_qft_n4")
        return results
    else:
        print(f"电路文件不存在: {circuit_path}")
        return None

def main():
    """主函数 - 运行所有示例"""
    print("QASMBench通用基准测试工具使用示例")
    print("="*60)
    
    # 示例1: 列出电路
    example_1_list_circuits()
    
    # 示例2: 测试特定电路
    # example_2_test_specific_circuit()
    
    # 示例3: 通过名称测试
    # example_3_test_by_name()
    
    # 示例4: 自定义配置测试
    # example_4_custom_config()
    
    print("\n" + "="*60)
    print("使用说明:")
    print("1. 取消注释相应的示例函数来运行测试")
    print("2. 确保QASMBench目录结构正确")
    print("3. 根据需要调整配置参数")
    print("\n直接使用命令行:")
    print("  python qasmbench_runner.py --list")
    print("  python qasmbench_runner.py --circuit-name medium/qft_n18")

if __name__ == "__main__":
    main()