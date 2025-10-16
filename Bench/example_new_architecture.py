#!/usr/bin/env python3
"""
VQE基准测试新架构使用示例

该脚本展示了如何使用基于vqe_design.ipynb设计理念的新架构进行VQE基准测试。
新架构采用了面向对象的设计，包括FrameworkWrapper抽象基类、VQERunner执行引擎和BenchmarkController控制器。
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_start_example():
    """快速开始示例 - 使用默认配置"""
    print("=" * 60)
    print("快速开始示例 - 使用默认配置")
    print("=" * 60)
    
    from vqe_config import get_quick_start_config
    from vqe_bench_new import BenchmarkController
    
    # 获取快速开始配置
    config = get_quick_start_config()
    print(f"配置: {config}")
    
    # 创建基准测试控制器
    controller = BenchmarkController(config)
    
    # 运行基准测试
    results = controller.run_all_benchmarks()
    
    # 打印结果摘要
    print("\n基准测试结果摘要:")
    for framework in config["frameworks_to_test"]:
        print(f"\n{framework} 框架:")
        for n_qubits in config["n_qubits_range"]:
            if framework in results and n_qubits in results[framework]:
                data = results[framework][n_qubits]
                print(f"  {n_qubits} 量子比特:")
                print(f"    收敛率: {data['convergence_rate']:.1%}")
                if data['avg_time_to_solution'] is not None:
                    print(f"    求解时间: {data['avg_time_to_solution']:.3f} ± {data['std_time_to_solution']:.3f} 秒")
                print(f"    内存使用: {data['avg_peak_memory']:.1f} MB")
    
    return results

def custom_config_example():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("自定义配置示例")
    print("=" * 60)
    
    from vqe_config import merge_configs, CONFIG, ADVANCED_CONFIG
    from vqe_bench_new import BenchmarkController
    
    # 创建自定义配置
    custom_config = {
        "n_qubits_range": [4, 6],  # 只测试4和6个量子比特
        "frameworks_to_test": ["Qiskit"],  # 只测试Qiskit
        "ansatz_type": "QAOA",  # 使用QAOA ansatz
        "optimizer": "SPSA",  # 使用SPSA优化器
        "n_runs": 2,  # 只运行2次
        "experiment_name": "Custom_QAOA_Test"
    }
    
    # 合并配置
    full_config = merge_configs(custom_config)
    print(f"自定义配置: {custom_config}")
    
    # 创建基准测试控制器
    controller = BenchmarkController(full_config)
    
    # 运行基准测试
    results = controller.run_all_benchmarks()
    
    # 打印结果摘要
    print("\n自定义基准测试结果摘要:")
    for framework in custom_config["frameworks_to_test"]:
        print(f"\n{framework} 框架:")
        for n_qubits in custom_config["n_qubits_range"]:
            if framework in results and n_qubits in results[framework]:
                data = results[framework][n_qubits]
                print(f"  {n_qubits} 量子比特:")
                print(f"    收敛率: {data['convergence_rate']:.1%}")
                print(f"    总评估次数: {data['avg_total_evals']:.1f}")
    
    return results

def visualization_example():
    """可视化示例"""
    print("\n" + "=" * 60)
    print("可视化示例")
    print("=" * 60)
    
    from vqe_config import get_quick_start_config
    from vqe_bench_new import BenchmarkController, VQEBenchmarkVisualizer
    
    # 获取配置并运行基准测试
    config = get_quick_start_config()
    config["n_qubits_range"] = [4, 6]  # 减少量子比特数以加快演示
    config["n_runs"] = 2  # 减少运行次数
    
    controller = BenchmarkController(config)
    results = controller.run_all_benchmarks()
    
    # 创建可视化器并生成仪表盘
    visualizer = VQEBenchmarkVisualizer(results, config)
    
    # 生成并保存仪表盘
    output_dir = config.get("system", {}).get("output_dir", "./results/")
    visualizer.plot_dashboard(output_dir)
    
    print(f"可视化仪表盘已保存到: {output_dir}")
    
    return results

def framework_comparison_example():
    """框架比较示例"""
    print("\n" + "=" * 60)
    print("框架比较示例")
    print("=" * 60)
    
    from vqe_config import merge_configs
    from vqe_bench_new import BenchmarkController
    
    # 创建比较配置
    comparison_config = {
        "n_qubits_range": [4, 6],
        "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],  # 测试所有框架
        "ansatz_type": "HardwareEfficient",
        "optimizer": "COBYLA",
        "n_runs": 2,
        "experiment_name": "Framework_Comparison"
    }
    
    # 合并配置
    full_config = merge_configs(comparison_config)
    
    # 创建基准测试控制器
    controller = BenchmarkController(full_config)
    
    # 运行基准测试
    results = controller.run_all_benchmarks()
    
    # 比较框架性能
    print("\n框架性能比较:")
    for n_qubits in comparison_config["n_qubits_range"]:
        print(f"\n{n_qubits} 量子比特:")
        for framework in comparison_config["frameworks_to_test"]:
            if framework in results and n_qubits in results[framework]:
                data = results[framework][n_qubits]
                if data['avg_time_to_solution'] is not None:
                    print(f"  {framework}: 求解时间 {data['avg_time_to_solution']:.3f}s, 内存 {data['avg_peak_memory']:.1f}MB, 收敛率 {data['convergence_rate']:.1%}")
                else:
                    print(f"  {framework}: 未收敛")
    
    return results

def main():
    """主函数"""
    print("VQE基准测试新架构使用示例")
    print("该示例展示了如何使用新架构进行各种类型的基准测试")
    
    # 运行各种示例
    try:
        # 1. 快速开始示例
        quick_start_example()
        
        # 2. 自定义配置示例
        custom_config_example()
        
        # 3. 可视化示例
        visualization_example()
        
        # 4. 框架比较示例
        framework_comparison_example()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()