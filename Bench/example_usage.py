#!/usr/bin/env python3
"""
VQE分层配置系统使用示例

该脚本展示了如何使用新的分层配置系统进行VQE基准测试，
包括快速开始、自定义配置和高级研究用例。

使用方法:
    python example_usage.py
"""

import sys
import os

# 添加当前目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vqe_config import (
    merge_configs, validate_config, get_quick_start_config, 
    get_performance_config, get_legacy_config
)
from vqe_bench import VQEBenchmarkRunner, VQEBenchmarkVisualizer

def example_1_quick_start():
    """示例1: 快速开始 - 适合新用户"""
    print("=" * 60)
    print("示例1: 快速开始 - 适合新用户")
    print("=" * 60)
    
    # 获取快速开始配置
    config = get_quick_start_config()
    
    print("配置参数:")
    print(f"  量子比特数: {config['n_qubits_range']}")
    print(f"  测试框架: {config['frameworks_to_test']}")
    print(f"  算法类型: {config['ansatz_type']}")
    print(f"  优化器: {config['optimizer']}")
    print(f"  运行次数: {config['n_runs']}")
    
    # 验证配置
    is_valid, errors = validate_config(config)
    if not is_valid:
        print(f"配置验证失败: {errors}")
        return False
    
    print("\n✅ 配置验证通过")
    
    # 转换为兼容格式并运行基准测试
    legacy_config = get_legacy_config()
    legacy_config.update({
        "n_qubits_range": config["n_qubits_range"],
        "frameworks_to_test": config["frameworks_to_test"],
        "n_runs": config["n_runs"]
    })
    
    print("\n开始运行基准测试...")
    try:
        runner = VQEBenchmarkRunner(legacy_config)
        results = runner.run_all_benchmarks()
        
        # 生成可视化
        visualizer = VQEBenchmarkVisualizer(results, legacy_config)
        visualizer.plot_dashboard()
        
        print("✅ 快速开始示例完成")
        return True
    except Exception as e:
        print(f"❌ 运行基准测试时出错: {e}")
        return False

def example_2_custom_core_config():
    """示例2: 自定义核心配置"""
    print("\n" + "=" * 60)
    print("示例2: 自定义核心配置")
    print("=" * 60)
    
    # 自定义核心配置
    custom_core = {
        "n_qubits_range": [4, 6, 8],
        "frameworks_to_test": ["Qiskit", "Qibo"],
        "ansatz_type": "QAOA",  # 使用QAOA算法
        "optimizer": "SPSA",    # 使用SPSA优化器
        "n_runs": 5,
        "experiment_name": "Custom_QAOA_SPSA_Test"
    }
    
    print("自定义核心配置:")
    for key, value in custom_core.items():
        print(f"  {key}: {value}")
    
    # 合并配置（使用默认高级配置）
    config = merge_configs(core_config=custom_core)
    
    # 验证配置
    is_valid, errors = validate_config(config)
    if not is_valid:
        print(f"配置验证失败: {errors}")
        return False
    
    print("\n✅ 配置验证通过")
    
    # 显示合并后的配置
    print("\n合并后的配置摘要:")
    print(f"  量子比特数: {config['n_qubits_range']}")
    print(f"  测试框架: {config['frameworks_to_test']}")
    print(f"  算法类型: {config['ansatz_type']}")
    print(f"  优化器: {config['optimizer']}")
    print(f"  运行次数: {config['n_runs']}")
    print(f"  物理模型: {config['problem']['model_type']}")
    print(f"  边界条件: {config['problem']['boundary_conditions']}")
    print(f"  Ansatz层数: {config['ansatz_details']['n_layers']}")
    
    print("\n✅ 自定义核心配置示例完成")
    return True

def example_3_advanced_research():
    """示例3: 高级研究配置"""
    print("\n" + "=" * 60)
    print("示例3: 高级研究配置")
    print("=" * 60)
    
    # 自定义高级配置
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
    
    print("自定义高级配置:")
    for section, params in custom_advanced.items():
        print(f"  {section}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    # 合并配置（使用默认核心配置）
    config = merge_configs(advanced_config=custom_advanced)
    
    # 验证配置
    is_valid, errors = validate_config(config)
    if not is_valid:
        print(f"配置验证失败: {errors}")
        return False
    
    print("\n✅ 配置验证通过")
    
    # 显示合并后的配置
    print("\n合并后的配置摘要:")
    print(f"  量子比特数: {config['n_qubits_range']}")
    print(f"  测试框架: {config['frameworks_to_test']}")
    print(f"  边界条件: {config['problem']['boundary_conditions']}")
    print(f"  耦合强度: {config['problem']['j_coupling']}")
    print(f"  场强: {config['problem']['h_field']}")
    print(f"  无序强度: {config['problem']['disorder_strength']}")
    print(f"  Ansatz层数: {config['ansatz_details']['n_layers']}")
    print(f"  纠缠样式: {config['ansatz_details']['entanglement_style']}")
    print(f"  最大评估次数: {config['optimizer_details']['max_evaluations']}")
    print(f"  精度阈值: {config['optimizer_details']['accuracy_threshold']}")
    print(f"  模拟模式: {config['backend_details']['simulation_mode']}")
    print(f"  采样次数: {config['backend_details']['n_shots']}")
    
    print("\n✅ 高级研究配置示例完成")
    return True

def example_4_performance_evaluation():
    """示例4: 性能评估配置"""
    print("\n" + "=" * 60)
    print("示例4: 性能评估配置")
    print("=" * 60)
    
    # 获取性能评估配置
    config = get_performance_config()
    
    print("性能评估配置:")
    print(f"  量子比特数: {config['n_qubits_range']}")
    print(f"  测试框架: {config['frameworks_to_test']}")
    print(f"  运行次数: {config['n_runs']}")
    print(f"  最大评估次数: {config['optimizer_details']['max_evaluations']}")
    print(f"  精度阈值: {config['optimizer_details']['accuracy_threshold']}")
    print(f"  内存限制: {config['system']['max_memory_mb']} MB")
    print(f"  时间限制: {config['system']['max_time_seconds']} 秒")
    
    # 验证配置
    is_valid, errors = validate_config(config)
    if not is_valid:
        print(f"配置验证失败: {errors}")
        return False
    
    print("\n✅ 配置验证通过")
    
    print("\n✅ 性能评估配置示例完成")
    return True

def example_5_integrate_with_existing_code():
    """示例5: 与现有代码集成"""
    print("\n" + "=" * 60)
    print("示例5: 与现有代码集成")
    print("=" * 60)
    
    print("展示如何将新配置系统集成到现有的vqe_bench.py中...")
    
    # 方法1: 直接替换DEFAULT_CONFIG
    print("\n方法1: 直接替换DEFAULT_CONFIG")
    print("""
    # 在vqe_bench.py中，将原来的DEFAULT_CONFIG替换为:
    from vqe_config import get_legacy_config
    DEFAULT_CONFIG = get_legacy_config()
    """)
    
    # 方法2: 使用新配置系统但保持兼容性
    print("\n方法2: 使用新配置系统但保持兼容性")
    print("""
    # 在主函数中，使用新配置系统:
    from vqe_config import merge_configs, get_legacy_config
    
    # 获取新配置
    new_config = merge_configs()
    
    # 转换为兼容格式
    config = get_legacy_config()
    config.update({
        "n_qubits_range": new_config["n_qubits_range"],
        "frameworks_to_test": new_config["frameworks_to_test"],
        "n_runs": new_config["n_runs"]
    })
    
    # 创建并运行基准测试
    runner = VQEBenchmarkRunner(config)
    results = runner.run_all_benchmarks()
    """)
    
    # 方法3: 完全使用新配置系统
    print("\n方法3: 完全使用新配置系统")
    print("""
    # 创建一个适配器类，将新配置格式转换为VQEBenchmarkRunner需要的格式
    
    class ConfigAdapter:
        @staticmethod
        def adapt(new_config):
            # 将新配置格式转换为VQEBenchmarkRunner需要的格式
            return {
                "n_qubits_range": new_config["n_qubits_range"],
                "j_coupling": new_config["problem"]["j_coupling"],
                "h_field": new_config["problem"]["h_field"],
                "n_layers": new_config["ansatz_details"]["n_layers"],
                "optimizer": new_config["optimizer"],
                "max_evaluations": new_config["optimizer_details"]["max_evaluations"],
                "accuracy_threshold": new_config["optimizer_details"]["accuracy_threshold"],
                "n_runs": new_config["n_runs"],
                "frameworks_to_test": new_config["frameworks_to_test"],
                "seed": new_config["system"]["seed"],
                "max_memory_mb": new_config["system"]["max_memory_mb"],
                "max_time_seconds": new_config["system"]["max_time_seconds"],
            }
    
    # 使用适配器
    from vqe_config import merge_configs
    new_config = merge_configs()
    adapted_config = ConfigAdapter.adapt(new_config)
    runner = VQEBenchmarkRunner(adapted_config)
    """)
    
    print("\n✅ 与现有代码集成示例完成")
    return True

def main():
    """主函数"""
    print("VQE分层配置系统使用示例")
    print("=" * 60)
    
    # 运行所有示例
    examples = [
        example_1_quick_start,
        example_2_custom_core_config,
        example_3_advanced_research,
        example_4_performance_evaluation,
        example_5_integrate_with_existing_code
    ]
    
    passed = 0
    total = len(examples)
    
    for example in examples:
        try:
            if example():
                passed += 1
        except Exception as e:
            print(f"❌ 示例执行出错: {e}")
    
    print("\n" + "=" * 60)
    print(f"示例执行结果: {passed}/{total} 成功")
    
    if passed == total:
        print("✅ 所有示例执行成功！")
        print("\n您现在可以开始使用分层配置系统进行VQE基准测试了。")
    else:
        print("❌ 部分示例执行失败！请检查配置系统。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)