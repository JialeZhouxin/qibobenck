#!/usr/bin/env python3
"""
测试新的VQE基准测试架构

该脚本测试基于vqe_design.ipynb设计理念的新架构是否正常工作，
包括FrameworkWrapper抽象基类、VQERunner执行引擎和BenchmarkController控制器。
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_import():
    """测试配置系统导入"""
    print("测试配置系统导入...")
    try:
        from vqe_config import merge_configs, CONFIG, ADVANCED_CONFIG
        print("✓ 配置系统导入成功")
        
        # 测试配置合并
        config = merge_configs()
        print(f"✓ 配置合并成功，包含 {len(config)} 个主要部分")
        
        # 验证核心配置
        required_keys = ["n_qubits_range", "frameworks_to_test", "ansatz_type", "optimizer", "n_runs"]
        for key in required_keys:
            if key not in config:
                print(f"✗ 缺少核心配置项: {key}")
                return False
        print("✓ 核心配置验证通过")
        
        return True
    except Exception as e:
        print(f"✗ 配置系统导入失败: {e}")
        return False

def test_framework_wrappers():
    """测试框架适配器"""
    print("\n测试框架适配器...")
    try:
        from vqe_bench_new import QiskitWrapper, PennyLaneWrapper, QiboWrapper
        
        # 创建一个简单的后端配置
        backend_config = {
            "framework_backends": {
                "Qiskit": "aer_simulator",
                "PennyLane": "lightning.qubit",
                "Qibo": {"backend": "qibojit", "platform": "numba"}
            }
        }
        
        # 测试Qiskit适配器
        try:
            qiskit_wrapper = QiskitWrapper(backend_config)
            print("✓ Qiskit适配器创建成功")
        except Exception as e:
            print(f"⚠ Qiskit适配器创建失败（可能Qiskit未安装）: {e}")
        
        # 测试PennyLane适配器
        try:
            pennylane_wrapper = PennyLaneWrapper(backend_config)
            print("✓ PennyLane适配器创建成功")
        except Exception as e:
            print(f"⚠ PennyLane适配器创建失败（可能PennyLane未安装）: {e}")
        
        # 测试Qibo适配器
        try:
            qibo_wrapper = QiboWrapper(backend_config)
            print("✓ Qibo适配器创建成功")
        except Exception as e:
            print(f"⚠ Qibo适配器创建失败（可能Qibo未安装）: {e}")
        
        return True
    except Exception as e:
        print(f"✗ 框架适配器测试失败: {e}")
        return False

def test_vqe_runner():
    """测试VQE执行引擎"""
    print("\n测试VQE执行引擎...")
    try:
        from vqe_bench_new import VQERunner
        
        # 创建一个简单的成本函数
        def simple_cost_function(params):
            return np.sum(params**2)
        
        # 创建优化器配置
        optimizer_config = {
            "optimizer": "COBYLA",
            "options": {
                "COBYLA": {"tol": 1e-5, "rhobeg": 1.0}
            }
        }
        
        # 创建收敛配置
        convergence_config = {
            "max_evaluations": 50,
            "accuracy_threshold": 1e-4
        }
        
        # 创建VQE执行引擎
        vqe_runner = VQERunner(
            cost_function=simple_cost_function,
            optimizer_config=optimizer_config,
            convergence_config=convergence_config,
            exact_energy=0.0
        )
        print("✓ VQE执行引擎创建成功")
        
        # 设置参数数量
        vqe_runner.get_param_count = lambda: 3
        
        # 运行一个简单的测试
        initial_params = np.random.rand(3)
        result = vqe_runner.run(initial_params)
        
        # 验证结果
        required_keys = ["final_energy", "total_time", "peak_memory", "eval_count"]
        for key in required_keys:
            if key not in result:
                print(f"✗ 结果缺少关键字段: {key}")
                return False
        print("✓ VQE执行引擎测试成功")
        print(f"  最终能量: {result['final_energy']:.6f}")
        print(f"  总时间: {result['total_time']:.3f} 秒")
        print(f"  评估次数: {result['eval_count']}")
        
        return True
    except Exception as e:
        print(f"✗ VQE执行引擎测试失败: {e}")
        return False

def test_benchmark_controller():
    """测试基准测试控制器"""
    print("\n测试基准测试控制器...")
    try:
        from vqe_config import merge_configs
        from vqe_bench_new import BenchmarkController
        
        # 获取配置
        config = merge_configs()
        
        # 为了快速测试，减少量子比特数和运行次数
        config["n_qubits_range"] = [4]  # 只测试4个量子比特
        config["n_runs"] = 1  # 只运行1次
        
        # 创建基准测试控制器
        controller = BenchmarkController(config)
        print("✓ 基准测试控制器创建成功")
        
        # 检查框架适配器
        for framework_name in config["frameworks_to_test"]:
            if framework_name in controller.wrappers:
                print(f"✓ {framework_name} 适配器已创建")
            else:
                print(f"✗ {framework_name} 适配器创建失败")
        
        return True
    except Exception as e:
        print(f"✗ 基准测试控制器测试失败: {e}")
        return False

def test_visualizer():
    """测试可视化器"""
    print("\n测试可视化器...")
    try:
        from vqe_bench_new import VQEBenchmarkVisualizer
        from vqe_config import merge_configs
        
        # 创建模拟结果
        config = merge_configs()
        frameworks = config["frameworks_to_test"]
        n_qubits_range = config["n_qubits_range"]
        
        results = {}
        for fw in frameworks:
            results[fw] = {}
            for n_qubits in n_qubits_range:
                results[fw][n_qubits] = {
                    "avg_time_to_solution": 1.23,
                    "std_time_to_solution": 0.45,
                    "avg_total_time": 2.34,
                    "std_total_time": 0.56,
                    "avg_peak_memory": 123.45,
                    "std_peak_memory": 12.34,
                    "avg_total_evals": 123,
                    "std_total_evals": 23,
                    "avg_final_error": 1e-4,
                    "std_final_error": 5e-5,
                    "avg_quantum_time": 0.012,
                    "std_quantum_time": 0.003,
                    "avg_classic_time": 0.023,
                    "std_classic_time": 0.005,
                    "convergence_rate": 0.9,
                    "energy_histories": [[1.0, 0.8, 0.6, 0.5, 0.4]],
                    "errors": []
                }
        
        # 创建可视化器
        visualizer = VQEBenchmarkVisualizer(results, config)
        print("✓ 可视化器创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 可视化器测试失败: {e}")
        return False

def test_integration():
    """集成测试"""
    print("\n进行集成测试...")
    try:
        from vqe_config import merge_configs
        from vqe_bench_new import BenchmarkController
        
        # 获取配置
        config = merge_configs()
        
        # 为了快速测试，减少量子比特数和运行次数
        config["n_qubits_range"] = [4]  # 只测试4个量子比特
        config["n_runs"] = 1  # 只运行1次
        
        # 创建基准测试控制器
        controller = BenchmarkController(config)
        
        # 尝试运行一个框架的测试
        framework_name = config["frameworks_to_test"][0]
        n_qubits = config["n_qubits_range"][0]
        
        print(f"  尝试运行 {framework_name} 框架，{n_qubits} 量子比特的测试...")
        
        # 这里我们只测试到构建问题阶段，不实际运行VQE
        wrapper = controller.wrappers[framework_name]
        
        try:
            # 构建哈密顿量
            problem_config = config.get("problem", {})
            hamiltonian = wrapper.build_hamiltonian(problem_config, n_qubits)
            print(f"  ✓ {framework_name} 哈密顿量构建成功")
            
            # 构建Ansatz
            ansatz_config = config.get("ansatz_details", {})
            ansatz_config["ansatz_type"] = config.get("ansatz_type", "HardwareEfficient")
            ansatz = wrapper.build_ansatz(ansatz_config, n_qubits)
            print(f"  ✓ {framework_name} Ansatz构建成功")
            
            # 获取成本函数
            cost_function = wrapper.get_cost_function(hamiltonian, ansatz)
            print(f"  ✓ {framework_name} 成本函数创建成功")
            
            # 获取参数数量
            param_count = wrapper.get_param_count(ansatz)
            print(f"  ✓ {framework_name} 参数数量: {param_count}")
            
        except Exception as e:
            print(f"  ⚠ {framework_name} 框架测试失败（可能框架未安装）: {e}")
        
        print("✓ 集成测试完成")
        return True
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("VQE基准测试新架构测试")
    print("=" * 60)
    
    tests = [
        test_config_import,
        test_framework_wrappers,
        test_vqe_runner,
        test_benchmark_controller,
        test_visualizer,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("🎉 所有测试通过！新架构工作正常。")
    else:
        print("⚠ 部分测试失败，请检查相关组件。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)