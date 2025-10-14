#!/usr/bin/env python3
"""
测试改进后的qibo_profiler功能
"""

import sys
import os
import numpy as np
from qibo.models import Circuit
from qibo import gates

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qibo_profiler_improved import (
        ProfilerConfig, 
        ProfilerPipeline, 
        profile_circuit,
        InputValidator,
        ThreadSafeCache,
        EnvironmentCache
    )
    print("✅ 成功导入改进后的模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        # 创建一个简单的量子电路
        circuit = Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        circuit.add(gates.M(0, 1))
        
        print(f"✅ 创建了 {circuit.nqubits} 量子比特的电路，深度: {circuit.depth}")
        
        # 测试配置验证
        config = ProfilerConfig(n_runs=2, mode='basic', calculate_fidelity=True)
        print("✅ 配置验证通过")
        
        # 测试输入验证
        InputValidator.validate_circuit(circuit)
        print("✅ 电路验证通过")
        
        # 测试缓存
        cache = ThreadSafeCache()
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value", "缓存测试失败"
        print("✅ 线程安全缓存测试通过")
        
        # 测试环境缓存
        env_info = EnvironmentCache.get_environment_info()
        assert "qibo_backend" in env_info, "环境信息缺少必要字段"
        print("✅ 环境缓存测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_profiler_pipeline():
    """测试分析器管道"""
    print("\n🧪 测试分析器管道...")
    
    try:
        # 创建测试电路
        circuit = Circuit(3)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        circuit.add(gates.CNOT(1, 2))
        circuit.add(gates.M(0, 1, 2))
        
        # 创建配置
        config = ProfilerConfig(
            n_runs=2,
            mode='basic',
            calculate_fidelity=False,  # 简化测试
            timeout_seconds=60.0
        )
        
        # 创建管道
        pipeline = ProfilerPipeline()
        
        # 执行分析
        print("🔄 执行性能分析...")
        report = pipeline.execute(circuit, config)
        
        # 验证报告结构
        required_sections = ["metadata", "inputs", "results"]
        for section in required_sections:
            assert section in report, f"报告缺少 {section} 部分"
        
        assert report["metadata"]["profiler_version"] == "1.0", "版本信息不正确"
        assert report["inputs"]["circuit_properties"]["n_qubits"] == 3, "量子比特数不正确"
        assert "summary" in report["results"], "缺少结果摘要"
        
        print("✅ 分析器管道测试通过")
        print(f"   - 运行时间: {report['results']['summary']['runtime_avg']['value']:.4f} 秒")
        print(f"   - 内存使用: {report['results']['summary']['memory_usage_avg']['value']:.2f} MiB")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析器管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理...")
    
    try:
        # 测试无效配置
        try:
            invalid_config = ProfilerConfig(n_runs=-1)
            print("❌ 应该抛出配置验证错误")
            return False
        except ValueError:
            print("✅ 无效配置正确被拒绝")
        
        # 测试无效电路
        try:
            InputValidator.validate_circuit(None)  # type: ignore
            print("❌ 应该抛出电路验证错误")
            return False
        except Exception:
            print("✅ 无效电路正确被拒绝")
        
        # 测试无效模式
        try:
            invalid_config = ProfilerConfig(mode='invalid_mode')
            print("❌ 应该抛出模式验证错误")
            return False
        except ValueError:
            print("✅ 无效模式正确被拒绝")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def test_api_compatibility():
    """测试API兼容性"""
    print("\n🧪 测试API兼容性...")
    
    try:
        # 创建测试电路
        circuit = Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        
        # 使用新的API
        report_new = profile_circuit(
            circuit=circuit,
            n_runs=1,
            mode='basic',
            calculate_fidelity=False
        )
        
        # 验证报告结构
        assert "metadata" in report_new, "新API报告结构不正确"
        assert "inputs" in report_new, "新API报告结构不正确"
        assert "results" in report_new, "新API报告结构不正确"
        
        print("✅ API兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ API兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("🚀 开始测试改进后的qibo_profiler")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_profiler_pipeline,
        test_error_handling,
        test_api_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！改进成功！")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        return 1

if __name__ == "__main__":
    sys.exit(main())
