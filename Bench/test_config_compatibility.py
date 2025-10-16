#!/usr/bin/env python3
"""
测试vqe_config.py与vqe_bench.py的兼容性

该脚本验证新的分层配置系统是否与现有的vqe_bench.py兼容，
确保可以通过简单的导入替换现有的配置系统。

使用方法:
    python test_config_compatibility.py
"""

import sys
import os

# 添加当前目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vqe_config import merge_configs, validate_config, get_legacy_config
from vqe_bench import VQEBenchmarkRunner

def test_legacy_compatibility():
    """测试与vqe_bench.py的兼容性"""
    print("测试与vqe_bench.py的兼容性...")
    
    # 获取兼容格式的配置
    legacy_config = get_legacy_config()
    
    # 验证配置格式
    required_keys = [
        "n_qubits_range", "j_coupling", "h_field", "n_layers", 
        "optimizer", "max_evaluations", "accuracy_threshold",
        "n_runs", "frameworks_to_test", "seed", 
        "max_memory_mb", "max_time_seconds"
    ]
    
    missing_keys = [key for key in required_keys if key not in legacy_config]
    if missing_keys:
        print(f"❌ 兼容性测试失败：缺少必需的键: {missing_keys}")
        return False
    
    print("✅ 兼容性测试通过：所有必需的键都存在")
    
    # 尝试创建VQEBenchmarkRunner实例
    try:
        runner = VQEBenchmarkRunner(legacy_config)
        print("✅ 兼容性测试通过：VQEBenchmarkRunner可以成功创建")
        return True
    except Exception as e:
        print(f"❌ 兼容性测试失败：创建VQEBenchmarkRunner时出错: {e}")
        return False

def test_new_config_format():
    """测试新的分层配置格式"""
    print("\n测试新的分层配置格式...")
    
    # 获取合并后的配置
    new_config = merge_configs()
    
    # 验证配置
    is_valid, errors = validate_config(new_config)
    if not is_valid:
        print(f"❌ 新配置格式测试失败：配置验证错误: {errors}")
        return False
    
    print("✅ 新配置格式测试通过：配置验证成功")
    
    # 检查核心参数
    core_params = ["n_qubits_range", "frameworks_to_test", "ansatz_type", "optimizer", "n_runs"]
    missing_core = [param for param in core_params if param not in new_config]
    if missing_core:
        print(f"❌ 新配置格式测试失败：缺少核心参数: {missing_core}")
        return False
    
    print("✅ 新配置格式测试通过：所有核心参数都存在")
    
    # 检查高级配置部分
    advanced_sections = ["problem", "ansatz_details", "optimizer_details", "backend_details", "system"]
    missing_sections = [section for section in advanced_sections if section not in new_config]
    if missing_sections:
        print(f"❌ 新配置格式测试失败：缺少高级配置部分: {missing_sections}")
        return False
    
    print("✅ 新配置格式测试通过：所有高级配置部分都存在")
    return True

def test_quick_start_config():
    """测试快速开始配置"""
    print("\n测试快速开始配置...")
    
    from vqe_config import get_quick_start_config
    
    # 获取快速开始配置
    quick_config = get_quick_start_config()
    
    # 验证配置
    is_valid, errors = validate_config(quick_config)
    if not is_valid:
        print(f"❌ 快速开始配置测试失败：配置验证错误: {errors}")
        return False
    
    print("✅ 快速开始配置测试通过：配置验证成功")
    
    # 尝试创建VQEBenchmarkRunner实例
    try:
        # 使用兼容格式转换
        legacy_config = get_legacy_config()
        # 更新为快速开始配置的核心参数
        legacy_config.update({
            "n_qubits_range": quick_config["n_qubits_range"],
            "frameworks_to_test": quick_config["frameworks_to_test"],
            "n_runs": quick_config["n_runs"]
        })
        
        runner = VQEBenchmarkRunner(legacy_config)
        print("✅ 快速开始配置测试通过：VQEBenchmarkRunner可以成功创建")
        return True
    except Exception as e:
        print(f"❌ 快速开始配置测试失败：创建VQEBenchmarkRunner时出错: {e}")
        return False

def test_config_customization():
    """测试配置自定义功能"""
    print("\n测试配置自定义功能...")
    
    # 自定义核心配置
    custom_core = {
        "n_qubits_range": [4, 6],
        "frameworks_to_test": ["Qiskit"],
        "ansatz_type": "QAOA",
        "optimizer": "SPSA",
        "n_runs": 5
    }
    
    # 自定义高级配置
    custom_advanced = {
        "problem": {
            "j_coupling": 0.5,
            "h_field": 1.5
        },
        "optimizer_details": {
            "max_evaluations": 1000
        }
    }
    
    # 合并配置
    custom_config = merge_configs(core_config=custom_core, advanced_config=custom_advanced)
    
    # 验证配置
    is_valid, errors = validate_config(custom_config)
    if not is_valid:
        print(f"❌ 配置自定义测试失败：配置验证错误: {errors}")
        return False
    
    print("✅ 配置自定义测试通过：配置验证成功")
    
    # 检查自定义参数是否正确应用
    if custom_config["n_qubits_range"] != [4, 6]:
        print(f"❌ 配置自定义测试失败：n_qubits_range 未正确应用")
        return False
    
    if custom_config["frameworks_to_test"] != ["Qiskit"]:
        print(f"❌ 配置自定义测试失败：frameworks_to_test 未正确应用")
        return False
    
    if custom_config["ansatz_type"] != "QAOA":
        print(f"❌ 配置自定义测试失败：ansatz_type 未正确应用")
        return False
    
    if custom_config["optimizer"] != "SPSA":
        print(f"❌ 配置自定义测试失败：optimizer 未正确应用")
        return False
    
    if custom_config["n_runs"] != 5:
        print(f"❌ 配置自定义测试失败：n_runs 未正确应用")
        return False
    
    if custom_config["problem"]["j_coupling"] != 0.5:
        print(f"❌ 配置自定义测试失败：problem.j_coupling 未正确应用")
        return False
    
    if custom_config["problem"]["h_field"] != 1.5:
        print(f"❌ 配置自定义测试失败：problem.h_field 未正确应用")
        return False
    
    if custom_config["optimizer_details"]["max_evaluations"] != 1000:
        print(f"❌ 配置自定义测试失败：optimizer_details.max_evaluations 未正确应用")
        return False
    
    print("✅ 配置自定义测试通过：所有自定义参数都正确应用")
    return True

def main():
    """主测试函数"""
    print("VQE配置系统兼容性测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_legacy_compatibility,
        test_new_config_format,
        test_quick_start_config,
        test_config_customization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 所有测试通过！配置系统与vqe_bench.py完全兼容。")
        return True
    else:
        print("❌ 部分测试失败！请检查配置系统。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)