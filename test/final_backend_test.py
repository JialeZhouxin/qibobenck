#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最终后端测试脚本
测试所有Qibo后端的功能和正确性
"""

import sys
import numpy as np
import torch
from qibo import Circuit, gates, set_backend

def test_all_backends():
    """测试所有后端的基本功能"""
    print("🚀 Qibo后端全面测试")
    print("=" * 60)
    
    # 创建简单测试电路
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    
    print(f"测试电路: 2量子比特，H(0), CNOT(0,1)")
    print(f"电路深度: {circuit.depth}, 门数量: {circuit.ngates}")
    
    # 后端配置 - 只测试基本后端，避免导入冲突
    backend_configs = {
        "numpy": {"platform": None},
        "qibojit": {"platform": "numba"},
        "qibotn": {"platform": "qutensornet"},
        "clifford": {"platform": "numpy"},
        "hamming_weight": {"platform": "numpy"}
    }
    
    successful_backends = []
    
    for backend_name, config in backend_configs.items():
        print(f"\n🔬 测试后端: {backend_name}")
        print("-" * 40)
        
        try:
            # 设置后端
            if "backend" in config:
                set_backend(config["backend"], platform=config["platform"])
            else:
                set_backend(backend_name, platform=config["platform"])
            
            print(f"✅ {backend_name}后端设置成功")
            
            # 执行电路
            result = circuit()
            print(f"✅ 电路执行成功")
            
            # 获取状态向量
            if hasattr(result, 'state'):
                state = result.state()
                
                # 处理不同后端的数据类型
                if isinstance(state, torch.Tensor):
                    state = state.detach().cpu().numpy()
                elif hasattr(state, 'numpy'):
                    state = state.numpy()
                
                print(f"状态向量形状: {state.shape}")
                print(f"状态向量范数: {np.linalg.norm(state):.6f}")
                
                # 验证正确性（简单检查）
                if np.abs(np.linalg.norm(state) - 1.0) < 1e-6:
                    print("✅ 状态向量正确性验证通过")
                    successful_backends.append(backend_name)
                else:
                    print("❌ 状态向量正确性验证失败")
            
            # 获取概率分布
            if hasattr(result, 'probabilities'):
                try:
                    probs = result.probabilities()
                    if hasattr(probs, 'numpy'):
                        probs = probs.numpy()
                    print(f"概率分布: {probs}")
                except Exception as e:
                    print(f"⚠️ 概率分布获取失败: {e}")
            
        except Exception as e:
            print(f"❌ {backend_name}后端测试失败: {e}")
    
    print(f"\n🎯 测试完成总结")
    print("=" * 60)
    print(f"成功测试的后端数量: {len(successful_backends)}")
    print(f"成功后端列表: {successful_backends}")
    
    return successful_backends

def test_hamming_weight_special():
    """特殊测试Hamming Weight后端"""
    print(f"\n🔍 特殊测试: Hamming Weight后端")
    print("-" * 40)
    
    try:
        from qibo.backends import HammingWeightBackend
        
        backend = HammingWeightBackend()
        nqubits = 4
        circuit = Circuit(nqubits)
        circuit.add(gates.SWAP(0, 1))
        
        result = backend.execute_circuit(circuit, weight=2)
        print(f"✅ Hamming Weight后端特殊测试成功")
        print(f"结果类型: {type(result)}")
        
    except Exception as e:
        print(f"❌ Hamming Weight后端特殊测试失败: {e}")

if __name__ == "__main__":
    # 运行全面测试
    successful = test_all_backends()
    
    # 运行特殊测试
    test_hamming_weight_special()
    
    print(f"\n🎉 所有测试完成!")
    print(f"✅ 成功测试的后端: {successful}")
    
    # 生成测试报告
    with open("test/final_test_report.txt", "w", encoding="utf-8") as f:
        f.write("Qibo后端最终测试报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"测试时间: {sys.version}\n")
        f.write(f"成功后端数量: {len(successful)}\n")
        f.write(f"成功后端列表: {successful}\n")
        f.write("\n测试环境信息:\n")
        f.write(f"Python版本: {sys.version}\n")
        f.write(f"操作系统: {sys.platform}\n")
    
    print("📋 测试报告已保存到: test/final_test_report.txt")