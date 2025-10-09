#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
严格正确性验证测试
使用状态向量对比方法验证各后端的计算结果一致性
"""

import numpy as np
from qibo import Circuit, gates, set_backend

def compute_reference_state(circuit_qasm):
    """使用numpy后端计算参考状态向量"""
    set_backend("numpy")
    circuit = Circuit.from_qasm(circuit_qasm)
    result = circuit()
    return result.state()

def validate_backend_accuracy(backend_name, platform_name=None, circuit_qasm=None):
    """验证特定后端的计算准确性"""
    print(f"\n🔬 验证 {backend_name} 后端准确性")
    
    if circuit_qasm is None:
        # 创建测试电路
        circuit = Circuit(4)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        circuit.add(gates.RY(2, theta=0.5))
        circuit.add(gates.CZ(1, 3))
        circuit_qasm = circuit.to_qasm()
    
    # 计算参考状态（numpy后端）
    reference_state = compute_reference_state(circuit_qasm)
    
    try:
        # 设置测试后端
        if platform_name:
            set_backend(backend_name, platform=platform_name)
        else:
            set_backend(backend_name)
        
        # 使用测试后端计算
        test_circuit = Circuit.from_qasm(circuit_qasm)
        test_result = test_circuit()
        test_state = test_result.state()
        
        # 计算状态向量差异
        state_diff = np.linalg.norm(reference_state - test_state)
        
        # 计算概率分布差异
        ref_prob = np.abs(reference_state)**2
        test_prob = np.abs(test_state)**2
        prob_diff = np.linalg.norm(ref_prob - test_prob)
        
        # 检查状态向量范数（应该接近1）
        test_norm = np.linalg.norm(test_state)
        norm_error = abs(test_norm - 1.0)
        
        print(f"✅ {backend_name} 验证结果:")
        print(f"   状态向量差异: {state_diff:.2e}")
        print(f"   概率分布差异: {prob_diff:.2e}")
        print(f"   状态向量范数误差: {norm_error:.2e}")
        
        # 判断准确性
        if state_diff < 1e-10 and prob_diff < 1e-10 and norm_error < 1e-10:
            return "高精度"
        elif state_diff < 1e-6 and prob_diff < 1e-6:
            return "可接受精度"
        else:
            return f"低精度 (差异较大)"
            
    except Exception as e:
        print(f"❌ {backend_name} 验证失败: {e}")
        return "验证失败"

def test_all_backends():
    """测试所有后端的准确性"""
    print("🚀 开始严格正确性验证测试")
    print("=" * 60)
    
    # 后端配置
    backends = [
        {"name": "numpy", "platform": None, "desc": "基准后端"},
        {"name": "qibojit", "platform": "numba", "desc": "加速后端"},
        {"name": "qibotn", "platform": "qutensornet", "desc": "张量网络后端"},
        {"name": "qiboml", "platform": "jax", "desc": "JAX后端"},
        {"name": "qiboml", "platform": "pytorch", "desc": "PyTorch后端"},
        {"name": "qiboml", "platform": "tensorflow", "desc": "TensorFlow后端"},
    ]
    
    results = {}
    
    # 创建测试电路
    test_circuit = Circuit(6)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.CNOT(0, 1))
    test_circuit.add(gates.RY(2, theta=0.3))
    test_circuit.add(gates.CZ(1, 3))
    test_circuit.add(gates.SWAP(4, 5))
    test_circuit.add(gates.H(5))
    circuit_qasm = test_circuit.to_qasm()
    
    print(f"测试电路: 6量子比特，包含H、CNOT、RY、CZ、SWAP门")
    print(f"电路深度: {test_circuit.depth}, 门数量: {test_circuit.ngates}")
    
    # 测试每个后端
    for backend in backends:
        backend_key = f"{backend['name']} ({backend['platform']})" if backend['platform'] else backend['name']
        accuracy = validate_backend_accuracy(
            backend['name'], 
            backend['platform'],
            circuit_qasm
        )
        results[backend_key] = {
            "accuracy": accuracy,
            "description": backend['desc']
        }
    
    # 输出总结
    print("\n📊 正确性验证总结")
    print("=" * 60)
    for backend_key, result in results.items():
        print(f"{backend_key:25} | {result['accuracy']:15} | {result['description']}")

def test_qft_circuit_validation():
    """专门测试QFT电路的正确性"""
    print("\n🎯 QFT电路专门验证")
    print("=" * 50)
    
    # 加载QFT电路
    qasm_file = "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"
    
    try:
        with open(qasm_file, "r") as f:
            qft_qasm = f.read()
        
        # 清理QASM代码
        lines = qft_qasm.split('\n')
        filtered_lines = [line for line in lines if 'barrier' not in line and line.strip()]
        clean_qasm = '\n'.join(filtered_lines)
        
        print("✅ 成功加载QFT电路")
        
        # 测试关键后端
        key_backends = [
            ("numpy", None),
            ("qibojit", "numba"),
            ("qibotn", "qutensornet")
        ]
        
        for backend_name, platform in key_backends:
            accuracy = validate_backend_accuracy(backend_name, platform, clean_qasm)
            print(f"QFT电路 - {backend_name}: {accuracy}")
            
    except Exception as e:
        print(f"❌ QFT电路验证失败: {e}")

if __name__ == "__main__":
    # 运行所有测试
    test_all_backends()
    test_qft_circuit_validation()
    
    print("\n🎯 严格正确性验证完成！")