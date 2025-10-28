#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
qibotn后端警告测试脚本
专门测试qibotn后端的警告信息和正确性验证
"""

import warnings
import sys
import os
import numpy as np
from qibo import Circuit, gates, set_backend, get_backend

def test_qibotn_warnings():
    """测试qibotn后端的警告信息"""
    print("🔍 测试qibotn后端警告信息")
    
    # 捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # 设置qibotn后端
            set_backend("qibotn", platform="qutensornet")
            print("✅ qibotn后端设置成功")
            
            # 创建小型测试电路
            circuit = Circuit(4)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.H(2))
            
            # 执行电路
            result = circuit()
            print("✅ 电路执行成功")
            
            # 检查警告
            if w:
                print(f"⚠️ 发现 {len(w)} 个警告:")
                for warning in w:
                    print(f"   - {warning.category.__name__}: {warning.message}")
                    print(f"     文件: {warning.filename}:{warning.lineno}")
            else:
                print("✅ 没有发现警告")
                
            # 验证结果正确性
            if hasattr(result, 'state'):
                state = result.state()
                print(f"✅ 状态向量维度: {len(state)}")
                print(f"✅ 状态向量范数: {np.linalg.norm(state):.6f}")
            else:
                print("❌ 无法获取状态向量")
                
        except Exception as e:
            print(f"❌ qibotn测试失败: {e}")
            return False
    
    return True

def compare_with_numpy():
    """对比qibotn和numpy的结果"""
    print("\n🔬 对比qibotn和numpy结果")
    
    # 测试电路
    circuit_desc = "H(0), CNOT(0,1), H(2)"
    print(f"测试电路: {circuit_desc}")
    
    try:
        # 使用numpy后端计算基准
        set_backend("numpy")
        circuit_numpy = Circuit(4)
        circuit_numpy.add(gates.H(0))
        circuit_numpy.add(gates.CNOT(0, 1))
        circuit_numpy.add(gates.H(2))
        result_numpy = circuit_numpy()
        state_numpy = result_numpy.state()
        
        # 使用qibotn后端计算
        set_backend("qibotn", platform="qutensornet")
        circuit_qibotn = Circuit(4)
        circuit_qibotn.add(gates.H(0))
        circuit_qibotn.add(gates.CNOT(0, 1))
        circuit_qibotn.add(gates.H(2))
        result_qibotn = circuit_qibotn()
        state_qibotn = result_qibotn.state()
        
        # 对比结果
        diff = np.linalg.norm(state_numpy - state_qibotn)
        print(f"✅ 状态向量差异: {diff:.2e}")
        
        if diff < 1e-10:
            print("✅ 结果一致 - qibotn计算正确")
        else:
            print(f"⚠️ 结果有差异: {diff:.2e}")
            
        # 检查概率分布
        prob_numpy = np.abs(state_numpy)**2
        prob_qibotn = np.abs(state_qibotn)**2
        prob_diff = np.linalg.norm(prob_numpy - prob_qibotn)
        print(f"✅ 概率分布差异: {prob_diff:.2e}")
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")

def test_qibotn_limitations():
    """测试qibotn的可能限制"""
    print("\n🔧 测试qibotn功能限制")
    
    limitations = []
    
    try:
        # 测试大电路
        set_backend("qibotn", platform="qutensornet")
        circuit_large = Circuit(12)  # 中等大小电路
        for i in range(12):
            circuit_large.add(gates.H(i))
        result_large = circuit_large()
        print("✅ 支持12量子比特电路")
        
    except Exception as e:
        limitations.append(f"大电路限制: {e}")
        print(f"⚠️ 大电路可能受限: {e}")
    
    # 测试复杂门操作
    try:
        circuit_complex = Circuit(4)
        circuit_complex.add(gates.RY(0, theta=0.5))
        circuit_complex.add(gates.CZ(0, 1))
        circuit_complex.add(gates.SWAP(2, 3))
        result_complex = circuit_complex()
        print("✅ 支持复杂门操作")
        
    except Exception as e:
        limitations.append(f"复杂门限制: {e}")
        print(f"⚠️ 复杂门操作可能受限: {e}")
    
    if limitations:
        print(f"🔍 发现的限制: {len(limitations)}")
        for limit in limitations:
            print(f"   - {limit}")
    else:
        print("✅ 未发现明显功能限制")

if __name__ == "__main__":
    print("🚀 qibotn后端专项测试")
    print("=" * 50)
    
    # 运行测试
    test_qibotn_warnings()
    compare_with_numpy() 
    test_qibotn_limitations()
    
    print("\n🎯 qibotn测试完成")