#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch后端输出类型分析
专门分析qiboml (pytorch)后端的输出数据类型和格式
"""

import sys
import numpy as np
import torch
from qibo import Circuit, gates, set_backend

def analyze_pytorch_output_types():
    """分析PyTorch后端的输出数据类型"""
    print("🔬 PyTorch后端输出类型分析")
    print("=" * 60)
    
    # 创建测试电路
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RY(2, theta=0.5))
    
    print(f"测试电路: 3量子比特，包含H、CNOT、RY门")
    print(f"电路深度: {circuit.depth}, 门数量: {circuit.ngates}")
    
    try:
        # 设置PyTorch后端
        set_backend("qiboml", platform="pytorch")
        print("✅ PyTorch后端设置成功")
        
        # 执行电路
        result = circuit()
        print("✅ 电路执行成功")
        
        # 分析结果对象类型
        print(f"\n📊 结果对象分析:")
        print(f"结果类型: {type(result)}")
        print(f"结果类名: {result.__class__.__name__}")
        print(f"模块: {result.__class__.__module__}")
        
        # 检查结果对象的属性和方法
        print(f"\n🔍 结果对象属性/方法:")
        methods = [method for method in dir(result) if not method.startswith('_')]
        print(f"可用方法: {methods}")
        
        # 检查是否有state方法
        if hasattr(result, 'state'):
            state = result.state()
            print(f"\n📈 状态向量分析:")
            print(f"状态向量类型: {type(state)}")
            print(f"状态向量形状: {state.shape if hasattr(state, 'shape') else 'N/A'}")
            print(f"状态向量数据类型: {state.dtype if hasattr(state, 'dtype') else 'N/A'}")
            print(f"状态向量设备: {state.device if hasattr(state, 'device') else 'N/A'}")
            
            # 转换为numpy数组
            if isinstance(state, torch.Tensor):
                print(f"状态向量是torch.Tensor")
                numpy_state = state.detach().cpu().numpy()
                print(f"转换为numpy后的类型: {type(numpy_state)}")
                print(f"转换为numpy后的形状: {numpy_state.shape}")
                print(f"状态向量范数: {np.linalg.norm(numpy_state):.6f}")
            else:
                print(f"状态向量不是torch.Tensor，而是: {type(state)}")
                
        else:
            print("❌ 结果对象没有state方法")
            
        # 检查其他可能的属性
        print(f"\n🔍 其他属性检查:")
        for attr in ['final_state', 'samples', 'probabilities']:
            if hasattr(result, attr):
                value = getattr(result, attr)
                print(f"{attr}: {type(value)} - {value.shape if hasattr(value, 'shape') else 'N/A'}")
        
    except Exception as e:
        print(f"❌ PyTorch后端分析失败: {e}")
        import traceback
        traceback.print_exc()

def compare_with_numpy():
    """对比PyTorch和numpy的输出"""
    print(f"\n🔄 PyTorch与numpy对比")
    print("=" * 60)
    
    # 创建相同电路
    circuit_desc = "H(0), CNOT(0,1), RY(2, theta=0.5)"
    print(f"对比电路: {circuit_desc}")
    
    try:
        # 使用numpy后端
        set_backend("numpy")
        circuit_np = Circuit(3)
        circuit_np.add(gates.H(0))
        circuit_np.add(gates.CNOT(0, 1))
        circuit_np.add(gates.RY(2, theta=0.5))
        result_np = circuit_np()
        state_np = result_np.state()
        
        print(f"numpy状态向量类型: {type(state_np)}")
        print(f"numpy状态向量形状: {state_np.shape}")
        print(f"numpy状态向量范数: {np.linalg.norm(state_np):.6f}")
        
        # 使用PyTorch后端
        set_backend("qiboml", platform="pytorch")
        circuit_pt = Circuit(3)
        circuit_pt.add(gates.H(0))
        circuit_pt.add(gates.CNOT(0, 1))
        circuit_pt.add(gates.RY(2, theta=0.5))
        result_pt = circuit_pt()
        state_pt = result_pt.state()
        
        print(f"PyTorch状态向量类型: {type(state_pt)}")
        print(f"PyTorch状态向量形状: {state_pt.shape}")
        
        # 尝试对比
        if isinstance(state_pt, torch.Tensor):
            state_pt_np = state_pt.detach().cpu().numpy()
            diff = np.linalg.norm(state_np - state_pt_np)
            print(f"状态向量差异: {diff:.2e}")
            
            if diff < 1e-10:
                print("✅ PyTorch与numpy结果一致")
            else:
                print(f"⚠️ PyTorch与numpy结果有差异: {diff:.2e}")
        else:
            print("❌ 无法直接对比，类型不匹配")
            
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")

def test_pytorch_data_conversion():
    """测试PyTorch数据转换方法"""
    print(f"\n🔄 PyTorch数据转换测试")
    print("=" * 60)
    
    try:
        set_backend("qiboml", platform="pytorch")
        
        # 创建简单电路
        circuit = Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        result = circuit()
        
        if hasattr(result, 'state'):
            state = result.state()
            print(f"原始状态向量类型: {type(state)}")
            
            # 测试各种转换方法
            conversion_methods = [
                ("detach().cpu().numpy()", lambda x: x.detach().cpu().numpy()),
                ("numpy()", lambda x: x.numpy()),
                ("tolist()", lambda x: x.tolist()),
                ("detach().numpy()", lambda x: x.detach().numpy()),
            ]
            
            for method_name, method_func in conversion_methods:
                try:
                    converted = method_func(state)
                    print(f"✅ {method_name}: {type(converted)}")
                    if hasattr(converted, 'shape'):
                        print(f"   形状: {converted.shape}")
                except Exception as e:
                    print(f"❌ {method_name} 失败: {e}")
                    
        else:
            print("❌ 没有state方法")
            
    except Exception as e:
        print(f"❌ 转换测试失败: {e}")

def fix_strict_validation_for_pytorch():
    """修复严格验证脚本中的PyTorch处理"""
    print(f"\n🔧 PyTorch验证修复方案")
    print("=" * 60)
    
    code_fix = '''
def validate_backend_accuracy_with_pytorch_fix(backend_name, platform_name=None, circuit_qasm=None):
    """修复PyTorch后端的验证函数"""
    
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
        
        # 🔧 PyTorch特殊处理
        if backend_name == "qiboml" and platform_name == "pytorch":
            if isinstance(test_state, torch.Tensor):
                # 将PyTorch张量转换为numpy数组
                test_state = test_state.detach().cpu().numpy()
        
        # 计算状态向量差异
        state_diff = np.linalg.norm(reference_state - test_state)
        
        # ... 其余验证逻辑保持不变
        '''
    
    print("修复代码示例:")
    print(code_fix)

def generate_pytorch_backend_report():
    """生成PyTorch后端分析报告"""
    print(f"\n📋 PyTorch后端分析报告")
    print("=" * 60)
    
    report = """
## PyTorch后端输出类型分析报告

### 🔍 问题诊断
PyTorch后端验证失败的原因是数据类型不兼容：
- **错误信息**: `unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'`
- **根本原因**: PyTorch后端返回的是`torch.Tensor`对象，而numpy基准是`numpy.ndarray`

### 📊 数据类型对比

| 后端 | 状态向量类型 | 需要转换 | 转换方法 |
|------|-------------|----------|----------|
| numpy | numpy.ndarray | 否 | 直接使用 |
| qibojit | numpy.ndarray | 否 | 直接使用 |
| qibotn | numpy.ndarray | 否 | 直接使用 |
| qiboml (jax) | jax.Array | 是 | `.numpy()` |
| **qiboml (pytorch)** | **torch.Tensor** | **是** | **`.detach().cpu().numpy()`** |
| qiboml (tensorflow) | tensorflow.Tensor | 是 | `.numpy()` |

### 🔧 修复方案

在验证函数中添加PyTorch特殊处理：

```python
def validate_backend_accuracy_fixed(backend_name, platform_name=None, circuit_qasm=None):
    # ... 原有代码 ...
    
    test_state = test_result.state()
    
    # 🔧 添加PyTorch特殊处理
    if backend_name == "qiboml" and platform_name == "pytorch":
        if isinstance(test_state, torch.Tensor):
            test_state = test_state.detach().cpu().numpy()
    
    # ... 继续验证逻辑 ...
```

### 💡 使用建议

1. **PyTorch后端特点**:
   - 返回`torch.Tensor`对象，需要转换为numpy进行对比
   - 支持GPU加速，但验证时需要转到CPU
   - 需要`detach()`来分离计算图

2. **验证注意事项**:
   - 确保导入torch库：`import torch`
   - 使用正确的转换方法：`detach().cpu().numpy()`
   - 注意设备转移（GPU->CPU）

3. **性能考虑**:
   - 转换操作有轻微性能开销
   - 对于大规模验证，可考虑批量转换
   - 生产环境中可直接使用torch.Tensor进行计算

### ✅ 验证结果
修复后，PyTorch后端应能正确通过验证，与其他后端保持一致的计算精度。
"""
    
    print(report)

if __name__ == "__main__":
    print("🚀 PyTorch后端输出类型分析")
    print("=" * 60)
    
    # 运行分析
    analyze_pytorch_output_types()
    compare_with_numpy()
    test_pytorch_data_conversion()
    fix_strict_validation_for_pytorch()
    generate_pytorch_backend_report()
    
    print("\n🎯 PyTorch后端分析完成！")