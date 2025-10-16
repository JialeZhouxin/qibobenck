#!/usr/bin/env python3
"""
测试QiboQulacs后端的可用性
"""

import sys

def test_qibo_qulacs():
    """测试QiboQulacs后端是否可用"""
    try:
        import qibo
        print(f"✅ Qibo {qibo.__version__} 可用")
        
        # 尝试设置qulacs后端
        try:
            qibo.set_backend("qulacs")
            print("✅ QiboQulacs后端设置成功")
            
            # 创建一个简单电路进行测试
            from qibo import Circuit, gates
            c = Circuit(2)
            c.add(gates.H(0))
            c.add(gates.CNOT(0, 1))
            
            # 执行电路
            result = c()
            print(f"✅ QiboQulacs电路执行成功")
            try:
                final_state = result.state()
                print(f"   状态向量形状: {final_state.shape}")
                print(f"   状态向量范数: {abs(final_state.conj().T @ final_state)}")
            except Exception as e:
                print(f"   状态向量获取失败: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ QiboQulacs后端设置失败: {e}")
            print("   可能的解决方案:")
            print("   1. 安装qibo-qulacs: pip install qibo-qulacs")
            print("   2. 安装qulacs: pip install qulacs")
            print("   3. 确保qulacs已正确编译")
            return False
            
    except ImportError as e:
        print(f"❌ Qibo不可用: {e}")
        return False

def main():
    """主函数"""
    print("=== QiboQulacs后端测试 ===")
    print()
    
    success = test_qibo_qulacs()
    
    print()
    if success:
        print("🎉 QiboQulacs后端测试通过！")
        print()
        print("现在可以使用以下命令进行基准测试：")
        print("python run_benchmarks.py --simulators qibo-qulacs --qubits 2 3 4 --verbose")
        return 0
    else:
        print("⚠️ QiboQulacs后端测试失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())