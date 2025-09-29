#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Qibo不同后端的脚本
"""

import time
import sys
import numpy as np
from qibo import Circuit, gates, set_backend
from qibo.backends import HammingWeightBackend

# 检查qiboml是否已安装
try:
    import qiboml
    QIBOML_AVAILABLE = True
except ImportError:
    QIBOML_AVAILABLE = False


def test_clifford_backend():
    """测试Clifford后端"""
    print("\n===== 测试 Clifford 后端 (numpy平台) =====")
    set_backend("clifford", platform="numpy")
    backend_name = "clifford (numpy)"
    
    # 创建一个简单的电路
    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add([
        gates.H(0),
        gates.CNOT(0, 1),
        gates.X(2),
        gates.CNOT(2, 3)
    ])
    
    # 执行电路
    start_time = time.time()
    result = circuit()
    end_time = time.time()
    
    print(f"电路包含 {nqubits} 个量子比特")
    print(f"执行时间: {end_time - start_time:.6f} 秒")
    print(f"结果类型: {type(result)}")
    print(f"结果形状: {result.shape if hasattr(result, 'shape') else '无形状信息'}")
    
    return backend_name


def test_hamming_weight_backend():
    """测试HammingWeight后端"""
    print("\n===== 测试 Hamming Weight 后端 (numpy平台) =====")
    set_backend("hamming_weight", platform="numpy")
    backend_name = "hamming_weight (numpy)"
    
    # 创建一个简单的电路
    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add(gates.SWAP(0, 1))
    
    # 使用HammingWeightBackend执行
    backend = HammingWeightBackend()
    start_time = time.time()
    result = backend.execute_circuit(circuit, weight=2)
    end_time = time.time()
    
    print(f"电路包含 {nqubits} 个量子比特")
    print(f"执行时间: {end_time - start_time:.6f} 秒")
    print(f"结果类型: {type(result)}")
    print(f"结果形状: {result.shape if hasattr(result, 'shape') else '无形状信息'}")
    print(f"结果: {result}")
    
    return backend_name


def test_numpy_backend():
    """测试numpy后端"""
    print("\n===== 测试 numpy 后端  =====")
    set_backend("numpy")
    backend_name = "numpy"
    
    # 创建一个简单的电路
    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add([
        gates.H(0),
        gates.CNOT(0, 1),
        gates.X(2),
        gates.CNOT(2, 3)
    ])
    
    # 执行电路
    start_time = time.time()
    result = circuit()
    end_time = time.time()
    
    print(f"电路包含 {nqubits} 个量子比特")
    print(f"执行时间: {end_time - start_time:.6f} 秒")
    print(f"结果类型: {type(result)}")
    print(f"结果形状: {result.shape if hasattr(result, 'shape') else '无形状信息'}")
    
    return backend_name


def test_numba_backend():
    """测试qibojit后端"""
    print("\n===== 测试 qibojit 后端 (numba平台) =====")
    set_backend("qibojit", platform="numba")
    backend_name = "qibojit (numba)"
    
    # 创建一个简单的电路
    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add([
        gates.H(0),
        gates.CNOT(0, 1),
        gates.X(2),
        gates.CNOT(2, 3)
    ])
    
    # 执行电路
    start_time = time.time()
    result = circuit()
    end_time = time.time()
    
    print(f"电路包含 {nqubits} 个量子比特")
    print(f"执行时间: {end_time - start_time:.6f} 秒")
    print(f"结果类型: {type(result)}")
    print(f"结果形状: {result.shape if hasattr(result, 'shape') else '无形状信息'}")
    
    return backend_name


def test_qibotn_backend():
    """测试qibotn后端"""
    print("\n===== 测试 qibotn 后端 (qutensornet平台) =====")
    backend_name = "qibotn (qutensornet)"
    
    computation_settings = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }

    # Set the quimb backend
    set_backend(
        backend="qibotn", platform="qutensornet", runcard=computation_settings
    )
    
    # 创建一个简单的电路
    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add([
        gates.H(0),
        gates.CNOT(0, 1),
        gates.X(2),
        gates.CNOT(2, 3)
    ])
    
    # 执行电路
    start_time = time.time()
    result = circuit()
    end_time = time.time()
    
    print(f"电路包含 {nqubits} 个量子比特")
    print(f"执行时间: {end_time - start_time:.6f} 秒")
    print(f"结果类型: {type(result)}")
    print(f"结果形状: {result.shape if hasattr(result, 'shape') else '无形状信息'}")
    
    return backend_name


def test_qiboml_backends():
    """测试QiboML不同平台的后端"""
    successful_backends = []
    
    if not QIBOML_AVAILABLE:
        print("\n===== QiboML 后端测试 =====")
        print("QiboML 未安装。要使用 QiboML 后端，请先安装 qiboml 包。")
        print("可以使用以下命令安装：pip install qiboml")
        return successful_backends
        
    platforms = ["jax", "pytorch", "tensorflow"]
    
    for platform in platforms:
        print(f"\n===== 测试 QiboML 后端 ({platform}平台) =====")
        try:
            set_backend("qiboml", platform=platform)
            
            # 创建一个简单的电路
            nqubits = 4
            circuit = Circuit(nqubits)
            circuit.add([
                gates.H(0),
                gates.CNOT(0, 1),
                gates.X(2),
                gates.CNOT(2, 3)
            ])
            
            # 执行电路
            start_time = time.time()
            result = circuit()
            end_time = time.time()
            
            print(f"电路包含 {nqubits} 个量子比特")
            print(f"执行时间: {end_time - start_time:.6f} 秒")
            print(f"结果类型: {type(result)}")
            print(f"结果形状: {result.shape if hasattr(result, 'shape') else '无形状信息'}")
            
            successful_backends.append(f"qiboml ({platform})")
            
        except Exception as e:
            print(f"在 {platform} 平台上测试 QiboML 后端时出错: {str(e)}")
    
    return successful_backends


def get_qibo_version():
    """获取Qibo版本信息"""
    import qibo
    return qibo.__version__


def get_available_backends():
    """获取可用的后端列表"""
    try:
        from qibo.backends import MetaBackend
        available_backends = MetaBackend().list_available()
        return available_backends
    except Exception:
        return ["无法获取后端列表"]


if __name__ == "__main__":
    print("开始测试Qibo不同后端...")
    print(f"Qibo版本: {get_qibo_version()}")
    print(f"Python版本: {sys.version}")
    print(f"NumPy版本: {np.__version__}")
    print(f"QiboML是否可用: {'是' if QIBOML_AVAILABLE else '否'}")
    print(f"理论可用后端: {get_available_backends()}")
    print("-" * 50)
    
    # 用于存储成功测试的后端
    successful_backends = []
    
    try:
        backend = test_numpy_backend()
        successful_backends.append(backend)
    except Exception as e:
        print(f"测试numpy后端时出错: {str(e)}")
    
    try:
        backend = test_numba_backend()
        successful_backends.append(backend)
    except Exception as e:
        print(f"测试qibojit后端时出错: {str(e)}")

    try:
        backend = test_qibotn_backend()
        successful_backends.append(backend)
    except Exception as e:
        print(f"测试qibotn后端时出错: {str(e)}")

    try:
        backend = test_clifford_backend()
        successful_backends.append(backend)
    except Exception as e:
        print(f"测试Clifford后端时出错: {str(e)}")
    
    try:
        backend = test_hamming_weight_backend()
        successful_backends.append(backend)
    except Exception as e:
        print(f"测试HammingWeight后端时出错: {str(e)}")
    
    try:
        qiboml_backends = test_qiboml_backends()
        if qiboml_backends:
            successful_backends.extend(qiboml_backends)
    except Exception as e:
        print(f"测试QiboML后端时出错: {str(e)}")
    
    print("\n所有测试完成!")
    print("\n===== 成功测试的后端 =====")
    if successful_backends:
        for i, backend in enumerate(successful_backends, 1):
            print(f"{i}. {backend}")
    else:
        print("没有成功测试的后端")