#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
遍历测试所有后端配置的QFT电路性能
"""

import time
import sys
import os
from qibo import Circuit, gates, set_backend

# 后端配置字典
backend_configs = {
    "numpy": {"backend_name": "numpy", "platform_name": None},
    "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
    "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
    "clifford (numpy)": {"backend_name": "clifford", "platform_name": "numpy"},
    "hamming_weight (numpy)": {"backend_name": "hamming_weight", "platform_name": "numpy"},
    "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
    "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
    "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"}
}

def load_qft_circuit():
    """加载QFT电路（从run_qft_modified copy.py中提取的逻辑）"""
    qasm_file = "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"
    
    if not os.path.exists(qasm_file):
        print(f"错误: 找不到文件 {qasm_file}")
        return None
    
    print(f"正在读取QASM文件: {qasm_file}")
    with open(qasm_file, "r") as file:
        qasm_code = file.read()
    
    print("移除barrier语句...")
    lines = qasm_code.split('\n')
    filtered_lines = [line for line in lines if 'barrier' not in line]
    clean_qasm_code = '\n'.join(filtered_lines)
    
    print("正在加载电路...")
    circuit = Circuit.from_qasm(clean_qasm_code)
    return circuit

def run_qft_with_backend(backend_name, platform_name=None):
    """使用指定后端运行QFT电路"""
    print(f"\n{'='*60}")
    print(f"正在测试后端: {backend_name}")
    if platform_name:
        print(f"平台: {platform_name}")
    
    # 设置后端
    if platform_name is not None:
        set_backend(backend_name, platform=platform_name)
    else:
        set_backend(backend_name)
    
    # 加载电路
    circuit = load_qft_circuit()
    if circuit is None:
        return None
    
    # 打印电路信息
    print(f"电路包含 {circuit.nqubits} 个量子比特")
    print(f"电路深度: {circuit.depth}")
    print(f"电路门数量: {circuit.ngates}")
    
    # 执行电路模拟
    print("开始执行电路模拟...")
    start_time = time.time()
    try:
        result = circuit()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"模拟完成，耗时: {execution_time:.4f} 秒")
        return execution_time, result
    except Exception as e:
        print(f"模拟失败: {str(e)}")
        return None, None

def test_all_backends():
    """测试所有后端配置"""
    print("开始遍历测试所有后端配置...")
    print(f"Python版本: {sys.version}")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    for backend_key, config in backend_configs.items():
        backend_name = config["backend_name"]
        platform_name = config["platform_name"]
        
        execution_time, result = run_qft_with_backend(backend_name, platform_name)
        
        if execution_time is not None:
            results[backend_key] = {
                "execution_time": execution_time,
                "success": True,
                "result_type": str(type(result)) if result else "None"
            }
        else:
            results[backend_key] = {
                "execution_time": None,
                "success": False,
                "result_type": "None"
            }
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("后端测试汇总结果:")
    print("="*60)
    
    successful_backends = {k: v for k, v in results.items() if v["success"]}
    failed_backends = {k: v for k, v in results.items() if not v["success"]}
    
    if successful_backends:
        print("\n成功测试的后端 (按执行时间排序):")
        sorted_successful = sorted(successful_backends.items(), 
                                  key=lambda x: x[1]["execution_time"])
        for i, (backend, data) in enumerate(sorted_successful, 1):
            print(f"{i}. {backend}: {data['execution_time']:.4f}秒")
    
    if failed_backends:
        print("\n测试失败的后端:")
        for i, (backend, data) in enumerate(failed_backends.items(), 1):
            print(f"{i}. {backend}: 失败")
    
    return results

if __name__ == "__main__":
    test_all_backends()
