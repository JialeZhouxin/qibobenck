"""
使用qibojit后端运行qft_n18_transpiled.qasm文件中的QFT电路
移除Qibo不支持的barrier语句
"""
import os
from pdb import run
import time
import numpy as np
from qibo import Circuit, set_backend




# 定义QASM文件路径
qasm_file = "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"

# 检查文件是否存在
if not os.path.exists(qasm_file):
    print(f"错误: 找不到文件 {qasm_file}")
    exit(1)

# 读取QASM文件内容
print(f"正在读取QASM文件: {qasm_file}")
with open(qasm_file, "r") as file:
    qasm_code = file.read()

# 移除所有barrier语句（Qibo不支持）
print("移除barrier语句...")
lines = qasm_code.split('\n')
filtered_lines = [line for line in lines if 'barrier' not in line]
clean_qasm_code = '\n'.join(filtered_lines)

# 使用qibo加载清理后的QASM代码
print("正在加载电路...")
circuit = Circuit.from_qasm(clean_qasm_code)

def run_qft_modified(backend_name = "qibojit",platform_name = None):
    """
    使用qibojit后端运行qft_n18_transpiled.qasm文件中的QFT电路

    移除Qibo不支持的barrier语句
    """
    # 设置使用qibojit后端
    if platform_name is not None:
        set_backend(backend_name,platform_name)
    else:
        set_backend(backend_name)
    # 打印电路信息
    print(f"电路包含 {circuit.nqubits} 个量子比特")
    print(f"电路深度: {circuit.depth}")
    print(f"电路门数量: {circuit.ngates}")

    # 执行电路模拟
    print("开始执行电路模拟...")
    start_time = time.time()
    result = circuit()
    end_time = time.time()

    print(f"模拟完成，耗时: {end_time - start_time:.4f} 秒")

    # 获取最终状态向量
    #state_vector = result.state()
    #print("\n最终状态向量的前10个元素:")
    #print(state_vector[:10])

    # 计算所有可能测量结果的概率
    #probabilities = np.abs(state_vector)**2
    #print("\n前10个最大概率的测量结果:")
    #top_indices = np.argsort(-probabilities)[:10]
    #for i in top_indices:
        #binary_repr = format(i, f'0{circuit.nqubits}b')
        #print(f"|{i}>: {probabilities[i]:.6f}")

    #print("\n模拟完成!")

run_qft_modified("qibotn","qutensornet")