#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qibo 0.2.20 版本的QAOA基准测试 (带CPU监控版本)
使用qibojit后端加速计算
收集不同量子比特数和QAOA层数下的性能数据，包括CPU利用率
解决环形图的最大割问题
增强版：添加实时进程可视化和监控功能
"""

import time
import csv
import os
import psutil
import numpy as np
import qibo
from qibo import models, gates
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import threading
import datetime
import sys
import numpy as np  # 再次导入NumPy库（重复导入，可能是冗余的）
import torch  # 导入PyTorch库，用于深度学习计算
import jax  # 导入JAX库，用于高性能数值计算
import tensorflow as tf  # 导入TensorFlow库，用于深度学习计算
# 定义参数范围
QUBITS_RANGE = range(4, 14)  # 4到15个量子比特
LAYERS_RANGE = range(1, 3)   # 1到5层QAOA

# 全局变量，用于进程监控
current_task = "初始化"
progress_value = 0
total_tasks = len(QUBITS_RANGE) * len(LAYERS_RANGE)
start_time_global = time.time()
estimated_time_remaining = "计算中..."
current_memory_usage = 0
current_cpu_usage = 0



# CPU监控类
class CPUMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_percentages = []
        self.running = False
        self.thread = None
        self.process = psutil.Process(os.getpid())
        # 初始化CPU监控
        self.process.cpu_percent()
        time.sleep(0.1)  # 等待初始化
        
    def start(self):
        self.running = True
        self.cpu_percentages = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _monitor(self):
        while self.running:
            try:
                # 获取进程级CPU使用率
                cpu_percent = self.process.cpu_percent()
                self.cpu_percentages.append(cpu_percent)
            except Exception as e:
                print(f"CPU监控错误: {str(e)}")
            time.sleep(self.interval)
            
    def get_stats(self):
        if not self.cpu_percentages:
            return {'avg': 0.0, 'max': 0.0, 'cores': psutil.cpu_count(logical=False)}
            
        return {
            'avg': sum(self.cpu_percentages) / len(self.cpu_percentages),
            'max': max(self.cpu_percentages),
            'cores': psutil.cpu_count(logical=False)
        }

def format_time(seconds):
    """将秒数格式化为可读时间格式"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

def monitor_process():
    """实时监控进程状态并显示"""
    global current_task, progress_value, estimated_time_remaining, current_memory_usage, current_cpu_usage
    
    process = psutil.Process(os.getpid())
    
    while True:
        # 清空终端
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/MacOS
            os.system('clear')
        
        # 获取当前CPU和内存使用情况
        current_cpu_usage = process.cpu_percent(interval=0.1)
        current_memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        # 计算进度条
        progress_bar_length = 50
        filled_length = int(progress_bar_length * progress_value / total_tasks)
        bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
        
        # 计算已用时间
        elapsed_time = time.time() - start_time_global
        
        # 打印状态信息
        print('=' * 70)
        print(f"QAOA基准测试进度监控 - Qibo {qibo.__version__} with qibojit")
        print('=' * 70)
        print(f"当前任务: {current_task}")
        print(f"进度: [{bar}] {progress_value}/{total_tasks} ({progress_value/total_tasks*100:.1f}%)")
        print(f"已用时间: {format_time(elapsed_time)}")
        print(f"预计剩余时间: {estimated_time_remaining}")
        print("")
        print(f"CPU使用率: {current_cpu_usage:.1f}%")
        print(f"内存使用: {current_memory_usage:.1f} MB")
        print('=' * 70)
        
        time.sleep(1)  # 每秒更新一次

# 创建环形图
def create_ring_edges(n_nodes):
    """创建n_nodes节点的环形图边集合"""
    return [(i, (i+1) % n_nodes) for i in range(n_nodes)]

# QAOA电路构建 - 使用基本量子门
def create_qaoa_circuit(params, edges, nqubits, nlayers=1):
    """创建QAOA电路"""
    beta, gamma = params[:nlayers], params[nlayers:]
    
    circuit = models.Circuit(nqubits)
    
    # 初始态 - 均匀叠加态
    circuit.add(gates.H(i) for i in range(nqubits))
    
    for l in range(nlayers):
        # 问题哈密顿量演化 - 使用ZZ旋转门
        for i, j in edges:
            circuit.add(gates.CNOT(i, j))
            circuit.add(gates.RZ(j, 2 * gamma[l]))
            circuit.add(gates.CNOT(i, j))
        
        # 混合哈密顿量演化 - 使用X旋转门
        for i in range(nqubits):
            circuit.add(gates.RX(i, 2 * beta[l]))
    
    # 添加测量
    circuit.add(gates.M(*range(nqubits)))
    
    return circuit

# 计算MAX-CUT目标函数
def calculate_cut_value(bitstring, edges):
    """计算给定比特串的割值"""
    cut_value = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cut_value += 1
    return cut_value

# 目标函数
def objective(params, edges, nqubits, nlayers=1, shots=1024):
    """计算期望值"""
    circuit = create_qaoa_circuit(params, edges, nqubits, nlayers)
    result = circuit(nshots=shots)
    # 获取状态向量并计算概率
    state = convert_to_numpy(result.state())
    probabilities = np.abs(state) ** 2
    
    # 计算期望值
    expectation = 0
    for i, prob in enumerate(probabilities):
        bitstring = np.binary_repr(i, nqubits)
        cut_value = calculate_cut_value([int(bit) for bit in bitstring], edges)
        expectation += cut_value * prob
    
    return -expectation  # 负号是因为我们的目标函数是负的割值

def convert_to_numpy(array):
        """将不同框架的数组转换为NumPy数组"""
        # 处理NumPy数组
        if isinstance(array, np.ndarray):
            return array
            
        # 处理PyTorch Tensor
        elif isinstance(array, torch.Tensor):
            # 如果需要梯度，先分离
            if array.requires_grad:
                array = array.detach()
                # 如果在GPU上，移动到CPU
            if array.is_cuda:
                array = array.cpu()
            return array.numpy()
            
            # 处理JAX数组
        elif 'jaxlib.xla_extension.ArrayImpl' in str(type(array)):
            return jax.device_get(array)
            
            # 处理TensorFlow Tensor
        elif isinstance(array, tf.Tensor):
            return array.numpy()
            
            # 处理其他情况，尝试直接转换
        else:
            try:
                return np.array(array)
            except Exception as e:
                raise ValueError(f"无法将类型 {type(array)} 转换为NumPy数组: {str(e)}")

def run_benchmark(n_qubits, n_layers):
    """运行基准测试并返回结果"""
    global current_task, current_memory_usage, current_cpu_usage
    
    edges = create_ring_edges(n_qubits)
    current_task = f"运行测试: {n_qubits}个量子比特, {n_layers}层QAOA"
    
    # 记录开始时间和内存
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 启动CPU监控
    cpu_monitor = CPUMonitor(interval=0.05)
    cpu_monitor.start()
    start_time = time.time()
    
    # 优化
    initial_point = np.random.uniform(0, np.pi, 2 * n_layers)
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=100, desc=f"优化 {n_qubits}量子比特/{n_layers}层", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                file=sys.stdout)
    
    # 定义回调函数来更新进度条
    last_update = [time.time()]
    def callback(xk):
        current_time = time.time()
        if current_time - last_update[0] > 0.5:  # 每0.5秒更新一次
            pbar.update(1)
            last_update[0] = current_time
            # 更新全局监控变量
            current_memory_usage = process.memory_info().rss / (1024 * 1024)
            current_cpu_usage = process.cpu_percent(interval=0.01)
    
    # 优化
    res = minimize(objective, initial_point, args=(edges, n_qubits, n_layers), 
                  method='COBYLA', callback=callback)
    
    pbar.close()
    
    # 获取最终结果
    current_task = f"计算最终结果: {n_qubits}个量子比特, {n_layers}层QAOA"
    final_circuit = create_qaoa_circuit(res.x, edges, n_qubits, n_layers)
    result = final_circuit()
    final_state = result.state()
    final_state = convert_to_numpy(final_state)
    # 计算概率分布
    probabilities = np.abs(final_state)**2

    
    # 找到最优解
    best_index = np.argmax(probabilities)
    best_probability = probabilities[best_index]
    best_bitstring = np.binary_repr(best_index, n_qubits)
    best_energy = -calculate_cut_value(best_bitstring, edges)
    
    # 记录结束时间和内存
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 计算运行时间、内存使用和CPU利用率
    runtime = end_time - start_time
    memory_usage = end_memory - start_memory
    
    # 停止CPU监控
    cpu_monitor.stop()
    cpu_stats = cpu_monitor.get_stats()
    
    # 修正内存计算（确保非负）
    memory_usage = max(0, memory_usage)
    
    return {
        'best_energy': best_energy,
        'best_bitstring': best_bitstring,
        'best_probability': best_probability,
        'runtime': runtime,
        'memory_usage': memory_usage,
        'cpu_avg': cpu_stats['avg'],
        'cpu_max': cpu_stats['max'],
        'cpu_cores': cpu_stats.get('cores', psutil.cpu_count(logical=False))
    }


# 定义不同的后端配置
backend_configs = {
    "numpy": {"backend_name": "numpy", "platform_name": None},
    "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
    "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
    "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
    "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
    "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"},
    "qulacs": {"backend_name": "qulacs", "platform_name": None}
}



# 主循环
for backend_name, backend_config in backend_configs.items():
    try:
        # 设置当前后端
        qibo.set_backend(backend_config["backend_name"], platform=backend_config["platform_name"])
        
        # 创建特定后端的CSV文件
        csv_file = f'results/comparison/qaoa_qibo_benchmark_results_{backend_name.replace(" ", "_").replace("(", "").replace(")", "")}.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['num_qubits', 'qaoa_layers', 'best_energy', 'best_bitstring', 
                           'best_probability', 'runtime_sec', 'memory_usage_mb', 
                           'cpu_avg_percent', 'cpu_max_percent'])
        
        # 重置计数器
        completed_tasks = 0
        task_times = []
        start_time_global = time.time()
        
        print(f"\n开始使用 {backend_name} 后端进行QAOA基准测试...")
        time.sleep(2)  # 给用户时间阅读初始信息
        
        # 计算总任务数
        total_tasks = len(QUBITS_RANGE) * len(LAYERS_RANGE)
        
        for n_qubits in QUBITS_RANGE:
            for n_layers in LAYERS_RANGE:
                try:
                    # 更新全局状态
                    current_task = f"准备测试 ({backend_name}): {n_qubits}个量子比特, {n_layers}层QAOA"
                    
                    # 记录任务开始时间
                    task_start_time = time.time()
                    
                    # 运行基准测试
                    results = run_benchmark(n_qubits, n_layers)
                    
                    # 写入CSV
                    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            n_qubits,
                            n_layers,
                            results['best_energy'],
                            results['best_bitstring'],
                            results['best_probability'],
                            round(results['runtime'], 2),
                            round(results['memory_usage'], 2),
                            round(results['cpu_avg'], 2),
                            round(results['cpu_max'], 2)
                        ])
                    
                    # 更新进度
                    completed_tasks += 1
                    progress_value = completed_tasks
                    
                    # 记录任务执行时间
                    task_time = time.time() - task_start_time
                    task_times.append(task_time)
                    
                    # 估计剩余时间
                    if len(task_times) > 0:
                        avg_task_time = sum(task_times) / len(task_times)
                        remaining_tasks = total_tasks - completed_tasks
                        est_remaining_time = avg_task_time * remaining_tasks
                        estimated_time_remaining = format_time(est_remaining_time)
                    
                    # 输出结果摘要
                    current_task = f"完成测试 ({backend_name}): {n_qubits}个量子比特, {n_layers}层QAOA"
                    print("")
                    print('=' * 50)
                    print(f"完成测试 [{completed_tasks}/{total_tasks}] ({backend_name}): {n_qubits}个量子比特, {n_layers}层QAOA")
                    print(f"最佳能量: {results['best_energy']}")
                    print(f"最优比特串: {results['best_bitstring']}")
                    print(f"最优比特串概率: {results['best_probability']:.4f}")
                    print(f"运行时间: {results['runtime']:.2f}秒")
                    print(f"内存使用: {results['memory_usage']:.2f}MB")
                    print(f"CPU利用率 - 平均: {results['cpu_avg']:.2f}% 峰值: {results['cpu_max']:.2f}% (核心: {results['cpu_cores']})")
                    print('=' * 50)
                    
                except Exception as e:
                    print("!")
                    print('!' * 50)
                    print(f"测试失败 ({backend_name}): {n_qubits}个量子比特, {n_layers}层QAOA")
                    print(f"错误: {str(e)}")
                    print('!' * 50)
                    
                    # 记录错误到CSV
                    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([n_qubits, n_layers, "ERROR", str(e), "", "", "", ""])
                    
                    # 更新进度
                    completed_tasks += 1
                    progress_value = completed_tasks
                    
        print(f"\n{backend_name} 后端测试完成!")
        print(f"结果已保存到: {csv_file}")
        print(f"总执行时间: {format_time(time.time() - start_time_global)}")
        
    except Exception as e:
        print(f"\n后端 {backend_name} 初始化失败: {str(e)}")
        continue

# 完成所有测试
current_task = "所有基准测试完成!"
print("\n" + "#" * 70)
print("所有后端的基准测试已完成!")
print("#" * 70)




