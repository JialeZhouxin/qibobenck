#!/usr/bin/env python  # 指定Python解释器路径
# -*- coding: utf-8 -*-  # 指定文件编码为UTF-8

"""
QASMBench通用基准测试工具
支持加载QASMBench中的任意电路进行Qibo后端性能测试
"""

import time  # 导入time模块，用于计时和性能测量
import sys  # 导入sys模块，用于访问系统相关参数和功能
import os  # 导入os模块，用于处理文件和目录操作
import json  # 导入json模块，用于处理JSON格式的数据
import csv  # 导入csv模块，用于处理CSV格式的数据
import platform  # 导入platform模块，用于获取系统信息
import psutil  # 导入psutil模块，用于系统资源监控
import numpy as np  # 导入NumPy库，用于数值计算
from datetime import datetime  # 导入datetime类，用于处理日期和时间
from qibo import Circuit, gates, set_backend  # 从Qibo框架导入核心类和函数
from qibo.ui import plot_circuit  # 从Qibo UI模块导入电路绘图函数
import numpy as np  # 再次导入NumPy库（重复导入，可能是冗余的）
import torch  # 导入PyTorch库，用于深度学习计算
import jax  # 导入JAX库，用于高性能数值计算
import tensorflow as tf  # 导入TensorFlow库，用于深度学习计算

class QASMBenchConfig:
    """QASMBench基准测试配置类
    
    该类用于存储和管理QASMBench基准测试的配置参数，
    包括运行次数、输出格式、基准后端等设置。
    """
    def __init__(self):
        """初始化配置对象
        
        设置默认的基准测试配置参数。
        """
        self.num_runs = 5  # 每个后端正式运行的次数
        self.warmup_runs = 1  # 预热运行的次数，用于JIT编译等
        self.output_formats = ['csv', 'markdown', 'json']  # 支持的输出报告格式
        self.baseline_backend = "numpy"  # 作为性能比较基准的后端
        self.qasm_directory = "../QASMBench"  # QASMBench基准测试电路的根目录

class QASMBenchMetrics:
    """存储QASMBench基准测试指标
    
    该类用于存储和量化基准测试的各种指标，
    包括执行时间、内存使用、正确性验证等。
    """
    def __init__(self):
        """初始化指标对象
        
        创建并初始化所有指标的默认值。
        """
        # 核心指标
        self.execution_time_mean = None  # 平均执行时间（秒）
        self.execution_time_std = None  # 执行时间标准差（秒）
        self.peak_memory_mb = None  # 峰值内存使用量（MB）
        self.speedup = None  # 相对于基准后端的加速比
        self.correctness = "Unknown"  # 正确性验证结果
        
        # 电路信息
        self.circuit_parameters = {}
        self.backend_info = {}
        
        # 性能指标
        self.throughput_gates_per_sec = None
        self.jit_compilation_time = None
        self.environment_info = {}
        
        # 元数据
        self.circuit_build_time = None
        self.report_metadata = {}

class QASMBenchReporter:
    """生成QASMBench基准测试报告"""
    
    @staticmethod
    def generate_csv_report(results, circuit_name, filename=None):
        """生成CSV格式报告"""
        if filename is None:
            # 清理电路名称，移除路径分隔符
            clean_circuit_name = circuit_name.replace('/', '_').replace('\\', '_')
            # 创建专门的报告文件夹
            report_dir = f"qibobench/reports/{clean_circuit_name}"
            filename = f"{report_dir}/benchmark_report.csv"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            headers = [
                "后端名称", "执行时间均值(秒)", "执行时间标准差(秒)", 
                "峰值内存占用(MB)", "加速比", "正确性验证",
                "量子比特数", "电路深度", "门数量", "吞吐率(门/秒)",
                "JIT编译时间(秒)", "电路构建时间(秒)"
            ]
            writer.writerow(headers)
            
            # 写入数据
            for backend_name, metrics in results.items():
                if metrics.execution_time_mean is not None:
                    row = [
                        backend_name,
                        f"{metrics.execution_time_mean:.6f}",
                        f"{metrics.execution_time_std:.6f}",
                        f"{metrics.peak_memory_mb:.2f}",
                        f"{metrics.speedup:.2f}x" if metrics.speedup else "N/A",
                        metrics.correctness,
                        metrics.circuit_parameters.get('nqubits', 'N/A'),
                        metrics.circuit_parameters.get('depth', 'N/A'),
                        metrics.circuit_parameters.get('ngates', 'N/A'),
                        f"{metrics.throughput_gates_per_sec:.2f}" if metrics.throughput_gates_per_sec else "N/A",
                        f"{metrics.jit_compilation_time:.4f}" if metrics.jit_compilation_time else "N/A",
                        f"{metrics.circuit_build_time:.4f}" if metrics.circuit_build_time else "N/A"
                    ]
                    writer.writerow(row)
        
        print(f"CSV报告已生成: {filename}")
    
    @staticmethod
    def generate_markdown_report(results, circuit_name, filename=None):
        """生成Markdown格式报告"""
        if filename is None:
            # 清理电路名称，移除路径分隔符
            clean_circuit_name = circuit_name.replace('/', '_').replace('\\', '_')
            # 创建专门的报告文件夹
            report_dir = f"qibobench/reports/{clean_circuit_name}"
            filename = f"{report_dir}/benchmark_report.md"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as md_file:
            # 报告标题
            md_file.write(f"# QASMBench电路基准测试报告: {circuit_name}\n\n")
            md_file.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试电路信息
            md_file.write("## 测试电路参数\n\n")
            
            # 从第一个有效结果中获取电路参数
            circuit_params = None
            for metrics in results.values():
                if hasattr(metrics, 'circuit_parameters') and metrics.circuit_parameters:
                    circuit_params = metrics.circuit_parameters
                    break
            
            if circuit_params:
                md_file.write("| 参数 | 值 | 描述 |\n")
                md_file.write("|------|----|------|\n")
                md_file.write(f"| 电路名称 | {circuit_name} | QASMBench电路 |\n")
                md_file.write(f"| 量子比特数 | {circuit_params.get('nqubits', 'N/A')} | 电路的宽度 |\n")
                md_file.write(f"| 电路深度 | {circuit_params.get('depth', 'N/A')} | 电路的层数 |\n")
                md_file.write(f"| 门数量 | {circuit_params.get('ngates', 'N/A')} | 总门操作数 |\n")
                md_file.write(f"| 电路来源 | {circuit_params.get('source', 'N/A')} | QASM文件路径 |\n")
                md_file.write("\n")
            
            # 测试配置
            md_file.write("### 测试配置\n\n")
            md_file.write("- **运行次数**: 5次正式运行 + 1次预热运行\n")
            md_file.write("- **基准后端**: numpy (作为性能比较基准)\n")
            md_file.write("- **测试目标**: 比较不同后端在相同电路上的性能表现\n")
            md_file.write("- **输出格式**: CSV, Markdown, JSON\n\n")
            
            # 详细结果表格
            md_file.write("## 详细测试结果\n\n")
            md_file.write("| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |\n")
            md_file.write("|------|-------------|----------|--------|--------|---------------|\n")
            
            for backend_name, metrics in results.items():
                if metrics.execution_time_mean is not None:
                    time_str = f"{metrics.execution_time_mean:.4f} ± {metrics.execution_time_std:.4f}"
                    memory_str = f"{metrics.peak_memory_mb:.1f}"
                    speedup_str = f"{metrics.speedup:.2f}x" if metrics.speedup else "N/A"
                    throughput_str = f"{metrics.throughput_gates_per_sec:.0f}" if metrics.throughput_gates_per_sec else "N/A"
                    
                    md_file.write(f"| {backend_name} | {time_str} | {memory_str} | {speedup_str} | {metrics.correctness} | {throughput_str} |\n")
            
            # 环境信息
            md_file.write("\n## 测试环境\n\n")
            for backend_name, metrics in results.items():
                if metrics.environment_info:
                    md_file.write(f"### {backend_name} 环境\n")
                    for key, value in metrics.environment_info.items():
                        md_file.write(f"- {key}: {value}\n")
                    md_file.write("\n")
            
            # 性能分析
            md_file.write("## 性能分析\n\n")
            
            successful_results = {k: v for k, v in results.items() if v.execution_time_mean is not None}
            if successful_results:
                sorted_results = sorted(successful_results.items(), 
                                      key=lambda x: x[1].execution_time_mean)
                
                if len(sorted_results) > 1:
                    md_file.write("### 性能排名（从优到劣）\n")
                    for i, (backend_name, metrics) in enumerate(sorted_results, 1):
                        speedup_str = f" ({metrics.speedup:.2f}x)" if metrics.speedup else ""
                        md_file.write(f"{i}. **{backend_name}** - {metrics.execution_time_mean:.4f}秒{speedup_str}\n")
                    md_file.write("\n")
        
        print(f"Markdown报告已生成: {filename}")
    
    @staticmethod
    def generate_json_report(results, circuit_name, filename=None):
        """生成JSON格式报告"""
        if filename is None:
            # 清理电路名称，移除路径分隔符
            clean_circuit_name = circuit_name.replace('/', '_').replace('\\', '_')
            # 创建专门的报告文件夹
            report_dir = f"qibobench/reports/{clean_circuit_name}"
            filename = f"{report_dir}/benchmark_report.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        report_data = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "circuit_name": circuit_name,
                "qibo_version": "0.2.21",
                "python_version": platform.python_version()
            },
            "results": {}
        }
        
        for backend_name, metrics in results.items():
            report_data["results"][backend_name] = {
                "execution_time": {
                    "mean": metrics.execution_time_mean,
                    "std": metrics.execution_time_std
                },
                "memory_usage_mb": metrics.peak_memory_mb,
                "speedup": metrics.speedup,
                "correctness": metrics.correctness,
                "throughput_gates_per_sec": metrics.throughput_gates_per_sec,
                "jit_compilation_time": metrics.jit_compilation_time,
                "circuit_build_time": metrics.circuit_build_time,
                "circuit_parameters": metrics.circuit_parameters,
                "environment_info": metrics.environment_info
            }
        
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(report_data, json_file, indent=2, ensure_ascii=False)
        
        print(f"JSON报告已生成: {filename}")
    
    @staticmethod
    def save_circuit_diagram(circuit, circuit_name, filename=None):
        """保存电路图到文件"""
        if filename is None:
            # 清理电路名称，移除路径分隔符
            clean_circuit_name = circuit_name.replace('/', '_').replace('\\', '_')
            # 创建专门的报告文件夹
            report_dir = f"qibobench/reports/{clean_circuit_name}"
            filename = f"{report_dir}/{clean_circuit_name}_diagram.png"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 绘制电路图
        fig = plot_circuit(circuit)
        # 假设fig是通过类似 fig, ax = plt.subplots() 的方式创建的
        # 那么fig应该是元组中的第一个元素
        figure = fig[0]  # 或者根据实际情况选择正确的索引
        figure.figure.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"电路图已保存: {filename}")

class QASMBenchRunner:
    """运行QASMBench基准测试"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}

    def discover_qasm_circuits(self):
        """发现QASMBench中所有可用的电路"""
        circuits = {}
        
        # 搜索small、medium、large目录
        for size in ['small', 'medium', 'large']:
            size_dir = os.path.join(self.config.qasm_directory, size)
            if os.path.exists(size_dir):
                print(f"搜索目录: {size_dir}")
                for circuit_dir in os.listdir(size_dir):
                    circuit_path = os.path.join(size_dir, circuit_dir)
                    if os.path.isdir(circuit_path):
                        # 查找.qasm文件
                        qasm_files = []
                        for file in os.listdir(circuit_path):
                            if file.endswith('.qasm'):
                                qasm_files.append(file)
                        
                        if qasm_files:
                            # 优先使用transpiled版本（避免Qibo后端不支持的问题）
                            target_file = None
                            for file in qasm_files:
                                if 'transpiled' in file:
                                    target_file = file
                                    break
                            if target_file is None:
                                target_file = qasm_files[0]  # 使用第一个文件
                            
                            circuit_name = f"{size}/{circuit_dir}"
                            circuits[circuit_name] = {
                                'size': size,
                                'name': circuit_dir,
                                'path': os.path.join(circuit_path, target_file),
                                'full_path': circuit_path,
                                'available_files': qasm_files
                            }
        
        return circuits
    
    def load_qasm_circuit(self, qasm_file_path):
        """加载QASM电路文件"""
        if not os.path.exists(qasm_file_path):
            print(f"错误: 找不到文件 {qasm_file_path}")
            return None
        
        try:
            with open(qasm_file_path, "r") as file:
                qasm_code = file.read()
            
            # 移除barrier语句（Qibo不支持）
            lines = qasm_code.split('\n')
            filtered_lines = [line for line in lines if 'barrier' not in line]
            clean_qasm_code = '\n'.join(filtered_lines)
            
            circuit = Circuit.from_qasm(clean_qasm_code)
            return circuit
            
        except Exception as e:
            print(f"加载电路失败: {str(e)}")
            return None
    
    def measure_memory_usage(self):
        """测量内存使用情况"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # 转换为MB
    


    def validate_correctness(self, result, baseline_result=None):
        """验证计算结果的正确性"""
        try:
            if result is None:
                return "Failed"
            
            # 检查是否有状态向量方法
            if hasattr(result, 'state') and callable(getattr(result, 'state')):
                current_state = result.state()
                if current_state is None or len(current_state) == 0:
                    return "Failed - Invalid state"
                
                # 将当前状态转换为NumPy数组
                current_state_np = self._convert_to_numpy(current_state)
                
                # 如果有基准结果，进行状态向量比较
                if baseline_result is not None and hasattr(baseline_result, 'state'):
                    baseline_state = baseline_result.state()
                    if baseline_state is not None:
                        # 将基准状态转换为NumPy数组
                        baseline_state_np = self._convert_to_numpy(baseline_state)
                        
                        # 确保两个数组形状相同
                        if current_state_np.shape != baseline_state_np.shape:
                            return f"Failed - Shape mismatch: {current_state_np.shape} vs {baseline_state_np.shape}"
                        
                        # 计算两个状态向量的内积的绝对值
                        fidelity = np.abs(np.vdot(current_state_np, baseline_state_np))
                        # 如果保真度大于0.99，认为结果正确
                        if fidelity > 0.99:
                            return f"Passed (fidelity: {fidelity:.6f})"
                        else:
                            return f"Failed (fidelity: {fidelity:.6f})"
                
                # 如果没有基准结果，只检查状态向量是否有效
                return "Passed (no baseline)"
            else:
                return "Unknown - No state method"
                
        except Exception as e:
            return f"Failed - {str(e)}"
    
    def _convert_to_numpy(self, array):
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


    
    def run_benchmark_for_circuit(self, circuit_name, qasm_file_path):
        """为特定电路运行基准测试"""
        print(f"\n{'='*80}")
        print(f"开始基准测试电路: {circuit_name}")
        print(f"电路文件: {qasm_file_path}")
        print('='*80)
        
        results = {}
        baseline_result = None
        
        # 后端配置
        backend_configs = {
            "numpy": {"backend_name": "numpy", "platform_name": None},
            "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
            "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
            "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
            "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
            "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"},
            "qulacs": {"backend_name": "qulacs", "platform_name": None}
        }
        
        # 首先运行numpy后端作为基准
        if "numpy" in backend_configs:
            config = backend_configs["numpy"]
            result, metrics = self._run_single_backend_benchmark(
                "numpy", config["backend_name"], config["platform_name"], qasm_file_path
            )
            results["numpy"] = metrics
            if result is not None:
                baseline_result = result
        
        # 运行其他后端的基准测试
        for backend_key, config in backend_configs.items():
            if backend_key != "numpy":  # 跳过已经运行的numpy后端
                result, metrics = self._run_single_backend_benchmark(
                    backend_key, config["backend_name"], config["platform_name"], 
                    qasm_file_path, baseline_result
                )
                results[backend_key] = metrics
        
        # 计算加速比
        self._calculate_speedup(results)
        
        return results

    
    def _run_single_backend_benchmark(self, backend_key, backend_name, platform_name, qasm_file_path, baseline_result=None):
        """为单个后端运行基准测试"""
        metrics = QASMBenchMetrics()
        
        try:
            # 设置后端
            if platform_name is not None:
                set_backend(backend_name, platform=platform_name)
            else:
                set_backend(backend_name)
            
            # 记录环境信息
            metrics.environment_info = {
                "CPU": platform.processor(),
                "RAM_GB": psutil.virtual_memory().total / 1024**3,
                "Python": platform.python_version(),
                "Qibo": "0.2.21",
                "Backend": backend_name,
                "Platform": platform_name or "default"
            }
            
            # 加载电路并测量构建时间
            build_start = time.time()
            circuit = self.load_qasm_circuit(qasm_file_path)
            build_end = time.time()
            metrics.circuit_build_time = build_end - build_start
            
            if circuit is None:
                return metrics
            
            # 记录电路参数
            metrics.circuit_parameters = {
                "nqubits": circuit.nqubits,
                "depth": circuit.depth,
                "ngates": circuit.ngates,
                "source": qasm_file_path
            }
            
            # 预热运行（不记录时间）
            print(f"预热运行 {backend_key}...")
            for i in range(self.config.warmup_runs):
                _ = circuit()
            
            # 正式测试运行
            print(f"正式测试运行 {backend_key} ({self.config.num_runs}次)...")
            execution_times = []
            peak_memory = 0
            result = None
            
            for run in range(self.config.num_runs):
                # 测量内存使用前
                memory_before = self.measure_memory_usage()
                
                # 执行电路
                start_time = time.time()
                result = circuit()
                end_time = time.time()
                
                # 测量内存使用后
                memory_after = self.measure_memory_usage()
                peak_memory = max(peak_memory, memory_after - memory_before)
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                print(f"运行 {run+1}/{self.config.num_runs}: {execution_time:.4f}秒")
            
            # 计算统计指标
            metrics.execution_time_mean = np.mean(execution_times)
            metrics.execution_time_std = np.std(execution_times)
            metrics.peak_memory_mb = peak_memory
            
            # 计算吞吐率
            if metrics.execution_time_mean > 0:
                metrics.throughput_gates_per_sec = circuit.ngates / metrics.execution_time_mean
            
            # 验证正确性（传入基准结果）
            metrics.correctness = self.validate_correctness(result, baseline_result)
            
            # 记录后端信息
            metrics.backend_info = {
                "name": backend_name,
                "platform": platform_name
            }
            
            # 记录报告元数据
            metrics.report_metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs
            }
            
            print(f"\n✅ {backend_key} 基准测试完成")
            print(f"   执行时间: {metrics.execution_time_mean:.4f} ± {metrics.execution_time_std:.4f} 秒")
            print(f"   峰值内存: {metrics.peak_memory_mb:.2f} MB")
            print(f"   正确性: {metrics.correctness}")
            
            # 返回结果用于后续验证
            return result, metrics
            
        except Exception as e:
            print(f"❌ {backend_key} 基准测试失败: {str(e)}")
            metrics.correctness = "Failed"
            return None, metrics

    
    def _calculate_speedup(self, results):
        """计算相对于基准后端的加速比"""
        baseline_time = None
        
        # 查找基准后端的执行时间
        for backend_name, metrics in results.items():
            if backend_name == self.config.baseline_backend and metrics.execution_time_mean:
                baseline_time = metrics.execution_time_mean
                break
        
        if baseline_time:
            for backend_name, metrics in results.items():
                if metrics.execution_time_mean and backend_name != self.config.baseline_backend:
                    metrics.speedup = baseline_time / metrics.execution_time_mean
    
    def generate_reports(self, results, circuit_name, circuit=None):
        """生成所有配置的报告格式"""
        reporter = QASMBenchReporter()
        
        if 'csv' in self.config.output_formats:
            reporter.generate_csv_report(results, circuit_name)
        
        if 'markdown' in self.config.output_formats:
            reporter.generate_markdown_report(results, circuit_name)
        
        if 'json' in self.config.output_formats:
            reporter.generate_json_report(results, circuit_name)
        # 如果提供了电路对象，则保存电路图
        if circuit is not None:
            reporter.save_circuit_diagram(circuit, circuit_name)

def list_available_circuits():
    """列出所有可用的QASMBench电路"""
    config = QASMBenchConfig()
    runner = QASMBenchRunner(config)
    circuits = runner.discover_qasm_circuits()
    
    print("可用的QASMBench电路:")
    print("="*80)
    
    circuits_by_size = {}
    for circuit_name, info in circuits.items():
        size = info['size']
        if size not in circuits_by_size:
            circuits_by_size[size] = []
        circuits_by_size[size].append((circuit_name, info))
    
    for size in ['small', 'medium', 'large']:
        if size in circuits_by_size:
            print(f"\n{size.upper()} 规模电路 ({len(circuits_by_size[size])}个):")
            for circuit_name, info in sorted(circuits_by_size[size]):
                # 显示transpiled文件信息
                transpiled_files = [f for f in info['available_files'] if 'transpiled' in f]
                if transpiled_files:
                    print(f"  - {circuit_name} (推荐使用transpiled版本)")
                else:
                    print(f"  - {circuit_name} (使用原始版本)")
    
    return circuits

def find_circuit_by_name(circuit_name):
    """根据电路名称查找对应的电路文件"""
    config = QASMBenchConfig()
    runner = QASMBenchRunner(config)
    circuits = runner.discover_qasm_circuits()
    
    # 尝试精确匹配
    if circuit_name in circuits:
        info = circuits[circuit_name]
        # 强制使用transpiled版本，避免Qibo后端报错
        transpiled_files = [f for f in info['available_files'] if 'transpiled' in f]
        if transpiled_files:
            target_file = transpiled_files[0]
            print(f"✅ 找到transpiled电路文件: {target_file}")
        else:
            # 如果没有transpiled版本，尝试查找其他可用文件
            if info['available_files']:
                target_file = info['available_files'][0]
                print(f"⚠️ 警告: 未找到transpiled版本，使用原始文件: {target_file}")
            else:
                print(f"❌ 错误: 电路目录中没有可用的QASM文件")
                return None
        
        return os.path.join(info['full_path'], target_file)
    
    # 尝试部分匹配（只使用电路目录名）
    circuit_dir_name = circuit_name.split('/')[-1] if '/' in circuit_name else circuit_name
    
    for full_name, info in circuits.items():
        if circuit_dir_name == info['name']:
            # 强制使用transpiled版本，避免Qibo后端报错
            transpiled_files = [f for f in info['available_files'] if 'transpiled' in f]
            if transpiled_files:
                target_file = transpiled_files[0]
                print(f"✅ 找到transpiled电路文件: {target_file}")
            else:
                # 如果没有transpiled版本，尝试查找其他可用文件
                if info['available_files']:
                    target_file = info['available_files'][0]
                    print(f"⚠️ 警告: 未找到transpiled版本，使用原始文件: {target_file}")
                else:
                    print(f"❌ 错误: 电路目录中没有可用的QASM文件")
                    return None
            
            return os.path.join(info['full_path'], target_file)
    
    # 如果还是找不到，尝试大小写不敏感匹配
    for full_name, info in circuits.items():
        if circuit_dir_name.lower() == info['name'].lower():
            # 强制使用transpiled版本，避免Qibo后端报错
            transpiled_files = [f for f in info['available_files'] if 'transpiled' in f]
            if transpiled_files:
                target_file = transpiled_files[0]
                print(f"✅ 找到transpiled电路文件: {target_file}")
            else:
                # 如果没有transpiled版本，尝试查找其他可用文件
                if info['available_files']:
                    target_file = info['available_files'][0]
                    print(f"⚠️ 警告: 未找到transpiled版本，使用原始文件: {target_file}")
                else:
                    print(f"❌ 错误: 电路目录中没有可用的QASM文件")
                    return None
            
            return os.path.join(info['full_path'], target_file)
    
    print(f"❌ 错误: 未找到电路 '{circuit_name}'")
    print("💡 提示: 使用 --list 参数查看所有可用电路")
    return None

def run_benchmark_for_circuit(circuit_path):
    """为指定电路路径运行基准测试"""
    if not os.path.exists(circuit_path):
        print(f"错误: 电路文件不存在: {circuit_path}")
        return None
    
    # 从路径中提取电路名称
    circuit_name = os.path.basename(circuit_path).replace('.qasm', '')
    
    config = QASMBenchConfig()
    runner = QASMBenchRunner(config)
    
    print(f"🚀 开始QASMBench基准测试: {circuit_name}")
    print(f"电路文件: {circuit_path}")
    print('='*80)
    
    circuit = runner.load_qasm_circuit(circuit_path)
    if circuit is None:
        print(f"错误: 无法加载电路 {circuit_name}")
        return None
    
    results = runner.run_benchmark_for_circuit(circuit_name, circuit_path)
    
    # 生成报告
    runner.generate_reports(results, circuit_name, circuit)
    
    # 打印简要总结
    print("\n" + "="*80)
    print("📊 基准测试总结")
    print("="*80)
    
    successful_tests = {k: v for k, v in results.items() if v.execution_time_mean is not None}
    
    if successful_tests:
        print("成功测试的后端 (按执行时间排序):")
        sorted_results = sorted(successful_tests.items(), 
                               key=lambda x: x[1].execution_time_mean)
        
        for i, (backend_name, metrics) in enumerate(sorted_results, 1):
            speedup_str = f" ({metrics.speedup:.2f}x)" if metrics.speedup else ""
            print(f"{i}. {backend_name}: {metrics.execution_time_mean:.4f}秒{speedup_str}")
    
    print(f"\n报告文件已生成:")
    for fmt in config.output_formats:
        print(f"  - {circuit_name}_benchmark_report.{fmt}")
    
    print("\n🎯 基准测试完成!")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QASMBench电路基准测试工具')
    parser.add_argument('--list', action='store_true', help='列出所有可用电路')
    parser.add_argument('--circuit', type=str, help='指定QASM电路文件的完整路径进行基准测试')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_circuits()
    elif args.circuit:
        run_benchmark_for_circuit(args.circuit)

    else:
        print("使用方法:")
        print("  python qasmbench_runner.py --list                    # 列出所有电路")
        print("  python qasmbench_runner.py --circuit <文件路径>      # 测试指定电路")

        print("\n示例:")
        print("  python qasmbench_runner.py --list")

        print("  python qasmbench_runner.py --circuit QASMBench/medium/qft_n18/qft_n18.qasm")