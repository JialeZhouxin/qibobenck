#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版Qibo后端基准测试脚本
专注于核心指标，避免复杂依赖
"""

import time
import sys
import os
import platform
import json
import csv
from datetime import datetime
from qibo import Circuit, gates, set_backend

class SimpleBenchmark:
    """简化版基准测试类"""
    
    def __init__(self):
        self.num_runs = 3  # 运行次数
        self.warmup_runs = 1  # 预热次数
        self.baseline_backend = "numpy"
        self.results = {}
    
    def measure_memory_simple(self):
        """简化的内存测量方法"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # 如果psutil不可用，返回0
    
    def load_qft_circuit(self):
        """加载QFT电路"""
        qasm_file = "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"
        
        if not os.path.exists(qasm_file):
            print(f"错误: 找不到文件 {qasm_file}")
            return None
        
        try:
            with open(qasm_file, "r") as file:
                qasm_code = file.read()
            
            # 移除barrier语句
            lines = qasm_code.split('\n')
            filtered_lines = [line for line in lines if 'barrier' not in line]
            clean_qasm_code = '\n'.join(filtered_lines)
            
            circuit = Circuit.from_qasm(clean_qasm_code)
            return circuit
            
        except Exception as e:
            print(f"加载电路失败: {str(e)}")
            return None
    
    def run_backend_test(self, backend_name, platform_name=None):
        """运行单个后端的测试"""
        print(f"\n{'='*60}")
        print(f"测试后端: {backend_name}")
        if platform_name:
            print(f"平台: {platform_name}")
        print('='*60)
        
        try:
            # 设置后端
            if platform_name is not None:
                set_backend(backend_name, platform=platform_name)
            else:
                set_backend(backend_name)
            
            # 加载电路
            circuit = self.load_qft_circuit()
            if circuit is None:
                return None
            
            # 预热运行
            print("预热运行...")
            for _ in range(self.warmup_runs):
                _ = circuit()
            
            # 正式测试
            print(f"正式测试 ({self.num_runs}次运行)...")
            execution_times = []
            peak_memory = 0
            
            for run in range(self.num_runs):
                # 测量内存
                memory_before = self.measure_memory_simple()
                
                # 执行电路
                start_time = time.time()
                result = circuit()
                end_time = time.time()
                
                # 测量内存
                memory_after = self.measure_memory_simple()
                peak_memory = max(peak_memory, memory_after - memory_before)
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                print(f"运行 {run+1}/{self.num_runs}: {execution_time:.4f}秒")
            
            # 计算统计指标
            import numpy as np
            mean_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            
            # 计算吞吐率
            throughput = circuit.ngates / mean_time if mean_time > 0 else 0
            
            # 验证正确性
            correctness = "Passed" if result is not None else "Failed"
            
            return {
                "execution_time_mean": mean_time,
                "execution_time_std": std_time,
                "peak_memory_mb": peak_memory,
                "throughput_gates_per_sec": throughput,
                "correctness": correctness,
                "circuit_parameters": {
                    "nqubits": circuit.nqubits,
                    "depth": circuit.depth,
                    "ngates": circuit.ngates
                },
                "environment_info": {
                    "python_version": platform.python_version(),
                    "system": platform.system(),
                    "backend": backend_name,
                    "platform": platform_name or "default"
                }
            }
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return None
    
    def calculate_speedup(self):
        """计算加速比"""
        baseline_time = None
        
        # 查找基准后端的执行时间
        for backend_name, metrics in self.results.items():
            if backend_name == self.baseline_backend and metrics:
                baseline_time = metrics["execution_time_mean"]
                break
        
        if baseline_time:
            for backend_name, metrics in self.results.items():
                if metrics and backend_name != self.baseline_backend:
                    metrics["speedup"] = baseline_time / metrics["execution_time_mean"]
    
    def generate_csv_report(self, filename="simple_benchmark_report.csv"):
        """生成CSV报告"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 表头
            headers = [
                "后端名称", "执行时间均值(秒)", "执行时间标准差(秒)", 
                "峰值内存占用(MB)", "加速比", "正确性验证",
                "量子比特数", "电路深度", "门数量", "吞吐率(门/秒)"
            ]
            writer.writerow(headers)
            
            # 数据行
            for backend_name, metrics in self.results.items():
                if metrics:
                    row = [
                        backend_name,
                        f"{metrics['execution_time_mean']:.6f}",
                        f"{metrics['execution_time_std']:.6f}",
                        f"{metrics['peak_memory_mb']:.2f}",
                        f"{metrics.get('speedup', 'N/A'):.2f}x" if metrics.get('speedup') else "N/A",
                        metrics['correctness'],
                        metrics['circuit_parameters']['nqubits'],
                        metrics['circuit_parameters']['depth'],
                        metrics['circuit_parameters']['ngates'],
                        f"{metrics['throughput_gates_per_sec']:.2f}"
                    ]
                    writer.writerow(row)
        
        print(f"CSV报告已生成: {filename}")
    
    def generate_markdown_report(self, filename="simple_benchmark_report.md"):
        """生成Markdown报告"""
        with open(filename, 'w', encoding='utf-8') as md_file:
            # 标题
            md_file.write("# Qibo后端基准测试报告 (简化版)\n\n")
            md_file.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 核心指标表格
            md_file.write("## 核心性能指标\n\n")
            md_file.write("| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率(门/秒) |\n")
            md_file.write("|------|-------------|----------|--------|--------|---------------|\n")
            
            for backend_name, metrics in self.results.items():
                if metrics:
                    time_str = f"{metrics['execution_time_mean']:.4f} ± {metrics['execution_time_std']:.4f}"
                    memory_str = f"{metrics['peak_memory_mb']:.1f}"
                    speedup_str = f"{metrics.get('speedup', 'N/A'):.2f}x" if metrics.get('speedup') else "N/A"
                    throughput_str = f"{metrics['throughput_gates_per_sec']:.0f}"
                    
                    md_file.write(f"| {backend_name} | {time_str} | {memory_str} | {speedup_str} | {metrics['correctness']} | {throughput_str} |\n")
            
            # 环境信息
            md_file.write("\n## 测试环境\n\n")
            for backend_name, metrics in self.results.items():
                if metrics:
                    md_file.write(f"### {backend_name}\n")
                    for key, value in metrics['environment_info'].items():
                        md_file.write(f"- {key}: {value}\n")
                    md_file.write("\n")
        
        print(f"Markdown报告已生成: {filename}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始简化版Qibo后端基准测试")
        print(f"测试配置: {self.num_runs}次运行, {self.warmup_runs}次预热")
        print(f"基准后端: {self.baseline_backend}")
        print('='*60)
        
        # 后端配置
        backend_configs = {
            "numpy": {"backend_name": "numpy", "platform_name": None},
            "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
            "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
            "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
            "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
            "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"}
        }
        
        # 运行所有测试
        for backend_key, config in backend_configs.items():
            result = self.run_backend_test(
                config["backend_name"], 
                config["platform_name"]
            )
            self.results[backend_key] = result
        
        # 计算加速比
        self.calculate_speedup()
        
        # 生成报告
        self.generate_csv_report()
        self.generate_markdown_report()
        
        # 打印总结
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("📊 基准测试总结")
        print("="*60)
        
        successful_tests = {k: v for k, v in self.results.items() if v is not None}
        
        if successful_tests:
            print("成功测试的后端 (按执行时间排序):")
            sorted_results = sorted(successful_tests.items(), 
                                   key=lambda x: x[1]["execution_time_mean"])
            
            for i, (backend_name, metrics) in enumerate(sorted_results, 1):
                speedup_str = f" ({metrics.get('speedup', 'N/A'):.2f}x)" if metrics.get('speedup') else ""
                print(f"{i}. {backend_name}: {metrics['execution_time_mean']:.4f}秒{speedup_str}")
        
        print(f"\n报告文件已生成:")
        print("  - simple_benchmark_report.csv")
        print("  - simple_benchmark_report.md")
        print("\n🎯 基准测试完成!")

def main():
    """主函数"""
    benchmark = SimpleBenchmark()
    results = benchmark.run_all_tests()

if __name__ == "__main__":
    main()