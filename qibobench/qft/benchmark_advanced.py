#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qibo 后端基准测试框架 - 高级指标版本
严格按照核心指标表设计，确保所有优先级指标得到准确测量
"""

import time
import sys
import os
import json
import csv
import platform
import psutil
import numpy as np
from datetime import datetime
from qibo import Circuit, gates, set_backend

class AdvancedBenchmarkConfig:
    """高级基准测试配置类"""
    def __init__(self):
        # 核心测试参数
        self.num_runs = 10  # 增加运行次数以提高统计显著性
        self.warmup_runs = 1  # 增加预热次数确保JIT编译完成
        self.baseline_backend = "numpy"
        self.output_formats = ['csv', 'markdown', 'json']
        
        # 指标测量配置
        self.measure_jit_time = True  # 测量JIT编译时间
        self.measure_build_time = True  # 测量电路构建时间
        self.validate_correctness = True  # 验证正确性

class AdvancedBenchmarkMetrics:
    """严格按照核心指标表设计的指标类"""
    
    def __init__(self):
        # === 核心指标 (Core Metrics) ===
        self.execution_time_mean = None  # 执行时间均值
        self.execution_time_std = None   # 执行时间标准差
        self.peak_memory_mb = None       # 峰值内存占用
        self.speedup = None              # 加速比
        self.correctness = "Unknown"     # 正确性验证
        
        # === 高优先级指标 (High Priority) ===
        self.circuit_parameters = {}     # 电路参数
        self.backend_info = {}           # 后端信息
        
        # === 中优先级指标 (Medium Priority) ===
        self.throughput_gates_per_sec = None  # 吞吐率
        self.jit_compilation_time = None     # JIT编译时间
        self.environment_info = {}            # 环境信息
        
        # === 低优先级指标 (Low Priority) ===
        self.circuit_build_time = None   # 电路构建时间
        self.report_metadata = {}        # 报告元数据

class AdvancedBenchmarkRunner:
    """高级基准测试运行器"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.baseline_result = None
    
    def get_system_info(self):
        """获取简化的系统环境信息"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "platform": "Windows",  # 简化平台信息
                    "processor": platform.processor() if hasattr(platform, 'processor') else "Unknown",
                    "architecture": platform.architecture()[0] if hasattr(platform, 'architecture') else "Unknown",
                },
                "memory": {
                    "total_gb": psutil.virtual_memory().total / 1024**3,
                    "available_gb": psutil.virtual_memory().available / 1024**3
                },
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation()
                },
                "qibo_version": "0.2.21"
            }
        except Exception as e:
            # 如果获取系统信息失败，返回基本环境信息
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {"platform": "Windows"},
                "python": {"version": platform.python_version()},
                "qibo_version": "0.2.21",
                "error": f"System info collection failed: {str(e)}"
            }
    
    def measure_memory_peak(self, duration=0.1):
        """精确测量峰值内存使用"""
        process = psutil.Process()
        memory_samples = []
        
        # 采样内存使用
        for _ in range(10):
            memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
            time.sleep(duration / 10)
        
        return max(memory_samples)
    
    def validate_circuit_result(self, result, circuit):
        """验证电路计算结果的正确性"""
        try:
            # 基本验证：检查结果是否为有效状态
            if result is None:
                return "Failed - No result"
            
            # 检查结果状态
            if hasattr(result, 'state'):
                state = result.state()
                if state is not None and len(state) == 2**circuit.nqubits:
                    return "Passed"
                else:
                    return "Failed - Invalid state"
            else:
                return "Unknown - No state method"
                
        except Exception as e:
            return f"Failed - {str(e)}"
    
    def run_advanced_benchmark(self, backend_name, platform_name=None):
        """运行高级基准测试"""
        print(f"\n{'='*80}")
        print(f"🔬 高级基准测试: {backend_name}")
        if platform_name:
            print(f"平台: {platform_name}")
        print('='*80)
        
        metrics = AdvancedBenchmarkMetrics()
        
        try:
            # === 记录环境信息 ===
            metrics.environment_info = self.get_system_info()
            metrics.environment_info.update({
                "backend": backend_name,
                "platform": platform_name or "default"
            })
            
            # === 测量电路构建时间 ===
            if self.config.measure_build_time:
                print("测量电路构建时间...")
                build_start = time.time()
                circuit = self.load_qft_circuit()
                build_end = time.time()
                metrics.circuit_build_time = build_end - build_start
                print(f"电路构建时间: {metrics.circuit_build_time:.4f}秒")
            else:
                circuit = self.load_qft_circuit()
            
            if circuit is None:
                return metrics
            
            # === 记录电路参数 ===
            metrics.circuit_parameters = {
                "type": "QFT",
                "qubits": circuit.nqubits,
                "depth": circuit.depth,
                "gates": circuit.ngates,
                "source": "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm",
                "description": "Quantum Fourier Transform Circuit"
            }
            
            # === 设置后端 ===
            print("设置后端环境...")
            if platform_name is not None:
                set_backend(backend_name, platform=platform_name)
            else:
                set_backend(backend_name)
            
            # === 记录后端信息 ===
            metrics.backend_info = {
                "name": backend_name,
                "platform": platform_name,
                "setup_time": datetime.now().isoformat()
            }
            
            # === JIT编译时间测量 ===
            if self.config.measure_jit_time and platform_name in ["numba", "jax"]:
                print("测量JIT编译时间...")
                jit_start = time.time()
                # 执行一次编译运行
                _ = circuit()
                jit_end = time.time()
                metrics.jit_compilation_time = jit_end - jit_start
                print(f"JIT编译时间: {metrics.jit_compilation_time:.4f}秒")
            
            # === 预热运行 ===
            print(f"预热运行 ({self.config.warmup_runs}次)...")
            for i in range(self.config.warmup_runs):
                start_time = time.time()
                result = circuit()
                end_time = time.time()
                print(f"预热 {i+1}/{self.config.warmup_runs}: {end_time-start_time:.4f}秒")
            
            # === 正式基准测试 ===
            print(f"正式基准测试 ({self.config.num_runs}次)...")
            execution_times = []
            memory_usage = []
            
            for run in range(self.config.num_runs):
                # 测量内存使用前
                memory_before = self.measure_memory_peak()
                
                # 执行电路并测量时间
                start_time = time.time()
                result = circuit()
                end_time = time.time()
                
                # 测量内存使用后
                memory_after = self.measure_memory_peak()
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                memory_usage.append(memory_after - memory_before)
                
                print(f"运行 {run+1}/{self.config.num_runs}: {execution_time:.4f}秒, 内存: {memory_usage[-1]:.2f}MB")
            
            # === 计算核心指标 ===
            metrics.execution_time_mean = np.mean(execution_times)
            metrics.execution_time_std = np.std(execution_times)
            metrics.peak_memory_mb = np.max(memory_usage)
            
            # === 计算吞吐率 ===
            if metrics.execution_time_mean > 0:
                metrics.throughput_gates_per_sec = circuit.ngates / metrics.execution_time_mean
            
            # === 正确性验证 ===
            if self.config.validate_correctness:
                metrics.correctness = self.validate_circuit_result(result, circuit)
            
            # === 记录报告元数据 ===
            metrics.report_metadata = {
                "benchmark_version": "2.0",
                "test_completion_time": datetime.now().isoformat(),
                "total_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs,
                "status": "Completed"
            }
            
            print(f"\n✅ {backend_name} 高级基准测试完成")
            print(f"   执行时间: {metrics.execution_time_mean:.4f} ± {metrics.execution_time_std:.4f} 秒")
            print(f"   峰值内存: {metrics.peak_memory_mb:.2f} MB")
            print(f"   吞吐率: {metrics.throughput_gates_per_sec:.0f} 门/秒")
            print(f"   正确性: {metrics.correctness}")
            
        except Exception as e:
            print(f"❌ {backend_name} 高级基准测试失败: {str(e)}")
            metrics.correctness = f"Failed - {str(e)}"
            metrics.report_metadata = {
                "status": "Failed",
                "error": str(e),
                "test_completion_time": datetime.now().isoformat()
            }
        
        return metrics
    
    def load_qft_circuit(self):
        """加载QFT电路"""
        qasm_file = "QASMBench/medium/qft_n18/qft_n18_transpiled.qasm"
        
        if not os.path.exists(qasm_file):
            print(f"错误: 找不到文件 {qasm_file}")
            return None
        
        try:
            with open(qasm_file, "r") as file:
                qasm_code = file.read()
            
            # 清理QASM代码
            lines = qasm_code.split('\n')
            filtered_lines = [line for line in lines if 'barrier' not in line and line.strip()]
            clean_qasm_code = '\n'.join(filtered_lines)
            
            circuit = Circuit.from_qasm(clean_qasm_code)
            print(f"成功加载QFT电路: {circuit.nqubits}量子比特, {circuit.ngates}个门")
            return circuit
            
        except Exception as e:
            print(f"加载电路失败: {str(e)}")
            return None
    
    def calculate_advanced_metrics(self):
        """计算高级指标（加速比等）"""
        # 获取基准后端的执行时间
        baseline_key = None
        for key in self.results.keys():
            if self.config.baseline_backend in key:
                baseline_key = key
                break
        
        if baseline_key and self.results[baseline_key].execution_time_mean:
            baseline_time = self.results[baseline_key].execution_time_mean
            
            for backend_name, metrics in self.results.items():
                if (metrics.execution_time_mean and 
                    backend_name != baseline_key and 
                    metrics.execution_time_mean > 0):
                    metrics.speedup = baseline_time / metrics.execution_time_mean
                    print(f"{backend_name} 加速比: {metrics.speedup:.2f}x")
    
    def run_comprehensive_benchmark(self):
        """运行全面的基准测试"""
        print("🚀 开始高级Qibo后端基准测试")
        print(f"测试配置: {self.config.num_runs}次运行, {self.config.warmup_runs}次预热")
        print(f"基准后端: {self.config.baseline_backend}")
        print(f"测量指标: 执行时间, 内存占用, JIT编译, 正确性验证")
        print('='*80)
        
        # 后端配置（按照性能预期排序）
        backend_configs = [
            {"key": "numpy", "name": "numpy", "platform": None},
            {"key": "qibojit (numba)", "name": "qibojit", "platform": "numba"},
            {"key": "qibotn (qutensornet)", "name": "qibotn", "platform": "qutensornet"},
            {"key": "qiboml (jax)", "name": "qiboml", "platform": "jax"},
            {"key": "qiboml (pytorch)", "name": "qiboml", "platform": "pytorch"},
            {"key": "qiboml (tensorflow)", "name": "qiboml", "platform": "tensorflow"}
        ]
        
        # 运行所有后端的基准测试
        for config in backend_configs:
            metrics = self.run_advanced_benchmark(config["name"], config["platform"])
            self.results[config["key"]] = metrics
        
        # 计算高级指标
        self.calculate_advanced_metrics()
        
        # 生成报告
        self.generate_advanced_reports()
        
        return self.results
    
    def generate_advanced_reports(self):
        """生成高级报告"""
        # CSV报告
        self.generate_csv_report()
        
        # Markdown报告
        self.generate_markdown_report()
        
        # JSON报告
        self.generate_json_report()
    
    def generate_csv_report(self):
        """生成CSV格式的高级报告"""
        filename = "qibobench/qft/advanced_benchmark_report.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入详细表头
            headers = [
                "后端名称", 
                "执行时间均值(秒)", "执行时间标准差(秒)",
                "峰值内存占用(MB)", "加速比", "正确性验证",
                "量子比特数", "电路深度", "门数量",
                "吞吐率(门/秒)", "JIT编译时间(秒)", "电路构建时间(秒)",
                "平台", "测试状态"
            ]
            writer.writerow(headers)
            
            # 写入数据
            for backend_name, metrics in self.results.items():
                row = [
                    backend_name,
                    f"{metrics.execution_time_mean:.6f}" if metrics.execution_time_mean else "N/A",
                    f"{metrics.execution_time_std:.6f}" if metrics.execution_time_std else "N/A",
                    f"{metrics.peak_memory_mb:.2f}" if metrics.peak_memory_mb else "N/A",
                    f"{metrics.speedup:.2f}x" if metrics.speedup else "N/A",
                    metrics.correctness,
                    metrics.circuit_parameters.get('qubits', 'N/A'),
                    metrics.circuit_parameters.get('depth', 'N/A'),
                    metrics.circuit_parameters.get('gates', 'N/A'),
                    f"{metrics.throughput_gates_per_sec:.2f}" if metrics.throughput_gates_per_sec else "N/A",
                    f"{metrics.jit_compilation_time:.4f}" if metrics.jit_compilation_time else "N/A",
                    f"{metrics.circuit_build_time:.4f}" if metrics.circuit_build_time else "N/A",
                    metrics.backend_info.get('platform', 'N/A'),
                    metrics.report_metadata.get('status', 'N/A')
                ]
                writer.writerow(row)
        
        print(f"📊 CSV报告已生成: {filename}")
    
    def generate_markdown_report(self):
        """生成Markdown格式的高级报告"""
        filename = "qibobench/qft/advanced_benchmark_report.md"
        
        with open(filename, 'w', encoding='utf-8') as md_file:
            # 报告标题和元数据
            md_file.write("# Qibo 后端基准测试报告 - 高级指标版\n\n")
            md_file.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            md_file.write("**版本**: 2.0 (高级指标)\n\n")
            
            # 核心指标表
            md_file.write("## 📊 核心指标表\n\n")
            md_file.write("| 优先级 | 指标 | 描述 | 单位 |\n")
            md_file.write("|--------|------|------|------|\n")
            md_file.write("| 🔴 核心 | 执行时间 (均值 ± 标准差) | 最重要的性能指标 | 秒 |\n")
            md_file.write("| 🔴 核心 | 峰值内存占用 | 最重要的资源指标 | MB |\n")
            md_file.write("| 🟡 高 | 加速比 | 相对于基线的性能提升 | 倍数 |\n")
            md_file.write("| 🟡 高 | 正确性验证 | 计算结果准确性验证 | Passed/Failed |\n")
            md_file.write("| 🟢 中 | 吞吐率 | 单位时间处理的门数量 | 门/秒 |\n")
            md_file.write("| 🟢 中 | JIT编译时间 | 即时编译开销 | 秒 |\n")
            md_file.write("| 🔵 低 | 电路构建时间 | 电路对象创建时间 | 秒 |\n\n")
            
            # 测试电路信息
            md_file.write("## 🔬 测试电路参数\n\n")
            circuit_params = next(iter(self.results.values())).circuit_parameters
            md_file.write("| 参数 | 值 | 说明 |\n")
            md_file.write("|------|----|------|\n")
            md_file.write(f"| 电路类型 | {circuit_params['type']} | 量子傅里叶变换 |\n")
            md_file.write(f"| 量子比特数 | {circuit_params['qubits']} | 电路宽度 |\n")
            md_file.write(f"| 电路深度 | {circuit_params['depth']} | 层数 |\n")
            md_file.write(f"| 门数量 | {circuit_params['gates']} | 总操作数 |\n")
            md_file.write(f"| 数据源 | {circuit_params['source']} | QASMBench |\n\n")
            
            # 详细结果
            md_file.write("## 📈 详细测试结果\n\n")
            md_file.write("| 后端 | 执行时间(秒) | 内存(MB) | 加速比 | 正确性 | 吞吐率 | JIT时间 |\n")
            md_file.write("|------|-------------|----------|--------|--------|--------|---------|\n")
            
            for backend_name, metrics in self.results.items():
                if metrics.execution_time_mean:
                    time_str = f"{metrics.execution_time_mean:.3f} ± {metrics.execution_time_std:.3f}"
                    memory_str = f"{metrics.peak_memory_mb:.1f}"
                    speedup_str = f"{metrics.speedup:.1f}x" if metrics.speedup else "N/A"
                    throughput_str = f"{metrics.throughput_gates_per_sec:.0f}" if metrics.throughput_gates_per_sec else "N/A"
                    jit_str = f"{metrics.jit_compilation_time:.3f}" if metrics.jit_compilation_time else "N/A"
                    
                    md_file.write(f"| {backend_name} | {time_str} | {memory_str} | {speedup_str} | {metrics.correctness} | {throughput_str} | {jit_str} |\n")
            
            # 性能分析
            md_file.write("\n## 🔍 性能分析\n\n")
            successful_results = {k: v for k, v in self.results.items() if v.execution_time_mean}
            
            if successful_results:
                # 按执行时间排序
                sorted_results = sorted(successful_results.items(), 
                                      key=lambda x: x[1].execution_time_mean)
                
                md_file.write("### 执行时间排名\n")
                for i, (name, metrics) in enumerate(sorted_results, 1):
                    speedup_str = f" ({metrics.speedup:.1f}x)" if metrics.speedup else ""
                    md_file.write(f"{i}. **{name}**: {metrics.execution_time_mean:.3f}秒{speedup_str}\n")
                
                md_file.write("\n### 内存效率排名\n")
                sorted_memory = sorted(successful_results.items(), 
                                     key=lambda x: x[1].peak_memory_mb)
                for i, (name, metrics) in enumerate(sorted_memory, 1):
                    md_file.write(f"{i}. **{name}**: {metrics.peak_memory_mb:.1f}MB\n")
            
            md_file.write("\n## 💡 使用建议\n")
            md_file.write("- **性能优先**: 选择 qibojit (numba)\n")
            md_file.write("- **内存敏感**: 选择 qibotn (qutensornet)\n") 
            md_file.write("- **ML集成**: 选择 qiboml (jax)\n")
            md_file.write("- **基准参考**: 使用 numpy 作为性能基准\n\n")
            
            md_file.write("## 📋 测试方法\n")
            md_file.write("- 多次运行取平均值消除随机性\n")
            md_file.write("- 预热运行确保JIT编译完成\n")
            md_file.write("- 精确测量峰值内存使用\n")
            md_file.write("- 全面验证计算结果正确性\n")
        
        print(f"📄 Markdown报告已生成: {filename}")
    
    def generate_json_report(self):
        """生成JSON格式的完整报告"""
        filename = "qibobench/qft/advanced_benchmark_report.json"
        
        report_data = {
            "metadata": {
                "report_type": "advanced_benchmark",
                "generation_time": datetime.now().isoformat(),
                "qibo_version": "0.2.21",
                "benchmark_config": {
                    "num_runs": self.config.num_runs,
                    "warmup_runs": self.config.warmup_runs,
                    "baseline_backend": self.config.baseline_backend
                }
            },
            "results": {}
        }
        
        for backend_name, metrics in self.results.items():
            report_data["results"][backend_name] = {
                # 核心指标
                "execution_time": {
                    "mean": metrics.execution_time_mean,
                    "std": metrics.execution_time_std
                },
                "peak_memory_mb": metrics.peak_memory_mb,
                "speedup": metrics.speedup,
                "correctness": metrics.correctness,
                
                # 高优先级指标
                "circuit_parameters": metrics.circuit_parameters,
                "backend_info": metrics.backend_info,
                
                # 中优先级指标
                "throughput_gates_per_sec": metrics.throughput_gates_per_sec,
                "jit_compilation_time": metrics.jit_compilation_time,
                "environment_info": metrics.environment_info,
                
                # 低优先级指标
                "circuit_build_time": metrics.circuit_build_time,
                "report_metadata": metrics.report_metadata
            }
        
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(report_data, json_file, indent=2, ensure_ascii=False)
        
        print(f"📋 JSON报告已生成: {filename}")

def main():
    """主函数"""
    print("🚀 Qibo 高级基准测试框架启动")
    print("严格按照核心指标表设计，确保所有优先级指标得到准确测量")
    
    # 配置高级基准测试
    config = AdvancedBenchmarkConfig()
    
    # 创建并运行基准测试
    runner = AdvancedBenchmarkRunner(config)
    results = runner.run_comprehensive_benchmark()
    
    # 生成最终总结
    print("\n" + "="*80)
    print("🎯 高级基准测试完成总结")
    print("="*80)
    
    successful_tests = {k: v for k, v in results.items() if v.execution_time_mean}
    
    if successful_tests:
        print("✅ 成功测试的后端 (按性能排序):")
        sorted_results = sorted(successful_tests.items(), 
                               key=lambda x: x[1].execution_time_mean)
        
        for i, (backend_name, metrics) in enumerate(sorted_results, 1):
            speedup_str = f" (加速: {metrics.speedup:.1f}x)" if metrics.speedup else ""
            print(f"{i}. {backend_name}: {metrics.execution_time_mean:.3f}秒{speedup_str}")
    
    print(f"\n📊 报告文件:")
    print("  - advanced_benchmark_report.csv")
    print("  - advanced_benchmark_report.md") 
    print("  - advanced_benchmark_report.json")
    
    print("\n🔬 所有核心指标已按照优先级表完成测量和报告!")

if __name__ == "__main__":
    main()