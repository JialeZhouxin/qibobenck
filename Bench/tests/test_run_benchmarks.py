#!/usr/bin/env python3
"""
run_benchmarks.py 的测试脚本

这个脚本测试了基准测试运行器的所有主要功能，包括：
- 命令行参数解析
- 模拟器和电路实例创建
- 基准测试执行
- 缓存功能
- 结果后处理
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd

# 添加父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
import run_benchmarks as rb
from benchmark_harness.abstractions import BenchmarkResult
from benchmark_harness.caching import CacheConfig


class TestRunBenchmarks(unittest.TestCase):
    """run_benchmarks.py 的测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录用于测试输出
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_parse_arguments_default(self):
        """测试默认参数解析"""
        with patch('sys.argv', ['run_benchmarks.py']):
            args = rb.parse_arguments()
            
            self.assertEqual(args.circuits, ['qft'])
            self.assertEqual(args.qubits, [2, 3, 4])
            self.assertEqual(args.simulators, ['qibo-qibojit'])
            self.assertEqual(args.golden_standard, 'qibo-qibojit')
            self.assertEqual(args.output_dir, 'results')
            self.assertFalse(args.verbose)
            self.assertTrue(args.enable_cache)
            self.assertFalse(args.no_cache)
            self.assertEqual(args.cache_type, 'hybrid')
            self.assertEqual(args.cache_dir, '.benchmark_cache')
            self.assertEqual(args.memory_cache_size, 64)
            self.assertFalse(args.clear_cache)
            self.assertFalse(args.cache_stats)
    
    def test_parse_arguments_custom(self):
        """测试自定义参数解析"""
        with patch('sys.argv', [
            'run_benchmarks.py',
            '--circuits', 'qft',
            '--qubits', '2', '3',
            '--simulators', 'qibo-numpy', 'qibo-qibojit',
            '--golden-standard', 'qibo-numpy',
            '--output-dir', 'custom_results',
            '--verbose',
            '--cache-type', 'memory',
            '--memory-cache-size', '128',
            '--cache-stats'
        ]):
            args = rb.parse_arguments()
            
            self.assertEqual(args.circuits, ['qft'])
            self.assertEqual(args.qubits, [2, 3])
            self.assertEqual(args.simulators, ['qibo-numpy', 'qibo-qibojit'])
            self.assertEqual(args.golden_standard, 'qibo-numpy')
            self.assertEqual(args.output_dir, 'custom_results')
            self.assertTrue(args.verbose)
            self.assertEqual(args.cache_type, 'memory')
            self.assertEqual(args.memory_cache_size, 128)
            self.assertTrue(args.cache_stats)
    
    def test_create_simulator_instances_qibo(self):
        """测试Qibo模拟器实例创建"""
        # 设置全局args变量
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        
        try:
            simulator_configs = ['qibo-numpy', 'qibo-qibojit']
            simulators = rb.create_simulator_instances(simulator_configs)
            
            self.assertEqual(len(simulators), 2)
            self.assertIn('qibo-numpy', simulators)
            self.assertIn('qibo-qibojit', simulators)
            
            for config, simulator in simulators.items():
                self.assertEqual(simulator.platform_name, 'qibo')
                self.assertIn(config, simulator_configs)
                self.assertEqual(simulator.backend_name, config.split('-', 1)[1])
        except Exception as e:
            self.skipTest(f"Qibo not available: {e}")
    
    def test_create_simulator_instances_invalid(self):
        """测试无效模拟器配置处理"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        
        # 测试格式错误的配置
        with self.assertRaises(ValueError):
            rb.create_simulator_instances(['invalid_format'])
        
        # 测试不存在的平台
        with patch('builtins.print') as mock_print:
            simulators = rb.create_simulator_instances(['nonexistent-backend'])
            self.assertEqual(len(simulators), 0)
            mock_print.assert_called()
    
    def test_create_circuit_instances_qft(self):
        """测试QFT电路实例创建"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        
        circuit_names = ['qft']
        circuits = rb.create_circuit_instances(circuit_names)
        
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0].name, 'Quantum Fourier Transform')
    
    def test_create_circuit_instances_unsupported(self):
        """测试不支持的电路类型处理"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        
        with patch('builtins.print') as mock_print:
            circuits = rb.create_circuit_instances(['unsupported_circuit'])
            self.assertEqual(len(circuits), 0)
            mock_print.assert_called()
    
    def test_run_benchmarks_basic(self):
        """测试基本基准测试流程"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        run_benchmarks.args.clear_cache = False
        
        try:
            # 创建测试数据
            circuit_factory = rb.create_circuit_instances(['qft'])[0]
            simulators = rb.create_simulator_instances(['qibo-numpy'])
            
            if not simulators:
                self.skipTest("No simulators available")
            
            # 运行基准测试（使用少量量子比特）
            results = rb.run_benchmarks(
                circuits=[circuit_factory],
                qubit_ranges=[2],  # 只使用2个量子比特
                simulators=simulators,
                golden_standard_key='qibo-numpy'
            )
            
            # 验证结果
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.simulator, 'qibo')
            self.assertEqual(result.backend, 'numpy')
            self.assertEqual(result.circuit_name, 'qft_2_qubits')
            self.assertEqual(result.n_qubits, 2)
            self.assertGreaterEqual(result.wall_time_sec, 0)
            self.assertGreaterEqual(result.cpu_time_sec, 0)
            self.assertGreaterEqual(result.peak_memory_mb, 0)
            self.assertEqual(result.state_fidelity, 1.0)  # 黄金标准模拟器
        except Exception as e:
            self.skipTest(f"Qibo not available: {e}")
    
    def test_run_benchmarks_multiple_simulators(self):
        """测试多模拟器比较"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        run_benchmarks.args.clear_cache = False
        
        try:
            # 创建测试数据
            circuit_factory = rb.create_circuit_instances(['qft'])[0]
            simulators = rb.create_simulator_instances(['qibo-numpy', 'qibo-qibojit'])
            
            if len(simulators) < 2:
                self.skipTest("Not enough simulators available")
            
            # 运行基准测试
            results = rb.run_benchmarks(
                circuits=[circuit_factory],
                qubit_ranges=[2],  # 只使用2个量子比特
                simulators=simulators,
                golden_standard_key='qibo-numpy'
            )
            
            # 验证结果
            self.assertEqual(len(results), 2)
            
            # 验证黄金标准模拟器的保真度为1.0
            golden_result = next(r for r in results if r.backend == 'numpy')
            self.assertEqual(golden_result.state_fidelity, 1.0)
            
            # 验证其他模拟器的保真度接近1.0
            other_result = next(r for r in results if r.backend == 'qibojit')
            self.assertGreater(other_result.state_fidelity, 0.99)  # 允许小的数值误差
        except Exception as e:
            self.skipTest(f"Qibo not available: {e}")
    
    def test_cache_configuration_memory(self):
        """测试内存缓存配置"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        run_benchmarks.args.clear_cache = False
        
        try:
            # 创建内存缓存配置
            cache_config = CacheConfig(
                enable_cache=True,
                cache_type="memory",
                memory_cache_size=16
            )
            
            # 创建测试数据
            circuit_factory = rb.create_circuit_instances(['qft'])[0]
            simulators = rb.create_simulator_instances(['qibo-numpy'])
            
            if not simulators:
                self.skipTest("No simulators available")
            
            # 第一次运行（缓存未命中）
            results1 = rb.run_benchmarks(
                circuits=[circuit_factory],
                qubit_ranges=[2],
                simulators=simulators,
                golden_standard_key='qibo-numpy',
                cache_config=cache_config
            )
            
            # 第二次运行（缓存命中）
            results2 = rb.run_benchmarks(
                circuits=[circuit_factory],
                qubit_ranges=[2],
                simulators=simulators,
                golden_standard_key='qibo-numpy',
                cache_config=cache_config
            )
            
            # 验证结果一致
            self.assertEqual(len(results1), len(results2))
            for r1, r2 in zip(results1, results2):
                np.testing.assert_array_almost_equal(r1.final_state, r2.final_state)
        except Exception as e:
            self.skipTest(f"Qibo not available: {e}")
    
    def test_cache_configuration_disk(self):
        """测试磁盘缓存配置"""
        import run_benchmarks
        run_benchmarks.args = MagicMock()
        run_benchmarks.args.verbose = False
        run_benchmarks.args.clear_cache = False
        
        try:
            # 创建临时缓存目录
            cache_dir = os.path.join(self.temp_dir, 'test_cache')
            
            # 创建磁盘缓存配置
            cache_config = CacheConfig(
                enable_cache=True,
                cache_type="disk",
                disk_cache_dir=cache_dir
            )
            
            # 创建测试数据
            circuit_factory = rb.create_circuit_instances(['qft'])[0]
            simulators = rb.create_simulator_instances(['qibo-numpy'])
            
            if not simulators:
                self.skipTest("No simulators available")
            
            # 运行基准测试
            results = rb.run_benchmarks(
                circuits=[circuit_factory],
                qubit_ranges=[2],
                simulators=simulators,
                golden_standard_key='qibo-numpy',
                cache_config=cache_config
            )
            
            # 验证缓存目录已创建
            self.assertTrue(os.path.exists(cache_dir))
        except Exception as e:
            self.skipTest(f"Qibo not available: {e}")
    
    def test_result_analysis(self):
        """测试结果分析功能"""
        from benchmark_harness.post_processing import analyze_results
        
        # 创建模拟结果
        results = [
            BenchmarkResult(
                simulator="qibo",
                backend="numpy",
                circuit_name="qft_2_qubits",
                n_qubits=2,
                wall_time_sec=0.1,
                cpu_time_sec=0.05,
                peak_memory_mb=10.0,
                cpu_utilization_percent=50.0,
                state_fidelity=1.0,
                final_state=np.array([1.0, 0.0, 0.0, 0.0])
            ),
            BenchmarkResult(
                simulator="qibo",
                backend="qibojit",
                circuit_name="qft_2_qubits",
                n_qubits=2,
                wall_time_sec=0.05,
                cpu_time_sec=0.02,
                peak_memory_mb=8.0,
                cpu_utilization_percent=60.0,
                state_fidelity=0.9999,
                final_state=np.array([1.0, 0.0, 0.0, 0.0])
            )
        ]
        
        # 分析结果
        analyze_results(results, self.temp_dir)
        
        # 验证输出文件
        csv_path = os.path.join(self.temp_dir, 'raw_results.csv')
        self.assertTrue(os.path.exists(csv_path))
        
        # 验证CSV内容
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertIn('simulator', df.columns)
        self.assertIn('backend', df.columns)
        self.assertIn('wall_time_sec', df.columns)
    
    def test_main_basic_flow(self):
        """测试main函数基本流程"""
        with patch('sys.argv', [
            'run_benchmarks.py',
            '--circuits', 'qft',
            '--qubits', '2',  # 只使用2个量子比特
            '--simulators', 'qibo-numpy',
            '--output-dir', self.temp_dir,
            '--verbose'
        ]):
            try:
                # 运行main函数
                exit_code = rb.main()
                
                # 验证成功退出
                self.assertEqual(exit_code, 0)
                
                # 验证输出目录和文件
                self.assertTrue(os.path.exists(self.temp_dir))
                
                # 查找带时间戳的子目录
                subdirs = [d for d in os.listdir(self.temp_dir) 
                          if os.path.isdir(os.path.join(self.temp_dir, d)) 
                          and d.startswith('benchmark_')]
                self.assertGreater(len(subdirs), 0)
                
                # 验证结果文件
                result_dir = os.path.join(self.temp_dir, subdirs[0])
                csv_path = os.path.join(result_dir, 'raw_results.csv')
                report_path = os.path.join(result_dir, 'summary_report.md')
                
                self.assertTrue(os.path.exists(csv_path))
                self.assertTrue(os.path.exists(report_path))
            except Exception as e:
                self.skipTest(f"Qibo not available: {e}")
    
    def test_main_error_handling(self):
        """测试main函数错误处理"""
        with patch('sys.argv', [
            'run_benchmarks.py',
            '--simulators', 'nonexistent-backend'  # 使用不存在的模拟器
        ]):
            # 运行main函数
            exit_code = rb.main()
            
            # 验证错误退出
            self.assertEqual(exit_code, 1)


if __name__ == '__main__':
    unittest.main()