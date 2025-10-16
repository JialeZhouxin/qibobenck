# run_benchmarks.py 测试计划

## 概述

本文档描述了为 `Bench/run_benchmarks.py` 创建测试脚本的详细计划。测试脚本将验证基准测试运行器的所有主要功能，包括命令行参数解析、模拟器和电路实例创建、基准测试执行、缓存功能和结果后处理。

## 测试目标

1. 验证所有命令行参数的正确解析和处理
2. 测试模拟器实例的创建和配置
3. 测试电路实例的创建和验证
4. 测试基准测试核心功能（使用少量量子比特减少计算量）
5. 测试缓存系统的各种配置和功能
6. 测试结果后处理和报告生成
7. 测试完整流程的端到端集成

## 测试脚本结构

### 文件位置
- 文件名: `Bench/tests/test_run_benchmarks.py`
- 使用 Python 的 unittest 框架

### 导入模块
```python
import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入要测试的模块
from run_benchmarks import (
    parse_arguments, 
    create_simulator_instances, 
    create_circuit_instances,
    run_benchmarks,
    main
)
from benchmark_harness.abstractions import BenchmarkResult
from benchmark_harness.caching import CacheConfig
```

### 测试类结构
```python
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
    
    # 测试方法...
```

## 详细测试用例

### 1. 命令行参数解析测试

#### 1.1 测试默认参数
```python
def test_parse_arguments_default(self):
    """测试默认参数解析"""
    with patch('sys.argv', ['run_benchmarks.py']):
        args = parse_arguments()
        
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
```

#### 1.2 测试自定义参数
```python
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
        args = parse_arguments()
        
        self.assertEqual(args.circuits, ['qft'])
        self.assertEqual(args.qubits, [2, 3])
        self.assertEqual(args.simulators, ['qibo-numpy', 'qibo-qibojit'])
        self.assertEqual(args.golden_standard, 'qibo-numpy')
        self.assertEqual(args.output_dir, 'custom_results')
        self.assertTrue(args.verbose)
        self.assertEqual(args.cache_type, 'memory')
        self.assertEqual(args.memory_cache_size, 128)
        self.assertTrue(args.cache_stats)
```

### 2. 模拟器实例创建测试

#### 2.1 测试Qibo模拟器创建
```python
def test_create_simulator_instances_qibo(self):
    """测试Qibo模拟器实例创建"""
    # 设置全局args变量
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    
    simulator_configs = ['qibo-numpy', 'qibo-qibojit']
    simulators = create_simulator_instances(simulator_configs)
    
    self.assertEqual(len(simulators), 2)
    self.assertIn('qibo-numpy', simulators)
    self.assertIn('qibo-qibojit', simulators)
    
    for config, simulator in simulators.items():
        self.assertEqual(simulator.platform_name, 'qibo')
        self.assertIn(config, simulator_configs)
        self.assertEqual(simulator.backend_name, config.split('-', 1)[1])
```

#### 2.2 测试无效模拟器配置
```python
def test_create_simulator_instances_invalid(self):
    """测试无效模拟器配置处理"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    
    # 测试格式错误的配置
    with self.assertRaises(ValueError):
        create_simulator_instances(['invalid_format'])
    
    # 测试不存在的平台
    with patch('builtins.print') as mock_print:
        simulators = create_simulator_instances(['nonexistent-backend'])
        self.assertEqual(len(simulators), 0)
        mock_print.assert_called()
```

### 3. 电路实例创建测试

#### 3.1 测试QFT电路创建
```python
def test_create_circuit_instances_qft(self):
    """测试QFT电路实例创建"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    
    circuit_names = ['qft']
    circuits = create_circuit_instances(circuit_names)
    
    self.assertEqual(len(circuits), 1)
    self.assertEqual(circuits[0].name, 'Quantum Fourier Transform')
```

#### 3.2 测试不支持的电路类型
```python
def test_create_circuit_instances_unsupported(self):
    """测试不支持的电路类型处理"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    
    with patch('builtins.print') as mock_print:
        circuits = create_circuit_instances(['unsupported_circuit'])
        self.assertEqual(len(circuits), 0)
        mock_print.assert_called()
```

### 4. 基准测试核心功能测试

#### 4.1 测试基本基准测试流程
```python
def test_run_benchmarks_basic(self):
    """测试基本基准测试流程"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    run_benchmarks.args.clear_cache = False
    
    # 创建测试数据
    circuit_factory = create_circuit_instances(['qft'])[0]
    simulators = create_simulator_instances(['qibo-numpy'])
    
    # 运行基准测试（使用少量量子比特）
    results = run_benchmarks(
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
```

#### 4.2 测试多模拟器比较
```python
def test_run_benchmarks_multiple_simulators(self):
    """测试多模拟器比较"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    run_benchmarks.args.clear_cache = False
    
    # 创建测试数据
    circuit_factory = create_circuit_instances(['qft'])[0]
    simulators = create_simulator_instances(['qibo-numpy', 'qibo-qibojit'])
    
    # 运行基准测试
    results = run_benchmarks(
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
```

### 5. 缓存功能测试

#### 5.1 测试内存缓存配置
```python
def test_cache_configuration_memory(self):
    """测试内存缓存配置"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    run_benchmarks.args.clear_cache = False
    
    # 创建内存缓存配置
    cache_config = CacheConfig(
        enable_cache=True,
        cache_type="memory",
        memory_cache_size=16
    )
    
    # 创建测试数据
    circuit_factory = create_circuit_instances(['qft'])[0]
    simulators = create_simulator_instances(['qibo-numpy'])
    
    # 第一次运行（缓存未命中）
    results1 = run_benchmarks(
        circuits=[circuit_factory],
        qubit_ranges=[2],
        simulators=simulators,
        golden_standard_key='qibo-numpy',
        cache_config=cache_config
    )
    
    # 第二次运行（缓存命中）
    results2 = run_benchmarks(
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
```

#### 5.2 测试磁盘缓存配置
```python
def test_cache_configuration_disk(self):
    """测试磁盘缓存配置"""
    import run_benchmarks
    run_benchmarks.args = MagicMock()
    run_benchmarks.args.verbose = False
    run_benchmarks.args.clear_cache = False
    
    # 创建临时缓存目录
    cache_dir = os.path.join(self.temp_dir, 'test_cache')
    
    # 创建磁盘缓存配置
    cache_config = CacheConfig(
        enable_cache=True,
        cache_type="disk",
        disk_cache_dir=cache_dir
    )
    
    # 创建测试数据
    circuit_factory = create_circuit_instances(['qft'])[0]
    simulators = create_simulator_instances(['qibo-numpy'])
    
    # 运行基准测试
    results = run_benchmarks(
        circuits=[circuit_factory],
        qubit_ranges=[2],
        simulators=simulators,
        golden_standard_key='qibo-numpy',
        cache_config=cache_config
    )
    
    # 验证缓存目录已创建
    self.assertTrue(os.path.exists(cache_dir))
```

### 6. 结果后处理测试

#### 6.1 测试结果分析
```python
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
```

### 7. 完整流程集成测试

#### 7.1 测试main函数基本流程
```python
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
        # 运行main函数
        exit_code = main()
        
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
```

#### 7.2 测试错误处理
```python
def test_main_error_handling(self):
    """测试main函数错误处理"""
    with patch('sys.argv', [
        'run_benchmarks.py',
        '--simulators', 'nonexistent-backend'  # 使用不存在的模拟器
    ]):
        # 运行main函数
        exit_code = main()
        
        # 验证错误退出
        self.assertEqual(exit_code, 1)
```

## 运行测试

### 单独运行测试
```bash
cd Bench/tests
python -m unittest test_run_benchmarks.py
```

### 运行特定测试
```bash
python -m unittest test_run_benchmarks.TestRunBenchmarks.test_parse_arguments_default
```

### 运行所有测试并显示详细输出
```bash
python -m unittest -v test_run_benchmarks.py
```

## 测试注意事项

1. **资源使用**: 测试使用少量量子比特（2-3个）以减少计算资源使用
2. **临时文件**: 所有测试使用临时目录，测试后自动清理
3. **模拟器可用性**: 测试前确保Qibo和相关后端已安装
4. **平台兼容性**: 测试在不同操作系统上的兼容性
5. **错误处理**: 测试各种错误情况的处理

## 预期测试结果

所有测试应该通过，验证：
- 命令行参数正确解析
- 模拟器和电路实例正确创建
- 基准测试正确执行
- 缓存系统正常工作
- 结果正确处理和保存
- 错误情况得到妥善处理

## 后续改进

1. 添加性能回归测试
2. 添加更多电路类型的测试
3. 添加并行执行测试
4. 添加更多边界条件测试
5. 添加集成测试到CI/CD流程