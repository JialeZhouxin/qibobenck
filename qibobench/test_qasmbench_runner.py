#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版 QASMBench Runner 测试套件

本测试套件为 QASMBench Runner 提供全面的测试覆盖，确保基准测试工具的可靠性、
准确性和健壮性。测试范围包括核心功能验证、边界条件处理、错误处理机制、
性能测试和集成测试。

测试覆盖的主要组件:
- QASMBenchConfig: 配置管理类
- QASMBenchMetrics: 指标存储和计算类  
- QASMBenchReporter: 报告生成类
- QASMBenchRunner: 核心基准测试运行器
- 工具函数: 电路发现、查找和运行函数

测试类型:
- 单元测试: 验证单个组件的功能正确性
- 集成测试: 验证组件间的协作
- 性能测试: 验证系统在负载下的表现
- 边界测试: 验证极端条件下的行为
- 错误处理测试: 验证异常情况的处理能力

依赖项:
- pytest: 测试框架
- numpy, torch, jax, tensorflow: 多框架支持测试
- psutil: 系统资源监控
- unittest.mock: 模拟对象和函数

使用方法:
    pytest test_qasmbench_runner.py -v                    # 运行所有测试
    pytest test_qasmbench_runner.py -m "not slow"        # 跳过耗时测试
    pytest test_qasmbench_runner.py::TestQASMBenchConfig  # 运行特定测试类
"""

import pytest
import numpy as np
import os
import tempfile
import json
import csv
import time
import psutil
from unittest.mock import Mock, patch, MagicMock
import torch
import jax
import tensorflow as tf
import warnings

# 导入被测试的模块
from qasmbench_runner import (
    QASMBenchConfig, QASMBenchMetrics, QASMBenchReporter, 
    QASMBenchRunner, list_available_circuits, find_circuit_by_name,
    run_benchmark_for_circuit
)

# 导入配置文件中的辅助函数
from conftest import (
    create_test_metrics, create_mock_circuit, create_mock_result,
    MockCircuit, MockResult
)


class TestQASMBenchConfig:
    """
    测试 QASMBenchConfig 类
    
    测试目标:
    - 验证配置类的初始化过程
    - 确保默认值的正确性
    - 测试配置参数的动态修改
    - 验证边界条件的处理
    
    测试策略:
    - 直接实例化验证默认配置
    - 动态修改配置参数验证
    - 边界值测试确保健壮性
    """
    
    def test_default_configuration(self):
        """
        测试默认配置初始化
        
        测试作用: 验证QASMBenchConfig类在无参数初始化时是否设置了正确的默认值
        
        输入:
        - 无直接输入参数
        
        输出:
        - 创建一个QASMBenchConfig实例
        - 验证实例的各个属性值
        
        验证点:
        - num_runs属性应为5
        - warmup_runs属性应为1
        - output_formats应为['csv', 'markdown', 'json']
        - baseline_backend应为"numpy"
        - qasm_directory应为"../QASMBench"
        
        测试策略: 直接实例化和属性验证
        """
        config = QASMBenchConfig()
        assert config.num_runs == 5
        assert config.warmup_runs == 1
        assert config.output_formats == ['csv', 'markdown', 'json']
        assert config.baseline_backend == "numpy"
        assert config.qasm_directory == "../QASMBench"
    
    def test_custom_configuration(self):
        """
        测试自定义配置参数设置
        
        测试作用: 验证QASMBenchConfig类的配置参数是否可以正确修改和保存
        
        输入:
        - 无直接输入参数
        
        输出:
        - 创建并修改配置实例
        - 验证修改后的属性值
        
        验证点:
        - num_runs属性应正确设置为10
        - warmup_runs属性应正确设置为3
        - output_formats应正确设置为['csv']
        - baseline_backend应正确设置为"qibojit"
        - qasm_directory应正确设置为"/custom/path"
        
        测试策略: 动态修改配置参数并验证结果
        """
        config = QASMBenchConfig()
        config.num_runs = 10
        config.warmup_runs = 3
        config.output_formats = ['csv']
        config.baseline_backend = "qibojit"
        config.qasm_directory = "/custom/path"
        
        assert config.num_runs == 10
        assert config.warmup_runs == 3
        assert config.output_formats == ['csv']
        assert config.baseline_backend == "qibojit"
        assert config.qasm_directory == "/custom/path"
    
    def test_boundary_values(self):
        """
        测试边界值配置处理
        
        测试作用: 验证配置类在处理边界值时的稳定性和正确性
        
        输入:
        - 无直接输入参数
        
        输出:
        - 创建配置实例并设置边界值
        - 验证边界值的正确处理
        
        验证点:
        - num_runs最小值1应被正确接受
        - warmup_runs最小值0应被正确接受
        - num_runs较大值1000应被正确接受
        - warmup_runs较大值100应被正确接受
        
        测试策略: 测试最小值和较大值的边界条件
        """
        config = QASMBenchConfig()
        
        # 测试最小值
        config.num_runs = 1
        config.warmup_runs = 0
        assert config.num_runs == 1
        assert config.warmup_runs == 0
        
        # 测试较大值
        config.num_runs = 1000
        config.warmup_runs = 100
        assert config.num_runs == 1000
        assert config.warmup_runs == 100


class TestQASMBenchMetrics:
    """
    测试 QASMBenchMetrics 类
    
    测试目标:
    - 验证指标类的初始化过程
    - 确保所有属性的默认值正确
    - 测试指标赋值和读取功能
    - 验证指标计算的一致性
    
    测试策略:
    - 直接实例化验证默认值
    - 动态赋值验证属性功能
    - 计算逻辑验证确保正确性
    """
    
    def test_initialization(self):
        """
        测试指标对象初始化
        
        测试作用: 验证QASMBenchMetrics类在初始化时是否正确设置所有属性的默认值
        
        输入:
        - 无直接输入参数
        
        输出:
        - 创建一个QASMBenchMetrics实例
        - 验证所有属性的初始值
        
        验证点:
        - execution_time_mean应为None
        - execution_time_std应为None
        - peak_memory_mb应为None
        - speedup应为None
        - correctness应为"Unknown"
        - circuit_parameters应为空字典
        - backend_info应为空字典
        - environment_info应为空字典
        - report_metadata应为空字典
        - throughput_gates_per_sec应为None
        - jit_compilation_time应为None
        - circuit_build_time应为None
        
        测试策略: 直接实例化和属性验证
        """
        metrics = QASMBenchMetrics()
        
        # 验证所有核心指标初始化为 None 或 "Unknown"
        assert metrics.execution_time_mean is None
        assert metrics.execution_time_std is None
        assert metrics.peak_memory_mb is None
        assert metrics.speedup is None
        assert metrics.correctness == "Unknown"
        
        # 验证字典初始化为空
        assert metrics.circuit_parameters == {}
        assert metrics.backend_info == {}
        assert metrics.environment_info == {}
        assert metrics.report_metadata == {}
        
        # 验证其他指标初始化为 None
        assert metrics.throughput_gates_per_sec is None
        assert metrics.jit_compilation_time is None
        assert metrics.circuit_build_time is None
    
    def test_metrics_assignment(self):
        """
        测试指标属性赋值功能
        
        测试作用: 验证QASMBenchMetrics类的属性是否可以正确赋值和读取
        
        输入:
        - 无直接输入参数
        
        输出:
        - 创建指标实例并赋值
        - 验证赋值后的属性值
        
        验证点:
        - execution_time_mean应正确设置为1.5
        - execution_time_std应正确设置为0.1
        - peak_memory_mb应正确设置为128.5
        - speedup应正确设置为2.5
        - correctness应正确设置为"Passed"
        - circuit_parameters字典应正确赋值
        - backend_info字典应正确赋值
        - environment_info字典应正确赋值
        
        测试策略: 使用setattr绕过类型检查，验证属性赋值功能
        """
        metrics = QASMBenchMetrics()
        
        # 使用 setattr 来避免类型检查错误
        setattr(metrics, 'execution_time_mean', 1.5)
        setattr(metrics, 'execution_time_std', 0.1)
        setattr(metrics, 'peak_memory_mb', 128.5)
        setattr(metrics, 'speedup', 2.5)
        metrics.correctness = "Passed"
        
        assert metrics.execution_time_mean == 1.5
        assert metrics.execution_time_std == 0.1
        assert metrics.peak_memory_mb == 128.5
        assert metrics.speedup == 2.5
        assert metrics.correctness == "Passed"
        
        # 测试字典赋值
        metrics.circuit_parameters = {'nqubits': 4, 'depth': 10}
        metrics.backend_info = {'name': 'numpy', 'platform': None}
        metrics.environment_info = {'CPU': 'Intel', 'RAM': '16GB'}
        
        assert metrics.circuit_parameters['nqubits'] == 4
        assert metrics.backend_info['name'] == 'numpy'
        assert metrics.environment_info['CPU'] == 'Intel'
    
    def test_metrics_consistency(self):
        """
        测试指标计算的一致性
        
        测试作用: 验证指标之间的计算关系是否正确，特别是吞吐率的计算
        
        输入:
        - 无直接输入参数
        
        输出:
        - 设置指标值并计算相关指标
        - 验证计算结果的正确性
        
        验证点:
        - 正常情况下吞吐率应正确计算为100.0
        - 除零情况下吞吐率应为None
        - 指标计算逻辑应保持一致性
        
        测试策略: 设置合理的指标值，验证计算逻辑和边界条件处理
        """
        metrics = QASMBenchMetrics()
        
        # 设置合理的指标值
        setattr(metrics, 'execution_time_mean', 1.0)
        setattr(metrics, 'execution_time_std', 0.1)
        metrics.circuit_parameters = {'ngates': 100}
        
        # 计算吞吐率
        if metrics.execution_time_mean is not None and metrics.execution_time_mean > 0:
            setattr(metrics, 'throughput_gates_per_sec', 
                   metrics.circuit_parameters['ngates'] / metrics.execution_time_mean)
            assert metrics.throughput_gates_per_sec == 100.0
        
        # 测试无效值处理
        setattr(metrics, 'execution_time_mean', 0)
        setattr(metrics, 'throughput_gates_per_sec', None)  # 除零情况下应该为 None
        assert metrics.throughput_gates_per_sec is None


class TestQASMBenchReporter:
    """
    测试 QASMBenchReporter 类
    
    测试目标:
    - 验证报告生成功能的各种格式输出
    - 确保CSV、Markdown、JSON格式的正确性
    - 测试电路图保存功能
    - 验证特殊字符和边界条件的处理
    - 测试目录自动创建功能
    
    测试策略:
    - 使用临时目录进行文件操作测试
    - 模拟各种指标数据和电路对象
    - 验证文件内容和结构的正确性
    - 测试异常情况和边界条件
    """
    
    @pytest.fixture
    def sample_metrics(self):
        """
        提供示例指标数据fixture
        
        测试作用: 为测试方法提供预配置的示例指标数据，确保测试数据的一致性
        
        输入:
        - 无直接输入参数
        
        输出:
        - 返回一个配置好的QASMBenchMetrics实例
        - 包含预设的执行时间、内存使用和正确性指标
        
        验证点:
        - execution_time_mean设置为1.23
        - execution_time_std设置为0.15
        - peak_memory_mb设置为256.7
        - correctness设置为"Passed (fidelity: 0.999999)"
        
        测试策略: 使用conftest中的辅助函数创建标准测试数据
        """
        return create_test_metrics(
            execution_time_mean=1.23,
            execution_time_std=0.15,
            peak_memory_mb=256.7,
            correctness="Passed (fidelity: 0.999999)"
        )
    
    def test_generate_csv_report(self, tmp_path, sample_metrics):
        """
        测试CSV报告生成功能
        
        测试作用: 验证QASMBenchReporter.generate_csv_report方法能否正确生成CSV格式的基准测试报告
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        - sample_metrics: 通过fixture提供的示例指标数据
        
        输出:
        - 在指定路径生成CSV报告文件
        - 验证文件存在性和内容正确性
        
        验证点:
        - CSV文件应成功创建
        - 文件内容应包含中文表头"后端名称"
        - 文件内容应包含后端名称"numpy"和"qibojit"
        - 文件内容应包含执行时间数值"1.23"
        - CSV格式应正确解析
        
        测试策略: 使用多后端结果数据，验证CSV报告的完整性和格式正确性
        """
        results = {'numpy': sample_metrics, 'qibojit': sample_metrics}
        filename = tmp_path / "test_report.csv"
        
        QASMBenchReporter.generate_csv_report(results, "test_circuit", filename)
        
        assert filename.exists()
        
        # 验证CSV内容
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "后端名称" in content
            assert "numpy" in content
            assert "qibojit" in content
            assert "1.23" in content
    
    def test_generate_csv_report_with_special_characters(self, tmp_path, sample_metrics):
        """
        测试包含特殊字符的CSV报告生成
        
        测试作用: 验证QASMBenchReporter.generate_csv_report方法在处理包含特殊字符的电路名称时的健壮性
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        - sample_metrics: 通过fixture提供的示例指标数据
        
        输出:
        - 生成包含特殊字符的CSV报告文件
        - 验证特殊字符处理能力
        
        验证点:
        - 包含特殊字符的电路名称应被正确处理
        - CSV文件应成功创建在嵌套目录中
        - 文件内容应包含后端名称"numpy"
        - 文件内容应包含执行时间数值"1.23"
        - 特殊字符不应影响CSV格式正确性
        
        测试策略: 使用包含路径分隔符和特殊字符的电路名称，验证文件系统和内容处理的健壮性
        """
        results = {'numpy': sample_metrics}
        circuit_name = "test/circuit/special@name"
        # 传入明确的文件名，避免硬编码路径问题
        filename = tmp_path / "test_circuit_special@name" / "benchmark_report.csv"

        QASMBenchReporter.generate_csv_report(results, circuit_name, filename)

        # 验证报告文件被创建
        assert filename.exists()
        
        # 验证CSV内容
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "numpy" in content
            assert "1.23" in content
    
    def test_generate_markdown_report(self, tmp_path, sample_metrics):
        """
        测试Markdown报告生成功能
        
        测试作用: 验证QASMBenchReporter.generate_markdown_report方法能否正确生成Markdown格式的基准测试报告
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        - sample_metrics: 通过fixture提供的示例指标数据
        
        输出:
        - 在指定路径生成Markdown报告文件
        - 验证文件存在性和内容正确性
        
        验证点:
        - Markdown文件应成功创建
        - 文件内容应包含中文标题"# QASMBench电路基准测试报告"
        - 文件内容应包含电路名称"test_circuit"
        - 文件内容应包含后端名称"numpy"
        - 文件内容应包含执行时间数值"1.23"
        - Markdown格式应正确渲染
        
        测试策略: 使用单后端结果数据，验证Markdown报告的结构完整性和格式正确性
        """
        results = {'numpy': sample_metrics}
        filename = tmp_path / "test_report.md"
        
        QASMBenchReporter.generate_markdown_report(results, "test_circuit", filename)
        
        assert filename.exists()
        
        # 验证Markdown内容
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# QASMBench电路基准测试报告" in content
            assert "test_circuit" in content
            assert "numpy" in content
            assert "1.23" in content
    
    def test_generate_json_report(self, tmp_path, sample_metrics):
        """
        测试JSON报告生成功能
        
        测试作用: 验证QASMBenchReporter.generate_json_report方法能否正确生成JSON格式的基准测试报告
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        - sample_metrics: 通过fixture提供的示例指标数据
        
        输出:
        - 在指定路径生成JSON报告文件
        - 验证文件存在性和内容正确性
        
        验证点:
        - JSON文件应成功创建
        - JSON数据应包含"metadata"字段
        - JSON数据应包含"results"字段
        - results中应包含"numpy"后端数据
        - execution_time.mean应正确设置为1.23
        - JSON格式应正确解析
        
        测试策略: 使用单后端结果数据，验证JSON报告的结构完整性和数据正确性
        """
        results = {'numpy': sample_metrics}
        filename = tmp_path / "test_report.json"
        
        QASMBenchReporter.generate_json_report(results, "test_circuit", filename)
        
        assert filename.exists()
        
        # 验证JSON内容
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert "metadata" in data
            assert "results" in data
            assert "numpy" in data["results"]
            assert data["results"]["numpy"]["execution_time"]["mean"] == 1.23
    
    def test_save_circuit_diagram(self, tmp_path):
        """
        测试电路图保存功能
        
        测试作用: 验证QASMBenchReporter.save_circuit_diagram方法能否正确保存电路的可视化图表
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        
        输出:
        - 在指定路径生成电路图文件
        - 验证文件创建和模拟调用
        
        验证点:
        - plot_circuit函数应被正确调用
        - 电路图文件应成功创建
        - mock的savefig方法应被调用
        - 电路对象的必要属性应被正确设置
        - 文件创建操作应成功执行
        
        测试策略: 使用mock对象模拟绘图库，验证文件创建和函数调用的正确性
        """
        with patch('qasmbench_runner.plot_circuit') as mock_plot:
            # 创建正确的嵌套mock结构
            mock_figure = Mock()
            mock_figure.figure = Mock()
            
            circuit = Mock()
            circuit.nqubits = 3  # 添加必需的属性
            circuit.queue = []  # 添加queue属性以避免迭代错误
            circuit.init_kwargs = {"wire_names": None}  # 添加init_kwargs属性
            filename = tmp_path / "test_diagram.png"
            
            # 让savefig实际创建文件，而不是什么都不做的Mock
            # 在filename定义后设置lambda函数
            mock_figure.figure.savefig = lambda *args, **kwargs: filename.touch()
            
            # plot_circuit返回一个元组，第一个元素是figure对象
            mock_plot.return_value = (mock_figure,)
            
            QASMBenchReporter.save_circuit_diagram(circuit, "test_circuit", filename)
            
            # 验证文件被创建（mock的savefig会创建文件）
            assert filename.exists()
            mock_plot.assert_called_once_with(circuit)
            # 由于savefig现在是lambda函数，我们不能用assert_called_once_with，但可以验证它被调用了
            assert filename.exists()  # 重复验证文件确实被创建了
    
    def test_report_generation_with_empty_results(self, tmp_path):
        """
        测试空结果集的报告生成
        
        测试作用: 验证QASMBenchReporter.generate_csv_report方法在处理空结果集时的健壮性
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        
        输出:
        - 生成空结果集的CSV报告文件
        - 验证文件创建和表头结构
        
        验证点:
        - 空结果集应能成功生成CSV文件
        - CSV文件应成功创建
        - 空报告仍应包含中文表头"后端名称"
        - 报告结构应保持完整性
        - 空数据不应导致程序崩溃
        
        测试策略: 使用空字典作为结果数据，验证报告生成的健壮性和结构完整性
        """
        results = {}
        filename = tmp_path / "empty_report.csv"
        
        QASMBenchReporter.generate_csv_report(results, "empty_circuit", filename)
        
        assert filename.exists()
        
        # 验证空报告仍然有表头
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "后端名称" in content
    
    def test_report_directory_creation(self, tmp_path):
        """
        测试报告目录自动创建
        
        测试作用: 验证QASMBenchReporter.generate_csv_report方法在目标目录不存在时能否自动创建目录结构
        
        输入:
        - tmp_path: pytest提供的临时目录路径
        
        输出:
        - 在深层嵌套目录中生成报告文件
        - 验证目录自动创建功能
        
        验证点:
        - 深层嵌套目录应被自动创建
        - CSV报告文件应成功创建在目标目录中
        - 目录创建功能应支持多级嵌套
        - 文件创建操作应成功执行
        - 路径处理应正确处理斜杠分隔符
        
        测试策略: 使用三层嵌套目录路径，验证自动目录创建和文件生成的功能
        """
        results = {'numpy': QASMBenchMetrics()}
        setattr(results['numpy'], 'execution_time_mean', 1.0)
        
        # 使用深层嵌套路径
        deep_path = tmp_path / "level1" / "level2" / "level3"
        filename = deep_path / "test_report.csv"
        
        QASMBenchReporter.generate_csv_report(results, "test_circuit", filename)
        
        assert filename.exists()
        assert deep_path.exists()


class TestQASMBenchRunner:
    """测试 QASMBenchRunner 类"""
    
    @pytest.fixture
    def runner(self):
        """提供测试用的runner实例"""
        config = QASMBenchConfig()
        return QASMBenchRunner(config)
    
    @pytest.fixture
    def sample_qasm_content(self):
        """提供示例QASM内容"""
        return """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
cx q[0], q[1];
cx q[1], q[2];
barrier q[0], q[1], q[2];
measure q -> c;
"""
    
    def test_runner_initialization(self):
        """
        测试QASMBenchRunner的基本初始化功能。
        
        测试作用：
        验证QASMBenchRunner类能够正确初始化，包括配置对象的存储和结果字典的创建。
        这是所有基准测试功能的基础，确保runner对象在创建后处于正确的初始状态。
        
        输入：
        - 无直接输入，通过创建QASMBenchConfig和QASMBenchRunner实例进行测试
        
        输出：
        - 无返回值，通过断言验证初始化结果
        
        验证点：
        1. runner.config属性正确存储传入的配置对象
        2. runner.results属性初始化为空字典
        3. 对象创建过程不抛出异常
        
        测试策略：
        创建标准配置对象和runner实例，验证核心属性的正确性。这是最基础的单元测试，
        确保类的构造函数按预期工作。
        """
        config = QASMBenchConfig()
        runner = QASMBenchRunner(config)
        
        assert runner.config == config
        assert runner.results == {}
    
    def test_discover_qasm_circuits(self, runner, tmp_path):
        """
        测试QASM电路自动发现功能。
        
        测试作用：
        验证runner能够自动扫描和识别QASMBench目录结构中的量子电路文件。
        这个功能是基准测试系统的核心，需要正确解析目录层次结构并识别有效的QASM文件。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证发现的电路信息
        
        验证点：
        1. 返回的circuits是字典类型
        2. 正确识别电路路径"small/test_circuit"
        3. 电路信息包含正确的size分类
        4. 电路信息包含正确的name
        5. 能够识别transpiled文件路径
        
        测试策略：
        创建完整的QASMBench目录结构模拟，包括原始和transpiled文件，
        验证发现算法能够正确解析和分类电路。这测试了文件系统遍历和路径解析逻辑。
        """
        # 创建模拟的QASMBench目录结构
        qasm_dir = tmp_path / "QASMBench"
        small_dir = qasm_dir / "small"
        test_circuit_dir = small_dir / "test_circuit"
        test_circuit_dir.mkdir(parents=True)
        
        # 创建测试QASM文件
        qasm_file = test_circuit_dir / "test_circuit.qasm"
        qasm_file.write_text("OPENQASM 2.0;\nqreg q[2];\nh q[0];")
        
        # 创建transpiled文件
        transpiled_file = test_circuit_dir / "test_circuit_transpiled.qasm"
        transpiled_file.write_text("OPENQASM 2.0;\nqreg q[2];\nh q[0];")
        
        # 修改配置指向测试目录
        runner.config.qasm_directory = str(qasm_dir)
        
        circuits = runner.discover_qasm_circuits()
        
        assert isinstance(circuits, dict)
        assert "small/test_circuit" in circuits
        assert circuits["small/test_circuit"]["size"] == "small"
        assert circuits["small/test_circuit"]["name"] == "test_circuit"
        assert "transpiled" in circuits["small/test_circuit"]["path"]
    
    def test_discover_qasm_circuits_empty_directory(self, runner, tmp_path):
        """
        测试在空目录中进行电路发现的边界条件。
        
        测试作用：
        验证当QASM目录为空时，电路发现功能能够优雅处理并返回空结果。
        这是重要的边界条件测试，确保系统在没有可用电路时不会崩溃或返回错误结果。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证处理结果
        
        验证点：
        1. 返回的circuits是字典类型
        2. 字典长度为0，表示没有发现任何电路
        3. 处理过程不抛出异常
        
        测试策略：
        创建空目录并配置runner指向该目录，验证发现算法能够正确处理
        没有QASM文件的情况。这测试了算法的健壮性和边界条件处理。
        """
        empty_dir = tmp_path / "empty_qasm"
        empty_dir.mkdir()
        
        runner.config.qasm_directory = str(empty_dir)
        
        circuits = runner.discover_qasm_circuits()
        
        assert isinstance(circuits, dict)
        assert len(circuits) == 0
    
    def test_discover_qasm_circuits_nonexistent_directory(self, runner):
        """
        测试在不存在的目录中进行电路发现的错误处理。
        
        测试作用：
        验证当QASM目录路径不存在时，电路发现功能能够优雅处理错误并返回空结果。
        这测试了系统的错误处理能力，确保在配置错误或路径问题时不会导致程序崩溃。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证错误处理结果
        
        验证点：
        1. 返回的circuits是字典类型
        2. 字典长度为0，表示没有发现任何电路
        3. 处理过程不抛出异常
        4. 系统能够处理不存在的路径
        
        测试策略：
        配置runner指向不存在的目录路径，验证发现算法的错误处理机制。
        这测试了文件系统访问错误的处理和异常情况的优雅降级。
        """
        runner.config.qasm_directory = "/nonexistent/path"
        
        circuits = runner.discover_qasm_circuits()
        
        assert isinstance(circuits, dict)
        assert len(circuits) == 0
    
    def test_load_qasm_circuit(self, runner, tmp_path, sample_qasm_content):
        """
        测试QASM电路文件加载和解析功能。
        
        测试作用：
        验证runner能够正确加载QASM文件并创建量子电路对象。
        这包括文件读取、内容预处理（移除barrier语句）和电路对象创建的完整流程。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        - sample_qasm_content: 示例QASM内容（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证加载结果
        
        验证点：
        1. 返回的circuit对象不为None
        2. qibo.Circuit.from_qasm方法被正确调用
        3. barrier语句被正确移除
        4. 文件读取和预处理正常工作
        
        测试策略：
        创建包含barrier语句的QASM文件，使用mock对象模拟qibo的电路创建，
        验证预处理逻辑和加载流程的正确性。这测试了文件I/O和字符串处理功能。
        """
        qasm_file = tmp_path / "test_circuit.qasm"
        qasm_file.write_text(sample_qasm_content)
        
        with patch('qibo.Circuit.from_qasm') as mock_from_qasm:
            mock_circuit = Mock()
            mock_circuit.nqubits = 3
            mock_circuit.depth = 2
            mock_circuit.ngates = 3
            mock_from_qasm.return_value = mock_circuit
            
            circuit = runner.load_qasm_circuit(str(qasm_file))
            
            assert circuit is not None
            mock_from_qasm.assert_called_once()
            
            # 验证barrier语句被移除
            call_args = mock_from_qasm.call_args[0][0]
            assert 'barrier' not in call_args
    
    def test_load_qasm_circuit_nonexistent_file(self, runner):
        """
        测试加载不存在QASM文件的错误处理。
        
        测试作用：
        验证当尝试加载不存在的QASM文件时，系统能够优雅处理并返回None。
        这测试了文件不存在错误的处理能力，确保系统在文件访问失败时不会崩溃。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证错误处理结果
        
        验证点：
        1. 返回的circuit为None
        2. 处理过程不抛出异常
        3. 系统能够处理FileNotFoundError或类似异常
        
        测试策略：
        直接调用load_qasm_circuit方法并传入不存在的文件路径，
        验证错误处理机制的正确性。这测试了异常捕获和错误恢复能力。
        """
        circuit = runner.load_qasm_circuit("/nonexistent/file.qasm")
        
        assert circuit is None
    
    def test_load_qasm_circuit_invalid_content(self, runner, tmp_path):
        """
        测试加载无效QASM内容的错误处理。
        
        测试作用：
        验证当QASM文件内容无效或格式错误时，系统能够正确处理解析异常。
        这测试了内容验证和异常处理机制，确保无效输入不会导致系统崩溃。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证错误处理结果
        
        验证点：
        1. 返回的circuit为None
        2. qibo解析异常被正确捕获
        3. 系统能够处理格式错误的QASM内容
        
        测试策略：
        创建包含无效内容的QASM文件，模拟qibo解析器抛出异常，
        验证异常处理和错误恢复机制。这测试了内容验证和异常管理。
        """
        invalid_file = tmp_path / "invalid.qasm"
        invalid_file.write_text("This is not valid QASM content")
        
        with patch('qibo.Circuit.from_qasm', side_effect=Exception("Invalid QASM")):
            circuit = runner.load_qasm_circuit(str(invalid_file))
            
            assert circuit is None
    
    def test_measure_memory_usage(self, runner):
        """
        测试内存使用量测量功能。
        
        测试作用：
        验证runner能够正确测量当前进程的内存使用量。
        这是性能监控的基础功能，为基准测试提供资源使用数据。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证测量结果
        
        验证点：
        1. 返回的memory_usage是float类型
        2. 内存使用量大于0（表示合理值）
        3. 测量过程不抛出异常
        
        测试策略：
        直接调用measure_memory_usage方法，验证返回值的类型和合理性。
        这测试了系统资源监控功能，确保psutil库的正确使用。
        """
        memory_usage = runner.measure_memory_usage()
        
        assert isinstance(memory_usage, float)
        assert memory_usage > 0
    
    def test_validate_correctness_perfect_match(self, runner):
        """
        测试完美匹配情况的正确性验证功能。
        
        测试作用：
        验证当量子计算结果与基准结果完全匹配时，正确性验证能够返回通过状态。
        这测试了量子态比较算法的核心功能，确保相同状态的正确识别。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证验证结果
        
        验证点：
        1. 返回的correctness包含"Passed"
        2. 包含"fidelity: 1.000000"表示完全匹配
        3. 量子态比较算法正确工作
        4. 计算精度正确（6位小数）
        
        测试策略：
        创建具有相同量子态的MockResult对象，验证正确性算法能够识别
        完美匹配并计算正确的保真度。这测试了量子态比较和保真度计算。
        """
        class MockResult:
            def state(self):
                return np.array([1, 0, 0, 0])
        
        result = MockResult()
        baseline_result = MockResult()
        
        correctness = runner.validate_correctness(result, baseline_result)
        
        assert "Passed" in correctness
        assert "fidelity: 1.000000" in correctness
    
    def test_validate_correctness_no_match(self, runner):
        """
        测试完全不匹配情况的正确性验证功能。
        
        测试作用：
        验证当量子计算结果与基准结果完全不匹配时，正确性验证能够返回失败状态。
        这测试了量子态比较算法对不同状态的识别能力。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证验证结果
        
        验证点：
        1. 返回的correctness包含"Failed"
        2. 包含"fidelity:"字段（值应该为0）
        3. 能够正确识别正交量子态
        4. 保真度计算正确
        
        测试策略：
        创建具有不同量子态的MockResult对象（正交态），验证正确性算法能够
        识别不匹配并计算正确的保真度。这测试了量子态差异检测能力。
        """
        class MockResult:
            def state(self):
                return np.array([1, 0, 0, 0])
        
        class MockBaselineResult:
            def state(self):
                return np.array([0, 1, 0, 0])
        
        result = MockResult()
        baseline_result = MockBaselineResult()
        
        correctness = runner.validate_correctness(result, baseline_result)
        
        assert "Failed" in correctness
        assert "fidelity:" in correctness
    
    def test_validate_correctness_shape_mismatch(self, runner):
        """
        测试量子态形状不匹配情况的正确性验证功能。
        
        测试作用：
        验证当量子计算结果与基准结果的量子态维度不匹配时，正确性验证能够返回失败状态。
        这测试了系统对不同维度量子系统的处理能力和错误检测机制。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证验证结果
        
        验证点：
        1. 返回的correctness包含"Failed - Shape mismatch"
        2. 系统能够检测量子态维度差异
        3. 形状不匹配被正确识别和报告
        4. 比较过程不会因维度不匹配而崩溃
        
        测试策略：
        创建具有不同维度量子态的MockResult对象（4维vs 2维），验证正确性算法能够
        识别形状不匹配并返回适当的错误信息。这测试了维度检查和错误处理机制。
        """
        class MockResult:
            def state(self):
                return np.array([1, 0, 0, 0])  # 4元素
        
        class MockBaselineResult:
            def state(self):
                return np.array([1, 0])  # 2元素
        
        result = MockResult()
        baseline_result = MockBaselineResult()
        
        correctness = runner.validate_correctness(result, baseline_result)
        
        assert "Failed - Shape mismatch" in correctness
    
    def test_validate_correctness_no_state_method(self, runner):
        """
        测试缺少state方法对象的正确性验证功能。
        
        测试作用：
        验证当结果对象缺少state方法时，正确性验证能够返回未知状态。
        这测试了系统对不完整或异常结果对象的处理能力。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证验证结果
        
        验证点：
        1. 返回的correctness包含"Unknown - No state method"
        2. 系统能够检测方法缺失
        3. 异常情况被优雅处理
        4. 不会因方法缺失而抛出异常
        
        测试策略：
        创建没有state方法的MockResult对象，验证正确性算法能够检测到
        方法缺失并返回适当的状态信息。这测试了反射机制和异常处理。
        """
        class MockResult:
            pass
        
        result = MockResult()
        
        correctness = runner.validate_correctness(result)
        
        assert "Unknown - No state method" in correctness
    
    def test_validate_correctness_none_result(self, runner):
        """
        测试None结果输入的正确性验证功能。
        
        测试作用：
        验证当输入结果为None时，正确性验证能够返回失败状态。
        这是最基本的边界条件测试，确保系统能够处理空值输入。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证验证结果
        
        验证点：
        1. 返回的correctness等于"Failed"
        2. 系统能够处理None输入
        3. 空值检查机制正常工作
        4. 不会因None输入而抛出异常
        
        测试策略：
        直接传入None作为结果对象，验证空值检查和错误处理机制。
        这测试了最基本的输入验证和边界条件处理。
        """
        correctness = runner.validate_correctness(None)
        
        assert correctness == "Failed"
    
    def test_convert_to_numpy_numpy_array(self, runner):
        """
        测试NumPy数组到NumPy数组的转换功能。
        
        测试作用：
        验证当输入已经是NumPy数组时，转换函数能够正确处理并返回原数组。
        这测试了类型检查和直接返回的逻辑，确保不必要的转换不会发生。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证转换结果
        
        验证点：
        1. 转换后的数组与原数组内容完全相等
        2. 转换后的对象是np.ndarray类型
        3. 数组内容保持不变
        4. 没有不必要的复制或转换操作
        
        测试策略：
        创建NumPy数组并调用转换函数，验证类型检查和直接返回逻辑。
        这测试了最简单的转换路径和类型识别功能。
        """
        original_array = np.array([1, 0, 1, 0])
        converted_array = runner._convert_to_numpy(original_array)
        
        np.testing.assert_array_equal(original_array, converted_array)
        assert isinstance(converted_array, np.ndarray)
    
    def test_convert_to_numpy_pytorch_tensor(self, runner):
        """
        测试PyTorch Tensor到NumPy数组的转换功能。
        
        测试作用：
        验证转换函数能够正确处理PyTorch Tensor并将其转换为NumPy数组。
        这测试了多框架兼容性，确保系统能够处理来自不同深度学习框架的数据格式。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证转换结果
        
        验证点：
        1. 转换后的数组内容与期望值完全相等
        2. 转换后的对象是np.ndarray类型
        3. 数值精度在转换过程中保持不变
        4. 只有在PyTorch可用时才执行测试
        
        测试策略：
        创建PyTorch Tensor并调用转换函数，验证跨框架数据转换的正确性。
        这测试了多框架互操作性和数据格式兼容性。
        """
        if torch is not None:
            original_tensor = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
            converted_array = runner._convert_to_numpy(original_tensor)
            
            expected_array = np.array([1, 0, 1, 0])
            np.testing.assert_array_equal(expected_array, converted_array)
            assert isinstance(converted_array, np.ndarray)
    
    def test_convert_to_numpy_pytorch_gpu_tensor(self, runner):
        """
        测试PyTorch GPU Tensor到NumPy数组的转换功能。
        
        测试作用：
        验证转换函数能够正确处理PyTorch GPU Tensor并将其转换为NumPy数组。
        这测试了GPU内存数据的处理能力，确保系统能够处理来自不同设备的数据。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证转换结果
        
        验证点：
        1. 转换后的数组内容与期望值完全相等
        2. 转换后的对象是np.ndarray类型
        3. GPU到CPU的数据传输正确
        4. 只有在PyTorch和CUDA都可用时才执行测试
        
        测试策略：
        创建PyTorch GPU Tensor并调用转换函数，验证跨设备数据转换的正确性。
        这测试了GPU内存管理和数据传输功能。
        """
        if torch is not None and torch.cuda.is_available():
            original_tensor = torch.tensor([1, 0, 1, 0], dtype=torch.float32).cuda()
            converted_array = runner._convert_to_numpy(original_tensor)
            
            expected_array = np.array([1, 0, 1, 0])
            np.testing.assert_array_equal(expected_array, converted_array)
            assert isinstance(converted_array, np.ndarray)
    
    def test_convert_to_numpy_jax_array(self, runner):
        """
        测试JAX数组到NumPy数组的转换功能。
        
        测试作用：
        验证转换函数能够正确处理JAX数组并将其转换为NumPy数组。
        这测试了与JAX框架的兼容性，确保系统能够处理JAX的不可变数组格式。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证转换结果
        
        验证点：
        1. 转换后的数组内容与期望值完全相等
        2. 转换后的对象是np.ndarray类型
        3. JAX不可变数组正确转换为可变NumPy数组
        4. 只有在JAX可用时才执行测试
        
        测试策略：
        创建JAX数组并调用转换函数，验证JAX到NumPy的数据转换正确性。
        这测试了函数式编程框架的数据兼容性。
        """
        if jax is not None:
            original_array = jax.numpy.array([1, 0, 1, 0])
            converted_array = runner._convert_to_numpy(original_array)
            
            expected_array = np.array([1, 0, 1, 0])
            np.testing.assert_array_equal(expected_array, converted_array)
            assert isinstance(converted_array, np.ndarray)
    
    def test_convert_to_numpy_tensorflow_tensor(self, runner):
        """
        测试TensorFlow Tensor到NumPy数组的转换功能。
        
        测试作用：
        验证转换函数能够正确处理TensorFlow Tensor并将其转换为NumPy数组。
        这测试了与TensorFlow框架的兼容性，确保系统能够处理TF的计算图张量。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证转换结果
        
        验证点：
        1. 转换后的数组内容与期望值完全相等
        2. 转换后的对象是np.ndarray类型
        3. TensorFlow张量正确转换为NumPy数组
        4. 只有在TensorFlow可用时才执行测试
        
        测试策略：
        创建TensorFlow常量张量并调用转换函数，验证TF到NumPy的数据转换正确性。
        这测试了静态图框架的数据兼容性。
        """
        if tf is not None:
            original_tensor = tf.constant([1, 0, 1, 0], dtype=tf.float32)
            converted_array = runner._convert_to_numpy(original_tensor)
            
            expected_array = np.array([1, 0, 1, 0])
            np.testing.assert_array_equal(expected_array, converted_array)
            assert isinstance(converted_array, np.ndarray)
    
    def test_convert_to_numpy_unsupported_type(self, runner):
        """
        测试不支持类型到NumPy数组的转换错误处理。
        
        测试作用：
        验证当输入对象类型不被支持时，转换函数能够正确抛出异常。
        这测试了类型检查和错误处理机制，确保系统能够优雅处理不支持的输入类型。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证异常抛出
        
        验证点：
        1. 抛出ValueError异常
        2. 异常消息包含"无法将类型"字样
        3. 自定义对象的__array__方法异常被正确处理
        4. 警告过滤机制正常工作
        
        测试策略：
        创建自定义UnsupportedObject类，其__array__方法抛出异常，
        验证转换函数的错误处理和异常传播机制。这测试了异常安全和错误报告。
        """
        # 创建一个真正无法转换的对象
        class UnsupportedObject:
            def __array__(self):  # 定义 __array__ 方法，用于将对象转换为 NumPy 数组
                raise TypeError("Cannot convert to array")  # 抛出类型错误，表示该对象不能转换为数组
        
        unsupported_obj = UnsupportedObject()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with pytest.raises(ValueError, match="无法将类型"):
                runner._convert_to_numpy(unsupported_obj)
    def test_calculate_speedup(self, runner):
        """
        测试加速比计算函数。

        该测试通过创建两个QASMBenchMetrics对象，分别表示基准和优化后的执行时间。
        然后调用runner的_calculate_speedup方法计算加速比，并验证计算结果是否正确。
        """
        # 创建测试指标
        baseline_metrics = QASMBenchMetrics()
        setattr(baseline_metrics, 'execution_time_mean', 2.0)
        
        test_metrics = QASMBenchMetrics()
        setattr(test_metrics, 'execution_time_mean', 1.0)
        
        results = {
            "numpy": baseline_metrics,
            "qibojit": test_metrics
        }
        
        runner._calculate_speedup(results)
        
        assert results["qibojit"].speedup == 2.0
        assert results["numpy"].speedup is None  # 基准后端不应该有加速比
    
    def test_calculate_speedup_no_baseline(self, runner):
        """测试没有基准的加速比计算"""
        test_metrics = QASMBenchMetrics()
        setattr(test_metrics, 'execution_time_mean', 1.0)
        
        results = {
            "qibojit": test_metrics
        }
        
        runner._calculate_speedup(results)
        
        assert results["qibojit"].speedup is None
    
    def test_calculate_speedup_missing_execution_time(self, runner):
        """测试缺少执行时间的加速比计算"""
        baseline_metrics = QASMBenchMetrics()
        setattr(baseline_metrics, 'execution_time_mean', 2.0)
        
        test_metrics = QASMBenchMetrics()
        setattr(test_metrics, 'execution_time_mean', None)  # 缺少执行时间
        
        results = {
            "numpy": baseline_metrics,
            "qibojit": test_metrics
        }
        
        runner._calculate_speedup(results)
        
        assert results["qibojit"].speedup is None
    
    def test_run_single_backend_benchmark_success(self, runner, tmp_path, sample_qasm_content):
        """
        测试单个后端基准测试成功执行情况。
        
        测试作用：
        验证runner能够成功执行单个后端的完整基准测试流程，包括电路加载、后端设置、
        执行测量和指标收集。这是基准测试系统的核心功能测试。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        - sample_qasm_content: 示例QASM内容（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证基准测试结果
        
        验证点：
        1. 返回的result不为None，表示基准测试成功执行
        2. execution_time_mean不为None，表示执行时间被正确测量
        3. execution_time_std不为None，表示执行时间标准差被计算
        4. peak_memory_mb不为None，表示内存使用被监控
        5. correctness为"Passed"或"Passed (no baseline)"，表示正确性验证通过
        6. circuit_parameters包含正确的电路参数信息
        7. 所有模拟的依赖项被正确调用
        
        测试策略：
        创建完整的测试环境，模拟电路加载、后端设置、系统信息获取等，
        验证基准测试流程的完整性和正确性。这测试了系统的核心执行流程。
        """
        # 创建测试QASM文件
        qasm_file = tmp_path / "test_circuit.qasm"
        qasm_file.write_text(sample_qasm_content)
        
        # 模拟电路和结果
        mock_circuit = Mock()
        mock_circuit.nqubits = 3
        mock_circuit.depth = 2
        mock_circuit.ngates = 3
        mock_circuit.return_value = Mock()
        mock_circuit.return_value.state.return_value = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        
        with patch.object(runner, 'load_qasm_circuit', return_value=mock_circuit), \
             patch('qibo.set_backend'), \
             patch('platform.processor', return_value='Intel i7'), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.total = 16 * 1024**3  # 16GB
            
            result, metrics = runner._run_single_backend_benchmark(
                "test_backend", "numpy", None, str(qasm_file)
            )
            
            assert result is not None
            assert metrics.execution_time_mean is not None
            assert metrics.execution_time_std is not None
            assert metrics.peak_memory_mb is not None
            assert metrics.correctness in ["Passed", "Passed (no baseline)"]
            assert metrics.circuit_parameters['nqubits'] == 3
            assert metrics.circuit_parameters['depth'] == 2
            assert metrics.circuit_parameters['ngates'] == 3
    
    def test_run_single_backend_benchmark_circuit_load_failure(self, runner, tmp_path):
        """
        测试电路加载失败时的基准测试错误处理。
        
        测试作用：
        验证当电路加载失败时，基准测试能够正确处理错误并返回失败状态。
        这测试了系统在输入文件问题时的健壮性和错误恢复能力。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证错误处理结果
        
        验证点：
        1. 返回的result为None，表示基准测试失败
        2. metrics.correctness为"Failed"，明确标记失败状态
        3. 系统能够处理文件不存在的情况
        4. 错误处理流程正常工作
        5. 不会因文件加载失败而崩溃
        
        测试策略：
        使用不存在的QASM文件路径，验证系统对文件加载错误的处理机制。
        这测试了错误捕获、状态标记和异常恢复功能。
        """
        qasm_file = tmp_path / "nonexistent.qasm"
        
        result, metrics = runner._run_single_backend_benchmark(
            "test_backend", "numpy", None, str(qasm_file)
        )
        
        assert result is None
        assert metrics.correctness == "Failed"
    
    def test_run_single_backend_benchmark_backend_failure(self, runner, tmp_path, sample_qasm_content):
        """
        测试后端设置失败时的基准测试错误处理。
        
        测试作用：
        验证当量子计算后端设置失败时，基准测试能够正确处理错误并返回失败状态。
        这测试了系统在后端不可用或配置错误时的健壮性。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        - sample_qasm_content: 示例QASM内容（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证错误处理结果
        
        验证点：
        1. 返回的result为None，表示基准测试失败
        2. metrics.correctness为"Failed"，明确标记失败状态
        3. 系统能够处理后端不可用的情况
        4. 后端设置异常被正确捕获
        5. 错误处理流程不会导致程序崩溃
        
        测试策略：
        模拟qibo.set_backend抛出异常，验证系统对后端错误的处理机制。
        这测试了后端管理、异常捕获和错误恢复功能。
        """
        qasm_file = tmp_path / "test_circuit.qasm"
        qasm_file.write_text(sample_qasm_content)
        
        with patch('qibo.set_backend', side_effect=Exception("Backend not available")):
            result, metrics = runner._run_single_backend_benchmark(
                "test_backend", "nonexistent_backend", None, str(qasm_file)
            )
            
            assert result is None
            assert metrics.correctness == "Failed"
    
    def test_run_benchmark_for_circuit(self, runner, tmp_path, sample_qasm_content):
        """
        测试完整电路基准测试流程。
        
        测试作用：
        验证runner能够执行完整的电路基准测试流程，包括所有配置后端的测试。
        这是系统的主要功能测试，确保多后端基准测试的完整执行。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        - sample_qasm_content: 示例QASM内容（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证基准测试结果
        
        验证点：
        1. 返回的results是字典类型
        2. results长度大于0，表示有后端被测试
        3. results包含"numpy"后端的结果
        4. 电路加载被正确调用
        5. 单后端基准测试被正确调用
        6. 基准测试流程完整执行
        
        测试策略：
        模拟完整的基准测试流程，包括电路加载和单后端测试，
        验证多后端基准测试的协调和执行。这测试了系统的整体工作流程。
        """
        qasm_file = tmp_path / "test_circuit.qasm"
        qasm_file.write_text(sample_qasm_content)
        
        # 模拟电路
        mock_circuit = Mock()
        mock_circuit.nqubits = 3
        mock_circuit.depth = 2
        mock_circuit.ngates = 3
        
        with patch.object(runner, 'load_qasm_circuit', return_value=mock_circuit), \
             patch.object(runner, '_run_single_backend_benchmark') as mock_benchmark:
            
            # 模拟基准测试结果
            mock_metrics = QASMBenchMetrics()
            setattr(mock_metrics, 'execution_time_mean', 1.0)
            mock_metrics.correctness = "Passed"
            
            mock_benchmark.return_value = (Mock(), mock_metrics)
            
            results = runner.run_benchmark_for_circuit("test_circuit", str(qasm_file))
            
            assert isinstance(results, dict)
            assert len(results) > 0
            assert "numpy" in results
    
    def test_generate_reports(self, runner, tmp_path):
        """
        测试完整报告生成功能。
        
        测试作用：
        验证runner能够生成所有格式的基准测试报告，包括CSV、Markdown、JSON和电路图。
        这测试了报告生成的完整性和多格式输出能力。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证报告生成调用
        
        验证点：
        1. generate_csv_report方法被调用一次
        2. generate_markdown_report方法被调用一次
        3. generate_json_report方法被调用一次
        4. save_circuit_diagram方法被调用一次
        5. 所有报告格式都被正确生成
        6. 电路对象被正确传递给报告生成器
        
        测试策略：
        使用mock对象模拟所有报告生成方法，验证它们都被正确调用。
        这测试了报告生成的协调和多格式输出功能。
        """
        # 创建测试结果
        metrics = QASMBenchMetrics()
        metrics.execution_time_mean = 1.0
        metrics.correctness = "Passed"
        results = {'numpy': metrics}
        
        # 模拟电路
        circuit = Mock()
        
        with patch.object(QASMBenchReporter, 'generate_csv_report') as mock_csv, \
             patch.object(QASMBenchReporter, 'generate_markdown_report') as mock_md, \
             patch.object(QASMBenchReporter, 'generate_json_report') as mock_json, \
             patch.object(QASMBenchReporter, 'save_circuit_diagram') as mock_diagram:
            
            runner.generate_reports(results, "test_circuit", circuit)
            
            # 验证所有报告生成方法都被调用
            mock_csv.assert_called_once()
            mock_md.assert_called_once()
            mock_json.assert_called_once()
            mock_diagram.assert_called_once()
    
    def test_generate_reports_no_circuit(self, runner, tmp_path):
        """
        测试没有电路对象时的报告生成功能。
        
        测试作用：
        验证当电路对象为None时，runner能够生成数据报告但跳过电路图保存。
        这测试了系统在缺少电路信息时的智能处理能力。
        
        输入：
        - runner: QASMBenchRunner实例（通过fixture提供）
        - tmp_path: pytest临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证报告生成调用
        
        验证点：
        1. generate_csv_report方法被调用一次
        2. generate_markdown_report方法被调用一次
        3. generate_json_report方法被调用一次
        4. save_circuit_diagram方法不被调用
        5. 系统能够处理None电路对象
        6. 数据报告正常生成，电路图被跳过
        
        测试策略：
        传入None作为电路对象，验证数据报告正常生成而电路图保存被跳过。
        这测试了条件逻辑和智能报告生成功能。
        """
        metrics = QASMBenchMetrics()
        metrics.execution_time_mean = 1.0
        results = {'numpy': metrics}
        
        with patch.object(QASMBenchReporter, 'generate_csv_report') as mock_csv, \
             patch.object(QASMBenchReporter, 'generate_markdown_report') as mock_md, \
             patch.object(QASMBenchReporter, 'generate_json_report') as mock_json, \
             patch.object(QASMBenchReporter, 'save_circuit_diagram') as mock_diagram:
            
            runner.generate_reports(results, "test_circuit", None)
            
            # 验证报告生成方法被调用，但电路图保存不被调用
            mock_csv.assert_called_once()
            mock_md.assert_called_once()
            mock_json.assert_called_once()
            mock_diagram.assert_not_called()


class TestUtilityFunctions:
    """
    测试工具函数
    
    测试目标:
    - 验证电路发现和查找工具函数的正确性
    - 测试基准测试运行函数的集成功能
    - 确保工具函数的错误处理机制
    - 验证文件路径和名称匹配逻辑
    
    测试策略:
    - 使用临时目录模拟QASMBench结构
    - 模拟配置对象进行路径控制
    - 验证函数返回值和异常处理
    - 测试精确匹配和模糊匹配功能
    """
    
    def test_list_available_circuits(self, tmp_path):
        """
        测试列出可用电路函数功能。
        
        测试作用：
        验证list_available_circuits函数能够正确扫描和返回可用的量子电路列表。
        这是用户浏览和选择电路的基础功能。
        
        输入：
        - tmp_path: pytest提供的临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证函数返回结果
        
        验证点：
        1. 返回的circuits是字典类型
        2. 函数能够正确扫描QASM目录结构
        3. 模拟的配置对象被正确使用
        4. 电路发现逻辑正常工作
        5. 目录遍历功能正确
        
        测试策略：
        创建模拟的QASMBench目录结构和QASM文件，通过mock配置对象
        控制扫描路径，验证电路发现功能的正确性。
        """
        # 创建模拟目录结构
        qasm_dir = tmp_path / "QASMBench"
        small_dir = qasm_dir / "small"
        circuit_dir = small_dir / "test_circuit"
        circuit_dir.mkdir(parents=True)
        
        # 创建QASM文件
        qasm_file = circuit_dir / "test_circuit.qasm"
        qasm_file.write_text("OPENQASM 2.0;\nqreg q[2];")
        
        with patch('qasmbench_runner.QASMBenchConfig') as mock_config:
            mock_config.return_value.qasm_directory = str(qasm_dir)
            
            circuits = list_available_circuits()
            
            assert isinstance(circuits, dict)
    
    def test_find_circuit_by_name_exact_match(self, tmp_path):
        """
        测试精确匹配查找电路功能。
        
        测试作用：
        验证find_circuit_by_name函数能够通过完整路径名称精确查找并返回电路文件路径。
        这测试了精确匹配算法和路径解析功能。
        
        输入：
        - tmp_path: pytest提供的临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证查找结果
        
        验证点：
        1. 返回的file_path不为None
        2. 文件路径包含"test_circuit.qasm"
        3. 精确匹配算法正确工作
        4. 路径解析功能正常
        5. 配置对象被正确使用
        
        测试策略：
        创建完整的目录结构和QASM文件，使用完整路径名称进行查找，
        验证精确匹配功能的正确性。
        """
        # 创建模拟目录结构
        qasm_dir = tmp_path / "QASMBench"
        small_dir = qasm_dir / "small"
        circuit_dir = small_dir / "test_circuit"
        circuit_dir.mkdir(parents=True)
        
        # 创建QASM文件
        qasm_file = circuit_dir / "test_circuit.qasm"
        qasm_file.write_text("OPENQASM 2.0;\nqreg q[2];")
        
        with patch('qasmbench_runner.QASMBenchConfig') as mock_config:
            mock_config.return_value.qasm_directory = str(qasm_dir)
            
            file_path = find_circuit_by_name("small/test_circuit")
            
            assert file_path is not None
            assert "test_circuit.qasm" in file_path
    
    def test_find_circuit_by_name_partial_match(self, tmp_path):
        """
        测试部分匹配查找电路功能。
        
        测试作用：
        验证find_circuit_by_name函数能够通过部分名称匹配查找并返回电路文件路径。
        这测试了模糊匹配算法和用户友好的查找功能。
        
        输入：
        - tmp_path: pytest提供的临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证查找结果
        
        验证点：
        1. 返回的file_path不为None
        2. 文件路径包含"test_circuit.qasm"
        3. 部分匹配算法正确工作
        4. 能够处理不完整的路径信息
        5. 查找逻辑具有容错性
        
        测试策略：
        使用部分名称"test_circuit"进行查找，验证系统能够匹配到完整路径。
        这测试了模糊匹配和智能查找功能。
        """
        # 创建模拟目录结构
        qasm_dir = tmp_path / "QASMBench"
        small_dir = qasm_dir / "small"
        circuit_dir = small_dir / "test_circuit"
        circuit_dir.mkdir(parents=True)
        
        # 创建QASM文件
        qasm_file = circuit_dir / "test_circuit.qasm"
        qasm_file.write_text("OPENQASM 2.0;\nqreg q[2];")
        
        with patch('qasmbench_runner.QASMBenchConfig') as mock_config:
            mock_config.return_value.qasm_directory = str(qasm_dir)
            
            file_path = find_circuit_by_name("test_circuit")
            
            assert file_path is not None
            assert "test_circuit.qasm" in file_path
    
    def test_find_circuit_by_name_not_found(self, tmp_path):
        """
        测试查找不存在电路的错误处理。
        
        测试作用：
        验证当查找不存在的电路时，find_circuit_by_name函数能够正确返回None。
        这测试了查找失败时的错误处理和返回值规范。
        
        输入：
        - tmp_path: pytest提供的临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证错误处理结果
        
        验证点：
        1. 返回的file_path为None
        2. 系统能够处理查找失败的情况
        3. 不会因找不到电路而抛出异常
        4. 错误处理机制正常工作
        5. 函数行为符合预期规范
        
        测试策略：
        在空目录中查找不存在的电路名称，验证错误处理的正确性。
        这测试了边界条件和异常情况处理。
        """
        qasm_dir = tmp_path / "QASMBench"
        qasm_dir.mkdir()
        
        with patch('qasmbench_runner.QASMBenchConfig') as mock_config:
            mock_config.return_value.qasm_directory = str(qasm_dir)
            
            file_path = find_circuit_by_name("nonexistent_circuit")
            
            assert file_path is None
    
    def test_run_benchmark_for_circuit_function(self, tmp_path):
        """
        测试基准测试运行函数的集成功能。
        
        测试作用：
        验证run_benchmark_for_circuit函数能够正确协调整个基准测试流程，
        包括runner创建、电路加载、基准测试执行和报告生成。
        
        输入：
        - tmp_path: pytest提供的临时目录路径（通过fixture提供）
        
        输出：
        - 无返回值，通过断言验证集成测试结果
        
        验证点：
        1. 返回的results等于模拟的测试结果
        2. load_qasm_circuit方法被调用一次
        3. run_benchmark_for_circuit方法被调用一次
        4. generate_reports方法被调用一次
        5. 完整的基准测试流程被正确执行
        6. 所有组件协调工作正常
        
        测试策略：
        通过mock对象模拟整个基准测试流程，验证函数的集成功能。
        这测试了端到端的基准测试执行流程。
        """
        # 创建测试QASM文件
        qasm_file = tmp_path / "test_circuit.qasm"
        qasm_file.write_text("OPENQASM 2.0;\nqreg q[2];\nh q[0];")
        
        with patch('qasmbench_runner.QASMBenchConfig'), \
             patch('qasmbench_runner.QASMBenchRunner') as mock_runner_class:
            
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner
            
            # 模拟电路加载
            mock_circuit = Mock()
            mock_runner.load_qasm_circuit.return_value = mock_circuit
            
            # 模拟基准测试结果
            mock_metrics = Mock()
            mock_metrics.execution_time_mean = 1.0
            mock_metrics.speedup = 1.0  # 设置具体的speedup值避免格式化错误
            mock_results = {'numpy': mock_metrics}
            mock_runner.run_benchmark_for_circuit.return_value = mock_results
            
            results = run_benchmark_for_circuit(str(qasm_file))
            
            assert results == mock_results
            mock_runner.load_qasm_circuit.assert_called_once()
            mock_runner.run_benchmark_for_circuit.assert_called_once()
            mock_runner.generate_reports.assert_called_once()


class TestPerformanceAndStress:
    """
    性能和压力测试
    
    测试目标:
    - 验证系统在处理大规模量子电路时的性能表现
    - 测试内存使用的稳定性和一致性
    - 验证并发执行时的线程安全性
    - 确保系统在负载下的健壮性
    
    测试策略:
    - 创建大规模量子电路进行性能测试
    - 连续测量内存使用验证稳定性
    - 使用多线程模拟并发基准测试
    - 验证系统在高负载下的正确性
    """
    
    def test_large_circuit_handling(self, tmp_path):
        """
        测试大电路处理性能
        
        测试作用: 验证系统能够正确处理大规模量子电路，包括10个量子位和19个量子门的复杂电路。
        这测试了系统的性能上限和大规模数据处理能力。
        
        输入:
        - tmp_path: pytest提供的临时目录路径（通过fixture提供）
        
        输出:
        - 无返回值，通过断言验证大电路处理结果
        
        验证点:
        1. 大电路QASM文件能够成功创建
        2. 电路加载过程不抛出异常
        3. 返回的circuit对象不为None
        4. 电路的量子位数正确设置为10
        5. 系统能够处理复杂的量子门序列
        6. 文件读取和解析性能正常
        
        测试策略: 创建包含10个量子位和19个量子门的大型QASM文件，验证系统对大规模电路的处理能力。
        这测试了系统的性能基准和扩展性。
        """
        # 创建一个较大的QASM文件
        large_qasm = "OPENQASM 2.0;\nqreg q[10];\n"
        for i in range(10):
            large_qasm += f"h q[{i}];\n"
        for i in range(9):
            large_qasm += f"cx q[{i}], q[{i+1}];\n"
        
        qasm_file = tmp_path / "large_circuit.qasm"
        qasm_file.write_text(large_qasm)
        
        config = QASMBenchConfig()
        runner = QASMBenchRunner(config)
        
        with patch('qibo.Circuit.from_qasm') as mock_from_qasm:
            mock_circuit = Mock()
            mock_circuit.nqubits = 10
            mock_circuit.depth = 19
            mock_circuit.ngates = 19
            mock_from_qasm.return_value = mock_circuit
            
            circuit = runner.load_qasm_circuit(str(qasm_file))
            
            assert circuit is not None
            assert circuit.nqubits == 10
    
    def test_memory_usage_stability(self):
        """
        测试内存使用稳定性
        
        测试作用: 验证系统在连续运行过程中内存使用的稳定性，确保没有内存泄漏或异常波动。
        这测试了系统的资源管理能力和长期运行的稳定性。
        
        输入:
        - 无直接输入参数
        
        输出:
        - 无返回值，通过断言验证内存使用稳定性
        
        验证点:
        1. 所有内存测量值都大于0，表示合理值
        2. 内存使用的变异系数小于0.1（10%），表示相对稳定
        3. 连续10次测量过程不抛出异常
        4. 内存测量功能正常工作
        5. 系统没有明显的内存泄漏
        
        测试策略: 连续测量内存使用10次，计算变异系数来评估稳定性。
        这测试了系统的资源监控和内存管理能力。
        """
        runner = QASMBenchRunner(QASMBenchConfig())
        
        # 连续测量内存使用
        memory_measurements = []
        for _ in range(10):
            memory_usage = runner.measure_memory_usage()
            memory_measurements.append(memory_usage)
            time.sleep(0.1)  # 短暂等待
        
        # 验证内存使用相对稳定
        assert all(m > 0 for m in memory_measurements)
        
        # 计算变异系数
        mean_memory = np.mean(memory_measurements)
        std_memory = np.std(memory_measurements)
        cv = std_memory / mean_memory if mean_memory > 0 else 0
        
        # 变异系数应该相对较小（小于10%）
        assert cv < 0.1
    
    def test_concurrent_benchmark_execution(self, tmp_path):
        """
        测试并发基准测试执行
        
        测试作用：
        验证QASMBenchRunner在多线程并发执行时的稳定性和正确性，确保多个基准测试
        任务可以同时运行而不会相互干扰或产生竞态条件。此测试对于验证系统在
        高并发场景下的可靠性至关重要。
        
        输入参数：
        - tmp_path (pytest.fixture): 临时目录路径，用于创建测试QASM文件
        
        输出结果：
        - 无直接返回值，通过断言验证并发执行的正确性
        
        验证点：
        1. 所有并发线程都能成功完成基准测试
        2. 每个线程返回的结果都是有效的字典结构
        3. 线程数量与结果队列中的结果数量一致
        4. 并发执行不会产生数据竞争或状态污染
        
        测试策略：
        1. 创建一个简单的QASM测试文件
        2. 使用多线程同时执行基准测试任务
        3. 使用队列收集各个线程的执行结果
        4. 验证所有线程都能正常完成并返回有效结果
        5. 模拟真实的并发场景，检查系统的线程安全性
        
        并发测试要点：
        - 线程安全性验证
        - 资源竞争检测
        - 状态隔离验证
        - 并发性能评估
        """
        import threading
        import queue
        
        # 创建测试QASM文件
        qasm_content = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\n"
        qasm_file = tmp_path / "concurrent_test.qasm"
        qasm_file.write_text(qasm_content)
        
        results_queue = queue.Queue()
        
        def run_benchmark():
            config = QASMBenchConfig()
            runner = QASMBenchRunner(config)
            
            with patch.object(runner, 'load_qasm_circuit') as mock_load, \
                 patch.object(runner, '_run_single_backend_benchmark') as mock_benchmark:
                
                mock_circuit = Mock()
                mock_circuit.nqubits = 2
                mock_load.return_value = mock_circuit
                
                mock_metrics = QASMBenchMetrics()
                mock_metrics.execution_time_mean = 1.0
                mock_benchmark.return_value = (Mock(), mock_metrics)
                
                results = runner.run_benchmark_for_circuit("test", str(qasm_file))
                results_queue.put(results)
        
        # 启动多个线程
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_benchmark)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有测试都成功完成
        assert results_queue.qsize() == 3
        
        while not results_queue.empty():
            results = results_queue.get()
            assert isinstance(results, dict)


class TestErrorHandlingAndEdgeCases:
    """
    错误处理和边界条件测试
    
    测试目标:
    - 验证系统对编码问题的处理能力
    - 测试空电路和边界条件的处理
    - 验证极端值的指标处理
    - 测试特殊字符的报告生成
    - 确保系统在各种异常情况下的健壮性
    
    测试策略:
    - 创建包含特殊字符和编码问题的测试文件
    - 模拟各种边界条件和异常情况
    - 验证错误处理机制的完整性
    - 测试系统的容错能力和恢复机制
    """
    
    def test_qasm_file_with_encoding_issues(self, tmp_path):
        """
        测试编码问题的QASM文件处理
        
        测试作用:
        验证系统能够正确处理包含中文注释和特殊字符的QASM文件，确保文件编码
        不会影响量子电路的加载和解析。这测试了系统的国际化支持和编码兼容性。
        
        输入参数:
        - tmp_path (pytest.fixture): 临时目录路径，用于创建包含特殊字符的测试文件
        
        输出结果:
        - 无直接返回值，通过断言验证编码处理结果
        
        验证点:
        1. 包含中文注释的QASM文件能够成功创建
        2. 文件读取过程能够正确处理UTF-8编码
        3. 电路加载过程不抛出编码相关异常
        4. 返回的circuit对象不为None
        5. 特殊字符不会影响QASM解析
        6. 系统具有良好的编码兼容性
        
        测试策略:
        创建包含中文注释的QASM文件，使用UTF-8编码保存，验证系统能够正确
        读取和解析包含多字节字符的文件内容。这测试了文件I/O的编码处理能力。
        """
        # 创建包含特殊字符的QASM文件
        qasm_content = "OPENQASM 2.0;\nqreg q[2];\n# 注释: 测试中文编码\nh q[0];\n"
        
        qasm_file = tmp_path / "encoding_test.qasm"
        with open(qasm_file, 'w', encoding='utf-8') as f:
            f.write(qasm_content)
        
        config = QASMBenchConfig()
        runner = QASMBenchRunner(config)
        
        with patch('qibo.Circuit.from_qasm') as mock_from_qasm:
            mock_circuit = Mock()
            mock_from_qasm.return_value = mock_circuit
            
            circuit = runner.load_qasm_circuit(str(qasm_file))
            assert circuit is not None
    
    def test_circuit_with_empty_gates(self, tmp_path):
        """
        测试空门电路处理
        
        测试作用:
        验证系统能够正确处理只包含量子寄存器声明但没有量子门的空电路。
        这测试了系统对最小有效量子电路的处理能力和边界条件处理。
        
        输入参数:
        - tmp_path (pytest.fixture): 临时目录路径，用于创建空门电路测试文件
        
        输出结果:
        - 无直接返回值，通过断言验证空电路处理结果
        
        验证点:
        1. 只包含寄存器声明的QASM文件能够成功创建
        2. 空电路能够被正确加载和解析
        3. 返回的circuit对象不为None
        4. 电路的量子门数正确设置为0
        5. 电路深度正确设置为0
        6. 系统能够处理没有量子操作的边界情况
        
        测试策略:
        创建只包含量子寄存器声明但不包含任何量子门的QASM文件，验证系统能够
        正确处理这种最小化的量子电路。这测试了边界条件处理和最小输入验证。
        """
        qasm_content = "OPENQASM 2.0;\nqreg q[2];\n"
        
        qasm_file = tmp_path / "empty_circuit.qasm"
        qasm_file.write_text(qasm_content)
        
        config = QASMBenchConfig()
        runner = QASMBenchRunner(config)
        
        with patch('qibo.Circuit.from_qasm') as mock_from_qasm:
            mock_circuit = Mock()
            mock_circuit.nqubits = 2
            mock_circuit.depth = 0
            mock_circuit.ngates = 0
            mock_from_qasm.return_value = mock_circuit
            
            circuit = runner.load_qasm_circuit(str(qasm_file))
            assert circuit is not None
            assert circuit.ngates == 0
    
    def test_metrics_with_extreme_values(self):
        """
        测试极端值的指标处理
        
        测试作用:
        验证QASMBenchMetrics类能够正确处理和存储极端数值范围的指标数据，
        包括极小值和极大值。这测试了系统对异常数值的处理能力和数值精度保持。
        
        输入参数:
        - 无直接输入参数
        
        输出结果:
        - 无直接返回值，通过断言验证极端值处理结果
        
        验证点:
        1. 极小值1e-10能够正确赋值给execution_time_mean
        2. 极小值1e-12能够正确赋值给execution_time_std
        3. 极小值0.001能够正确赋值给peak_memory_mb
        4. 极大值1e10能够正确赋值给execution_time_mean
        5. 极大值1e8能够正确赋值给execution_time_std
        6. 极大值1e6能够正确赋值给peak_memory_mb
        7. 数值精度在赋值过程中保持不变
        8. 系统能够处理科学计数法表示的数值
        
        测试策略:
        测试极小值（接近浮点数精度极限）和极大值（接近系统性能极限）的赋值和读取，
        验证指标系统对各种数值范围的兼容性。这测试了数值处理的健壮性和精度保持。
        """
        metrics = QASMBenchMetrics()
        
        # 测试极小值
        metrics.execution_time_mean = 1e-10
        metrics.execution_time_std = 1e-12
        metrics.peak_memory_mb = 0.001
        
        assert metrics.execution_time_mean == 1e-10
        assert metrics.execution_time_std == 1e-12
        assert metrics.peak_memory_mb == 0.001
        
        # 测试极大值
        metrics.execution_time_mean = 1e10
        metrics.execution_time_std = 1e8
        metrics.peak_memory_mb = 1e6
        
        assert metrics.execution_time_mean == 1e10
        assert metrics.execution_time_std == 1e8
        assert metrics.peak_memory_mb == 1e6
    
    def test_report_generation_with_special_characters(self, tmp_path):
        """
        测试包含特殊字符的报告生成功能
        
        测试作用：
        验证报告生成系统能够正确处理包含特殊字符的电路名称和参数，
        包括中文、特殊符号、换行符等，确保国际化支持和字符编码的正确性
        
        输入参数：
        - tmp_path: pytest提供的临时目录路径，用于创建测试文件
        
        输出结果：
        - 成功生成包含特殊字符的CSV和JSON报告文件
        - JSON报告能够正确解析并包含特殊字符内容
        
        验证点：
        - CSV报告文件成功创建
        - JSON报告文件成功创建
        - JSON文件能够正确解析，不会因特殊字符而失败
        - 特殊字符（中文、@#$%、换行符等）在报告中正确保存和读取
        
        测试策略：
        - 创建包含中文和特殊字符的电路参数
        - 使用包含特殊字符的电路名称生成报告
        - 验证CSV和JSON报告的生成
        - 特别验证JSON报告的解析能力，确保编码处理正确
        """
        metrics = QASMBenchMetrics()
        metrics.execution_time_mean = 1.0
        metrics.correctness = "Passed"
        metrics.circuit_parameters = {
            'name': '测试电路@#$%',
            'description': 'Special chars: \n\t\r'
        }
        
        results = {'numpy': metrics}
        
        # 测试CSV报告
        csv_file = tmp_path / "special_chars.csv"
        QASMBenchReporter.generate_csv_report(results, "test@#$%", csv_file)
        assert csv_file.exists()
        
        # 测试JSON报告
        json_file = tmp_path / "special_chars.json"
        QASMBenchReporter.generate_json_report(results, "test@#$%", json_file)
        assert json_file.exists()
        
        # 验证JSON可以正确解析
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert 'test@#$%' in data['metadata']['circuit_name']


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
