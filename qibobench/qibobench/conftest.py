#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pytest 配置文件
为 QASMBench Runner 测试提供共享的 fixtures 和配置
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock

# 导入被测试的模块
from qasmbench_runner import QASMBenchConfig, QASMBenchMetrics


@pytest.fixture
def temp_qasm_dir():
    """创建临时QASM目录结构"""
    with tempfile.TemporaryDirectory() as temp_dir:
        qasm_dir = os.path.join(temp_dir, "QASMBench")
        
        # 创建small、medium、large目录
        for size in ['small', 'medium', 'large']:
            size_dir = os.path.join(qasm_dir, size)
            os.makedirs(size_dir, exist_ok=True)
            
            # 在每个目录中创建测试电路
            for i in range(2):
                circuit_name = f"test_circuit_{size}_{i}"
                circuit_dir = os.path.join(size_dir, circuit_name)
                os.makedirs(circuit_dir, exist_ok=True)
                
                # 创建原始QASM文件
                qasm_content = f"""
OPENQASM 2.0;
include "qelib1.inc";

qreg q[{2 if size == 'small' else 4 if size == 'medium' else 8}];
creg c[{2 if size == 'small' else 4 if size == 'medium' else 8}];

h q[0];
cx q[0], q[1];
"""
                qasm_file = os.path.join(circuit_dir, f"{circuit_name}.qasm")
                with open(qasm_file, 'w') as f:
                    f.write(qasm_content)
                
                # 创建transpiled QASM文件
                transpiled_file = os.path.join(circuit_dir, f"{circuit_name}_transpiled.qasm")
                with open(transpiled_file, 'w') as f:
                    f.write(qasm_content)
        
        yield qasm_dir


@pytest.fixture
def sample_qasm_content():
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


@pytest.fixture
def sample_metrics():
    """提供示例指标数据"""
    metrics = QASMBenchMetrics()
    # 使用 setattr 来避免类型检查错误
    setattr(metrics, 'execution_time_mean', 1.23)
    setattr(metrics, 'execution_time_std', 0.15)
    setattr(metrics, 'peak_memory_mb', 256.7)
    setattr(metrics, 'speedup', 2.5)
    metrics.correctness = "Passed (fidelity: 0.999999)"
    metrics.circuit_parameters = {
        'nqubits': 5,
        'depth': 12,
        'ngates': 25,
        'source': 'test_circuit.qasm'
    }
    setattr(metrics, 'throughput_gates_per_sec', 20.3)
    setattr(metrics, 'jit_compilation_time', 0.8)
    setattr(metrics, 'circuit_build_time', 0.2)
    metrics.environment_info = {
        'CPU': 'Intel i7',
        'RAM_GB': 16.0,
        'Python': '3.8.10',
        'Backend': 'numpy'
    }
    return metrics


@pytest.fixture
def mock_circuit():
    """提供模拟电路对象"""
    circuit = Mock()
    circuit.nqubits = 3
    circuit.depth = 2
    circuit.ngates = 3
    circuit.summary.return_value = "Circuit summary with Total number of gates = 3"
    return circuit


@pytest.fixture
def mock_result():
    """提供模拟结果对象"""
    result = Mock()
    result.state.return_value = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    return result


@pytest.fixture
def runner_config():
    """提供测试用的配置"""
    config = QASMBenchConfig()
    config.num_runs = 2  # 减少测试时间
    config.warmup_runs = 1
    return config


def create_test_metrics(execution_time_mean=1.0, execution_time_std=0.1, 
                       peak_memory_mb=100.0, correctness="Passed"):
    """创建测试指标的辅助函数"""
    metrics = QASMBenchMetrics()
    setattr(metrics, 'execution_time_mean', execution_time_mean)
    setattr(metrics, 'execution_time_std', execution_time_std)
    setattr(metrics, 'peak_memory_mb', peak_memory_mb)
    metrics.correctness = correctness
    metrics.circuit_parameters = {'nqubits': 2, 'depth': 3, 'ngates': 4}
    return metrics


def create_mock_circuit(nqubits=2, depth=3, ngates=4):
    """创建模拟电路的辅助函数"""
    circuit = Mock()
    circuit.nqubits = nqubits
    circuit.depth = depth
    circuit.ngates = ngates
    circuit.summary.return_value = f"Circuit summary with Total number of gates = {ngates}"
    return circuit


def create_mock_result(state_vector=None):
    """创建模拟结果的辅助函数"""
    result = Mock()
    if state_vector is None:
        state_vector = np.array([1, 0, 0, 0])
    result.state.return_value = state_vector
    return result


class MockCircuit:
    """更完整的模拟电路类"""
    def __init__(self, nqubits=2, depth=3, ngates=4):
        self.nqubits = nqubits
        self.depth = depth
        self.ngates = ngates
        self._execution_count = 0
    
    def __call__(self):
        """模拟电路执行"""
        self._execution_count += 1
        return MockResult(nqubits=self.nqubits)
    
    def summary(self):
        return f"Circuit summary with Total number of gates = {self.ngates}"


class MockResult:
    """更完整的模拟结果类"""
    def __init__(self, nqubits=2):
        self._state = np.zeros(2**nqubits)
        self._state[0] = 1.0  # 初始化为 |00...0> 状态
    
    def state(self):
        return self._state.copy()


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        # 为没有标记的测试添加unit标记
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # 为性能测试添加slow标记
        if "performance" in item.nodeid or "concurrent" in item.nodeid:
            item.add_marker(pytest.mark.slow)
