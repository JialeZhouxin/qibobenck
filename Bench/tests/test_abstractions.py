"""
抽象层测试

这个模块包含了核心抽象接口和数据结构的测试用例。
"""

import numpy as np
import pytest

from benchmark_harness.abstractions import (BenchmarkCircuit, BenchmarkResult,
                                            SimulatorInterface)


def test_benchmark_result_creation():
    """测试BenchmarkResult数据类的创建和属性访问"""
    final_state = np.array([1.0, 0.0])
    result = BenchmarkResult(
        simulator="qibo",
        backend="numpy",
        circuit_name="test_circuit",
        n_qubits=1,
        wall_time_sec=0.1,
        cpu_time_sec=0.05,
        peak_memory_mb=10.0,
        cpu_utilization_percent=50.0,
        state_fidelity=1.0,
        final_state=final_state,
    )

    assert result.simulator == "qibo"
    assert result.backend == "numpy"
    assert result.circuit_name == "test_circuit"
    assert result.n_qubits == 1
    assert result.wall_time_sec == 0.1
    assert result.cpu_time_sec == 0.05
    assert result.peak_memory_mb == 10.0
    assert result.cpu_utilization_percent == 50.0
    assert result.state_fidelity == 1.0
    np.testing.assert_array_equal(result.final_state, final_state)


def test_simulator_interface_is_abstract():
    """测试SimulatorInterface无法直接实例化"""
    with pytest.raises(TypeError):
        SimulatorInterface()


def test_benchmark_circuit_is_abstract():
    """测试BenchmarkCircuit无法直接实例化"""
    with pytest.raises(TypeError):
        BenchmarkCircuit()


class TestSimulatorImplementation(SimulatorInterface):
    """用于测试的SimulatorInterface实现"""

    platform_name = "test"

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def execute(self, circuit, n_qubits, reference_state=None):
        return BenchmarkResult(
            simulator="test",
            backend=self.backend_name,
            circuit_name="test",
            n_qubits=n_qubits,
            wall_time_sec=0.0,
            cpu_time_sec=0.0,
            peak_memory_mb=0.0,
            cpu_utilization_percent=0.0,
            state_fidelity=1.0,
            final_state=np.array([1.0]),
        )


class TestCircuitImplementation(BenchmarkCircuit):
    """用于测试的BenchmarkCircuit实现"""

    name = "Test Circuit"

    def build(self, platform: str, n_qubits: int):
        return f"test_circuit_{platform}_{n_qubits}"


def test_simulator_interface_implementation():
    """测试SimulatorInterface的具体实现可以正常工作"""
    simulator = TestSimulatorImplementation("test_backend")
    assert simulator.platform_name == "test"
    assert simulator.backend_name == "test_backend"

    result = simulator.execute("dummy_circuit", 1)
    assert result.simulator == "test"
    assert result.backend == "test_backend"


def test_benchmark_circuit_implementation():
    """测试BenchmarkCircuit的具体实现可以正常工作"""
    circuit = TestCircuitImplementation()
    assert circuit.name == "Test Circuit"

    result = circuit.build("test_platform", 5)
    assert result == "test_circuit_test_platform_5"
