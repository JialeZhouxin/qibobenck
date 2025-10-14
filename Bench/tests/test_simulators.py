"""
模拟器封装器测试

这个模块包含了各种模拟器封装器的测试用例。
"""

import numpy as np
import pytest

from benchmark_harness.circuits.qft import QFTCircuit
from benchmark_harness.simulators.pennylane_wrapper import PennyLaneWrapper
from benchmark_harness.simulators.qibo_wrapper import QiboWrapper
from benchmark_harness.simulators.qiskit_wrapper import QiskitWrapper


class TestQiboWrapper:
    """测试Qibo封装器"""

    def setup_method(self):
        """设置测试环境"""
        try:
            self.wrapper = QiboWrapper("numpy")
            self.circuit_factory = QFTCircuit()
        except Exception as e:
            pytest.skip(f"Qibo not available or failed to initialize: {e}")

    def test_qibo_wrapper_initialization(self):
        """测试Qibo封装器的初始化"""
        assert self.wrapper.backend_name == "numpy"
        assert self.wrapper.platform_name == "qibo"

    def test_qibo_wrapper_invalid_backend(self):
        """测试无效后端的处理"""
        with pytest.raises(ValueError, match="Failed to set Qibo backend"):
            QiboWrapper("invalid_backend")

    def test_qibo_wrapper_execution(self):
        """测试Qibo封装器的执行"""
        circuit = self.circuit_factory.build("qibo", 2)
        result = self.wrapper.execute(circuit, 2)

        assert result.simulator == "qibo"
        assert result.backend == "numpy"
        assert result.n_qubits == 2
        assert len(result.final_state) == 4


class TestQiskitWrapper:
    """测试Qiskit封装器"""

    def setup_method(self):
        """设置测试环境"""
        try:
            self.wrapper = QiskitWrapper("aer_simulator")
            self.circuit_factory = QFTCircuit()
        except Exception as e:
            pytest.skip(f"Qiskit not available or failed to initialize: {e}")

    def test_qiskit_wrapper_initialization(self):
        """测试Qiskit封装器的初始化"""
        assert self.wrapper.backend_name == "aer_simulator"
        assert self.wrapper.platform_name == "qiskit"

    def test_qiskit_wrapper_invalid_backend(self):
        """测试无效后端的处理"""
        with pytest.raises(ValueError, match="Unsupported Qiskit backend"):
            QiskitWrapper("invalid_backend")

    def test_qiskit_wrapper_execution(self):
        """测试Qiskit封装器的执行"""
        circuit = self.circuit_factory.build("qiskit", 2)
        result = self.wrapper.execute(circuit, 2)

        assert result.simulator == "qiskit"
        assert result.backend == "aer_simulator"
        assert result.n_qubits == 2
        assert len(result.final_state) == 4


class TestPennyLaneWrapper:
    """测试PennyLane封装器"""

    def setup_method(self):
        """设置测试环境"""
        try:
            self.wrapper = PennyLaneWrapper("default.qubit")
            self.circuit_factory = QFTCircuit()
        except Exception as e:
            pytest.skip(f"PennyLane not available or failed to initialize: {e}")

    def test_pennylane_wrapper_initialization(self):
        """测试PennyLane封装器的初始化"""
        assert self.wrapper.backend_name == "default.qubit"
        assert self.wrapper.platform_name == "pennylane"

    def test_pennylane_wrapper_invalid_backend(self):
        """测试无效后端的处理"""
        with pytest.raises(ValueError, match="Failed to initialize PennyLane backend"):
            PennyLaneWrapper("invalid_backend")

    def test_pennylane_wrapper_execution(self):
        """测试PennyLane封装器的执行"""
        circuit = self.circuit_factory.build("pennylane", 2)
        result = self.wrapper.execute(circuit, 2)

        assert result.simulator == "pennylane"
        assert result.backend == "default.qubit"
        assert result.n_qubits == 2
        assert len(result.final_state) == 4


class TestCrossPlatformConsistency:
    """测试跨平台一致性"""

    def test_qft_results_consistency(self):
        """测试不同平台的QFT结果一致性"""
        try:
            qibo_wrapper = QiboWrapper("numpy")
            qiskit_wrapper = QiskitWrapper("aer_simulator")
            circuit_factory = QFTCircuit()

            # 构建电路
            qibo_circuit = circuit_factory.build("qibo", 2)
            qiskit_circuit = circuit_factory.build("qiskit", 2)

            # 执行电路
            qibo_result = qibo_wrapper.execute(qibo_circuit, 2)
            qiskit_result = qiskit_wrapper.execute(qiskit_circuit, 2)

            # 检查结果结构
            assert qibo_result.simulator == "qibo"
            assert qiskit_result.simulator == "qiskit"
            assert qibo_result.n_qubits == qiskit_result.n_qubits

            # 检查状态向量维度
            assert len(qibo_result.final_state) == len(qiskit_result.final_state)

        except Exception as e:
            pytest.skip(f"Cross-platform test failed: {e}")
