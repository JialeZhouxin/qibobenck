"""
集成测试

这个模块包含了不同组件之间协同工作的测试用例。
"""

import numpy as np
import pytest

from benchmark_harness.circuits.qft import QFTCircuit
from benchmark_harness.simulators.qibo_wrapper import QiboWrapper


class TestQiboIntegration:
    """测试Qibo封装器和QFT电路的集成"""

    def setup_method(self):
        """设置测试环境"""
        try:
            self.wrapper = QiboWrapper("numpy")
            self.circuit_factory = QFTCircuit()
        except Exception as e:
            pytest.skip(f"Qibo not available or failed to initialize: {e}")

    def test_qibo_qft_circuit_creation(self):
        """测试Qibo QFT电路的创建"""
        circuit = self.circuit_factory.build("qibo", 3)
        assert circuit is not None
        assert circuit.nqubits == 3

    def test_qibo_qft_circuit_execution(self):
        """测试Qibo QFT电路的执行"""
        circuit = self.circuit_factory.build("qibo", 3)
        result = self.wrapper.execute(circuit, 3)

        # 验证结果结构
        assert result.simulator == "qibo"
        assert result.backend == "numpy"
        assert result.circuit_name == "qft_3_qubits"
        assert result.n_qubits == 3
        assert result.wall_time_sec >= 0
        assert result.cpu_time_sec >= 0
        assert result.peak_memory_mb >= 0
        assert 0 <= result.cpu_utilization_percent <= 100
        assert result.state_fidelity == -1.0  # 没有参考态时为-1
        assert result.final_state is not None
        assert len(result.final_state) == 2**3  # 3量子比特对应8维状态向量

    def test_qibo_qft_with_reference_state(self):
        """测试使用参考态的Qibo QFT电路执行"""
        circuit = self.circuit_factory.build("qibo", 2)

        # 首先执行一次获取参考态
        reference_result = self.wrapper.execute(circuit, 2)
        reference_state = reference_result.final_state

        # 再次执行并使用参考态计算保真度
        result = self.wrapper.execute(circuit, 2, reference_state=reference_state)

        # 保真度应该为1（相同电路的执行结果应该完全相同）
        assert abs(result.state_fidelity - 1.0) < 1e-10

    def test_qibo_qft_different_qubit_counts(self):
        """测试不同量子比特数的QFT电路"""
        for n_qubits in [1, 2, 3, 4]:
            circuit = self.circuit_factory.build("qibo", n_qubits)
            result = self.wrapper.execute(circuit, n_qubits)

            assert result.n_qubits == n_qubits
            assert len(result.final_state) == 2**n_qubits

    def test_qibo_backend_configuration(self):
        """测试Qibo后端配置"""
        wrapper = QiboWrapper("numpy")
        assert wrapper.backend_name == "numpy"
        assert wrapper.platform_name == "qibo"


class TestQFTCircuitFactory:
    """测试QFT电路工厂"""

    def test_qft_circuit_qibo_platform(self):
        """测试QFT电路在Qibo平台上的构建"""
        circuit_factory = QFTCircuit()
        circuit = circuit_factory.build("qibo", 3)

        assert circuit is not None
        assert circuit.nqubits == 3

    def test_qft_circuit_qiskit_platform(self):
        """测试QFT电路在Qiskit平台上的构建"""
        try:
            from qiskit import QuantumCircuit

            circuit_factory = QFTCircuit()
            circuit = circuit_factory.build("qiskit", 3)

            assert circuit is not None
            assert circuit.num_qubits == 3
        except ImportError:
            pytest.skip("Qiskit not available")

    def test_qft_circuit_unsupported_platform(self):
        """测试不支持的平台"""
        circuit_factory = QFTCircuit()

        with pytest.raises(ValueError, match="Unsupported platform"):
            circuit_factory.build("unsupported", 3)
