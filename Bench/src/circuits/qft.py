"""
量子傅里叶变换电路实现

这个模块实现了量子傅里叶变换(QFT)电路，支持多个量子计算平台。
"""

from typing import Any

import numpy as np
import qibo

from ..abstractions import BenchmarkCircuit
from ..constants import PI, QFT_PHASE_FACTOR, get_circuit_name


class QFTCircuit(BenchmarkCircuit):
    """量子傅里叶变换电路实现"""

    name = "Quantum Fourier Transform"

    def build(self, platform: str, n_qubits: int) -> Any:
        """为指定平台构建并返回QFT电路"""
        if platform == "qibo":
            return self._build_qibo_qft(n_qubits)
        elif platform == "qiskit":
            return self._build_qiskit_qft(n_qubits)
        elif platform == "pennylane":
            return self._build_pennylane_qft(n_qubits)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _build_qibo_qft(self, n_qubits: int) -> qibo.models.Circuit:
        """构建Qibo平台的QFT电路"""
        c = qibo.models.Circuit(n_qubits)

        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            c.add(qibo.gates.H(j))
            for k in range(j + 1, n_qubits):
                theta = QFT_PHASE_FACTOR / (2 ** (k - j - 1))
                c.add(qibo.gates.CU1(k, j, theta))

        # 应用交换门
        for j in range(n_qubits // 2):
            c.add(qibo.gates.SWAP(j, n_qubits - j - 1))

        c.name = get_circuit_name("qft", n_qubits)
        return c

    def _build_qiskit_qft(self, n_qubits: int):
        """构建Qiskit平台的QFT电路"""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n_qubits)

        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                theta = QFT_PHASE_FACTOR / (2 ** (k - j - 1))
                qc.cp(theta, k, j)  # 使用cp代替cu1

        # 应用交换门
        for j in range(n_qubits // 2):
            qc.swap(j, n_qubits - j - 1)

        qc.name = get_circuit_name("qft", n_qubits)

        # 创建一个包装类来实现__call__方法
        class CallableQFT:
            def __init__(self, circuit, name):
                self.circuit = circuit
                self.name = name
                self.nqubits = n_qubits

            def __call__(self, *args, **kwargs):
                return self.circuit

        return CallableQFT(qc, qc.name)

    def _build_pennylane_qft(self, n_qubits: int):
        """构建PennyLane平台的QFT电路"""
        import pennylane as qml

        def qft_circuit(wires):
            """PennyLane QFT电路函数"""
            # 应用Hadamard门和受控相位门
            for j in range(n_qubits):
                qml.Hadamard(wires=wires[j])
                for k in range(j + 1, n_qubits):
                    theta = QFT_PHASE_FACTOR / (2 ** (k - j - 1))
                    qml.ControlledPhaseShift(theta, wires=[wires[k], wires[j]])

            # 应用交换门
            for j in range(n_qubits // 2):
                qml.SWAP(wires=[wires[j], wires[n_qubits - j - 1]])

        qft_circuit.name = get_circuit_name("qft", n_qubits)

        # 创建一个包装类来实现__call__方法
        class CallableQFT:
            def __init__(self, circuit_func, name):
                self.circuit_func = circuit_func
                self.name = name
                self.nqubits = n_qubits

            def __call__(self, *args, **kwargs):
                return self.circuit_func(*args, **kwargs)

        return CallableQFT(qft_circuit, qft_circuit.name)

    def __call__(self, *args, **kwargs):
        """执行QFT电路（委托给实际的电路对象）"""
        # 创建临时电路实例来执行
        circuit = self.build("qibo", self.nqubits)
        return circuit(*args, **kwargs)
