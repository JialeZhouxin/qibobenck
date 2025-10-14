"""
量子傅里叶变换电路实现

这个模块实现了量子傅里叶变换(QFT)电路，支持多个量子计算平台。
"""

from typing import Any

import numpy as np
import qibo
from qiskit import QuantumCircuit

from benchmark_harness.abstractions import BenchmarkCircuit


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
                theta = np.pi / (2 ** (k - j))
                c.add(qibo.gates.CU1(k, j, theta))

        # 应用交换门
        for j in range(n_qubits // 2):
            c.add(qibo.gates.SWAP(j, n_qubits - j - 1))

        c.name = f"qft_{n_qubits}_qubits"
        return c

    def _build_qiskit_qft(self, n_qubits: int) -> QuantumCircuit:
        """构建Qiskit平台的QFT电路"""
        qc = QuantumCircuit(n_qubits)

        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                theta = np.pi / (2 ** (k - j))
                qc.cp(theta, k, j)  # 使用cp代替cu1

        # 应用交换门
        for j in range(n_qubits // 2):
            qc.swap(j, n_qubits - j - 1)

        qc.name = f"qft_{n_qubits}_qubits"
        return qc

    def _build_pennylane_qft(self, n_qubits: int):
        """构建PennyLane平台的QFT电路"""
        import pennylane as qml

        def qft_circuit(wires):
            """PennyLane QFT电路函数"""
            # 应用Hadamard门和受控相位门
            for j in range(n_qubits):
                qml.Hadamard(wires=wires[j])
                for k in range(j + 1, n_qubits):
                    theta = np.pi / (2 ** (k - j))
                    qml.ControlledPhaseShift(theta, wires=[wires[k], wires[j]])

            # 应用交换门
            for j in range(n_qubits // 2):
                qml.SWAP(wires=[wires[j], wires[n_qubits - j - 1]])

        qft_circuit.name = f"qft_{n_qubits}_qubits"
        return qft_circuit
