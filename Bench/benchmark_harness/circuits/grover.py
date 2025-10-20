"""
Grover搜索算法电路实现

这个模块实现了Grover搜索算法(grover)电路，支持多个量子计算平台。
"""


from typing import Any

import numpy as np
import qibo


from benchmark_harness.abstractions import BenchmarkCircuit


class GroverCircuit(BenchmarkCircuit):
    """Grover搜索算法电路实现"""

    name = "Grover's Search Algorithm"

    def build(self, platform: str, n_qubits: int) -> Any:
        """为指定平台构建并返回Grover电路"""
        if platform == "qibo":
            return self._build_qibo_grover(n_qubits)
        elif platform == "qiskit":
            return self._build_qiskit_grover(n_qubits)
        elif platform == "pennylane":
            return self._build_pennylane_grover(n_qubits)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _build_qibo_grover(self, n_qubits: int) -> qibo.models.Circuit:
        """构建Qibo平台的grover电路"""
        O_grover = qibo.models.Circuit(n_qubits)
        O_grover.add(qibo.gates.Z(0).controlled_by(*range(1,n_qubits)))
        for j in range(n_qubits):
            O_grover.add(qibo.gates.H(j))
            O_grover.add(qibo.gates.X(j))
        O_grover.add(qibo.gates.Z(0).controlled_by(*range(1,n_qubits)))
        for j in range(n_qubits):
            O_grover.add(qibo.gates.X(j))
            O_grover.add(qibo.gates.H(j))
            
        c = qibo.models.Circuit(n_qubits )

        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            c.add(qibo.gates.H(j))
        # Grover迭代次数应该是整数 R_opt = round( ( π / (2 * arcsin(1/√N)) - 1 ) / 2 )，其中 N 是搜索空间的大小
        n_iterations = round((np.pi/(2* np.arcsin(1/ np.sqrt(2**n_qubits)) ) -1 )/2)
        for _ in range(n_iterations):
            c += O_grover
        c.add(qibo.gates.M(*range(n_qubits)))

        c.name = f"grover_{n_qubits}_qubits"
        return c

    def _build_qiskit_grover(self, n_qubits: int):
        """构建Qiskit平台的grover电路"""
        from qiskit import QuantumCircuit
        # 构建Grover算子O_grover (对应Qibo中的O_grover)
        O_grover = QuantumCircuit(n_qubits)
        
        # 添加受控Z门: qibo.gates.Z(0).controlled_by(*range(1,n_qubits))
        # 在Qiskit中使用多控制Z门，可以通过Hadamard门和受控X门实现
        O_grover.h(0)
        O_grover.mcx(list(range(1, n_qubits)), 0)  # 多控制X门
        O_grover.h(0)
        
        # 添加Hadamard门和X门
        for j in range(n_qubits):
            O_grover.h(j)
            O_grover.x(j)
        
        # 再次添加受控Z门
        O_grover.h(0)
        O_grover.mcx(list(range(1, n_qubits)), 0)
        O_grover.h(0)
        
        # 再次添加X门和Hadamard门
        for j in range(n_qubits):
            O_grover.x(j)
            O_grover.h(j)
        
        # 创建主电路c (对应Qibo中的c)
        c = QuantumCircuit(n_qubits, n_qubits)  # 包含经典寄存器用于测量
        
        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            c.h(j)
        
        # 计算Grover迭代次数
        n_iterations = round((np.pi/(2* np.arcsin(1/ np.sqrt(2**n_qubits)) ) -1 )/2)
        
        # 应用Grover算子
        for _ in range(n_iterations):
            c.compose(O_grover, inplace=True)
        
        # 添加测量门: qibo.gates.M(*range(n_qubits))
        c.measure(range(n_qubits), range(n_qubits))

        c.name = f"grover_{n_qubits}_qubits"
        return c

    def _build_pennylane_grover(self, n_qubits: int):
        """构建PennyLane平台的grover电路"""
        import pennylane as qml

        def oracle():
            """Oracle函数"""
            # Multi-controlled Z gate
            qml.Hadamard(wires=0)
            qml.MultiControlledX(wires=list(range(1, n_qubits)) + [0])
            qml.Hadamard(wires=0)

        def grover_diffusion():
            """Grover扩散算子"""
            # Hadamard gates
            for j in range(n_qubits):
                qml.Hadamard(wires=j)
            
            # X gates
            for j in range(n_qubits):
                qml.PauliX(wires=j)
            
            # Multi-controlled Z gate
            qml.Hadamard(wires=0)
            qml.MultiControlledX(wires=list(range(1, n_qubits)) + [0])
            qml.Hadamard(wires=0)

            # X gates
            for j in range(n_qubits):
                qml.PauliX(wires=j)
            
            # Hadamard gates
            for j in range(n_qubits):
                qml.Hadamard(wires=j)
        
        # 计算Grover迭代次数
        n_iterations = round(np.round((np.pi/(2 * np.arcsin(1/ np.sqrt(2**n_qubits))) - 1)/2))
        
        def grover_circuit():
            # 初始化Hadamard门
            for j in range(n_qubits):
                qml.Hadamard(wires=j)
            
            # Grover迭代
            for _ in range(n_iterations):
                oracle()
                grover_diffusion()
        
        grover_circuit.name = f"grover_{n_qubits}_qubits"
        return grover_circuit
