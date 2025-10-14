"""
Qiskit模拟器封装器

这个模块实现了Qiskit量子计算框架的封装器，用于基准测试。
"""

from typing import Any, Optional

import numpy as np

from benchmark_harness.abstractions import BenchmarkResult, SimulatorInterface
from benchmark_harness.metrics import MetricsCollector

# 尝试导入Qiskit依赖
try:
    from qiskit_aer import AerSimulator
except ImportError:
    try:
        from qiskit.providers.aer import AerSimulator
    except ImportError:
        AerSimulator = None

try:
    from qiskit.quantum_info import Statevector
except ImportError:
    Statevector = None


class QiskitWrapper(SimulatorInterface):
    """Qiskit模拟器的封装器实现"""

    platform_name = "qiskit"

    def __init__(self, backend_name: str):
        """初始化Qiskit封装器并配置后端"""
        if AerSimulator is None or Statevector is None:
            raise ImportError(
                "Qiskit is not available. Please install qiskit and qiskit-aer."
            )

        self.backend_name = backend_name
        try:
            if backend_name == "aer_simulator":
                self.backend_instance = AerSimulator()
            else:
                # 可以在这里添加其他Qiskit后端的支持
                raise ValueError(f"Unsupported Qiskit backend: {backend_name}")
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Qiskit backend '{backend_name}': {e}"
            )

    def execute(
        self, circuit: Any, n_qubits: int, reference_state: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """执行Qiskit电路并返回基准测试结果"""
        collector = MetricsCollector()

        with collector:
            try:
                # 执行电路
                circuit.save_statevector()
                job = self.backend_instance.run(circuit, shots=1)
                result = job.result()

                # 获取状态向量
                if hasattr(result, "get_statevector"):
                    final_state = result.get_statevector()
                else:
                    # 备用方法
                    final_state = Statevector.from_instruction(circuit).data

            except Exception as e:
                raise RuntimeError(f"Failed to execute Qiskit circuit: {e}")

        # 获取性能指标
        metrics = collector.get_results()

        # 计算保真度
        fidelity = -1.0
        if reference_state is not None:
            try:
                fidelity = np.abs(np.vdot(reference_state, final_state)) ** 2
            except Exception as e:
                raise ValueError(f"Failed to calculate fidelity: {e}")

        # 创建并返回基准测试结果
        return BenchmarkResult(
            simulator="qiskit",
            backend=self.backend_name,
            circuit_name=getattr(circuit, "name", "unknown"),
            n_qubits=n_qubits,
            wall_time_sec=metrics.get("wall_time_sec", 0.0),
            cpu_time_sec=metrics.get("cpu_time_sec", 0.0),
            peak_memory_mb=metrics.get("peak_memory_mb", 0.0),
            cpu_utilization_percent=metrics.get("cpu_utilization_percent", 0.0),
            state_fidelity=fidelity,
            final_state=final_state,
        )
