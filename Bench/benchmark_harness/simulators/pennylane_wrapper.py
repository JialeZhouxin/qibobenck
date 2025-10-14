"""
PennyLane模拟器封装器

这个模块实现了PennyLane量子计算框架的封装器，用于基准测试。
"""

from typing import Any, Optional

import numpy as np

from benchmark_harness.abstractions import BenchmarkResult, SimulatorInterface
from benchmark_harness.metrics import MetricsCollector

# 尝试导入PennyLane依赖
try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    PENNYLANE_AVAILABLE = False


class PennyLaneWrapper(SimulatorInterface):
    """PennyLane模拟器的封装器实现"""

    platform_name = "pennylane"

    def __init__(self, backend_name: str):
        """初始化PennyLane封装器并配置后端"""
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is not available. Please install pennylane.")

        self.backend_name = backend_name
        try:
            # 测试后端是否可用
            dev = qml.device(backend_name, wires=1)
            if dev is None:
                raise ValueError(f"Backend '{backend_name}' is not available")
        except Exception as e:
            raise ValueError(
                f"Failed to initialize PennyLane backend '{backend_name}': {e}"
            )

    def execute(
        self, circuit: Any, n_qubits: int, reference_state: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """执行PennyLane电路并返回基准测试结果"""
        collector = MetricsCollector()

        with collector:
            try:
                # 创建设备
                dev = qml.device(self.backend_name, wires=n_qubits)

                # 创建QNode
                @qml.qnode(dev)
                def qnode():
                    # 执行电路
                    circuit(dev.wires)
                    return qml.state()

                # 执行并获取状态
                final_state = qnode()

            except Exception as e:
                raise RuntimeError(f"Failed to execute PennyLane circuit: {e}")

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
            simulator="pennylane",
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
