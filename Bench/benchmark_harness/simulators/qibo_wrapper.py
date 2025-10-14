"""
Qibo模拟器封装器

这个模块实现了Qibo量子计算框架的封装器，用于基准测试。
"""

from typing import Optional

import numpy as np
import qibo

from benchmark_harness.abstractions import BenchmarkResult, SimulatorInterface
from benchmark_harness.metrics import MetricsCollector


class QiboWrapper(SimulatorInterface):
    """Qibo模拟器的封装器实现"""

    platform_name = "qibo"

    def __init__(self, backend_name: str):
        """初始化Qibo封装器并配置后端"""
        self.backend_name = backend_name
        try:
            qibo.set_backend(backend_name)
        except Exception as e:
            raise ValueError(f"Failed to set Qibo backend '{backend_name}': {e}")

    def execute(
        self,
        circuit: qibo.models.Circuit,
        n_qubits: int,
        reference_state: Optional[np.ndarray] = None,
    ) -> BenchmarkResult:
        """执行Qibo电路并返回基准测试结果"""
        collector = MetricsCollector()

        with collector:
            try:
                # 执行电路
                qibo_result = circuit(nshots=1)
                final_state = qibo_result.state()
            except Exception as e:
                raise RuntimeError(f"Failed to execute Qibo circuit: {e}")

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
            simulator="qibo",
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
