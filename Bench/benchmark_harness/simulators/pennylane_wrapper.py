"""
PennyLane模拟器封装器

这个模块实现了PennyLane量子计算框架的封装器，用于基准测试。
"""

from typing import Any, Optional, List

import numpy as np
from scipy import stats

from benchmark_harness.abstractions import BenchmarkResult, SimulatorInterface
from benchmark_harness.metrics import MetricsCollector

# 尝试导入PennyLane依赖
try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    PENNYLANE_AVAILABLE = False


class PennylaneWrapper(SimulatorInterface):
    """Pennylane模拟器的封装器实现"""

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
        self,
        circuit: Any,
        n_qubits: int,
        reference_state: Optional[np.ndarray] = None,
        repeat: int = 1,
        warmup_runs: int = 0
    ) -> List[BenchmarkResult]:
        """执行PennyLane电路并返回基准测试结果列表"""
        if repeat <= 0:
            raise ValueError(f"repeat must be positive, got {repeat}")
        if warmup_runs < 0:
            raise ValueError(f"warmup_runs must be non-negative, got {warmup_runs}")
            
        results = []
        
        # 预热运行
        for _ in range(warmup_runs):
            try:
                dev = qml.device(self.backend_name, wires=n_qubits)
                @qml.qnode(dev)
                def qnode():
                    circuit(dev.wires)
                    return qml.state()
                qnode()
            except Exception:
                pass  # 忽略预热运行的错误
        
        # 正式运行
        wall_times = []
        cpu_times = []
        memory_usages = []
        cpu_utilizations = []
        fidelities = []
        final_states = []
        
        for run_id in range(repeat):
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
                    final_states.append(final_state)

                except Exception as e:
                    raise RuntimeError(f"Failed to execute PennyLane circuit (run {run_id+1}): {e}")
            
            metrics = collector.get_results()
            
            # 计算保真度
            fidelity = -1.0
            if reference_state is not None:
                try:
                    fidelity = np.abs(np.vdot(reference_state, final_state)) ** 2
                except Exception as e:
                    raise ValueError(f"Failed to calculate fidelity (run {run_id+1}): {e}")
            
            # 收集指标
            wall_times.append(metrics.get("wall_time_sec", 0.0))
            cpu_times.append(metrics.get("cpu_time_sec", 0.0))
            memory_usages.append(metrics.get("peak_memory_mb", 0.0))
            cpu_utilizations.append(metrics.get("cpu_utilization_percent", 0.0))
            fidelities.append(fidelity)
            
            # 创建单次运行结果
            result = BenchmarkResult(
                simulator="pennylane",
                backend=self.backend_name,
                circuit_name=getattr(circuit, "name", "unknown"),
                n_qubits=n_qubits,
                run_id=run_id + 1,
                wall_time_sec=wall_times[-1],
                cpu_time_sec=cpu_times[-1],
                peak_memory_mb=memory_usages[-1],
                cpu_utilization_percent=cpu_utilizations[-1],
                state_fidelity=fidelities[-1],
                final_state=final_state,
            )
            results.append(result)
        
        # 如果多次运行，计算统计信息并更新第一个结果
        if repeat > 1:
            # 计算统计量
            wall_times_arr = np.array(wall_times)
            cpu_times_arr = np.array(cpu_times)
            memory_arr = np.array(memory_usages)
            fidelities_arr = np.array(fidelities)
            
            # 计算置信区间（95%）
            try:
                wall_ci = stats.t.interval(0.95, len(wall_times)-1,
                                         loc=wall_times_arr.mean(),
                                         scale=stats.sem(wall_times_arr))
            except (ValueError, ZeroDivisionError):
                # 如果无法计算统计量，使用简单估计
                wall_ci = (wall_times_arr.min(), wall_times_arr.max())
            
            # 更新第一个结果为汇总结果
            results[0].wall_time_mean = float(wall_times_arr.mean())
            results[0].wall_time_std = float(wall_times_arr.std(ddof=1))
            results[0].wall_time_min = float(wall_times_arr.min())
            results[0].wall_time_max = float(wall_times_arr.max())
            results[0].cpu_time_mean = float(cpu_times_arr.mean())
            results[0].cpu_time_std = float(cpu_times_arr.std(ddof=1))
            results[0].memory_mean = float(memory_arr.mean())
            results[0].memory_std = float(memory_arr.std(ddof=1))
            results[0].fidelity_mean = float(fidelities_arr.mean())
            results[0].fidelity_std = float(fidelities_arr.std(ddof=1))
            results[0].confidence_interval = wall_ci
        
        return results
