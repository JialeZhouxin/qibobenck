"""
Qibo模拟器封装器

这个模块实现了Qibo量子计算框架的封装器，用于基准测试。
"""

from typing import Optional, List

import numpy as np
import qibo
from scipy import stats

from ..abstractions import BenchmarkResult, SimulatorInterface
from ..metrics import MetricsCollector
from ..exceptions import BackendError, ExecutionError, ValidationError, MetricsError, format_error_message
from ..constants import DEFAULT_REPEAT, DEFAULT_WARMUP_RUNS, FIDELITY_INVALID, FIDELITY_MIN, FIDELITY_MAX
from ..validation import validate_execution_parameters


class QiboWrapper(SimulatorInterface):
    """Qibo模拟器的封装器实现"""

    platform_name = "qibo"

    def __init__(self, backend_name: str):
        """初始化Qibo封装器并配置后端"""
        self.backend_name = backend_name
        try:
            qibo.set_backend(backend_name)
        except qibo.backends.BackendNotFoundError as e:
            raise BackendError(
                format_error_message("Backend not found", f"Qibo backend '{backend_name}'", str(e)),
                details={"backend_name": backend_name, "available_backends": qibo.get_available_backends()}
            )
        except qibo.backends.BackendError as e:
            raise BackendError(
                format_error_message("Backend configuration error", f"Qibo backend '{backend_name}'", str(e)),
                details={"backend_name": backend_name}
            )
        except Exception as e:
            raise BackendError(
                format_error_message("Unexpected backend error", f"Qibo backend '{backend_name}'", str(e)),
                details={"backend_name": backend_name}
            )

    def execute(
        self,
        circuit: qibo.models.Circuit,
        n_qubits: int,
        reference_state: Optional[np.ndarray] = None,
        repeat: int = DEFAULT_REPEAT,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
    ) -> List[BenchmarkResult]:
        """执行Qibo电路并返回基准测试结果列表"""
        # 输入验证
        validate_execution_parameters(n_qubits, repeat, warmup_runs, reference_state)

        results = []

        # 预热运行
        self._perform_warmup_runs(circuit, warmup_runs)
        
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
                    qibo_result = circuit(nshots=1)
                    final_state = qibo_result.state()
                    final_states.append(final_state)
                except Exception as e:
                    raise RuntimeError(f"Failed to execute Qibo circuit (run {run_id+1}): {e}")
            
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
                simulator="qibo",
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

    
    def _perform_warmup_runs(self, circuit: qibo.models.Circuit, warmup_runs: int) -> None:
        """执行预热运行"""
        for run_id in range(warmup_runs):
            try:
                circuit(nshots=1)
            except qibo.backends.BackendError as e:
                # 预热运行中的后端错误记录但不中断
                print(f"Warning: Warmup run {run_id+1} failed with backend error: {e}")
            except Exception as e:
                # 其他类型的错误也记录但继续
                print(f"Warning: Warmup run {run_id+1} failed: {e}")

    def _execute_single_run(
        self,
        circuit: qibo.models.Circuit,
        run_id: int,
        reference_state: Optional[np.ndarray],
        n_qubits: int
    ) -> tuple:
        """执行单次运行并返回结果"""
        collector = MetricsCollector()

        try:
            with collector:
                qibo_result = circuit(nshots=1)
                final_state = qibo_result.state()
        except qibo.backends.BackendError as e:
            raise ExecutionError(
                format_error_message("Backend execution error", f"circuit execution (run {run_id+1})", str(e)),
                details={"run_id": run_id, "backend": self.backend_name}
            )
        except Exception as e:
            raise ExecutionError(
                format_error_message("Unexpected execution error", f"circuit execution (run {run_id+1})", str(e)),
                details={"run_id": run_id, "backend": self.backend_name}
            )

        try:
            metrics = collector.get_results()
        except Exception as e:
            raise MetricsError(
                format_error_message("Metrics collection error", f"run {run_id+1}", str(e)),
                details={"run_id": run_id}
            )

        # 计算保真度
        fidelity = self._calculate_fidelity(final_state, reference_state, run_id)

        return final_state, metrics, fidelity

    def _calculate_fidelity(
        self,
        final_state: np.ndarray,
        reference_state: Optional[np.ndarray],
        run_id: int
    ) -> float:
        """计算保真度"""
        if reference_state is None:
            return FIDELITY_INVALID

        try:
            fidelity = np.abs(np.vdot(reference_state, final_state)) ** 2
            # 确保保真度在合理范围内
            if not (FIDELITY_MIN <= fidelity <= FIDELITY_MAX):
                print(f"Warning: Fidelity out of range [{fidelity}] in run {run_id+1}, setting to {FIDELITY_INVALID}")
                return FIDELITY_INVALID
            return fidelity
        except ValueError as e:
            raise ValidationError(
                format_error_message("Fidelity calculation error", f"run {run_id+1}", str(e)),
                details={"run_id": run_id, "final_state_shape": final_state.shape,
                        "reference_state_shape": reference_state.shape if reference_state is not None else None}
            )
        except Exception as e:
            raise ValidationError(
                format_error_message("Unexpected fidelity error", f"run {run_id+1}", str(e)),
                details={"run_id": run_id}
            )
