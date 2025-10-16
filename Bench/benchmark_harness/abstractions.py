"""
核心抽象层定义

这个模块定义了基准测试平台的所有核心抽象接口和数据结构。
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class BenchmarkResult:
    """标准化的基准测试结果数据结构"""

    # 元数据
    simulator: str
    backend: str
    circuit_name: str
    n_qubits: int

    # 速度指标
    wall_time_sec: float
    cpu_time_sec: float

    # 资源指标
    peak_memory_mb: float
    cpu_utilization_percent: float

    # 正确性指标
    state_fidelity: float

    # 原始输出供参考
    final_state: np.ndarray

    # 运行ID，用于区分多次运行
    run_id: int = 1

    # 统计信息（当repeat > 1时使用）
    wall_time_mean: Optional[float] = None
    wall_time_std: Optional[float] = None
    wall_time_min: Optional[float] = None
    wall_time_max: Optional[float] = None
    cpu_time_mean: Optional[float] = None
    cpu_time_std: Optional[float] = None
    memory_mean: Optional[float] = None
    memory_std: Optional[float] = None
    fidelity_mean: Optional[float] = None
    fidelity_std: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class SimulatorInterface(ABC):
    """模拟器封装器的统一接口"""

    platform_name: str

    @abstractmethod
    def __init__(self, backend_name: str):
        """初始化封装器并配置特定后端"""
        self.backend_name = backend_name
        pass

    @abstractmethod
    def execute(
        self,
        circuit: Any,
        n_qubits: int,
        reference_state: Optional[np.ndarray] = None,
        repeat: int = 1,
        warmup_runs: int = 0
    ) -> list[BenchmarkResult]:
        """执行给定电路并返回结果列表
        
        Args:
            circuit: 要执行的电路
            n_qubits: 量子比特数
            reference_state: 参考态（用于保真度计算）
            repeat: 重复运行次数
            warmup_runs: 预热运行次数
            
        Returns:
            list[BenchmarkResult]: 包含所有运行结果的列表
        """
        pass


class BenchmarkCircuit(ABC):
    """基准测试电路的工厂接口"""

    name: str = "Abstract Benchmark Circuit"

    @abstractmethod
    def build(self, platform: str, n_qubits: int) -> Any:
        """为指定平台构建并返回原生电路对象"""
        pass
