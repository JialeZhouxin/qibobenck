"""
类型定义模块

定义了基准测试框架中使用的类型别名和协议。
"""

from typing import (
    Any, Union, Optional, List, Dict, Tuple, Callable, Protocol, TypeVar,
    runtime_checkable, Generic, NewType, TypedDict
)
import numpy as np
import pandas as pd

# =============================================================================
# 基础类型别名
# =============================================================================
Number = Union[int, float]
Array = np.ndarray
DataFrame = pd.DataFrame
Path = str

# 平台和后端类型
Platform = NewType('Platform', str)
Backend = NewType('Backend', str)
CircuitType = NewType('CircuitType', str)

# 性能指标类型
Seconds = NewType('Seconds', float)
Megabytes = NewType('Megabytes', float)
Percentage = NewType('Percentage', float)
Fidelity = NewType('Fidelity', float)

# 标识符类型
RunId = NewType('RunId', int)
QubitCount = NewType('QubitCount', int)

# =============================================================================
# 协议定义
# =============================================================================

@runtime_checkable
class CircuitProtocol(Protocol):
    """电路协议接口"""
    name: str
    nqubits: int

    def execute(self, **kwargs) -> Any:
        """执行电路"""
        ...

    def __call__(self, **kwargs) -> Any:
        """调用电路"""
        ...


@runtime_checkable
class SimulatorProtocol(Protocol):
    """模拟器协议接口"""
    platform_name: str
    backend_name: str

    def execute(self, circuit: Any, n_qubits: int, **kwargs) -> List[Any]:
        """执行电路"""
        ...


@runtime_checkable
class MetricsProtocol(Protocol):
    """指标收集协议接口"""
    def __enter__(self) -> 'MetricsProtocol':
        """进入上下文"""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文"""
        ...

    def get_results(self) -> Dict[str, Any]:
        """获取结果"""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """缓存协议接口"""
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        ...

    def set(self, key: str, value: Any) -> bool:
        """设置缓存值"""
        ...

    def clear(self) -> None:
        """清空缓存"""
        ...

# =============================================================================
# 泛型类型变量
# =============================================================================
T = TypeVar('T')
CircuitT = TypeVar('CircuitT', bound=CircuitProtocol)
SimulatorT = TypeVar('SimulatorT', bound=SimulatorProtocol)
MetricsT = TypeVar('MetricsT', bound=MetricsProtocol)
CacheT = TypeVar('CacheT', bound=CacheProtocol)

# =============================================================================
# 配置类型
# =============================================================================

class CircuitConfig(TypedDict, total=False):
    """电路配置类型"""
    name: str
    n_qubits: int
    depth: Optional[int]
    parameters: Optional[Dict[str, Any]]


class SimulatorConfig(TypedDict, total=False):
    """模拟器配置类型"""
    platform: str
    backend: str
    shots: Optional[int]
    device: Optional[str]


class BenchmarkConfig(TypedDict, total=False):
    """基准测试配置类型"""
    repeat: int
    warmup_runs: int
    timeout: Optional[float]
    enable_metrics: bool
    enable_cache: bool


# =============================================================================
# 结果类型
# =============================================================================

class PerformanceMetrics(TypedDict):
    """性能指标类型"""
    wall_time_sec: Seconds
    cpu_time_sec: Seconds
    peak_memory_mb: Megabytes
    cpu_utilization_percent: Percentage


class StatisticalMetrics(TypedDict, total=False):
    """统计指标类型"""
    mean: float
    std: float
    min: float
    max: float
    confidence_interval: Optional[Tuple[float, float]]


class ExecutionResult(TypedDict):
    """执行结果类型"""
    success: bool
    data: Optional[Any]
    error: Optional[str]
    metrics: Optional[PerformanceMetrics]


# =============================================================================
# 缓存相关类型
# =============================================================================

class CacheKey(TypedDict):
    """缓存键类型"""
    circuit_type: str
    n_qubits: int
    backend: str
    parameters_hash: str


class CacheEntry(TypedDict):
    """缓存条目类型"""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    expires_at: Optional[float]


class CacheStats(TypedDict):
    """缓存统计类型"""
    hits: int
    misses: int
    size: int
    memory_usage_mb: Megabytes
    hit_rate: Percentage

# =============================================================================
# 回调和事件类型
# =============================================================================

ProgressCallback = Callable[[float, str], None]  # (progress, message) -> None
ErrorCallback = Callable[[Exception, str], None]  # (error, context) -> None
CompletionCallback = Callable[[List[Any]], None]  # (results) -> None

class BenchmarkEvent(TypedDict):
    """基准测试事件类型"""
    type: str
    timestamp: float
    data: Optional[Dict[str, Any]]
    message: Optional[str]

# =============================================================================
# 可视化类型
# =============================================================================

PlotStyle = Dict[str, Any]
ColorScheme = List[str]
FigureSize = Tuple[float, float]

class PlotConfig(TypedDict, total=False):
    """绘图配置类型"""
    figure_size: FigureSize
    dpi: int
    style: str
    color_scheme: ColorScheme
    title: Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    legend: bool
    grid: bool

# =============================================================================
# 实用工具函数类型
# =============================================================================

ValidatorFunc = Callable[[Any], bool]
TransformerFunc = Callable[[Any], Any]
FilterFunc = Callable[[Any], bool]

# =============================================================================
# 类型检查函数
# =============================================================================

def is_valid_platform(platform: Any) -> bool:
    """检查是否为有效的平台类型"""
    return isinstance(platform, str) and len(platform.strip()) > 0


def is_valid_backend(backend: Any) -> bool:
    """检查是否为有效的后端类型"""
    return isinstance(backend, str) and len(backend.strip()) > 0


def is_valid_circuit(circuit: Any) -> bool:
    """检查是否为有效的电路对象"""
    return hasattr(circuit, 'execute') or hasattr(circuit, '__call__')


def is_valid_metrics(metrics: Any) -> bool:
    """检查是否为有效的指标字典"""
    return isinstance(metrics, dict) and 'wall_time_sec' in metrics


def is_positive_number(value: Any) -> bool:
    """检查是否为正数"""
    return isinstance(value, (int, float)) and value > 0


def is_non_negative_number(value: Any) -> bool:
    """检查是否为非负数"""
    return isinstance(value, (int, float)) and value >= 0


def is_valid_fidelity(fidelity: Any) -> bool:
    """检查是否为有效的保真度值"""
    return isinstance(fidelity, (int, float)) and 0.0 <= fidelity <= 1.0


def is_valid_array(array: Any) -> bool:
    """检查是否为有效的numpy数组"""
    return isinstance(array, np.ndarray) and array.size > 0


def is_valid_qubit_count(n_qubits: Any) -> bool:
    """检查是否为有效的量子比特数"""
    return isinstance(n_qubits, int) and 1 <= n_qubits <= 30


# =============================================================================
# 类型转换函数
# =============================================================================

def to_platform(value: Any) -> Platform:
    """转换为平台类型"""
    if not is_valid_platform(value):
        raise ValueError(f"Invalid platform: {value}")
    return Platform(str(value))


def to_backend(value: Any) -> Backend:
    """转换为后端类型"""
    if not is_valid_backend(value):
        raise ValueError(f"Invalid backend: {value}")
    return Backend(str(value))


def to_qubit_count(value: Any) -> QubitCount:
    """转换为量子比特数类型"""
    if not is_valid_qubit_count(value):
        raise ValueError(f"Invalid qubit count: {value}")
    return QubitCount(int(value))


def to_fidelity(value: Any) -> Fidelity:
    """转换为保真度类型"""
    if not is_valid_fidelity(value):
        raise ValueError(f"Invalid fidelity: {value}")
    return Fidelity(float(value))


def to_seconds(value: Any) -> Seconds:
    """转换为秒类型"""
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Invalid time value: {value}")
    return Seconds(float(value))


def to_megabytes(value: Any) -> Megabytes:
    """转换为兆字节类型"""
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Invalid memory value: {value}")
    return Megabytes(float(value))


def to_percentage(value: Any) -> Percentage:
    """转换为百分比类型"""
    if not isinstance(value, (int, float)) or not (0.0 <= value <= 100.0):
        raise ValueError(f"Invalid percentage value: {value}")
    return Percentage(float(value))

# =============================================================================
# 运行时类型检查装饰器
# =============================================================================

from functools import wraps

def validate_types(**type_hints):
    """类型验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查参数类型
            for arg_name, (arg_value, expected_type) in zip(
                list(type_hints.keys())[:len(args)],
                zip(args, list(type_hints.values())[:len(args)])
            ):
                if not isinstance(arg_value, expected_type):
                    raise TypeError(
                        f"Argument '{arg_name}' must be {expected_type}, "
                        f"got {type(arg_value).__name__}"
                    )

            return func(*args, **kwargs)
        return wrapper
    return decorator