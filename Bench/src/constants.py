"""
常量定义模块

定义了基准测试框架中使用的所有常量。
"""

import numpy as np
from typing import Final

# =============================================================================
# 数学常量
# =============================================================================
PI: Final[float] = np.pi
QFT_PHASE_FACTOR: Final[float] = np.pi / 2
SQRT_2: Final[float] = np.sqrt(2.0)

# =============================================================================
# 基准测试参数限制
# =============================================================================
MIN_QUBITS: Final[int] = 1
MAX_QUBITS: Final[int] = 30
MIN_REPEAT: Final[int] = 1
MAX_REPEAT: Final[int] = 1000
MIN_WARMUP_RUNS: Final[int] = 0
MAX_WARMUP_RUNS: Final[int] = 100

# 默认参数
DEFAULT_REPEAT: Final[int] = 5
DEFAULT_WARMUP_RUNS: Final[int] = 1
DEFAULT_N_SHOTS: Final[int] = 1024

# =============================================================================
# 性能阈值
# =============================================================================
FIDELITY_MIN: Final[float] = 0.0
FIDELITY_MAX: Final[float] = 1.0
FIDELITY_INVALID: Final[float] = -1.0

# 内存限制 (MB)
MAX_MEMORY_USAGE_MB: Final[int] = 8192  # 8GB
MIN_MEMORY_USAGE_MB: Final[float] = 0.1

# CPU使用率阈值
CPU_UTILIZATION_MIN: Final[float] = 0.0
CPU_UTILIZATION_MAX: Final[float] = 100.0

# 执行时间阈值 (秒)
MAX_EXECUTION_TIME_SEC: Final[float] = 300.0  # 5分钟
MIN_EXECUTION_TIME_SEC: Final[float] = 1e-6  # 1微秒

# =============================================================================
# 缓存配置
# =============================================================================
DEFAULT_MEMORY_CACHE_SIZE: Final[int] = 64
DEFAULT_CACHE_MAX_AGE_DAYS: Final[int] = 30
DEFAULT_CACHE_VERSION: Final[str] = "v1"

# 缓存文件大小限制 (MB)
MAX_CACHE_FILE_SIZE_MB: Final[float] = 100.0

# =============================================================================
# 统计分析
# =============================================================================
CONFIDENCE_LEVEL: Final[float] = 0.95  # 95%置信区间
STATISTICAL_SIGNIFICANCE_THRESHOLD: Final[int] = 3  # 最小样本数

# =============================================================================
# 输出格式
# =============================================================================
PLOT_DPI: Final[int] = 300
PLOT_FIGURE_SIZE: Final[tuple] = (10, 6)
PLOT_BBOX_INCHES: Final[str] = "tight"

# CSV输出配置
CSV_FLOAT_FORMAT: Final[str] = "%.6f"
CSV_INDEX: Final[bool] = False

# =============================================================================
# 日志和报告
# =============================================================================
LOG_LEVELS: Final[dict] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# =============================================================================
# 电路名称模板
# =============================================================================
CIRCUIT_NAME_TEMPLATES: Final[dict] = {
    "qft": "qft_{n_qubits}_qubits",
    "grover": "grover_{n_qubits}_qubits",
    "hardware_efficient": "he_ansatz_{n_qubits}_qubits",
    "qaoa": "qaoa_{n_qubits}_qubits"
}

# =============================================================================
# 后端名称
# =============================================================================
SUPPORTED_BACKENDS: Final[dict] = {
    "qibo": ["numpy", "qibojit", "tensorflow"],
    "qiskit": ["aer_simulator", "qasm_simulator", "statevector_simulator"],
    "pennylane": ["default.qubit", "lightning.qubit", "lightning.gpu"]
}

DEFAULT_BACKENDS: Final[dict] = {
    "qibo": "numpy",
    "qiskit": "aer_simulator",
    "pennylane": "default.qubit"
}

# =============================================================================
# 文件路径模式
# =============================================================================
RESULTS_DIR_PATTERN: Final[str] = "benchmark_{timestamp}"
CACHE_DIR_PATTERN: Final[str] = ".benchmark_cache"

# 输出文件名模式
OUTPUT_FILE_PATTERNS: Final[dict] = {
    "raw_results": "raw_results.csv",
    "detailed_runs": "detailed_runs.csv",
    "summary_report": "summary_report.md",
    "fidelity_plot": "fidelity.png",
    "wall_time_plot": "wall_time_scaling.png",
    "memory_plot": "memory_scaling.png",
    "cpu_time_plot": "cpu_time_scaling.png",
    "cpu_utilization_plot": "cpu_utilization.png",
    "execution_stability_plot": "execution_stability.png",
    "confidence_intervals_plot": "confidence_intervals.png"
}

# =============================================================================
# 错误消息模板
# =============================================================================
ERROR_MESSAGES: Final[dict] = {
    "invalid_parameter": "Invalid {parameter_name}: {details}",
    "backend_not_found": "Backend '{backend_name}' not found",
    "execution_failed": "Execution failed on run {run_id}: {details}",
    "cache_error": "Cache operation failed: {details}",
    "validation_failed": "Validation failed: {details}",
    "metric_collection_failed": "Metrics collection failed: {details}"
}

# =============================================================================
# 配置验证规则
# =============================================================================
VALIDATION_RULES: Final[dict] = {
    "n_qubits": {"min": MIN_QUBITS, "max": MAX_QUBITS, "type": int},
    "repeat": {"min": MIN_REPEAT, "max": MAX_REPEAT, "type": int},
    "warmup_runs": {"min": MIN_WARMUP_RUNS, "max": MAX_WARMUP_RUNS, "type": int},
    "fidelity": {"min": FIDELITY_MIN, "max": FIDELITY_MAX, "type": float},
    "memory_mb": {"min": MIN_MEMORY_USAGE_MB, "max": MAX_MEMORY_USAGE_MB, "type": float},
    "cpu_utilization": {"min": CPU_UTILIZATION_MIN, "max": CPU_UTILIZATION_MAX, "type": float},
    "execution_time": {"min": MIN_EXECUTION_TIME_SEC, "max": MAX_EXECUTION_TIME_SEC, "type": float}
}

# =============================================================================
# 支持的量子电路类型
# =============================================================================
SUPPORTED_CIRCUITS: Final[set] = {
    "qft",
    "grover",
    "hardware_efficient",
    "qaoa",
    "vqe_uccsd"
}

# =============================================================================
# 平台特定的常量
# =============================================================================
SUPPORTED_PLATFORMS: Final[dict] = {
    "qibo": "qibo",
    "qiskit": "qiskit",
    "pennylane": "pennylane"
}

PLATFORM_NAMES: Final[dict] = {
    "qibo": "qibo",
    "qiskit": "qiskit",
    "pennylane": "pennylane"
}

def get_circuit_name(circuit_type: str, n_qubits: int) -> str:
    """获取电路名称

    Args:
        circuit_type: 电路类型
        n_qubits: 量子比特数

    Returns:
        格式化的电路名称
    """
    template = CIRCUIT_NAME_TEMPLATES.get(circuit_type, "{circuit_type}_{n_qubits}_qubits")
    return template.format(circuit_type=circuit_type, n_qubits=n_qubits)

def validate_parameter(parameter_name: str, value: int | float) -> bool:
    """验证参数是否在有效范围内

    Args:
        parameter_name: 参数名称
        value: 参数值

    Returns:
        是否有效
    """
    if parameter_name not in VALIDATION_RULES:
        return False

    rules = VALIDATION_RULES[parameter_name]

    # 检查类型
    if not isinstance(value, rules["type"]):
        return False

    # 检查范围
    if "min" in rules and value < rules["min"]:
        return False
    if "max" in rules and value > rules["max"]:
        return False

    return True