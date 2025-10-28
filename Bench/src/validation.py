"""
输入验证模块

提供基准测试框架中使用的各种输入验证功能。
"""

from typing import Any, Optional, List, Union
import numpy as np

from .constants import (
    VALIDATION_RULES, MIN_QUBITS, MAX_QUBITS, MIN_REPEAT, MAX_REPEAT,
    MIN_WARMUP_RUNS, MAX_WARMUP_RUNS, FIDELITY_MIN, FIDELITY_MAX,
    SUPPORTED_PLATFORMS, SUPPORTED_CIRCUITS
)
from .exceptions import ValidationError, format_error_message


def validate_n_qubits(n_qubits: int) -> None:
    """验证量子比特数参数

    Args:
        n_qubits: 量子比特数

    Raises:
        ValidationError: 如果参数无效
    """
    if not isinstance(n_qubits, int):
        raise ValidationError(
            format_error_message("Invalid type", "n_qubits", f"expected int, got {type(n_qubits).__name__}"),
            details={"n_qubits": n_qubits, "type": type(n_qubits).__name__}
        )

    if n_qubits < MIN_QUBITS or n_qubits > MAX_QUBITS:
        raise ValidationError(
            format_error_message("Invalid n_qubits", "validation", f"must be {MIN_QUBITS}-{MAX_QUBITS}, got {n_qubits}"),
            details={"n_qubits": n_qubits, "min": MIN_QUBITS, "max": MAX_QUBITS}
        )


def validate_repeat_parameters(repeat: int, warmup_runs: int) -> None:
    """验证重复运行参数

    Args:
        repeat: 重复运行次数
        warmup_runs: 预热运行次数

    Raises:
        ValidationError: 如果参数无效
    """
    # 验证repeat
    if not isinstance(repeat, int):
        raise ValidationError(
            format_error_message("Invalid type", "repeat", f"expected int, got {type(repeat).__name__}"),
            details={"repeat": repeat, "type": type(repeat).__name__}
        )

    if repeat < MIN_REPEAT or repeat > MAX_REPEAT:
        raise ValidationError(
            format_error_message("Invalid repeat", "validation", f"must be {MIN_REPEAT}-{MAX_REPEAT}, got {repeat}"),
            details={"repeat": repeat, "min": MIN_REPEAT, "max": MAX_REPEAT}
        )

    # 验证warmup_runs
    if not isinstance(warmup_runs, int):
        raise ValidationError(
            format_error_message("Invalid type", "warmup_runs", f"expected int, got {type(warmup_runs).__name__}"),
            details={"warmup_runs": warmup_runs, "type": type(warmup_runs).__name__}
        )

    if warmup_runs < MIN_WARMUP_RUNS or warmup_runs > MAX_WARMUP_RUNS:
        raise ValidationError(
            format_error_message("Invalid warmup_runs", "validation", f"must be {MIN_WARMUP_RUNS}-{MAX_WARMUP_RUNS}, got {warmup_runs}"),
            details={"warmup_runs": warmup_runs, "min": MIN_WARMUP_RUNS, "max": MAX_WARMUP_RUNS}
        )


def validate_reference_state(reference_state: Optional[np.ndarray], n_qubits: int) -> None:
    """验证参考态

    Args:
        reference_state: 参考态数组，可以为None
        n_qubits: 量子比特数

    Raises:
        ValidationError: 如果参考态无效
    """
    if reference_state is None:
        return

    if not isinstance(reference_state, np.ndarray):
        raise ValidationError(
            format_error_message("Invalid type", "reference_state", f"expected numpy.ndarray, got {type(reference_state).__name__}"),
            details={"reference_state_type": type(reference_state).__name__}
        )

    expected_size = 2 ** n_qubits
    if reference_state.size != expected_size:
        raise ValidationError(
            format_error_message("Invalid reference_state size", "validation",
                              f"expected {expected_size}, got {reference_state.size}"),
            details={"n_qubits": n_qubits, "expected_size": expected_size, "actual_size": reference_state.size}
        )

    # 检查是否为归一化的量子态
    norm = np.linalg.norm(reference_state)
    if not np.isclose(norm, 1.0, atol=1e-8):
        raise ValidationError(
            format_error_message("Reference state not normalized", "validation", f"norm = {norm}"),
            details={"norm": norm, "tolerance": 1e-8}
        )


def validate_fidelity(fidelity: float) -> None:
    """验证保真度值

    Args:
        fidelity: 保真度值

    Raises:
        ValidationError: 如果保真度无效
    """
    if not isinstance(fidelity, (int, float)):
        raise ValidationError(
            format_error_message("Invalid type", "fidelity", f"expected float, got {type(fidelity).__name__}"),
            details={"fidelity": fidelity, "type": type(fidelity).__name__}
        )

    if not (FIDELITY_MIN <= fidelity <= FIDELITY_MAX):
        raise ValidationError(
            format_error_message("Invalid fidelity", "validation", f"must be {FIDELITY_MIN}-{FIDELITY_MAX}, got {fidelity}"),
            details={"fidelity": fidelity, "min": FIDELITY_MIN, "max": FIDELITY_MAX}
        )


def validate_platform(platform: str) -> None:
    """验证量子计算平台

    Args:
        platform: 平台名称

    Raises:
        ValidationError: 如果平台不支持
    """
    if not isinstance(platform, str):
        raise ValidationError(
            format_error_message("Invalid type", "platform", f"expected str, got {type(platform).__name__}"),
            details={"platform": platform, "type": type(platform).__name__}
        )

    if platform.lower() not in SUPPORTED_PLATFORMS:
        raise ValidationError(
            format_error_message("Unsupported platform", "validation", f"supported platforms: {list(SUPPORTED_PLATFORMS.keys())}"),
            details={"platform": platform, "supported_platforms": list(SUPPORTED_PLATFORMS.keys())}
        )


def validate_circuit_type(circuit_type: str) -> None:
    """验证电路类型

    Args:
        circuit_type: 电路类型

    Raises:
        ValidationError: 如果电路类型不支持
    """
    if not isinstance(circuit_type, str):
        raise ValidationError(
            format_error_message("Invalid type", "circuit_type", f"expected str, got {type(circuit_type).__name__}"),
            details={"circuit_type": circuit_type, "type": type(circuit_type).__name__}
        )

    if circuit_type.lower() not in SUPPORTED_CIRCUITS:
        raise ValidationError(
            format_error_message("Unsupported circuit type", "validation", f"supported circuits: {sorted(SUPPORTED_CIRCUITS)}"),
            details={"circuit_type": circuit_type, "supported_circuits": sorted(SUPPORTED_CIRCUITS)}
        )


def validate_backend(backend: str, platform: str) -> None:
    """验证后端

    Args:
        backend: 后端名称
        platform: 平台名称

    Raises:
        ValidationError: 如果后端无效
    """
    if not isinstance(backend, str):
        raise ValidationError(
            format_error_message("Invalid type", "backend", f"expected str, got {type(backend).__name__}"),
            details={"backend": backend, "type": type(backend).__name__}
        )

    # 这里可以添加平台特定的后端验证逻辑
    # 暂时只做基本验证
    if not backend.strip():
        raise ValidationError(
            format_error_message("Empty backend", "validation", "backend name cannot be empty"),
            details={"backend": backend}
        )


def validate_array(array: np.ndarray, name: str, min_dim: int = 1, max_dim: Optional[int] = None) -> None:
    """验证numpy数组

    Args:
        array: 要验证的数组
        name: 数组名称（用于错误消息）
        min_dim: 最小维度
        max_dim: 最大维度（可选）

    Raises:
        ValidationError: 如果数组无效
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(
            format_error_message("Invalid type", name, f"expected numpy.ndarray, got {type(array).__name__}"),
            details={name: array, "type": type(array).__name__}
        )

    if array.ndim < min_dim:
        raise ValidationError(
            format_error_message("Insufficient dimensions", name, f"expected at least {min_dim}D, got {array.ndim}D"),
            details={name: array, "min_dim": min_dim, "actual_dim": array.ndim}
        )

    if max_dim is not None and array.ndim > max_dim:
        raise ValidationError(
            format_error_message("Too many dimensions", name, f"expected at most {max_dim}D, got {array.ndim}D"),
            details={name: array, "max_dim": max_dim, "actual_dim": array.ndim}
        )


def validate_positive_number(value: Union[int, float], name: str) -> None:
    """验证正数

    Args:
        value: 要验证的值
        name: 参数名称

    Raises:
        ValidationError: 如果值不是正数
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            format_error_message("Invalid type", name, f"expected number, got {type(value).__name__}"),
            details={name: value, "type": type(value).__name__}
        )

    if value <= 0:
        raise ValidationError(
            format_error_message("Non-positive value", name, f"must be positive, got {value}"),
            details={name: value}
        )


def validate_non_negative_number(value: Union[int, float], name: str) -> None:
    """验证非负数

    Args:
        value: 要验证的值
        name: 参数名称

    Raises:
        ValidationError: 如果值是负数
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            format_error_message("Invalid type", name, f"expected number, got {type(value).__name__}"),
            details={name: value, "type": type(value).__name__}
        )

    if value < 0:
        raise ValidationError(
            format_error_message("Negative value", name, f"must be non-negative, got {value}"),
            details={name: value}
        )


def validate_range(value: Union[int, float], name: str, min_val: float, max_val: float) -> None:
    """验证数值范围

    Args:
        value: 要验证的值
        name: 参数名称
        min_val: 最小值
        max_val: 最大值

    Raises:
        ValidationError: 如果值超出范围
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            format_error_message("Invalid type", name, f"expected number, got {type(value).__name__}"),
            details={name: value, "type": type(value).__name__}
        )

    if not (min_val <= value <= max_val):
        raise ValidationError(
            format_error_message("Value out of range", name, f"must be {min_val}-{max_val}, got {value}"),
            details={name: value, "min": min_val, "max": max_val}
        )


def validate_list_items(items: List[Any], name: str, item_validator: Optional[callable] = None) -> None:
    """验证列表中的每个项目

    Args:
        items: 要验证的列表
        name: 列表名称
        item_validator: 项目验证函数（可选）

    Raises:
        ValidationError: 如果列表或项目无效
    """
    if not isinstance(items, list):
        raise ValidationError(
            format_error_message("Invalid type", name, f"expected list, got {type(items).__name__}"),
            details={name: items, "type": type(items).__name__}
        )

    if not items:
        raise ValidationError(
            format_error_message("Empty list", name, "list cannot be empty"),
            details={name: items}
        )

    if item_validator is not None:
        for i, item in enumerate(items):
            try:
                item_validator(item)
            except ValidationError as e:
                raise ValidationError(
                    format_error_message("Invalid list item", f"{name}[{i}]", str(e)),
                    details={f"{name}[{i}]": item, "original_error": e.message}
                )


def validate_file_path(path: str, name: str, must_exist: bool = False) -> None:
    """验证文件路径

    Args:
        path: 文件路径
        name: 参数名称
        must_exist: 是否要求文件必须存在

    Raises:
        ValidationError: 如果路径无效
    """
    if not isinstance(path, str):
        raise ValidationError(
            format_error_message("Invalid type", name, f"expected str, got {type(path).__name__}"),
            details={name: path, "type": type(path).__name__}
        )

    if not path.strip():
        raise ValidationError(
            format_error_message("Empty path", name, "path cannot be empty"),
            details={name: path}
        )

    if must_exist:
        from pathlib import Path
        if not Path(path).exists():
            raise ValidationError(
                format_error_message("File not found", name, f"path does not exist: {path}"),
                details={name: path}
            )


# 组合验证函数
def validate_execution_parameters(
    n_qubits: int,
    repeat: int = 1,
    warmup_runs: int = 0,
    reference_state: Optional[np.ndarray] = None
) -> None:
    """验证执行参数的完整集合

    Args:
        n_qubits: 量子比特数
        repeat: 重复运行次数
        warmup_runs: 预热运行次数
        reference_state: 参考态（可选）

    Raises:
        ValidationError: 如果任何参数无效
    """
    validate_n_qubits(n_qubits)
    validate_repeat_parameters(repeat, warmup_runs)
    validate_reference_state(reference_state, n_qubits)


def validate_circuit_parameters(platform: str, circuit_type: str, backend: str, n_qubits: int) -> None:
    """验证电路相关参数

    Args:
        platform: 量子计算平台
        circuit_type: 电路类型
        backend: 后端名称
        n_qubits: 量子比特数

    Raises:
        ValidationError: 如果任何参数无效
    """
    validate_platform(platform)
    validate_circuit_type(circuit_type)
    validate_backend(backend, platform)
    validate_n_qubits(n_qubits)