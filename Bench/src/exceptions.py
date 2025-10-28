"""
自定义异常类模块

定义了基准测试框架中使用的特定异常类型。
"""

from typing import Optional, Any


class BenchmarkError(Exception):
    """基准测试基础异常类"""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class SimulatorError(BenchmarkError):
    """模拟器相关异常"""
    pass


class CircuitError(BenchmarkError):
    """电路相关异常"""
    pass


class ConfigurationError(BenchmarkError):
    """配置相关异常"""
    pass


class CacheError(BenchmarkError):
    """缓存相关异常"""
    pass


class ValidationError(BenchmarkError):
    """验证相关异常"""
    pass


class PerformanceError(BenchmarkError):
    """性能监控相关异常"""
    pass


class BackendError(SimulatorError):
    """后端设置异常"""
    pass


class ExecutionError(SimulatorError):
    """执行异常"""
    pass


class MetricsError(PerformanceError):
    """指标收集异常"""
    pass


class ImportWarning(UserWarning):
    """导入警告"""
    pass


def format_error_message(error_type: str, context: str, details: Optional[str] = None) -> str:
    """格式化错误消息

    Args:
        error_type: 错误类型
        context: 错误上下文
        details: 详细信息

    Returns:
        格式化的错误消息
    """
    message = f"{error_type} in {context}"
    if details:
        message += f": {details}"
    return message