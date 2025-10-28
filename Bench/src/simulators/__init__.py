"""
模拟器封装器模块

这个模块包含了不同量子计算框架的封装器实现。
"""

from .qibo_wrapper import QiboWrapper

# 尝试导入其他封装器，如果失败则跳过
try:
    from .qiskit_wrapper import QiskitWrapper

    _QISKIT_AVAILABLE = True
except ImportError:
    QiskitWrapper = None
    _QISKIT_AVAILABLE = False

try:
    from .pennylane_wrapper import PennyLaneWrapper

    _PENNYLANE_AVAILABLE = True
except ImportError:
    PennyLaneWrapper = None
    _PENNYLANE_AVAILABLE = False

__all__ = ["QiboWrapper", "QiskitWrapper", "PennyLaneWrapper"]
