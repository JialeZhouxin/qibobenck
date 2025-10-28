"""
Benchmark Harness - 模块化量子模拟器基准测试平台

这个包提供了一个用于测试和比较不同量子计算模拟器性能的框架。
"""

from .abstractions import BenchmarkCircuit, BenchmarkResult, SimulatorInterface

__all__ = ["BenchmarkResult", "SimulatorInterface", "BenchmarkCircuit"]
