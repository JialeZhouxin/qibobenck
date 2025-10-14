"""
性能指标收集器模块

这个模块提供了用于测量代码块性能和资源消耗的工具。
"""

import time
import tracemalloc
from typing import Any, Dict

import psutil


class MetricsCollector:
    """用于分析代码块性能和资源的上下文管理器"""

    def __init__(self):
        """初始化指标收集器"""
        self.process = psutil.Process()
        self.results: Dict[str, Any] = {}

    def __enter__(self):
        """启动所有监控器"""
        # 初始化CPU监控（非阻塞调用）
        self.process.cpu_percent(interval=None)
        # 启动内存跟踪
        tracemalloc.start()
        # 记录开始时间
        self.cpu_time_start = self.process.cpu_times()
        self.wall_time_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止所有监控器并计算结果"""
        # 记录结束时间
        self.wall_time_end = time.perf_counter()
        self.cpu_time_end = self.process.cpu_times()
        # 获取内存使用情况
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 计算指标
        self.results["wall_time_sec"] = self.wall_time_end - self.wall_time_start
        self.results["cpu_time_sec"] = (
            self.cpu_time_end.user - self.cpu_time_start.user
        ) + (self.cpu_time_end.system - self.cpu_time_start.system)
        self.results["peak_memory_mb"] = peak / (1024 * 1024)
        self.results["cpu_utilization_percent"] = self.process.cpu_percent(
            interval=None
        )

    def get_results(self) -> Dict[str, Any]:
        """返回收集的指标"""
        return self.results
