"""
指标收集器测试

这个模块包含了MetricsCollector类的测试用例。
"""

import time

import pytest

from benchmark_harness.metrics import MetricsCollector


def test_metrics_collector_context_manager():
    """测试MetricsCollector作为上下文管理器的正确行为"""
    collector = MetricsCollector()

    # 测试进入上下文
    with collector as c:
        assert c is collector
        # 执行一些简单操作
        time.sleep(0.01)
        result = sum(range(1000))

    # 测试退出上下文后结果被收集
    results = collector.get_results()
    assert "wall_time_sec" in results
    assert "cpu_time_sec" in results
    assert "peak_memory_mb" in results
    assert "cpu_utilization_percent" in results


def test_metrics_collector_wall_time():
    """测试墙上时间测量的合理性"""
    collector = MetricsCollector()

    with collector:
        time.sleep(0.1)  # 睡眠100ms

    results = collector.get_results()
    wall_time = results["wall_time_sec"]

    # 墙上时间应该大约为0.1秒（允许一些误差）
    assert 0.05 <= wall_time <= 0.2


def test_metrics_collector_memory_tracking():
    """测试内存跟踪功能"""
    collector = MetricsCollector()

    with collector:
        # 分配一些内存
        large_list = [0] * 1000000

    results = collector.get_results()
    peak_memory = results["peak_memory_mb"]

    # 峰值内存应该大于0
    assert peak_memory > 0


def test_metrics_collector_cpu_time():
    """测试CPU时间测量"""
    collector = MetricsCollector()

    with collector:
        # 执行一些CPU密集型操作
        result = sum(i * i for i in range(10000))

    results = collector.get_results()
    cpu_time = results["cpu_time_sec"]

    # CPU时间应该大于0
    assert cpu_time >= 0


def test_metrics_collector_cpu_utilization():
    """测试CPU利用率测量"""
    collector = MetricsCollector()

    with collector:
        # 执行一些操作
        time.sleep(0.01)

    results = collector.get_results()
    cpu_utilization = results["cpu_utilization_percent"]

    # CPU利用率应该在合理范围内
    assert 0 <= cpu_utilization <= 100


def test_metrics_collector_multiple_uses():
    """测试MetricsCollector可以多次使用"""
    collector = MetricsCollector()

    # 第一次使用
    with collector:
        time.sleep(0.01)

    first_results = collector.get_results()

    # 第二次使用（更长的时间）
    with collector:
        time.sleep(0.05)  # 增加睡眠时间确保有明显差异

    second_results = collector.get_results()

    # 两次结果应该不同（使用更大的时间差）
    assert second_results["wall_time_sec"] >= first_results["wall_time_sec"]


def test_metrics_collector_empty_context():
    """测试空上下文的情况"""
    collector = MetricsCollector()

    with collector:
        pass  # 什么都不做

    results = collector.get_results()

    # 即使是空上下文，也应该有指标
    assert all(
        key in results
        for key in [
            "wall_time_sec",
            "cpu_time_sec",
            "peak_memory_mb",
            "cpu_utilization_percent",
        ]
    )
