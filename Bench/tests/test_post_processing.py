"""
测试后处理模块
"""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest

from benchmark_harness.abstractions import BenchmarkResult
from benchmark_harness.post_processing import (analyze_results,
                                               generate_summary_report)


@pytest.fixture
def sample_results():
    """创建示例基准测试结果"""
    results = []

    # 添加Qibo结果
    for n_qubits in [2, 3, 4]:
        result = Mock(spec=BenchmarkResult)
        result.simulator = "qibo"
        result.backend = "numpy"
        result.circuit_name = "qft"
        result.n_qubits = n_qubits
        result.wall_time_sec = 0.1 * n_qubits
        result.cpu_time_sec = 0.08 * n_qubits
        result.peak_memory_mb = 10.0 * n_qubits
        result.cpu_utilization_percent = 80.0 + n_qubits
        result.state_fidelity = 0.999
        results.append(result)

    # 添加Qiskit结果
    for n_qubits in [2, 3, 4]:
        result = Mock(spec=BenchmarkResult)
        result.simulator = "qiskit"
        result.backend = "aer_simulator"
        result.circuit_name = "qft"
        result.n_qubits = n_qubits
        result.wall_time_sec = 0.15 * n_qubits
        result.cpu_time_sec = 0.12 * n_qubits
        result.peak_memory_mb = 15.0 * n_qubits
        result.cpu_utilization_percent = 75.0 + n_qubits
        result.state_fidelity = 0.998
        results.append(result)

    return results


def test_analyze_results_with_empty_list():
    """测试空结果列表"""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyze_results([], temp_dir)
        # 应该不会抛出异常，但也不会创建文件


def test_analyze_results_with_sample_data(sample_results):
    """测试使用示例数据进行分析"""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyze_results(sample_results, temp_dir)

        # 检查CSV文件是否创建
        csv_path = os.path.join(temp_dir, "raw_results.csv")
        assert os.path.exists(csv_path)

        # 检查图表文件是否创建
        plot_files = [
            "fidelity.png",
            "wall_time_scaling.png",
            "memory_scaling.png",
            "cpu_time_scaling.png",
            "cpu_utilization.png",
        ]

        for plot_file in plot_files:
            plot_path = os.path.join(temp_dir, plot_file)
            assert os.path.exists(plot_path)


def test_generate_summary_report(sample_results):
    """测试生成摘要报告"""
    import pandas as pd

    # 创建DataFrame
    data = []
    for result in sample_results:
        data.append(
            {
                "simulator": result.simulator,
                "backend": result.backend,
                "circuit_name": result.circuit_name,
                "n_qubits": result.n_qubits,
                "wall_time_sec": result.wall_time_sec,
                "cpu_time_sec": result.cpu_time_sec,
                "peak_memory_mb": result.peak_memory_mb,
                "cpu_utilization_percent": result.cpu_utilization_percent,
                "state_fidelity": result.state_fidelity,
            }
        )

    df = pd.DataFrame(data)
    df["runner_id"] = df["simulator"] + "-" + df["backend"]

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_summary_report(df, temp_dir)

        # 检查报告文件是否创建
        report_path = os.path.join(temp_dir, "summary_report.md")
        assert os.path.exists(report_path)

        # 检查报告内容
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "量子模拟器基准测试报告" in content
            assert "基本统计" in content
            assert "性能指标" in content
            assert "扩展性分析" in content
            assert "建议" in content
