"""
重复运行功能的基本测试

这个模块包含了对重复运行功能的基本测试用例，
验证新功能的正确性和向后兼容性。
"""

import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_harness.abstractions import BenchmarkResult
from benchmark_harness.post_processing import analyze_results, generate_summary_report


def test_benchmark_result_with_repeat_data():
    """测试包含重复运行数据的BenchmarkResult"""
    # 创建包含统计信息的BenchmarkResult
    result = BenchmarkResult(
        simulator="qibo",
        backend="numpy",
        circuit_name="test_circuit",
        n_qubits=2,
        wall_time_sec=0.1,
        cpu_time_sec=0.05,
        peak_memory_mb=10.0,
        cpu_utilization_percent=50.0,
        state_fidelity=0.99,
        final_state=np.array([1, 0, 0, 0]),
        run_id=1,
        wall_time_mean=0.1,
        wall_time_std=0.01,
        wall_time_min=0.08,
        wall_time_max=0.12,
        confidence_interval=(0.09, 0.11)
    )
    
    assert result.run_id == 1
    assert result.wall_time_mean == 0.1
    assert result.wall_time_std == 0.01
    assert result.confidence_interval == (0.09, 0.11)
    print("✓ BenchmarkResult with repeat data test passed")


def test_benchmark_result_backward_compatibility():
    """测试BenchmarkResult的向后兼容性"""
    # 创建不包含新字段的BenchmarkResult
    result = BenchmarkResult(
        simulator="qibo",
        backend="numpy",
        circuit_name="test_circuit",
        n_qubits=2,
        wall_time_sec=0.1,
        cpu_time_sec=0.05,
        peak_memory_mb=10.0,
        cpu_utilization_percent=50.0,
        state_fidelity=0.99,
        final_state=np.array([1, 0, 0, 0])
    )
    
    # 验证新字段有默认值
    assert result.run_id == 1
    assert result.wall_time_mean is None
    assert result.wall_time_std is None
    print("✓ BenchmarkResult backward compatibility test passed")


def create_mock_repeat_results():
    """创建模拟的重复运行结果"""
    results = []
    for i in range(5):
        result = BenchmarkResult(
            simulator="qibo",
            backend="numpy",
            circuit_name="test_circuit",
            n_qubits=2,
            run_id=i + 1,
            wall_time_sec=0.1 + i * 0.01,
            cpu_time_sec=0.05 + i * 0.005,
            peak_memory_mb=10.0 + i,
            cpu_utilization_percent=50.0 + i * 2,
            state_fidelity=0.99 - i * 0.001,
            final_state=np.array([1, 0, 0, 0])
        )
        results.append(result)
    
    # 为第一个结果添加统计信息
    wall_times = [r.wall_time_sec for r in results]
    results[0].wall_time_mean = sum(wall_times) / len(wall_times)
    results[0].wall_time_std = np.std(wall_times, ddof=1)
    results[0].wall_time_min = min(wall_times)
    results[0].wall_time_max = max(wall_times)
    
    return results


def test_analyze_results_with_repeat_data():
    """测试包含重复数据的analyze_results函数"""
    # 创建模拟的重复运行结果
    results = create_mock_repeat_results()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyze_results(results, temp_dir, repeat=5)
        
        # 验证文件生成
        assert os.path.exists(os.path.join(temp_dir, "raw_results.csv"))
        assert os.path.exists(os.path.join(temp_dir, "detailed_runs.csv"))
        
        # 验证CSV内容
        df = pd.read_csv(os.path.join(temp_dir, "raw_results.csv"))
        assert "wall_time_std" in df.columns
        assert "repeat" in df.columns
        assert len(df) == 1  # 只包含汇总结果
        
        detailed_df = pd.read_csv(os.path.join(temp_dir, "detailed_runs.csv"))
        assert len(detailed_df) == 5  # 5次运行
        assert "run_id" in detailed_df.columns
        
    print("✓ analyze_results with repeat data test passed")


def test_analyze_results_single_run():
    """测试单次运行的analyze_results函数"""
    # 创建单次运行结果
    result = BenchmarkResult(
        simulator="qibo",
        backend="numpy",
        circuit_name="test_circuit",
        n_qubits=2,
        wall_time_sec=0.1,
        cpu_time_sec=0.05,
        peak_memory_mb=10.0,
        cpu_utilization_percent=50.0,
        state_fidelity=0.99,
        final_state=np.array([1, 0, 0, 0])
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyze_results([result], temp_dir, repeat=1)
        
        # 验证文件生成
        assert os.path.exists(os.path.join(temp_dir, "raw_results.csv"))
        # 单次运行不应该生成详细运行文件
        assert not os.path.exists(os.path.join(temp_dir, "detailed_runs.csv"))
        
        # 验证CSV内容
        df = pd.read_csv(os.path.join(temp_dir, "raw_results.csv"))
        assert "wall_time_std" not in df.columns or df["wall_time_std"].iloc[0] == 0.0
        
    print("✓ analyze_results single run test passed")


def test_generate_summary_report_with_repeat_data():
    """测试包含重复数据的报告生成"""
    # 创建模拟的重复运行DataFrame
    data = {
        "simulator": ["qibo"],
        "backend": ["numpy"],
        "circuit_name": ["test_circuit"],
        "n_qubits": [2],
        "wall_time_sec": [0.1],
        "wall_time_std": [0.01],
        "cpu_time_sec": [0.05],
        "cpu_time_std": [0.005],
        "peak_memory_mb": [10.0],
        "memory_std": [1.0],
        "cpu_utilization_percent": [50.0],
        "state_fidelity": [0.99],
        "fidelity_std": [0.001]
    }
    df = pd.DataFrame(data)
    df["runner_id"] = df["simulator"] + "-" + df["backend"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_summary_report(df, temp_dir, repeat=5)
        
        report_path = os.path.join(temp_dir, "summary_report.md")
        assert os.path.exists(report_path)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "重复运行次数: 5" in content
            assert "稳定性分析" in content
            assert "变异系数" in content
            
    print("✓ generate_summary_report with repeat data test passed")


def test_generate_summary_report_single_run():
    """测试单次运行的报告生成"""
    # 创建单次运行DataFrame
    data = {
        "simulator": ["qibo"],
        "backend": ["numpy"],
        "circuit_name": ["test_circuit"],
        "n_qubits": [2],
        "wall_time_sec": [0.1],
        "cpu_time_sec": [0.05],
        "peak_memory_mb": [10.0],
        "cpu_utilization_percent": [50.0],
        "state_fidelity": [0.99]
    }
    df = pd.DataFrame(data)
    df["runner_id"] = df["simulator"] + "-" + df["backend"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_summary_report(df, temp_dir, repeat=1)
        
        report_path = os.path.join(temp_dir, "summary_report.md")
        assert os.path.exists(report_path)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "重复运行次数: 1" in content
            # 单次运行不应该包含稳定性分析
            assert "稳定性分析" not in content
            
    print("✓ generate_summary_report single run test passed")


def run_basic_tests():
    """运行所有基本测试"""
    print("开始运行重复运行功能的基本测试...")
    print("=" * 50)
    
    try:
        test_benchmark_result_with_repeat_data()
        test_benchmark_result_backward_compatibility()
        test_analyze_results_with_repeat_data()
        test_analyze_results_single_run()
        test_generate_summary_report_with_repeat_data()
        test_generate_summary_report_single_run()
        
        print("=" * 50)
        print("✅ 所有基本测试通过！")
        print("重复运行功能的基本实现正确。")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_basic_tests()