# 重复运行功能测试设计

## 测试概述

本文档描述了为重复运行功能设计的测试用例，确保功能的正确性、可靠性和性能。

## 测试策略

### 1. 单元测试

#### 1.1 BenchmarkResult 扩展测试

**测试文件**: `tests/test_repeat_runs.py`

**测试用例**:

```python
def test_benchmark_result_with_repeat_data():
    """测试包含重复运行数据的BenchmarkResult"""
    # 创建包含统计信息的BenchmarkResult
    result = BenchmarkResult(
        simulator="qibo",
        backend="numpy",
        circuit_name="test_circuit",
        n_qubits=2,
        run_id=1,
        wall_time_sec=0.1,
        cpu_time_sec=0.05,
        peak_memory_mb=10.0,
        cpu_utilization_percent=50.0,
        state_fidelity=0.99,
        final_state=np.array([1, 0, 0, 0]),
        wall_time_mean=0.1,
        wall_time_std=0.01,
        wall_time_min=0.08,
        wall_time_max=0.12,
        confidence_interval=(0.09, 0.11)
    )
    
    assert result.wall_time_mean == 0.1
    assert result.wall_time_std == 0.01
    assert result.confidence_interval == (0.09, 0.11)

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
```

#### 1.2 模拟器包装器测试

```python
def test_qibo_wrapper_single_run():
    """测试QiboWrapper单次运行"""
    wrapper = QiboWrapper("numpy")
    circuit = create_test_circuit(2)
    
    results = wrapper.execute(circuit, 2, repeat=1)
    
    assert len(results) == 1
    assert results[0].run_id == 1
    assert results[0].wall_time_sec > 0

def test_qibo_wrapper_multiple_runs():
    """测试QiboWrapper多次运行"""
    wrapper = QiboWrapper("numpy")
    circuit = create_test_circuit(2)
    
    results = wrapper.execute(circuit, 2, repeat=5)
    
    assert len(results) == 5
    
    # 验证每个结果有不同的run_id
    run_ids = [r.run_id for r in results]
    assert sorted(run_ids) == [1, 2, 3, 4, 5]
    
    # 验证第一个结果包含统计信息
    assert results[0].wall_time_mean is not None
    assert results[0].wall_time_std is not None
    assert results[0].wall_time_min is not None
    assert results[0].wall_time_max is not None

def test_qibo_wrapper_with_warmup():
    """测试QiboWrapper预热运行"""
    wrapper = QiboWrapper("numpy")
    circuit = create_test_circuit(2)
    
    results = wrapper.execute(circuit, 2, repeat=3, warmup_runs=2)
    
    assert len(results) == 3
    # 预热运行不应包含在结果中
    
    # 验证统计计算
    assert results[0].wall_time_mean is not None
    assert results[0].wall_time_std >= 0

def test_statistical_calculations():
    """测试统计计算的准确性"""
    wrapper = QiboWrapper("numpy")
    circuit = create_test_circuit(2)
    
    results = wrapper.execute(circuit, 2, repeat=10)
    
    # 获取所有运行时间
    wall_times = [r.wall_time_sec for r in results]
    
    # 验证统计计算
    expected_mean = sum(wall_times) / len(wall_times)
    expected_std = np.std(wall_times, ddof=1)
    
    assert abs(results[0].wall_time_mean - expected_mean) < 1e-10
    assert abs(results[0].wall_time_std - expected_std) < 1e-10
    assert results[0].wall_time_min == min(wall_times)
    assert results[0].wall_time_max == max(wall_times)
```

#### 1.3 后处理模块测试

```python
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

def test_generate_summary_report_with_repeat_data():
    """测试包含重复数据的报告生成"""
    df = create_mock_repeat_dataframe()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_summary_report(df, temp_dir, repeat=5)
        
        report_path = os.path.join(temp_dir, "summary_report.md")
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            content = f.read()
            assert "重复运行次数: 5" in content
            assert "稳定性分析" in content
            assert "变异系数" in content
```

### 2. 集成测试

#### 2.1 端到端流程测试

```python
def test_end_to_end_single_run():
    """测试端到端单次运行流程"""
    # 使用临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python", "run_benchmarks.py",
            "--circuits", "qft",
            "--qubits", "2",
            "--simulators", "qibo-numpy",
            "--output-dir", temp_dir,
            "--repeat", "1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # 验证输出文件
        results_dir = os.path.join(temp_dir, "benchmark_")
        results_dirs = [d for d in os.listdir(temp_dir) if d.startswith("benchmark_")]
        assert len(results_dirs) == 1
        
        benchmark_dir = os.path.join(temp_dir, results_dirs[0])
        assert os.path.exists(os.path.join(benchmark_dir, "raw_results.csv"))
        assert not os.path.exists(os.path.join(benchmark_dir, "detailed_runs.csv"))

def test_end_to_end_multiple_runs():
    """测试端到端多次运行流程"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python", "run_benchmarks.py",
            "--circuits", "qft",
            "--qubits", "2",
            "--simulators", "qibo-numpy",
            "--output-dir", temp_dir,
            "--repeat", "5",
            "--warmup-runs", "2",
            "--verbose"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # 验证输出文件
        results_dirs = [d for d in os.listdir(temp_dir) if d.startswith("benchmark_")]
        benchmark_dir = os.path.join(temp_dir, results_dirs[0])
        
        assert os.path.exists(os.path.join(benchmark_dir, "raw_results.csv"))
        assert os.path.exists(os.path.join(benchmark_dir, "detailed_runs.csv"))
        
        # 验证CSV内容
        df = pd.read_csv(os.path.join(benchmark_dir, "raw_results.csv"))
        assert "wall_time_std" in df.columns
        assert df["repeat"].iloc[0] == 5
        
        detailed_df = pd.read_csv(os.path.join(benchmark_dir, "detailed_runs.csv"))
        assert len(detailed_df) == 5  # 5次运行
        assert "run_id" in detailed_df.columns

def test_backward_compatibility():
    """测试向后兼容性"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 不指定repeat参数，应该使用默认值1
        cmd = [
            "python", "run_benchmarks.py",
            "--circuits", "qft",
            "--qubits", "2",
            "--simulators", "qibo-numpy",
            "--output-dir", temp_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # 验证行为与之前一致
        results_dirs = [d for d in os.listdir(temp_dir) if d.startswith("benchmark_")]
        benchmark_dir = os.path.join(temp_dir, results_dirs[0])
        
        df = pd.read_csv(os.path.join(benchmark_dir, "raw_results.csv"))
        assert "wall_time_std" not in df.columns or df["wall_time_std"].iloc[0] == 0.0
```

### 3. 性能测试

#### 3.1 重复运行性能测试

```python
def test_repeat_performance_impact():
    """测试重复运行对性能的影响"""
    wrapper = QiboWrapper("numpy")
    circuit = create_test_circuit(4)  # 使用较大的电路
    
    # 测量单次运行时间
    start_time = time.time()
    single_results = wrapper.execute(circuit, 4, repeat=1)
    single_time = time.time() - start_time
    
    # 测量5次运行时间
    start_time = time.time()
    multi_results = wrapper.execute(circuit, 4, repeat=5)
    multi_time = time.time() - start_time
    
    # 验证时间比例合理（考虑统计计算开销）
    assert multi_time < single_time * 6  # 应该小于6倍单次时间
    
    # 验证结果质量
    assert len(multi_results) == 5
    assert multi_results[0].wall_time_std is not None

def test_cache_effectiveness_with_repeat():
    """测试缓存与重复运行的协同效果"""
    # 这个测试需要集成缓存系统
    # 验证重复运行不会影响缓存的有效性
    pass
```

### 4. 边界条件测试

#### 4.1 参数边界测试

```python
def test_edge_cases():
    """测试边界条件"""
    wrapper = QiboWrapper("numpy")
    circuit = create_test_circuit(2)
    
    # 测试repeat=0（应该抛出异常）
    with pytest.raises(ValueError):
        wrapper.execute(circuit, 2, repeat=0)
    
    # 测试负数warmup_runs（应该抛出异常）
    with pytest.raises(ValueError):
        wrapper.execute(circuit, 2, repeat=1, warmup_runs=-1)
    
    # 测试大量重复运行
    results = wrapper.execute(circuit, 2, repeat=100)
    assert len(results) == 100
    
    # 测试repeat=1的行为与原始行为一致
    repeat_results = wrapper.execute(circuit, 2, repeat=1)
    single_result = wrapper.execute(circuit, 2, repeat=1, warmup_runs=0)
    
    assert abs(repeat_results[0].wall_time_sec - single_result[0].wall_time_sec) < 1e-10

def test_error_handling():
    """测试错误处理"""
    wrapper = QiboWrapper("numpy")
    
    # 创建一个会失败的电路
    circuit = create_failing_circuit()
    
    # 验证错误处理
    with pytest.raises(RuntimeError):
        wrapper.execute(circuit, 2, repeat=5)
    
    # 验证部分失败的处理
    circuit = create_flaky_circuit()  # 有时会失败的电路
    
    # 这里需要根据实际实现调整错误处理策略
    pass
```

## 测试数据准备

### 辅助函数

```python
def create_test_circuit(n_qubits):
    """创建测试用电路"""
    import qibo
    circuit = qibo.models.Circuit(n_qubits)
    for i in range(n_qubits):
        circuit.add(qibo.gates.H(i))
    for i in range(n_qubits - 1):
        circuit.add(qibo.gates.CNOT(i, i + 1))
    circuit.name = f"test_circuit_{n_qubits}q"
    return circuit

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

def create_mock_repeat_dataframe():
    """创建模拟的重复运行DataFrame"""
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
        "fidelity_std": [0.001],
        "repeat": [5]
    }
    return pd.DataFrame(data)
```

## 测试执行计划

### 阶段1：单元测试开发
- 实现BenchmarkResult扩展测试
- 实现模拟器包装器测试
- 实现后处理模块测试

### 阶段2：集成测试开发
- 实现端到端流程测试
- 实现向后兼容性测试

### 阶段3：性能和边界测试
- 实现性能影响测试
- 实现边界条件测试
- 实现错误处理测试

### 阶段4：测试执行和验证
- 执行所有测试用例
- 验证测试覆盖率
- 修复发现的问题

## 预期测试覆盖率

- **单元测试覆盖率**: ≥ 90%
- **集成测试覆盖率**: ≥ 80%
- **整体测试覆盖率**: ≥ 85%

## 测试环境要求

- Python 3.12+
- 所有依赖的量子计算框架
- 测试数据存储空间
- 足够的计算资源进行性能测试

## 持续集成

测试用例应该集成到CI/CD流水线中，确保每次代码变更都能通过测试验证。

## 测试报告

测试执行后应该生成详细的测试报告，包括：
- 测试用例执行结果
- 代码覆盖率统计
- 性能基准测试结果
- 发现的问题和修复建议