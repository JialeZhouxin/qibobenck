# test_profiler.py

# 导入pytest和numpy，这是我们测试的基石
import pytest
import numpy as np

# 导入我们需要测试的脚本中的所有类和函数
from qibo_profiler import (
    MetadataCollector,
    InputAnalyzer,
    ResultProcessor,
    BenchmarkManager,
    convert_to_numpy,
    _flatten_dict
)

# 导入qibo，用于创建测试用的电路
from qibo.models import Circuit
from qibo import gates
import qibo

# 导入其他框架的库，用于测试convert_to_numpy函数
import torch
import jax.numpy as jnp
import tensorflow as tf

# ==============================================================================
# 1. Fixture: 我们的“标准实验样品”
# 这是一个完美的Fixtures使用场景。我们不希望在每个测试中都重复创建电路。
# 这个fixture会为需要它的测试提供一个标准的、内容已知的电路对象。
# ==============================================================================

@pytest.fixture
def sample_circuit():
    """
    创建一个简单、固定内容的量子电路，作为所有测试的“标准输入”。
    这样可以确保我们的测试是在一个已知的、不变的基础上进行的。
    """
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.M(0, 1))
    return c

# ==============================================================================
# 2. 针对辅助函数的单元测试
# 我们从最简单、最独立的函数开始测试，确保这些基础工具是可靠的。
# ==============================================================================

def test_flatten_dict_simple():
    """
    测试 _flatten_dict 函数是否能正确地将一个简单的嵌套字典压平。
    这是最基础的功能验证。
    """
    nested_dict = {'a': 1, 'b': {'c': 2}}
    expected_flat_dict = {'a': 1, 'b_c': 2}
    assert _flatten_dict(nested_dict) == expected_flat_dict

def test_flatten_dict_deeply_nested():
    """
    测试 _flatten_dict 函数处理更深层嵌套的能力。
    """
    nested_dict = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    expected_flat_dict = {'a': 1, 'b_c': 2, 'b_d_e': 3}
    assert _flatten_dict(nested_dict) == expected_flat_dict

# ==============================================================================
# 3. 针对 InputAnalyzer 类的测试
# 这是我们的核心逻辑之一，负责解析输入，我们要确保它能准确提取信息。
# ==============================================================================

class TestInputAnalyzer:

    def test_analyze_profiler_settings(self):
        """
        测试 _analyze_profiler_settings 方法能否正确解析配置字典，
        并能在缺少某些键时提供正确的默认值。
        """
        analyzer = InputAnalyzer()
        
        # 情况1: 提供了所有配置
        config1 = {"n_runs": 5, "mode": "full", "fidelity_calculated": False}
        settings1 = analyzer._analyze_profiler_settings(config1)
        assert settings1["n_runs"] == 5
        assert settings1["mode"] == "full"
        assert settings1["fidelity_calculated"] is False

        # 情况2: 配置不完整，测试默认值
        config2 = {"n_runs": 10}
        settings2 = analyzer._analyze_profiler_settings(config2)
        assert settings2["n_runs"] == 10
        assert settings2["mode"] == "basic"       # 验证默认值
        assert settings2["fidelity_calculated"] is True # 验证默认值

    def test_analyze_circuit(self, sample_circuit):
        """
        测试 _analyze_circuit 方法能否从我们固定的 sample_circuit 中
        提取出正确的属性。
        注意：这里我们直接使用了上面定义的 fixture `sample_circuit`！
        """
        analyzer = InputAnalyzer()
        properties = analyzer._analyze_circuit(sample_circuit)

        # 验证基本属性
        assert properties["n_qubits"] == 2
        assert properties["depth"] == 3
        assert properties["total_gates"] == 3
        
        # 验证门统计是否正确 (注意qibo会自动将门名转为小写)
        expected_gate_counts = {'h': 1, 'cx': 1, 'measure': 1}
        assert properties["gate_counts"] == expected_gate_counts

        # 验证哈希值。我们不验证哈希值的具体内容（因为它会随qibo版本变化），
        # 但我们验证它是一个64位的十六进制字符串。
        assert isinstance(properties["qasm_hash_sha256"], str)
        assert len(properties["qasm_hash_sha256"]) == 64

# ==============================================================================
# 4. 针对 ResultProcessor 类的测试
# 这个类负责将原始性能数据转换为最终报告。我们要确保它的计算是准确的。
# ==============================================================================

def test_result_processor_calculations():
    """
    测试 ResultProcessor.process 方法的数值计算是否正确。
    我们提供一份伪造的`raw_data`，然后手动计算预期结果，
    并使用 `pytest.approx` 来比较浮点数。
    """
    processor = ResultProcessor()
    
    # 准备一份“伪造”的原始性能数据
    raw_data = {
        "wall_runtimes": [1.0, 2.0, 3.0],
        "cpu_time_total": 4.5,
        "cpu_utils": [80.0, 90.0, 100.0],
        "memory_usages": [100.0, 110.0, 120.0],
        "peak_memory_usage": 150.0,
        "final_state_vector": np.array([1, 0, 0, 0]) # 伪造一个态矢量
    }

    # 准备一个伪造的基准态矢量
    benchmark_state = np.array([1, 0, 0, 0]) * np.exp(1j * 0.001) # 加一点相位差

    # 调用 process 方法进行处理
    results = processor.process(raw_data, benchmark_state)

    # --- 开始验证 ---
    summary = results["summary"]
    
    # 验证运行时间统计
    assert summary["runtime_avg"]["value"] == pytest.approx(2.0)
    assert summary["runtime_std_dev"]["value"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    
    # 验证CPU利用率统计 (假设逻辑核心数为4)
    # psutil.cpu_count = lambda logical: 4 # 这是更高级的“模拟”技巧，这里我们手动计算
    assert summary["cpu_utilization_psutil_avg"]["value"] == pytest.approx(90.0)
    
    # 验证内存统计
    assert summary["memory_usage_avg"]["value"] == pytest.approx(110.0)
    assert summary["memory_usage_peak"]["value"] == 150.0

    # 验证保真度计算
    # fidelity = |<psi_bench|psi_final>|^2
    expected_fidelity = np.abs(np.vdot(raw_data["final_state_vector"], benchmark_state)) ** 2
    assert summary["fidelity"]["value"] == pytest.approx(expected_fidelity)

# ==============================================================================
# 5. 针对多框架数组转换函数的测试
# 这是一个展示参数化 (@pytest.mark.parametrize) 的绝佳机会！
# 我们可以用同一个测试函数，测试来自不同框架的输入。
# ==============================================================================

# 准备用于参数化测试的各种类型的数组
numpy_arr = np.array([0.5, 0.5j], dtype=np.complex64)
torch_tensor_cpu = torch.tensor([0.5, 0.5j], dtype=torch.complex64)
jax_array = jnp.array([0.5, 0.5j], dtype=jnp.complex64)
tf_tensor = tf.constant([0.5 + 0.0j, 0.0 + 0.5j], dtype=tf.complex64)
python_list = [0.5 + 0.0j, 0.0 + 0.5j]

@pytest.mark.parametrize("input_array", [
    numpy_arr,
    torch_tensor_cpu,
    jax_array,
    tf_tensor,
    python_list
])
def test_convert_to_numpy(input_array):
    """
    测试 convert_to_numpy 函数能否将不同框架的数组统一转换为NumPy数组。
    Pytest 会自动用 @parametrize 列表中的每一个元素作为 input_array，
    来独立地运行这个测试。
    """
    # 1. 调用转换函数
    numpy_result = convert_to_numpy(input_array)
    
    # 2. 验证输出类型
    assert isinstance(numpy_result, np.ndarray)

    # 3. 验证数值是否一致 (使用我们信赖的黄金标准)
    np.testing.assert_allclose(numpy_result, numpy_arr)
    