# 导入必要的库
# 导入time模块，用于处理时间相关的操作
import time
# 导入psutil模块，用于获取系统性能信息
import psutil
# 导入platform模块，用于获取系统平台信息
import platform
# 导入cProfile模块，用于性能分析
import cProfile
# 导入io模块，用于处理I/O操作
import io
# 导入pstats模块，用于处理性能分析结果
import pstats
# 导入datetime模块，用于处理日期和时间
import datetime
# 导入hashlib模块，用于哈希计算
import hashlib
# 导入numpy模块，用于科学计算
import numpy as np
# 导入csv模块，用于处理CSV文件
import csv
# 导入json模块，用于处理JSON数据
import json
# 导入os模块，用于操作系统相关操作
import os
# 从qibo.models导入Circuit类，用于量子电路的构建
from qibo.models import Circuit
# 导入qibo模块，用于量子计算
import qibo
import torch  # 导入PyTorch库，用于深度学习计算
import jax  # 导入JAX库，用于高性能数值计算
import tensorflow as tf  # 导入TensorFlow库，用于深度学习计算
import cpuinfo  # 导入 py-cpuinfo 库

"""
# --- Target Data Structure Blueprint ---
{
    "metadata": {
        "profiler_version": "1.0",
        "timestamp_utc": "2025-10-10T09:54:11.247172Z"
    },
    "inputs": {
        "profiler_settings": {
            "n_runs": 3,
            "mode": "basic",
            "fidelity_calculated": true
        },
        "circuit_properties": {
            "n_qubits": 18,
            "depth": 138,
            "total_gates": 820,
            "gate_counts": {
                "rz": 495,
                "sx": 18,
                "cx": 306,
                "measure": 1
            },
            "qasm_hash_sha256": "0717ab030b23ba740b6163b072283083de0dbd64dadfd65f48972295fcbab918"
        },
        "environment": {
            "qibo_backend": "qiboml (pytorch)",
            "qibo_version": "0.2.21",
            "python_version": "3.12.0",
            "cpu_model": "Intel64 Family 6 Model 158 Stepping 9, GenuineIntel",
            "cpu_model_friendly": "Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz",
            "cpu_cores_physical": 4,
            "total_memory": {
                "value": 15.91,
                "unit": "GiB"
            }
        }
    },
    "results": {
        "summary": {
            "runtime_avg": {
                "value": 8.466669666673019,
                "unit": "seconds"
            },
            "runtime_std_dev": {
                "value": 4.993247684234495,
                "unit": "seconds"
            },
            "cpu_utilization_avg": {
                "value": 57.70175120799762,
                "unit": "percent"
            },
            "cpu_utilization_psutil_avg": {
                "value": 90.43333333333334,
                "unit": "percent"
            },
            "cpu_utilization_psutil_std_dev": {
                "value": 60.00096295523561,
                "unit": "percent"
            },
            "cpu_cores_logical": 4,
            "cpu_utilization_normalized": {
                "value": 22.608333333333334,
                "unit": "percent"
            },
            "memory_usage_avg": {
                "value": 1266.8515625,
                "unit": "MiB"
            },
            "memory_usage_peak": {
                "value": 4450.80078125,
                "unit": "MiB"
            },
            "fidelity": {
                "value": 1.0000000000004174,
                "unit": null
            }
        },
        "raw_metrics": {
            "runtime_per_run": {
                "values": [
                    15.111781800005701,
                    7.213086100004148,
                    3.075141100009205
                ],
                "unit": "seconds"
            }
        }
    },
    "error": null
}
"""
# ... existing code ...
backend_configs = {
    "numpy": {"backend_name": "numpy", "platform_name": None},
    "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
    "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
    "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
    "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
    "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"},
    "qulacs": {"backend_name": "qulacs", "platform_name": None}
    }
# 模块 1: MetadataCollector 类
# ... existing code ...

# 模块 1: MetadataCollector 类
class MetadataCollector:
    @staticmethod
    def collect():
        """
        收集元数据信息，包括分析器版本和当前时间戳（UTC 时间）。

        返回:
            dict: 包含以下键值对的字典：
                - profiler_version (str): 分析器的版本号，固定为 "1.0"。
                - timestamp_utc (str): 当前时间的ISO 8601格式字符串，使用协调世界时（UTC），并以 'Z' 结尾。
        """
        # 获取当前UTC时间，使用时区感知的对象
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)
        # 格式化时间戳，确保以 'Z' 结尾
        timestamp_utc = current_time_utc.isoformat().replace('+00:00', 'Z')
        return {
            "profiler_version": "1.0",
            "timestamp_utc": timestamp_utc
        }

# ... rest of code ...

# 模块 2: InputAnalyzer 类
class InputAnalyzer:
    def analyze(self, circuit: Circuit, config: dict):
        """
        分析输入的电路和配置文件，返回分析结果。

        参数:
            circuit (Circuit): 需要分析的量子电路。
            config (dict): 包含分析配置的字典。

        返回:
            dict: 包含分析结果的字典，包括 profiler 设置、电路属性和环境信息。
        """
        # 分析配置文件中的 profiler 设置
        profiler_settings = self._analyze_profiler_settings(config)
        # 分析电路的属性
        circuit_properties = self._analyze_circuit(circuit)
        # 扫描当前环境信息
        environment = self._scan_environment()
        # 返回分析结果
        return {
            "profiler_settings": profiler_settings,
            "circuit_properties": circuit_properties,
            "environment": environment
        }

    def _analyze_profiler_settings(self, config):
        """
        从配置文件中提取 profiler 设置。

        参数:
            config (dict): 包含分析配置的字典。

        返回:
            dict: 包含 profiler 设置的字典，包括运行次数、模式和是否计算保真度。
        """
        # 从配置文件中提取 profiler 设置
        return {
            "n_runs": config.get("n_runs", 1),  # 默认运行次数为1
            "mode": config.get("mode", "basic"),  # 默认模式为 "basic"
            "fidelity_calculated": config.get("fidelity_calculated", True)  # 默认计算保真度
        }

    def _analyze_circuit(self, circuit: Circuit):
        """
        分析电路的属性，包括门的数量、QASM 格式和哈希值等。

        参数:
            circuit (Circuit): 需要分析的量子电路。

        返回:
            dict: 包含电路属性的字典，包括量子比特数量、电路深度、总门数量、每种门的计数和 QASM 格式的哈希值。
        """
        # 统计电路中每个门的出现次数
        gate_counts = {}
        for gate in circuit.queue:
            gate_name = gate.name.lower()  # 将门的名称转换为小写以保持一致性
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        # 将电路转换为 QASM 格式并计算其 SHA256 哈希值
        qasm = circuit.to_qasm()
        qasm_hash_sha256 = hashlib.sha256(qasm.encode()).hexdigest()
        # 返回电路的属性
        return {
            "n_qubits": circuit.nqubits,  # 量子比特数量
            "depth": circuit.depth,  # 电路深度
            "total_gates": len(circuit.queue),  # 总门数量
            "gate_counts": gate_counts,  # 每种门的计数
            "qasm_hash_sha256": qasm_hash_sha256  # QASM 格式的哈希值
        }



    def _scan_environment(self):
        """
        扫描当前环境信息，包括 Qibo 后端类型、版本号、Python 版本、CPU 信息和内存信息。

        返回:
            dict: 包含当前环境信息的字典。
        """
        # 获取当前环境信息
        cpu_info = cpuinfo.get_cpu_info()  # 使用 py-cpuinfo 获取 CPU 信息
        return {
            "qibo_backend": str(qibo.get_backend()),    # 获取 Qibo 的后端类型，默认为 "default"
            "qibo_version": qibo.__version__,  # 获取 Qibo 的版本号，默认为 "unknown"
            "python_version": platform.python_version(),  # Python 版本
            "cpu_model": platform.processor(),  # 精确的 CPU 型号
            "cpu_model_friendly": cpu_info.get("brand_raw", "unknown"),  # 友好的 CPU 商品名称
            "cpu_cores_physical": psutil.cpu_count(logical=False),  # 物理 CPU 核心数
            "total_memory": {
                "value": round(psutil.virtual_memory().total / (1024 ** 3), 2),  # 总内存大小（GiB）
                "unit": "GiB"
            }
        }

# ... rest of code ...
# ... existing code ...

# 模块 3: BenchmarkManager 类
class BenchmarkManager:
    _GLOBAL_CACHE = {}

    def get_benchmark_state(self, circuit: Circuit, circuit_hash: str):
        """
        获取电路的基准状态，如果缓存中已有该电路的基准状态，则直接返回缓存结果。
        否则，根据当前后端配置计算基准状态，并将其缓存。

        参数:
            circuit (Circuit): 要获取基准状态的电路对象。
            circuit_hash (str): 电路的哈希值，用于缓存和查找。

        返回:
            state: 电路的基准状态。

        异常:
            ValueError: 如果无法找到当前后端的配置，则抛出此异常。
        """
        if circuit_hash in self._GLOBAL_CACHE:
            return self._GLOBAL_CACHE[circuit_hash]

        # 获取当前后端配置
        current_backend = qibo.get_backend()
        current_backend_name = current_backend.name
        current_platform = getattr(current_backend, 'platform', None)
        
        # 查找当前后端的配置
        original_backend_config = None
        for key, config in backend_configs.items():
            if config['backend_name'] == current_backend_name and config.get('platform_name') == current_platform:
                original_backend_config = config
                break
        
        if not original_backend_config:
            raise ValueError(f"未知的后端配置: {current_backend_name} with platform {current_platform}")
        
        if current_backend_name == "qibojit":
            # 如果当前就是基准后端，直接计算即可
            state = circuit(nshots=1).state()
            self._GLOBAL_CACHE[circuit_hash] = state
            return state
            
        try:
            # 安全地切换到基准后端
            qibo.set_backend("qibojit")
            state = circuit(nshots=1).state()
            self._GLOBAL_CACHE[circuit_hash] = state
            return state
        finally:
            # 无论成功或失败，都确保切换回原始后端
            qibo.set_backend(original_backend_config['backend_name'], platform=original_backend_config.get('platform_name'))



# ... rest of code ...
# ... existing code ...

# 模块 4: ExecutionEngine 类
class ExecutionEngine:
    def run_and_measure(self, circuit: Circuit, config: dict):
        """运行量子电路并测量其性能指标，包括内存使用情况。

        该方法执行给定的量子电路多次，并收集每次运行的挂钟时间（wall time），
        同时计算总的CPU时间（包括用户态和内核态时间）以及内存使用情况。最后返回包含性能指标
        和最终状态向量的字典。

        Args:
            circuit (Circuit): 要执行的量子电路对象
            config (dict): 配置字典，包含以下可选键：
                - n_runs (int, optional): 运行电路的次数，默认为1

        Returns:
            dict: 包含以下键的字典：
                - wall_runtimes (list): 每次运行的挂钟时间列表（秒）
                - cpu_time_total (float): 总CPU时间（秒）
                - cpu_utils (list): 每次运行的CPU利用率
                - memory_usages (list): 每次运行的内存使用增量（MiB）
                - peak_memory_usage (float): 最大内存使用量（MiB）
                - final_state_vector: 最后一次运行后的量子态向量
        """
        # 从配置字典中获取运行次数，默认为1
        n_runs = config.get("n_runs", 1)
        wall_runtimes = []
        cpu_utils = []  # 用于记录psutil获取的CPU利用率
        memory_usages = []  # 用于记录每次运行的内存使用增量
        peak_memory_usage = 0  # 用于记录最大内存使用量
        final_state_vector = None
        
        # 记录开始的CPU时间和内存使用情况
        process = psutil.Process()
        start_cpu_times = process.cpu_times()
        #initial_memory_usage = process.memory_info().rss / (1024 ** 2)  # 初始内存使用量（MiB）

        # 循环执行量子电路n_runs次
        for _ in range(n_runs):
            # 启动CPU利用率和内存使用监控
            process.cpu_percent()  # 第一次调用，启动计时器
            start_memory_usage = process.memory_info().rss / (1024 ** 2)  # 运行前的内存使用量（MiB）
            
            # 记录每次运行的挂钟时间
            start_time = time.perf_counter()
            state_vector = convert_to_numpy(circuit(nshots=1).state())
            end_time = time.perf_counter()
            cpu_util = process.cpu_percent()  # 获取CPU利用率
            end_memory_usage = process.memory_info().rss / (1024 ** 2)  # 运行后的内存使用量（MiB）
            
            # 计算内存使用增量
            memory_usage_diff = end_memory_usage - start_memory_usage
            memory_usages.append(memory_usage_diff)
            peak_memory_usage = max(peak_memory_usage, end_memory_usage)
            
            wall_runtimes.append(end_time - start_time)
            cpu_utils.append(cpu_util)
            final_state_vector = state_vector

        # 记录结束的CPU时间
        end_cpu_times = process.cpu_times()
        # 计算总的CPU时间（用户态时间 + 内核态时间）
        cpu_time_total = (end_cpu_times.user - start_cpu_times.user) + \
                         (end_cpu_times.system - start_cpu_times.system)

        # 返回包含性能指标和最终状态向量的字典
        return {
            "wall_runtimes": wall_runtimes,
            "cpu_time_total": cpu_time_total,
            "cpu_utils": cpu_utils,
            "memory_usages": memory_usages,
            "peak_memory_usage": peak_memory_usage,
            "final_state_vector": final_state_vector
        }

# 模块 5: ResultProcessor 类
class ResultProcessor:
    """
    处理结果数据的类，计算运行时间、CPU利用率和保真度等指标。

    Attributes:
        None
    """

    def process(self, raw_data: dict, benchmark_state: np.ndarray = None):
        """
        处理原始数据并生成摘要和原始指标。

        Args:
            raw_data (dict): 包含运行时间和CPU时间的原始数据字典。
            benchmark_state (np.ndarray, optional): 基准状态向量，用于计算保真度。

        Returns:
            dict: 包含摘要和原始指标的字典。
        """
        wall_runtimes = raw_data["wall_runtimes"]
        cpu_time_total = raw_data["cpu_time_total"]
        final_state_vector = raw_data["final_state_vector"]
        cpu_utils = raw_data["cpu_utils"]  # 获取psutil的CPU利用率

        # 计算逻辑核心数
        logical_cores = psutil.cpu_count(logical=True)

        summary = {
            "runtime_avg": {"value": np.mean(wall_runtimes), "unit": "seconds"},
            "runtime_std_dev": {"value": np.std(wall_runtimes), "unit": "seconds"},
            "cpu_utilization_avg": {"value": (cpu_time_total / np.sum(wall_runtimes)) * 100, "unit": "percent"},
            "cpu_utilization_psutil_avg": {"value": np.mean(cpu_utils), "unit": "percent"},
            "cpu_utilization_psutil_std_dev": {"value": np.std(cpu_utils), "unit": "percent"},
            "cpu_cores_logical": logical_cores,  # 添加逻辑核心数
            "cpu_utilization_normalized": {
                "value": (np.mean(cpu_utils) / logical_cores),
                "unit": "percent"
            },
            "memory_usage_avg": {"value": np.mean(raw_data["memory_usages"]), "unit": "MiB"},
            "memory_usage_peak": {"value": raw_data["peak_memory_usage"], "unit": "MiB"}
        }

        if benchmark_state is not None:
            fidelity = np.abs(np.vdot(final_state_vector, benchmark_state)) ** 2
            summary["fidelity"] = {"value": fidelity, "unit": None}

        raw_metrics = {
            "runtime_per_run": {"values": wall_runtimes, "unit": "seconds"}
        }

        return {
            "summary": summary,
            "raw_metrics": raw_metrics
        }

# ... rest of code ...
def convert_to_numpy(array):
        """将不同框架的数组转换为NumPy数组"""
        # 处理NumPy数组
        if isinstance(array, np.ndarray):
            return array
            
        # 处理PyTorch Tensor
        elif isinstance(array, torch.Tensor):
            # 如果需要梯度，先分离
            if array.requires_grad:
                array = array.detach()
                # 如果在GPU上，移动到CPU
            if array.is_cuda:
                array = array.cpu()
            return array.numpy()
            
            # 处理JAX数组
        elif 'jaxlib.xla_extension.ArrayImpl' in str(type(array)):
            return jax.device_get(array)
            
            # 处理TensorFlow Tensor
        elif isinstance(array, tf.Tensor):
            return array.numpy()
            
            # 处理其他情况，尝试直接转换
        else:
            try:
                return np.array(array)
            except Exception as e:
                raise ValueError(f"无法将类型 {type(array)} 转换为NumPy数组: {str(e)}")
# 辅助函数: 将嵌套字典压平
def _flatten_dict(d, parent_key='', sep='_'):
    """
    将嵌套的字典展平为单层字典。

    参数:
    d (dict): 需要展平的字典。
    parent_key (str): 父键的前缀，用于递归时拼接键名，默认为空字符串。
    sep (str): 键名之间的分隔符，默认为下划线。

    返回:
    dict: 展平后的字典，键名为原字典的嵌套键拼接而成。
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# 辅助函数: 导出报告为CSV
def _export_to_csv(flat_report, filepath):
    """
    将扁平化的报告导出到CSV文件中。

    参数:
    flat_report (dict): 扁平化的报告数据，键为列名，值为对应的数据。
    filepath (str): CSV文件的路径。

    功能:
    - 如果文件不存在，则创建文件并写入表头。
    - 如果文件已存在，则直接追加数据行。
    """
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=flat_report.keys())
        if not file_exists:
            writer.writeheader()  # 写入表头
        writer.writerow(flat_report)

# 更新 profile_circuit 函数以支持CSV导出
def profile_circuit(circuit: Circuit, n_runs=1, mode='basic', calculate_fidelity=True):
    """分析量子电路的性能和保真度。

    该函数对输入的量子电路进行性能分析，包括运行时间、资源使用情况等指标的测量。
    可以选择计算保真度，并与基准状态进行比较。

    参数:
        circuit (Circuit): 待分析的量子电路
        n_runs (int, optional): 运行次数，默认为1
        mode (str, optional): 分析模式，默认为'basic'
        calculate_fidelity (bool, optional): 是否计算保真度，默认为True

    返回:
        dict: 包含分析结果的字典，结构如下：
            - metadata: 元数据信息
            - inputs: 输入电路的分析结果
            - results: 性能分析结果
            - error: 错误信息（如果有）
    """
    config = {
        "n_runs": n_runs,
        "mode": mode,
        "fidelity_calculated": calculate_fidelity
    }

    # 初始化报告骨架
    report = {
        "metadata": {},
        "inputs": {},
        "results": {},
        "error": None
    }

    try:
        # 收集元数据
        report["metadata"] = MetadataCollector.collect()

        # 分析输入
        input_analyzer = InputAnalyzer()
        report["inputs"] = input_analyzer.analyze(circuit, config)

        # 获取基准态矢量（如果需要）
        benchmark_state = None
        if calculate_fidelity:
            benchmark_manager = BenchmarkManager()
            circuit_hash = report["inputs"]["circuit_properties"]["qasm_hash_sha256"]
            benchmark_state = benchmark_manager.get_benchmark_state(circuit, circuit_hash)

        # 执行电路并测量性能
        execution_engine = ExecutionEngine()
        raw_data = execution_engine.run_and_measure(circuit, config)

        # 处理结果
        result_processor = ResultProcessor()
        processed_results = result_processor.process(raw_data, benchmark_state)
        report["results"] = processed_results

    except Exception as e:
        # 捕获异常并记录到报告中
        report["error"] = str(e)

    return report

# ... rest of code ...
# ... existing code ...
#if __name__ == "__main__":
    # 创建一个简单的 qibo 电路
    from qibo import gates, models
    from qibo.models import QFT
    from qibo import set_backend
    set_backend("numpy")
    # 定义一个简单的量子电路
    circuit = QFT(16)
    # 调用 profile_circuit 函数
    report = profile_circuit(
        circuit,
        n_runs=3,
        mode='basic',
        calculate_fidelity=True,
    )
#print(json.dumps(report, indent=4))
