# 导入必要的库
import time
import psutil
import platform
import datetime
import hashlib
import numpy as np
import json
import os
import gc
import threading
import logging
import weakref
from typing import Optional, Dict, Any, Union, List, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 从qibo.models导入Circuit类，用于量子电路的构建
from qibo.models import Circuit
# 导入qibo模块，用于量子计算
import qibo
import torch  # 导入PyTorch库，用于深度学习计算
import jax  # 导入JAX库，用于高性能数值计算
import tensorflow as tf  # 导入TensorFlow库，用于深度学习计算
import cpuinfo  # 导入 py-cpuinfo 库

# ============================================================================
# 常量定义和配置
# ============================================================================

MEMORY_UNIT_BYTES = 1024 ** 2  # MiB
MEMORY_UNIT_GIB = 1024 ** 3    # GiB
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 300  # 秒

SUPPORTED_BACKENDS = {
    "numpy": {"backend_name": "numpy", "platform_name": None},
    "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
    "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
    "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
    "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
    "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"},
    "qulacs": {"backend_name": "qulacs", "platform_name": None}
}

@dataclass
class ProfilerConfig:
    """分析器配置类"""
    n_runs: int = 1
    mode: str = 'basic'
    calculate_fidelity: bool = True
    timeout_seconds: float = 300.0
    version: str = "1.0"
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """验证配置参数"""
        if self.n_runs <= 0:
            raise ValueError("n_runs必须是正整数")
        if self.mode not in ['basic', 'detailed', 'comprehensive']:
            raise ValueError("mode必须是'basic', 'detailed'或'comprehensive'")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须是正数")

# ============================================================================
# 异常类定义
# ============================================================================

class ProfilerError(Exception):
    """基础分析器异常"""
    pass

class BackendError(ProfilerError):
    """后端相关异常"""
    pass

class CircuitValidationError(ProfilerError):
    """电路验证异常"""
    pass

class MeasurementError(ProfilerError):
    """测量过程异常"""
    pass

# ============================================================================
# 线程安全的缓存系统
# ============================================================================

class ThreadSafeCache:
    """线程安全的缓存实现"""
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if len(self._cache) >= self._max_size:
                # LRU清理策略 - 删除最旧的条目
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)

class EnvironmentCache:
    """环境信息缓存，避免重复扫描"""
    _cache = {}
    _cache_ttl = DEFAULT_CACHE_TTL
    
    @classmethod
    def get_environment_info(cls) -> dict:
        """获取环境信息（带缓存）"""
        current_time = time.time()
        cache_key = "environment_info"
        
        # 检查缓存是否有效
        if cache_key in cls._cache:
            cached_data, timestamp = cls._cache[cache_key]
            if current_time - timestamp < cls._cache_ttl:
                return cached_data
        
        # 重新获取环境信息
        env_info = cls._scan_environment()
        cls._cache[cache_key] = (env_info, current_time)
        return env_info
    
    @classmethod
    def _scan_environment(cls) -> dict:
        """扫描环境信息"""
        cpu_info = cpuinfo.get_cpu_info()
        return {
            "qibo_backend": str(qibo.get_backend()),
            "qibo_version": getattr(qibo, '__version__', 'unknown'),
            "python_version": platform.python_version(),
            "cpu_model": platform.processor(),
            "cpu_model_friendly": cpu_info.get("brand_raw", "unknown"),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "total_memory": {
                "value": round(psutil.virtual_memory().total / MEMORY_UNIT_GIB, 2),
                "unit": "GiB"
            }
        }

# ============================================================================
# 输入验证器
# ============================================================================

class InputValidator:
    """输入验证工具类"""
    
    @staticmethod
    def validate_circuit(circuit: Circuit) -> None:
        """验证量子电路"""
        if circuit is None:
            raise CircuitValidationError("电路不能为None")
        if not hasattr(circuit, 'nqubits') or circuit.nqubits <= 0:
            raise CircuitValidationError("量子比特数必须为正数")
        if not hasattr(circuit, 'queue') or not circuit.queue:
            raise CircuitValidationError("电路不能为空")
    
    @staticmethod
    def validate_initial_state(initial_state) -> None:
        """验证初始状态"""
        if initial_state is not None:
            if not isinstance(initial_state, (np.ndarray, list, tuple)):
                raise CircuitValidationError("初始状态必须是数组类型")
            if isinstance(initial_state, np.ndarray) and initial_state.size == 0:
                raise CircuitValidationError("初始状态不能为空数组")
    
    @staticmethod
    def validate_config(config: ProfilerConfig) -> None:
        """验证配置"""
        config.validate()

# ============================================================================
# 安全的后端管理器
# ============================================================================

class SafeBackendManager:
    """安全的后端管理器"""
    
    @staticmethod
    @contextmanager
    def with_backend_safety(backend_name: str, platform: Optional[str] = None):
        """上下文管理器确保后端安全切换"""
        class BackendContext:
            def __init__(self, backend_name: str, platform: Optional[str] = None):
                self.backend_name = backend_name
                self.platform = platform
                self.original_backend = None
            
            def __enter__(self):
                try:
                    self.original_backend = qibo.get_backend()
                    qibo.set_backend(self.backend_name, platform=self.platform)
                    return qibo.get_backend()
                except Exception as e:
                    raise BackendError(f"无法切换到后端 {backend_name}: {str(e)}")
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.original_backend:
                    try:
                        original_name = getattr(self.original_backend, 'name', 'numpy')
                        original_platform = getattr(self.original_backend, 'platform', None)
                        qibo.set_backend(original_name, platform=original_platform)
                    except Exception as e:
                        # 记录错误但不抛出异常，避免掩盖原始错误
                        logging.warning(f"警告：无法恢复原始后端: {str(e)}")
        
        context = BackendContext(backend_name, platform)
        yield context

# ============================================================================
# 精确的性能测量工具
# ============================================================================

class PrecisionMeasurement:
    """高精度性能测量工具"""
    
    @staticmethod
    @contextmanager
    def measure_performance() -> Generator[Dict[str, Any], None, None]:
        """上下文管理器进行精确测量"""
        # 强制垃圾回收，减少内存测量噪声
        gc.collect()
        
        process = psutil.Process()
        
        # 记录初始状态
        start_time = time.perf_counter()
        start_cpu_times = process.cpu_times()
        start_memory = process.memory_info().rss
        
        # 启动CPU监控
        process.cpu_percent()
        
        measurement_data = {
            "start_time": start_time,
            "start_cpu": start_cpu_times,
            "start_memory": start_memory
        }
        
        try:
            yield measurement_data
        finally:
            # 记录结束状态
            end_time = time.perf_counter()
            end_cpu_times = process.cpu_times()
            end_memory = process.memory_info().rss
            cpu_util = process.cpu_percent()
            
            # 再次垃圾回收
            gc.collect()
            
            # 计算差值
            wall_time = end_time - start_time
            cpu_time = (end_cpu_times.user - start_cpu_times.user) + \
                      (end_cpu_times.system - start_cpu_times.system)
            memory_delta = end_memory - start_memory
            
            # 更新测量数据
            measurement_data.update({
                "wall_time": wall_time,
                "cpu_time": cpu_time,
                "memory_delta": memory_delta,
                "end_memory": end_memory,
                "cpu_util": cpu_util
            })

# ============================================================================
# 日志管理器
# ============================================================================

class ProfilerLogger:
    """统一的日志管理"""
    
    def __init__(self, name: str = "qibo_profiler", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

# ============================================================================
# 改进的核心组件
# ============================================================================

class MetadataCollector:
    """元数据收集器"""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
    
    def collect(self) -> dict:
        """收集元数据信息"""
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)
        timestamp_utc = current_time_utc.isoformat().replace('+00:00', 'Z')
        return {
            "profiler_version": self.config.version,
            "timestamp_utc": timestamp_utc
        }

class InputAnalyzer:
    """输入分析器"""
    
    def analyze(self, circuit: Circuit, config: ProfilerConfig) -> dict:
        """分析输入的电路和配置文件"""
        # 验证输入
        InputValidator.validate_circuit(circuit)
        InputValidator.validate_config(config)
        
        profiler_settings = self._analyze_profiler_settings(config)
        circuit_properties = self._analyze_circuit(circuit)
        environment = EnvironmentCache.get_environment_info()
        
        return {
            "profiler_settings": profiler_settings,
            "circuit_properties": circuit_properties,
            "environment": environment
        }
    
    def _analyze_profiler_settings(self, config: ProfilerConfig) -> dict:
        """分析profiler设置"""
        return {
            "n_runs": config.n_runs,
            "mode": config.mode,
            "fidelity_calculated": config.calculate_fidelity
        }
    
    def _analyze_circuit(self, circuit: Circuit) -> dict:
        """分析电路属性"""
        gate_counts = {}
        for gate in circuit.queue:
            gate_name = gate.name.lower()
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        qasm = circuit.to_qasm()
        qasm_hash_sha256 = hashlib.sha256(qasm.encode()).hexdigest()
        
        return {
            "n_qubits": circuit.nqubits,
            "depth": circuit.depth,
            "total_gates": len(circuit.queue),
            "gate_counts": gate_counts,
            "qasm_hash_sha256": qasm_hash_sha256
        }

class BenchmarkManager:
    """基准管理器（改进版）"""
    
    def __init__(self):
        self._cache = ThreadSafeCache()
        self.logger = ProfilerLogger("BenchmarkManager")
    
    def get_benchmark_state(self, circuit: Circuit, circuit_hash: str, initial_state=None) -> np.ndarray:
        """获取电路的基准状态"""
        # 检查缓存
        cached_state = self._cache.get(circuit_hash)
        if cached_state is not None:
            return cached_state
        
        # 获取当前后端配置
        current_backend = qibo.get_backend()
        current_backend_name = current_backend.name
        current_platform = getattr(current_backend, 'platform', None)
        
        # 查找当前后端的配置
        original_backend_config = None
        for key, config in SUPPORTED_BACKENDS.items():
            if config['backend_name'] == current_backend_name and config.get('platform_name') == current_platform:
                original_backend_config = config
                break
        
        if not original_backend_config:
            raise BackendError(f"未知的后端配置: {current_backend_name} with platform {current_platform}")
        
        # 计算基准状态
        if current_backend_name == "qibojit":
            # 如果当前就是基准后端，直接计算
            state = self._compute_state(circuit, initial_state)
            norm_state = state / np.linalg.norm(state)
            self._cache.set(circuit_hash, norm_state)
            return norm_state
        else:
            # 切换到基准后端
            try:
                with SafeBackendManager.with_backend_safety("qibojit"):
                    state = self._compute_state(circuit, initial_state)
                    norm_state = state / np.linalg.norm(state)
                    self._cache.set(circuit_hash, norm_state)
                    return norm_state
            except Exception as e:
                self.logger.error(f"计算基准状态失败: {str(e)}")
                raise BackendError(f"无法计算基准状态: {str(e)}")
    
    def _compute_state(self, circuit: Circuit, initial_state=None) -> np.ndarray:
        """计算电路状态"""
        try:
            state = circuit(nshots=1, initial_state=initial_state).state()
            return convert_to_numpy(state)
        except Exception as e:
            raise MeasurementError(f"状态计算失败: {str(e)}")

class ExecutionEngine:
    """执行引擎（改进版）"""
    
    def __init__(self):
        self.logger = ProfilerLogger("ExecutionEngine")
    
    def run_and_measure(self, circuit: Circuit, config: ProfilerConfig, initial_state=None) -> dict:
        """运行量子电路并测量其性能指标"""
        # 验证输入
        InputValidator.validate_circuit(circuit)
        InputValidator.validate_config(config)
        
        wall_runtimes = []
        cpu_utils = []
        memory_usages = []
        peak_memory_usage = 0
        final_state_vector = None
        
        process = psutil.Process()
        start_cpu_times = process.cpu_times()
        
        try:
            for run_idx in range(config.n_runs):
                self.logger.info(f"执行第 {run_idx + 1} 次运行")
                
                with PrecisionMeasurement.measure_performance() as measurement:
                    # 执行电路
                    result = circuit(nshots=1, initial_state=initial_state)
                    state_vector = convert_to_numpy(result.state())
                    final_state_vector = state_vector / np.linalg.norm(state_vector)
                
                # 记录测量结果
                wall_runtimes.append(measurement["wall_time"])
                cpu_utils.append(measurement["cpu_util"])
                memory_usages.append(measurement["memory_delta"])
                peak_memory_usage = max(peak_memory_usage, measurement["end_memory"])
            
            # 计算总CPU时间
            end_cpu_times = process.cpu_times()
            cpu_time_total = (end_cpu_times.user - start_cpu_times.user) + \
                           (end_cpu_times.system - start_cpu_times.system)
            
            return {
                "wall_runtimes": wall_runtimes,
                "cpu_time_total": cpu_time_total,
                "cpu_utils": cpu_utils,
                "memory_usages": memory_usages,
                "peak_memory_usage": peak_memory_usage / MEMORY_UNIT_BYTES,  # 转换为MiB
                "final_state_vector": final_state_vector
            }
            
        except Exception as e:
            self.logger.error(f"电路执行失败: {str(e)}")
            raise MeasurementError(f"电路执行失败: {str(e)}")

class ResultProcessor:
    """结果处理器（改进版）"""
    
    def __init__(self):
        self.logger = ProfilerLogger("ResultProcessor")
    
    def process(self, raw_data: dict, benchmark_state: Optional[np.ndarray] = None) -> dict:
        """处理原始数据并生成摘要和原始指标"""
        try:
            wall_runtimes = raw_data["wall_runtimes"]
            cpu_time_total = raw_data["cpu_time_total"]
            final_state_vector = raw_data["final_state_vector"]
            cpu_utils = raw_data["cpu_utils"]
            
            logical_cores = psutil.cpu_count(logical=True)
            
            summary = {
                "runtime_avg": {"value": np.mean(wall_runtimes), "unit": "seconds"},
                "runtime_std_dev": {"value": np.std(wall_runtimes), "unit": "seconds"},
                "cpu_utilization_avg": {"value": (cpu_time_total / np.sum(wall_runtimes)) * 100, "unit": "percent"},
                "cpu_utilization_psutil_avg": {"value": np.mean(cpu_utils), "unit": "percent"},
                "cpu_utilization_psutil_std_dev": {"value": np.std(cpu_utils), "unit": "percent"},
                "cpu_cores_logical": logical_cores,
                "cpu_utilization_normalized": {
                    "value": (np.mean(cpu_utils) / logical_cores),
                    "unit": "percent"
                },
                "memory_usage_avg": {"value": np.mean(raw_data["memory_usages"]) / MEMORY_UNIT_BYTES, "unit": "MiB"},
                "memory_usage_peak": {"value": raw_data["peak_memory_usage"], "unit": "MiB"}
            }
            
            # 计算保真度
            if benchmark_state is not None:
                try:
                    fidelity = np.abs(np.vdot(final_state_vector, benchmark_state)) ** 2
                    summary["fidelity"] = {"value": fidelity, "unit": None}
                except Exception as e:
                    self.logger.warning(f"保真度计算失败: {str(e)}")
                    summary["fidelity"] = {"value": None, "unit": None, "error": str(e)}
            
            raw_metrics = {
                "runtime_per_run": {"values": wall_runtimes, "unit": "seconds"}
            }
            
            return {
                "summary": summary,
                "raw_metrics": raw_metrics
            }
            
        except Exception as e:
            self.logger.error(f"结果处理失败: {str(e)}")
            raise ProfilerError(f"结果处理失败: {str(e)}")

# ============================================================================
# 分析器管道
# ============================================================================

class ProfilerPipeline:
    """分析器管道，使用依赖注入"""
    
    def __init__(self, 
                 metadata_collector: Optional[MetadataCollector] = None,
                 input_analyzer: Optional[InputAnalyzer] = None,
                 benchmark_manager: Optional[BenchmarkManager] = None,
                 execution_engine: Optional[ExecutionEngine] = None,
                 result_processor: Optional[ResultProcessor] = None):
        
        self.metadata_collector = metadata_collector or MetadataCollector(ProfilerConfig())
        self.input_analyzer = input_analyzer or InputAnalyzer()
        self.benchmark_manager = benchmark_manager or BenchmarkManager()
        self.execution_engine = execution_engine or ExecutionEngine()
        self.result_processor = result_processor or ResultProcessor()
        self.logger = ProfilerLogger("ProfilerPipeline")
    
    def execute(self, circuit: Circuit, config: ProfilerConfig, initial_state=None) -> dict:
        """执行完整的分析管道"""
        # 验证输入
        InputValidator.validate_circuit(circuit)
        InputValidator.validate_config(config)
        InputValidator.validate_initial_state(initial_state)
        
        report = {
            "metadata": {},
            "inputs": {},
            "results": {},
            "error": None
        }
        
        try:
            self.logger.info("开始执行性能分析")
            
            # 执行管道
            report["metadata"] = self.metadata_collector.collect()
            report["inputs"] = self.input_analyzer.analyze(circuit, config)
            
            if config.calculate_fidelity:
                circuit_hash = report["inputs"]["circuit_properties"]["qasm_hash_sha256"]
                benchmark_state = self.benchmark_manager.get_benchmark_state(
                    circuit, circuit_hash, initial_state
                )
            else:
                benchmark_state = None
            
            raw_data = self.execution_engine.run_and_measure(circuit, config, initial_state)
            report["results"] = self.result_processor.process(raw_data, benchmark_state)
            
            self.logger.info("性能分析完成")
            
        except Exception as e:
            self.logger.error(f"分析管道执行失败: {str(e)}")
            report["error"] = str(e)
            # 添加详细的错误上下文
            report["error_context"] = {
                "circuit_info": {
                    "n_qubits": getattr(circuit, 'nqubits', 'unknown'),
                    "depth": getattr(circuit, 'depth', 'unknown'),
                    "gate_count": len(getattr(circuit, 'queue', []))
                },
                "config": config.__dict__
            }
        
        return report

# ============================================================================
# 辅助函数
# ============================================================================

def convert_to_numpy(array) -> np.ndarray:
    """将不同框架的数组转换为NumPy数组"""
    # 处理NumPy数组
    if isinstance(array, np.ndarray):
        return array
    
    # 处理PyTorch Tensor
    elif isinstance(array, torch.Tensor):
        if array.requires_grad:
            array = array.detach()
        if array.is_cuda:
            array = array.cpu()
        return array.numpy()
    
    # 处理JAX数组
    elif 'jaxlib.xla_extension.ArrayImpl' in str(type(array)):
        return jax.device_get(array)
    
    # 处理TensorFlow Tensor
    elif isinstance(array, tf.Tensor):
        return array.numpy()
    
    # 处理其他情况
    else:
        try:
            return np.array(array)
        except Exception as e:
            raise ValueError(f"无法将类型 {type(array)} 转换为NumPy数组: {str(e)}")

def _flatten_dict(d, parent_key='', sep='_'):
    """将嵌套的字典展平为单层字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# ============================================================================
# 主要API函数
# ============================================================================

def profile_circuit(circuit: Circuit, 
                   n_runs: int = 1, 
                   mode: str = 'basic', 
                   calculate_fidelity: bool = True, 
                   initial_state=None,
                   timeout_seconds: float = 300.0) -> dict:
    """分析量子电路的性能和保真度（改进版）
    
    参数:
        circuit (Circuit): 待分析的量子电路
        n_runs (int, optional): 运行次数，默认为1
        mode (str, optional): 分析模式，默认为'basic'
        calculate_fidelity (bool, optional): 是否计算保真度，默认为True
        initial_state: 初始状态，默认为None
        timeout_seconds (float, optional): 超时时间（秒），默认为300.0
    
    返回:
        dict: 包含分析结果的字典
    """
    # 创建配置
    config = ProfilerConfig(
        n_runs=n_runs,
        mode=mode,
        calculate_fidelity=calculate_fidelity,
        timeout_seconds=timeout_seconds
    )
    
    # 创建并执行管道
    pipeline = ProfilerPipeline()
    return pipeline.execute(circuit, config, initial_state)

def generate_markdown_report(report: dict, 
                           output_path: Optional[str] = None, 
                           default_dir: Optional[str] = None) -> str:
    """将性能分析报告转换为 Markdown 格式文档"""
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = default_dir if default_dir else os.getcwd()
        backend_name = report.get('inputs', {}).get('environment', {}).get('qibo_backend', 'unknown').replace(' ', '_').replace('(', '').replace(')', '')
        output_path = os.path.join(base_dir, f"qibo_report_{backend_name}_{timestamp}.md")
    
    markdown_lines = [
        "# 量子电路性能分析报告",
        "",
        "## 元数据",
        f"- 分析器版本: {report['metadata']['profiler_version']}",
        f"- 生成时间: {report['metadata']['timestamp_utc']}",
        "",
        "## 输入参数",
        "### 分析器设置",
        f"- 运行次数: {report['inputs']['profiler_settings']['n_runs']}",
        f"- 分析模式: {report['inputs']['profiler_settings']['mode']}",
        f"- 保真度计算: {'是' if report['inputs']['profiler_settings']['fidelity_calculated'] else '否'}",
        "",
        "### 电路属性",
        f"- 量子比特数: {report['inputs']['circuit_properties']['n_qubits']}",
        f"- 电路深度: {report['inputs']['circuit_properties']['depth']}",
        f"- 总门数: {report['inputs']['circuit_properties']['total_gates']}",
        "",
        "#### 门统计",
        "| 门类型 | 数量 |",
        "|--------|------|"
    ]
    
    # 添加门统计表格
    for gate, count in report['inputs']['circuit_properties']['gate_counts'].items():
        markdown_lines.append(f"| {gate} | {count} |")
    
    # 添加环境信息
    markdown_lines.extend([
        "",
        "### 运行环境",
        f"- Qibo 后端: {report['inputs']['environment']['qibo_backend']}",
        f"- Qibo 版本: {report['inputs']['environment']['qibo_version']}",
        f"- Python 版本: {report['inputs']['environment']['python_version']}",
        f"- CPU 型号: {report['inputs']['environment']['cpu_model_friendly']}",
        f"- 物理核心数: {report['inputs']['environment']['cpu_cores_physical']}",
        f"- 总内存: {report['inputs']['environment']['total_memory']['value']} GiB",
        "",
        "## 性能结果",
        "### 摘要统计",
        f"- 平均运行时间: {report['results']['summary']['runtime_avg']['value']:.2f} 秒",
        f"- 运行时间标准差: {report['results']['summary']['runtime_std_dev']['value']:.2f} 秒",
        f"- 平均 CPU 利用率: {report['results']['summary']['cpu_utilization_avg']['value']:.2f}%",
        f"- 平均每核CPU利用率: {report['results']['summary']['cpu_utilization_normalized']['value']:.2f}%",
        f"- 平均内存使用: {report['results']['summary']['memory_usage_avg']['value']:.2f} MiB",
        f"- 峰值内存使用: {report['results']['summary']['memory_usage_peak']['value']:.2f} MiB"
    ])
    
    # 如果有保真度信息，添加保真度数据
    if 'fidelity' in report['results']['summary']:
        fidelity_value = report['results']['summary']['fidelity']['value']
        if fidelity_value is not None:
            markdown_lines.append(f"- 保真度: {fidelity_value:.6f}")
        else:
            error_msg = report['results']['summary']['fidelity'].get('error', '未知错误')
            markdown_lines.append(f"- 保真度: 计算失败 ({error_msg})")
    
    # 添加每次运行的详细数据
    markdown_lines.extend([
        "",
        "### 详细运行数据",
        "#### 每次运行时间",
        "| 运行次数 | 时间 (秒) |",
        "|----------|-----------|"
    ])
    
    for i, runtime in enumerate(report['results']['raw_metrics']['runtime_per_run']['values'], 1):
        markdown_lines.append(f"| 第 {i} 次 | {runtime:.2f} |")
    
    # 如果有错误信息，添加错误部分
    if report.get('error'):
        markdown_lines.extend([
            "",
            "## 错误信息",
            f"```\n{report['error']}\n```"
        ])
        
        # 添加错误上下文
        if 'error_context' in report:
            markdown_lines.extend([
                "",
                "### 错误上下文",
                f"```json\n{json.dumps(report['error_context'], indent=2, ensure_ascii=False)}\n```"
            ])
    
    markdown_content = '\n'.join(markdown_lines)
    
    # 写入文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    except Exception as e:
        logging.error(f"无法写入报告文件 {output_path}: {str(e)}")
        raise
    
    return output_path

# ============================================================================
# 向后兼容的函数
# ============================================================================

# 为了保持向后兼容性，提供原始API的包装器
def profile_circuit_legacy(circuit: Circuit, n_runs=1, mode='basic', calculate_fidelity=True, initial_state=None):
    """向后兼容的性能分析函数"""
    return profile_circuit(circuit, n_runs, mode, calculate_fidelity, initial_state)
