# 量子模拟器基准测试平台 - 详细实施计划

## 阶段1: 核心架构与抽象层定义

### 目标
建立项目的基石，定义所有模块间通信所依赖的、不可变更的数据结构和接口。

### 具体任务

#### 1.1 创建项目基础结构
- 创建 `benchmark_harness/` 目录及其子目录
- 创建 `tests/` 目录
- 创建 `results/` 目录
- 创建 `__init__.py` 文件使目录成为Python包
- 创建干净的 `requirements.txt` 文件

#### 1.2 实现抽象层 (abstractions.py)
```python
import dataclasses
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

@dataclasses.dataclass
class BenchmarkResult:
    """标准化的基准测试结果数据结构"""
    # 元数据
    simulator: str
    backend: str
    circuit_name: str
    n_qubits: int
    
    # 速度指标
    wall_time_sec: float
    cpu_time_sec: float
    
    # 资源指标
    peak_memory_mb: float
    cpu_utilization_percent: float
    
    # 正确性指标
    state_fidelity: float
    
    # 原始输出供参考
    final_state: np.ndarray

class SimulatorInterface(ABC):
    """模拟器封装器的统一接口"""
    platform_name: str
    
    @abstractmethod
    def __init__(self, backend_name: str):
        """初始化封装器并配置特定后端"""
        pass
    
    @abstractmethod
    def execute(self, circuit: Any, n_qubits: int, reference_state: np.ndarray = None) -> BenchmarkResult:
        """执行给定电路并返回综合结果对象"""
        pass

class BenchmarkCircuit(ABC):
    """基准测试电路的工厂接口"""
    name: str = "Abstract Benchmark Circuit"
    
    @abstractmethod
    def build(self, platform: str, n_qubits: int) -> Any:
        """为指定平台构建并返回原生电路对象"""
        pass
```

#### 1.3 创建抽象层测试 (test_abstractions.py)
- 测试 `BenchmarkResult` 数据类创建和属性访问
- 测试 `SimulatorInterface` 无法直接实例化
- 测试 `BenchmarkCircuit` 无法直接实例化

### 成功标准
1. 所有抽象类和数据类被精确定义并写入 `abstractions.py` 文件
2. `tests/test_abstractions.py` 文件被创建，并包含验证抽象类无法被直接实例化的测试

## 阶段2: 可扩展的指标收集器模块

### 目标
创建一个专用的 `MetricsCollector` 类来封装所有性能和资源指标的测量逻辑。

### 具体任务

#### 2.1 实现指标收集器 (metrics.py)
```python
import time
import psutil
import tracemalloc
from typing import Dict, Any

class MetricsCollector:
    """用于分析代码块性能和资源的上下文管理器"""
    def __init__(self):
        self.process = psutil.Process()
        self.results: Dict[str, Any] = {}

    def __enter__(self):
        """启动所有监控器"""
        self.process.cpu_percent(interval=None)
        tracemalloc.start()
        self.cpu_time_start = self.process.cpu_times()
        self.wall_time_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止所有监控器并计算结果"""
        self.wall_time_end = time.perf_counter()
        self.cpu_time_end = self.process.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 计算指标
        self.results['wall_time_sec'] = self.wall_time_end - self.wall_time_start
        self.results['cpu_time_sec'] = (self.cpu_time_end.user - self.cpu_time_start.user) + \
                                       (self.cpu_time_end.system - self.cpu_time_start.system)
        self.results['peak_memory_mb'] = peak / (1024 * 1024)
        self.results['cpu_utilization_percent'] = self.process.cpu_percent(interval=None)
        
    def get_results(self) -> Dict[str, Any]:
        """返回收集的指标"""
        return self.results
```

#### 2.2 创建指标收集器测试 (test_metrics.py)
- 测试上下文管理器正确进入和退出
- 测试所有指标都被正确计算
- 测试指标值的合理性（非负数、合理范围）

### 成功标准
1. `MetricsCollector` 类被完整实现并写入 `metrics.py` 文件
2. `tests/test_metrics.py` 文件被创建，并包含验证所有指标准确性的单元测试

## 阶段3: 第一个具体实现 (Qibo封装与QFT电路)

### 目标
实现 `QiboWrapper` 和 `QFTCircuit`，打通从电路构建到指标返回的完整流程。

### 具体任务

#### 3.1 实现Qibo封装器 (simulators/qibo_wrapper.py)
```python
import qibo
import numpy as np
from benchmark_harness.abstractions import SimulatorInterface, BenchmarkResult
from benchmark_harness.metrics import MetricsCollector

class QiboWrapper(SimulatorInterface):
    platform_name = 'qibo'

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        qibo.set_backend(backend_name)

    def execute(self, circuit: qibo.models.Circuit, n_qubits: int, reference_state: np.ndarray = None) -> BenchmarkResult:
        collector = MetricsCollector()
        with collector:
            qibo_result = circuit(nshots=1)
            final_state = qibo_result.state()
        
        metrics = collector.get_results()
        fidelity = -1.0
        if reference_state is not None:
            fidelity = np.abs(np.vdot(reference_state, final_state))**2

        return BenchmarkResult(
            simulator='qibo', backend=self.backend_name, circuit_name=circuit.name,
            n_qubits=n_qubits, wall_time_sec=metrics.get('wall_time_sec'),
            cpu_time_sec=metrics.get('cpu_time_sec'), peak_memory_mb=metrics.get('peak_memory_mb'),
            cpu_utilization_percent=metrics.get('cpu_utilization_percent'),
            state_fidelity=fidelity, final_state=final_state
        )
```

#### 3.2 实现QFT电路 (circuits/qft.py)
```python
import numpy as np
import qibo
from qiskit import QuantumCircuit
from benchmark_harness.abstractions import BenchmarkCircuit

class QFTCircuit(BenchmarkCircuit):
    """量子傅里叶变换电路实现"""
    name = "Quantum Fourier Transform"

    def build(self, platform: str, n_qubits: int):
        if platform == 'qibo':
            return self._build_qibo_qft(n_qubits)
        elif platform == 'qiskit':
            return self._build_qiskit_qft(n_qubits)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def _build_qibo_qft(self, n_qubits: int):
        """构建Qibo平台的QFT电路"""
        c = qibo.models.Circuit(n_qubits)
        
        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            c.add(qibo.gates.H(j))
            for k in range(j + 1, n_qubits):
                theta = np.pi / (2 ** (k - j))
                c.add(qibo.gates.CU1(k, j, theta))
        
        # 应用交换门
        for j in range(n_qubits // 2):
            c.add(qibo.gates.SWAP(j, n_qubits - j - 1))
        
        c.name = f"qft_{n_qubits}_qubits"
        return c
    
    def _build_qiskit_qft(self, n_qubits: int):
        """构建Qiskit平台的QFT电路"""
        qc = QuantumCircuit(n_qubits)
        
        # 应用Hadamard门和受控相位门
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                theta = np.pi / (2 ** (k - j))
                qc.cu1(theta, k, j)
        
        # 应用交换门
        for j in range(n_qubits // 2):
            qc.swap(j, n_qubits - j - 1)
        
        qc.name = f"qft_{n_qubits}_qubits"
        return qc
```

#### 3.3 创建集成测试 (test_integration.py)
- 测试 `QiboWrapper` 和 `QFTCircuit` 可以协同工作
- 测试完整的执行流程返回有效的 `BenchmarkResult`

### 成功标准
1. `QiboWrapper` 实现了 `SimulatorInterface`，能够配置后端并使用 `MetricsCollector`
2. `QFTCircuit` 实现了 `BenchmarkCircuit`，能够为Qibo平台构建电路
3. 集成测试验证了两者可以协同工作

## 阶段4: 扩展至Qiskit和PennyLane

### 目标
通过实现 `QiskitWrapper` 验证架构的可扩展性，并扩展 `QFTCircuit` 的工厂能力。

### 具体任务

#### 4.1 实现Qiskit封装器 (simulators/qiskit_wrapper.py)
- 实现 `QiskitWrapper` 类，结构类似于 `QiboWrapper`
- 支持配置不同的Qiskit后端

#### 4.2 实现PennyLane封装器 (simulators/pennylane_wrapper.py)
- 实现 `PennyLaneWrapper` 类
- 支持配置不同的PennyLane后端

#### 4.3 扩展QFT电路支持
- 在 `QFTCircuit.build` 方法中添加对PennyLane的支持
- 确保所有平台的QFT电路在逻辑上等价

#### 4.4 创建模拟器测试 (test_simulators.py)
- 测试所有模拟器封装器的基本功能
- 测试后端配置的正确性

### 成功标准
1. `QiskitWrapper` 成功实现，其内部逻辑结构与 `QiboWrapper` 完全一致
2. `PennyLaneWrapper` 成功实现
3. `QFTCircuit` 的 `build` 方法被扩展，支持生成所有平台的原生电路对象

## 阶段5: 运行器与命令行接口

### 目标
实现 `BenchmarkRunner` 的核心逻辑，支持"黄金标准"参考态的生成与分发。

### 具体任务

#### 5.1 实现命令行接口 (run_benchmarks.py)
- 使用 `argparse` 添加所有必需的参数
- 支持选择性运行特定电路和模拟器
- 支持配置量子比特数量范围
- 支持指定黄金标准参考态

#### 5.2 实现核心运行器算法
- 动态导入和实例化所有请求的电路和封装器类
- 实现两阶段执行逻辑：
  1. 生成参考态
  2. 在所有模拟器上运行完整基准测试

#### 5.3 创建运行器测试
- 测试命令行参数解析
- 测试动态导入和实例化
- 测试两阶段执行逻辑

### 成功标准
1. CLI增加 `--golden-standard` 参数
2. 运行器成功实现两阶段执行逻辑

## 阶段6: 结果后处理与可视化

### 目标
将收集到的原始数据转化为对人类直观、有洞察力的CSV文件和多维度图表。

### 具体任务

#### 6.1 实现后处理模块 (post_processing.py)
- 实现 `analyze_results` 函数
- 生成包含所有指标列的CSV文件
- 生成用于分析所有关键指标的图表：
  - 保真度检查（条形图）- 正确性
  - 墙上时间扩展（线图）- 速度
  - 峰值内存扩展（线图）- 资源
  - CPU时间扩展（线图）- 速度
  - CPU利用率（条形图）- 资源

#### 6.2 创建后处理测试
- 测试CSV文件生成
- 测试图表生成
- 测试数据分析的正确性

#### 6.3 创建项目文档 (README.md)
- 清晰说明如何设置虚拟环境
- 说明如何安装依赖
- 说明如何通过命令行运行基准测试

### 成功标准
1. CSV文件包含所有指标列
2. 生成一套完整的、用于分析所有关键指标的图表
3. 项目根目录下包含清晰的README.md文档

## 预提交验证检查清单

每个阶段完成后必须执行：
1. 运行 `pytest` 确保所有测试通过
2. 运行 `black .` 格式化代码
3. 运行 `flake8 .` 检查代码质量
4. 对照DefinitionOfDone验证阶段产出

## 最终验收标准

1. 所有新代码都必须拥有对应的Pytest单元测试和集成测试，并达到高覆盖率
2. 位于项目根目录的 `run_benchmarks.py` 脚本必须是可执行的，并能通过命令行参数进行完整配置
3. 脚本成功运行后，必须在 `results/` 目录下生成一个带时间戳的CSV文件和多个PNG格式的可视化图表
4. 整个代码库必须完全符合在 `CodingStandardsAndTooling` 部分定义的所有标准（通过black, flake8, isort检查）
5. 项目根目录下必须包含一个 `README.md` 文件，清晰地说明如何设置虚拟环境、安装依赖以及如何通过命令行运行基准测试
6. 代码中不允许存在任何没有关联问题编号的 `# TODO` 注释