

# **实施计划：模块化量子模拟器基准测试平台 (v4.0 - 最终蓝图版)**

**总体目标**: 创建一个可配置、可扩展、具有科研精度的平台，用于严谨地分析和比较Qibo、Qiskit、PennyLane模拟器的**性能、资源消耗和计算正确性**，重点识别Qibo在不同后端下的综合特性。

---

## 阶段 1: 核心架构与抽象层定义

**目标**: 建立项目的基石，定义所有模块间通信所依赖的、不可变更的数据结构和接口。

**成功标准**:
1.  所有抽象类和数据类被精确定义并写入`abstractions.py`文件。
2.  `tests/test_abstractions.py`文件被创建，并包含验证抽象类无法被直接实例化的测试。

**关键任务**:
-   [ ] 创建项目结构: `benchmark_harness/`, `tests/`, `results/`, `run_benchmarks.py`。
-   [ ] 在 `benchmark_harness/abstractions.py` 文件中写入以下完整代码：
    ```python
    import dataclasses
    import numpy as np
    from abc import ABC, abstractmethod
    from typing import Any

    @dataclasses.dataclass
    class BenchmarkResult:
        """A standardized data structure for storing all benchmark results."""
        # --- Metadata ---
        simulator: str
        backend: str
        circuit_name: str
        n_qubits: int
        
        # --- Speed Metrics ---
        wall_time_sec: float
        cpu_time_sec: float
        
        # --- Resource Metrics ---
        peak_memory_mb: float
        cpu_utilization_percent: float
        
        # --- Correctness Metrics ---
        state_fidelity: float
        
        # --- Raw Output for Reference ---
        final_state: np.ndarray

    class SimulatorInterface(ABC):
        """
        An interface for a simulator wrapper. It standardizes how benchmarks are
        executed and how different backends are handled.
        """
        platform_name: str # e.g., 'qibo', 'qiskit'. Used by the runner.
        
        @abstractmethod
        def __init__(self, backend_name: str):
            """Initializes the wrapper and configures the specific backend."""
            self.backend_name = backend_name
            raise NotImplementedError

        @abstractmethod
        def execute(self, circuit: Any, n_qubits: int, reference_state: np.ndarray = None) -> BenchmarkResult:
            """Executes a given circuit and returns a comprehensive result object."""
            raise NotImplementedError

    class BenchmarkCircuit(ABC):
        """
        An interface for an algorithm blueprint. It acts as a factory for creating
        the same logical circuit on different simulation platforms.
        """
        name: str = "Abstract Benchmark Circuit"

        @abstractmethod
        def build(self, platform: str, n_qubits: int) -> Any:
            """Builds and returns a native circuit object for the specified platform."""
            raise NotImplementedError
    ```

**状态**: [未开始]

---

## 阶段 2: 可扩展的指标收集器模块

**目标**: 创建一个专用的 `MetricsCollector` 类来封装所有性能和资源指标的测量逻辑。

**成功标准**:
1.  `MetricsCollector` 类被完整实现并写入`metrics.py`文件。
2.  `tests/test_metrics.py`文件被创建，并包含验证所有指标准确性的单元测试。

**关键任务**:
-   [ ] 在 `benchmark_harness/metrics.py` 文件中写入以下完整代码：
    ```python
    import time
    import psutil
    import tracemalloc
    from typing import Dict, Any

    class MetricsCollector:
        """A context manager to profile a block of code for performance and resources."""
        def __init__(self):
            self.process = psutil.Process()
            self.results: Dict[str, Any] = {}

        def __enter__(self):
            """Start all monitors."""
            self.process.cpu_percent(interval=None) # Non-blocking call to initialize
            tracemalloc.start()
            self.cpu_time_start = self.process.cpu_times()
            self.wall_time_start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Stop all monitors and calculate results."""
            self.wall_time_end = time.perf_counter()
            self.cpu_time_end = self.process.cpu_times()
            current, peak = tracemacoll.get_traced_memory()
            tracemalloc.stop()

            # Calculate metrics
            self.results['wall_time_sec'] = self.wall_time_end - self.wall_time_start
            self.results['cpu_time_sec'] = (self.cpu_time_end.user - self.cpu_time_start.user) + \
                                           (self.cpu_time_end.system - self.cpu_time_start.system)
            self.results['peak_memory_mb'] = peak / (1024 * 1024)
            self.results['cpu_utilization_percent'] = self.process.cpu_percent(interval=None)
            
        def get_results(self) -> Dict[str, Any]:
            """Return the collected metrics."""
            return self.results
    ```

**状态**: [未开始]

---

## 阶段 3: 第一个具体实现 (Qibo封装与QFT电路)

**目标**: 实现 `QiboWrapper` 和 `QFTCircuit`，打通从电路构建到指标返回的完整流程。

**成功标准**:
1.  `QiboWrapper` 实现了 `SimulatorInterface`，能够配置后端并使用 `MetricsCollector`。
2.  `QFTCircuit` 实现了 `BenchmarkCircuit`，能够为Qibo平台构建电路。
3.  集成测试验证了两者可以协同工作。

**关键任务**:
-   [ ] 在 `benchmark_harness/simulators/qibo_wrapper.py` 中创建 `QiboWrapper` 类。
    -   **核心逻辑**:
        ```python
        # In benchmark_harness/simulators/qibo_wrapper.py
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
-   [ ] 在 `benchmark_harness/circuits/qft.py` 中创建 `QFTCircuit` 类。**指令**: 必须使用基础门来构建QFT电路，以确保与其他平台比较的公平性。

**状态**: [未开始]

---

## 阶段 4: 扩展至Qiskit和PennyLane

**目标**: 通过实现 `QiskitWrapper` 验证架构的可扩展性，并扩展 `QFTCircuit` 的工厂能力。

**成功标准**:
1.  `QiskitWrapper` 成功实现，其内部逻辑结构与 `QiboWrapper` 完全一致。
2.  `QFTCircuit` 的 `build` 方法被扩展，支持生成Qiskit的原生电路对象。

**关键任务**:
-   [ ] 在 `benchmark_harness/simulators/qiskit_wrapper.py` 中创建 `QiskitWrapper` 类。
    -   **核心逻辑**:
        ```python
        # In benchmark_harness/simulators/qiskit_wrapper.py
        import numpy as np
        from qiskit import execute
        from qiskit.providers.aer import AerSimulator
        from benchmark_harness.abstractions import SimulatorInterface, BenchmarkResult
        from benchmark_harness.metrics import MetricsCollector
        
        class QiskitWrapper(SimulatorInterface):
            platform_name = 'qiskit'

            def __init__(self, backend_name: str):
                self.backend_name = backend_name
                if backend_name == 'aer_simulator':
                    self.backend_instance = AerSimulator()
                # Add other Qiskit backends here if needed

            def execute(self, circuit: QuantumCircuit, n_qubits: int, reference_state: np.ndarray = None) -> BenchmarkResult:
                collector = MetricsCollector()
                with collector:
                    job = execute(circuit, self.backend_instance, shots=1)
                    final_state = job.result().get_statevector()
                
                metrics = collector.get_results()
                fidelity = -1.0
                if reference_state is not None:
                    fidelity = np.abs(np.vdot(reference_state, final_state))**2

                return BenchmarkResult(
                    simulator='qiskit', backend=self.backend_name, circuit_name=circuit.name,
                    n_qubits=n_qubits, wall_time_sec=metrics.get('wall_time_sec'),
                    cpu_time_sec=metrics.get('cpu_time_sec'), peak_memory_mb=metrics.get('peak_memory_mb'),
                    cpu_utilization_percent=metrics.get('cpu_utilization_percent'),
                    state_fidelity=fidelity, final_state=final_state
                )
        ```
-   [ ] 对 `pennylane_wrapper.py` 执行相同的模式。
-   [ ] 扩展 `qft.py` 中的 `QFTCircuit.build` 方法，添加 `elif platform == 'qiskit': ...` 逻辑。

**状态**: [未开始]

---

## 阶段 5: 运行器与命令行接口

**目标**: 实现 `BenchmarkRunner` 的核心逻辑，支持“黄金标准”参考态的生成与分发。

**成功标准**:
1.  CLI增加 `--golden-standard` 参数。
2.  运行器成功实现**两阶段执行**逻辑。

**关键任务**:
-   [ ] 在 `run_benchmarks.py` 中使用 `argparse` 添加所有必需的参数。
-   [ ] **核心运行器算法实现**:
    ```python
    # In run_benchmarks.py
    # 1. Setup Phase
    #   - Parse CLI arguments (circuits, qubits, simulators, backends, golden-standard).
    #   - Dynamically import and instantiate all requested Circuit and Wrapper classes.
    #   - Create a dictionary of configured simulator instances:
    #     e.g., configured_simulators = {"qibo-numpy": QiboWrapper("numpy"), "qiskit-aer_simulator": QiskitWrapper("aer_simulator")}
    
    all_results = []
    
    # 2. Main Loop over each task
    for circuit_class_instance in requested_circuits:
        for n_qubits in requested_qubit_range:
            
            # --- PHASE A: Generate Reference State ---
            golden_standard_key = args.golden_standard
            golden_wrapper = configured_simulators[golden_standard_key]
            
            circuit_for_golden = circuit_class_instance.build(platform=golden_wrapper.platform_name, n_qubits=n_qubits)
            circuit_for_golden.name = circuit_class_instance.name # Assign name to circuit object

            # Execute once to get the reference state from the result object
            golden_result = golden_wrapper.execute(circuit_for_golden, n_qubits)
            reference_state = golden_result.final_state

            # --- PHASE B: Run Full Benchmark on All Simulators ---
            for runner_id, wrapper_instance in configured_simulators.items():
                
                circuit_for_current = circuit_class_instance.build(platform=wrapper_instance.platform_name, n_qubits=n_qubits)
                circuit_for_current.name = circuit_class_instance.name

                # If this is the golden standard, we can reuse its result. Otherwise, run fresh.
                if runner_id == golden_standard_key:
                    # Update fidelity of the golden run (it's always 1.0 against itself)
                    golden_result.state_fidelity = 1.0
                    all_results.append(golden_result)
                else:
                    result = wrapper_instance.execute(
                        circuit=circuit_for_current,
                        n_qubits=n_qubits,
                        reference_state=reference_state
                    )
                    all_results.append(result)

    # 3. Post-processing
    analyze_results(all_results)
    ```

**状态**: [未开始]

---

## 阶段 6: 结果后处理与可视化

**目标**: 将收集到的原始数据转化为对人类直观、有洞察力的CSV文件和多维度图表。

**成功标准**:
1.  CSV文件包含所有指标列。
2.  生成一套完整的、用于分析所有关键指标的图表。

**关键任务**:
-   [ ] 在 `benchmark_harness/post_processing.py` 中创建 `analyze_results` 函数。
-   [ ] **核心后处理逻辑**:
    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def analyze_results(results_list: list[BenchmarkResult], output_dir: str):
        df = pd.DataFrame(results_list)
        df['runner_id'] = df['simulator'] + '-' + df['backend']
        df.to_csv(f"{output_dir}/raw_results.csv", index=False)

        # Plot 1: Fidelity Check (Bar Plot) - Correctness
        plt.figure(figsize=(12, 7)); sns.barplot(data=df, x='runner_id', y='state_fidelity', hue='n_qubits');
        plt.title('State Fidelity vs Golden Standard'); plt.ylabel('Fidelity'); plt.xticks(rotation=45); plt.tight_layout();
        plt.savefig(f"{output_dir}/fidelity.png"); plt.close()

        # Plot 2: Wall Time Scaling (Line Plot) - Speed
        plt.figure(figsize=(12, 7)); sns.lineplot(data=df, x='n_qubits', y='wall_time_sec', hue='runner_id', marker='o');
        plt.title('Wall Clock Time vs Number of Qubits'); plt.ylabel('Time (seconds)'); plt.xlabel('Number of Qubits'); plt.grid(True);
        plt.savefig(f"{output_dir}/wall_time_scaling.png"); plt.close()

        # Plot 3: Peak Memory Scaling (Line Plot) - Resource
        plt.figure(figsize=(12, 7)); sns.lineplot(data=df, x='n_qubits', y='peak_memory_mb', hue='runner_id', marker='o');
        plt.title('Peak Memory Usage vs Number of Qubits'); plt.ylabel('Memory (MB)'); plt.xlabel('Number of Qubits'); plt.grid(True);
        plt.savefig(f"{output_dir}/memory_scaling.png"); plt.close()

        # Plot 4: CPU Time Scaling (Line Plot) - Speed
        plt.figure(figsize=(12, 7)); sns.lineplot(data=df, x='n_qubits', y='cpu_time_sec', hue='runner_id', marker='o');
        plt.title('CPU Time vs Number of Qubits'); plt.ylabel('CPU Time (seconds)'); plt.xlabel('Number of Qubits'); plt.grid(True);
        plt.savefig(f"{output_dir}/cpu_time_scaling.png"); plt.close()

        # Plot 5: CPU Utilization (Bar Plot) - Resource
        plt.figure(figsize=(12, 7)); sns.barplot(data=df, x='runner_id', y='cpu_utilization_percent', hue='n_qubits');
        plt.title('CPU Utilization'); plt.ylabel('CPU %'); plt.xticks(rotation=45); plt.tight_layout();
        plt.savefig(f"{output_dir}/cpu_utilization.png"); plt.close()
    ```

**状态**: [未开始]
```