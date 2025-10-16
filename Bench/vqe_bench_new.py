#!/usr/bin/env python3
"""
VQE框架性能基准测试脚本 - 基于分层配置设计的新架构

该脚本实现了基于vqe_design.ipynb中设计理念的分层配置系统，
采用面向对象的架构设计，包括FrameworkWrapper抽象基类、VQERunner执行引擎和BenchmarkController控制器。

主要功能：
- 支持多个量子比特数的扩展性测试
- 详细的性能指标收集（时间、内存、收敛性等）
- 多维度可视化分析仪表盘
- 资源限制保护（内存和时间限制）

使用示例：
    from vqe_config import merge_configs
    from vqe_bench_new import BenchmarkController
    
    config = merge_configs()
    controller = BenchmarkController(config)
    results = controller.run_all_benchmarks()

作者：量子计算研究团队
版本：2.0.0
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import threading
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
# 在文件顶部进行一次性导入
try:
    import qibo
    from qibo import Circuit, gates, hamiltonians
    QIBO_AVAILABLE = True
except ImportError:
    QIBO_AVAILABLE = False
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
try:
    from qiskit.primitives import StatevectorEstimator  # 使用新的StatevectorEstimator替代
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import EfficientSU2, TwoLocal
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
# 抑制一些常见的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 性能监控模块 ---

class MemoryMonitor(threading.Thread):
    """一个在后台监控主进程峰值内存使用的线程。"""
    def __init__(self, process_id, max_memory_mb=4096):
        super().__init__()
        self.process = psutil.Process(process_id)
        self.peak_memory = 0
        self.running = False
        self.daemon = True  # 主线程退出时该线程也退出
        self.max_memory_mb = max_memory_mb
        self.memory_exceeded = False

    def run(self):
        self.running = True
        while self.running:
            try:
                mem_info = self.process.memory_info()
                current_memory_mb = mem_info.rss / (1024 * 1024)
                self.peak_memory = max(self.peak_memory, current_memory_mb)
                
                # 检查内存限制
                if current_memory_mb > self.max_memory_mb:
                    self.memory_exceeded = True
                    print(f"警告：内存使用超过限制 ({current_memory_mb:.1f}MB > {self.max_memory_mb}MB)")
                    
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)  # 采样间隔

    def stop(self):
        self.running = False
    
    def get_peak_mb(self):
        return self.peak_memory
    
    def is_memory_exceeded(self):
        return self.memory_exceeded

class StopVQE(Exception):
    """自定义异常，用于在满足收敛条件时优雅地停止优化器。"""
    pass

# --- 2. FrameworkWrapper抽象基类 (框架适配器) ---

class FrameworkWrapper(ABC):
    """
    框架适配器抽象基类，定义了所有框架必须实现的通用接口
    
    这个类充当"翻译官"的角色，将统一的指令翻译成每个框架都能理解的具体代码
    """
    
    def __init__(self, backend_config: Dict[str, Any]):
        """
        初始化框架适配器
        
        Args:
            backend_config: 后端配置字典
        """
        self.backend = self.setup_backend(backend_config)
    
    @abstractmethod
    def setup_backend(self, backend_config: Dict[str, Any]) -> Any:
        """
        设置框架特定的后端
        
        Args:
            backend_config: 后端配置字典
            
        Returns:
            配置好的后端对象
        """
        pass
    
    @abstractmethod
    def build_hamiltonian(self, problem_config: Dict[str, Any], n_qubits: int) -> Any:
        """
        根据配置构建特定于框架的哈密顿量
        
        Args:
            problem_config: 问题配置字典
            n_qubits: 量子比特数
            
        Returns:
            框架特定的哈密顿量对象
        """
        pass
    
    @abstractmethod
    def build_ansatz(self, ansatz_config: Dict[str, Any], n_qubits: int) -> Any:
        """
        根据配置构建特定于框架的参数化量子电路
        
        Args:
            ansatz_config: Ansatz配置字典
            n_qubits: 量子比特数
            
        Returns:
            框架特定的Ansatz电路对象
        """
        pass
    
    @abstractmethod
    def get_cost_function(self, hamiltonian: Any, ansatz: Any, n_qubits: int) -> Callable:
        """
        返回一个可调用的成本函数 (energy_function(params))
        
        这个函数是连接量子电路和经典优化器的桥梁
        
        Args:
            hamiltonian: 框架特定的哈密顿量对象
            ansatz: 框架特定的Ansatz电路对象
            
        Returns:
            接受参数数组并返回能量期望值的函数
        """
        pass
    
    @abstractmethod
    def get_param_count(self, ansatz: Any,n_qubits: int) -> int:
        """
        获取Ansatz电路的参数数量
        
        Args:
            ansatz: 框架特定的Ansatz电路对象
            
        Returns:
            参数数量
        """
        pass

# --- 3. 具体框架适配器实现 ---

class QiskitWrapper(FrameworkWrapper):
    """Qiskit框架的适配器实现"""
    
    def __init__(self, backend_config: Dict[str, Any]):
        if not QISKIT_AVAILABLE:
            print("警告：Qiskit或其依赖项未安装，跳过Qiskit测试。")
            self.qiskit_available = False
            return
        self.qiskit_available = True
        super().__init__(backend_config)

    def setup_backend(self, backend_config: Dict[str, Any]) -> Any:
        """
        对于Qiskit Primitives，我们不需要设置一个“后端”实例。
        Estimator本身就是高级接口，它在内部处理后端。
        我们返回配置中的名称以保持一致性，但它不会被直接使用。
        """
        if not self.qiskit_available:
            return None
        
        framework_backends = backend_config.get("framework_backends", {})
        # `aer_simulator` 是默认的，但Estimator会自动使用它
        return framework_backends.get("Qiskit", "aer_simulator")

    def build_hamiltonian(self, problem_config: Dict[str, Any], n_qubits: int) -> Any:
        """构建Qiskit的哈密顿量"""
        if not self.qiskit_available:
            raise ImportError("Qiskit不可用")
        
        from qiskit.quantum_info import SparsePauliOp
        
        # 获取问题参数
        j_coupling = problem_config.get("j_coupling", 1.0)
        h_field = problem_config.get("h_field", 1.0)
        
        # 构建Pauli项列表
        pauli_terms = []
        coeffs = []
        
        # 使用列表推导式生成相互作用项和横向场项
        # 相互作用项 -J * sum(Z_i Z_{i+1})
        pauli_terms.extend([''.join(['Z' if j in (i, (i + 1) % n_qubits) else 'I' 
                                    for j in range(n_qubits)]) 
                        for i in range(n_qubits)])
        coeffs.extend([-j_coupling] * n_qubits)
        
        # 横向场项 -h * sum(X_i)
        pauli_terms.extend([''.join(['X' if j == i else 'I' 
                                    for j in range(n_qubits)]) 
                        for i in range(n_qubits)])
        coeffs.extend([-h_field] * n_qubits)
        
        return SparsePauliOp.from_list(zip(pauli_terms, coeffs))

    def build_ansatz(self, ansatz_config: Dict[str, Any], n_qubits: int) -> Any:
        """构建Qiskit的参数化Ansatz电路"""
        if not self.qiskit_available:
            raise ImportError("Qiskit不可用")
        
        ansatz_type = ansatz_config.get("ansatz_type", "HardwareEfficient")
        n_layers = ansatz_config.get("n_layers", 2)
        entanglement_style = ansatz_config.get("entanglement_style", "linear")
        
        # HardwareEfficient可以使用TwoLocal实现，更灵活
        if ansatz_type == "HardwareEfficient" or ansatz_type == "QAOA":
            # 为HardwareEfficient定义旋转门
            rotation_blocks = ['ry', 'rz']
            # QAOA 通常使用RX和RZ层
            if ansatz_type == "QAOA":
                rotation_blocks = ['rx', 'rz'] # 实际上QAOA结构更特殊，这里用TwoLocal近似

            ansatz = TwoLocal(
                num_qubits=n_qubits,
                rotation_blocks=rotation_blocks,
                entanglement_blocks='cx',
                entanglement=entanglement_style,
                reps=n_layers,
                insert_barriers=True # 插入障碍，方便可视化
            )
        else:
            raise ValueError(f"不支持的ansatz类型: {ansatz_type}")
        
        return ansatz
    
    def get_cost_function(self, hamiltonian: Any, ansatz: Any,n_qubits: int) -> Callable:
        """
        构建并返回一个与SciPy优化器兼容的成本函数。
        """
        if not self.qiskit_available:
            raise ImportError("Qiskit不可用")
        
        # StatevectorEstimator 已被新的 Estimator 替代，它能处理状态向量模拟
        # 我们在这里实例化它，以便它能被闭包捕获
        estimator = StatevectorEstimator()
        
        def cost_function(params):
            """
            这是传递给优化器的函数。
            它接收一个numpy数组 `params`。
            """
            try:
                # 正确的使用方式：将电路、观测量和参数值分别传入
                job = estimator.run([(ansatz, hamiltonian, params)])
                result = job.result()
                
                # 正确的结果访问方式
                energy = result[0].data.evs
                
                return float(energy)
                
            except Exception as e:
                # 提供更详细的错误信息
                print(f"在Qiskit成本函数中发生错误: {e}")
                print(f"  - Ansatz参数数量: {ansatz.num_parameters}")
                print(f"  - 传入参数数量: {len(params)}")
                raise

        return cost_function
    
    def get_param_count(self, ansatz: Any,n_qubits: int) -> int:
        """获取Qiskit Ansatz的参数数量"""
        if not self.qiskit_available:
            raise ImportError("Qiskit不可用")
        
        return ansatz.num_parameters

class PennyLaneWrapper(FrameworkWrapper):
    """PennyLane框架的适配器实现"""
    
    def __init__(self, backend_config: Dict[str, Any]):
        if not PENNYLANE_AVAILABLE:
            print("警告：PennyLane或其依赖项未安装，跳过PennyLane测试。")
            self.pennylane_available = False
            return
        self.pennylane_available = True
        super().__init__(backend_config)
    
    def setup_backend(self, backend_config: Dict[str, Any]) -> str:
        """设置PennyLane后端名称"""
        if not self.pennylane_available:
            return "pennylane后端不可用"
        
        framework_backends = backend_config.get("framework_backends", {})
        backend_name = framework_backends.get("PennyLane", "lightning.qubit")
        
        # 验证后端是否可用
        try:
            qml.device(backend_name, wires=1)
        except qml.DeviceError:
            print(f"警告：PennyLane后端 '{backend_name}' 不可用。回退到 'default.qubit'。")
            backend_name = "default.qubit"

        return backend_name
    
    def build_hamiltonian(self, problem_config: Dict[str, Any], n_qubits: int) -> qml.Hamiltonian:
        """手动构建PennyLane的哈密顿量，以确保公平性"""
        if not self.pennylane_available:
            raise ImportError("PennyLane不可用")
        
        j_coupling = problem_config.get("j_coupling", 1.0)
        h_field = problem_config.get("h_field", 1.0)
        
        coeffs = []
        ops = []
        
        # 相互作用项: -J * sum(Z_i Z_{i+1})
        for i in range(n_qubits):
            coeffs.append(-j_coupling)
            ops.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
            
        # 横向场项: -h * sum(X_i)
        for i in range(n_qubits):
            coeffs.append(-h_field)
            ops.append(qml.PauliX(i))

        return qml.Hamiltonian(coeffs, ops)
    def build_ansatz(self, ansatz_config: Dict[str, Any], n_qubits: int) -> Callable:
        """
        只构建一个纯粹的、与设备无关的Ansatz函数。
        这个函数描述了量子门的序列。
        """
        if not self.pennylane_available:
            raise ImportError("PennyLane不可用")
        
        ansatz_type = ansatz_config.get("ansatz_type", "HardwareEfficient")
        n_layers = ansatz_config.get("n_layers", 2)
        entanglement_style = ansatz_config.get("entanglement_style", "linear")
        
        def ansatz_circuit(params):
            """
            这是实际的电路构建函数。
            它接收参数并应用门。
            """
            param_idx = 0
            # 确保ansatz结构与Qiskit的TwoLocal(['ry', 'rz'])对等
            for _ in range(n_layers):
                # 旋转层
                for i in range(n_qubits):
                    qml.RY(params[param_idx], wires=i); param_idx += 1
                for i in range(n_qubits):
                    qml.RZ(params[param_idx], wires=i); param_idx += 1

                # 纠缠层
                if entanglement_style == "linear":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif entanglement_style == "circular":
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])
                elif entanglement_style == "full":
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            qml.CNOT(wires=[i, j])

        return ansatz_circuit
    
    def get_cost_function(self, hamiltonian: qml.Hamiltonian, ansatz_func: Callable, n_qubits: int) -> Callable:
        """
        将ansatz函数、哈密顿量和设备组装成一个可执行的QNode成本函数。
        """
        if not self.pennylane_available:
            raise ImportError("PennyLane不可用")
        
        # 在这里创建设备实例
        dev = qml.device(self.backend, wires=n_qubits)
        
        # 使用@qml.qnode装饰器将所有部分连接起来
        @qml.qnode(dev)
        def cost_function(params):
            ansatz_func(params)
            return qml.expval(hamiltonian)
        
        return cost_function
    
    def get_param_count(self, ansatz_config: Dict[str, Any], n_qubits: int) -> int:
        """
        根据配置直接计算参数数量，而不是依赖QNode对象。
        """
        if not self.pennylane_available:
            raise ImportError("PennyLane不可用")
            
        ansatz_type = ansatz_config.get("ansatz_type", "HardwareEfficient")
        n_layers = ansatz_config.get("n_layers", 2)

        if ansatz_type == "HardwareEfficient" or ansatz_type == "QAOA":
            # 每层有 n_qubits个RY门 和 n_qubits个RZ门
            params_per_layer = 2 * n_qubits
            return n_layers * params_per_layer
        else:
            raise ValueError(f"不支持的ansatz类型: {ansatz_type}")
    
class QiboWrapper(FrameworkWrapper):
    """Qibo框架的适配器实现"""
    
    def __init__(self, backend_config: Dict[str, Any]):
        if not QIBO_AVAILABLE:
            print("警告：Qibo或其依赖项未安装，跳过Qibo测试。")
            self.qibo_available = False
            return
        self.qibo_available = True
        super().__init__(backend_config)
    
    def setup_backend(self, backend_config: Dict[str, Any]) -> Dict[str, str]:
        """设置Qibo后端（这是一个全局设置）"""
        if not self.qibo_available:
            return {"status": "qibo后端不可用"}

        
        framework_backends = backend_config.get("framework_backends", {})
        qibo_config = framework_backends.get("Qibo", {"backend": "qibojit", "platform": "numba"})
        
        backend_name = qibo_config.get("backend", "qibojit")
        platform = qibo_config.get("platform", "numba")
        
        # set_backend是全局操作
        qibo.set_backend(backend=backend_name, platform=platform)
        print(f"Qibo后端已设置为: {qibo.get_backend()}")
        
        return {"backend": backend_name, "platform": platform}
    
    def build_hamiltonian(self, problem_config: Dict[str, Any], n_qubits: int) -> hamiltonians.SymbolicHamiltonian:
        """
        使用SymbolicHamiltonian构建Qibo哈密顿量，保证公平、高效和简洁。
        """
        if not self.qibo_available:
            raise ImportError("Qibo不可用")
        
        from qibo import symbols  # 导入Qibo符号模块
        
        j_coupling = problem_config.get("j_coupling", 1.0)
        h_field = problem_config.get("h_field", 1.0)
        
        # 使用Qibo符号构建哈密顿量
        ham_expr = 0
        
        # 相互作用项: -J * sum(Z_i Z_{i+1})
        for i in range(n_qubits):
            Z_i = symbols.Z(i)
            Z_ip1 = symbols.Z((i + 1) % n_qubits)
            ham_expr += -j_coupling * Z_i * Z_ip1
        
        # 横向场项: -h * sum(X_i)
        for i in range(n_qubits):
            X_i = symbols.X(i)
            ham_expr += -h_field * X_i
        
        return hamiltonians.SymbolicHamiltonian(ham_expr)
    
    def build_ansatz(self, ansatz_config: Dict[str, Any], n_qubits: int) -> Any:
        """构建Qibo的Ansatz电路"""
        if not self.qibo_available:
            raise ImportError("Qibo不可用")
        
        from qibo import Circuit, gates
        
        # 获取Ansatz参数
        ansatz_type = ansatz_config.get("ansatz_type", "HardwareEfficient")
        n_layers = ansatz_config.get("n_layers", 2)
        entanglement_style = ansatz_config.get("entanglement_style", "linear")
        
        circuit = Circuit(n_qubits)
        
        for l in range(n_layers):
            # 参数化的单比特旋转层
            for q in range(n_qubits):
                circuit.add(gates.RY(q, theta=0.0))  # 参数占位符
            for q in range(n_qubits):
                circuit.add(gates.RZ(q, theta=0.0))  # 参数占位符
            
            # 纠缠层
            if entanglement_style == "linear":
                for q in range(0, n_qubits - 1, 2):
                    circuit.add(gates.CNOT(q, q + 1))
                for q in range(1, n_qubits - 1, 2):
                    circuit.add(gates.CNOT(q, q + 1))
            elif entanglement_style == "circular":
                for q in range(0, n_qubits - 1, 2):
                    circuit.add(gates.CNOT(q, q + 1))
                for q in range(1, n_qubits - 1, 2):
                    circuit.add(gates.CNOT(q, q + 1))
                circuit.add(gates.CNOT(n_qubits - 1, 0))  # 添加环形连接
            elif entanglement_style == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        circuit.add(gates.CNOT(i, j))
        return circuit
    
    def get_cost_function(self, hamiltonian: hamiltonians.SymbolicHamiltonian, ansatz: Circuit,n_qubits: int) -> Callable:
        """构建Qibo的成本函数"""
        if not self.qibo_available:
            raise ImportError("Qibo不可用")
        
        def cost_function(params):
            """接收numpy数组参数，执行电路，并计算期望值"""
            # set_parameters可以直接接收一个扁平的numpy数组
            ansatz.set_parameters(params)
            
            # 执行电路得到最终态
            final_state = ansatz().state()
            
            # 计算哈密顿量的期望值
            energy = hamiltonian.expectation(final_state)
            
            return float(energy)

        return cost_function
    
    def get_param_count(self, ansatz_config: Dict[str, Any], n_qubits: int) -> int:
        """根据配置确定性地计算参数数量"""
        if not self.qibo_available:
            raise ImportError("Qibo不可用")
        
        n_layers = ansatz_config.get("n_layers", 2)
        # 每层有 n_qubits个RY 和 n_qubits个RZ
        params_per_layer = 2 * n_qubits
        return n_layers * params_per_layer

# --- 4. VQERunner执行引擎 ---

class VQERunner:
    """
    VQE执行引擎，封装了单次VQE运行的完整逻辑
    
    这个类是性能监测被"注入"的地方，它不关心是哪个框架，只关心执行优化循环
    """
    
    def __init__(self, cost_function: Callable, optimizer_config: Dict[str, Any], 
                 convergence_config: Dict[str, Any], exact_energy: float):
        """
        初始化VQE执行引擎
        
        Args:
            cost_function: 成本函数
            optimizer_config: 优化器配置
            convergence_config: 收敛配置
            exact_energy: 精确基态能量
        """
        self.cost_function = cost_function
        self.optimizer = self.setup_optimizer(optimizer_config)
        self.max_evals = convergence_config["max_evaluations"]
        self.accuracy_threshold = convergence_config["accuracy_threshold"]
        self.exact_energy = exact_energy
        
        # 性能监测数据的内部状态
        self.eval_count = 0
        self.convergence_history = []
        self.quantum_step_times = []
        self.classic_step_times = []
        self.converged = False
        self.time_to_solution = None
    
    def setup_optimizer(self, optimizer_config: Dict[str, Any]) -> Any:
        """
        根据配置选择优化器
        
        Args:
            optimizer_config: 优化器配置
            
        Returns:
            配置好的优化器对象
        """
        optimizer_type = optimizer_config.get("optimizer", "COBYLA")
        options = optimizer_config.get("options", {})
        
        if optimizer_type == "COBYLA":
            from scipy.optimize import minimize
            
            def optimizer_fun(cost_function, initial_params, callback):
                return minimize(
                    fun=cost_function,
                    x0=initial_params,
                    method='COBYLA',
                    options={'maxiter': self.max_evals, 'disp': False, **options.get("COBYLA", {})},
                    callback=callback
                )
            
            return optimizer_fun
        
        elif optimizer_type == "SPSA":
            from scipy.optimize import minimize
            
            def spsa_optimizer(cost_function, initial_params, callback):
                # SPSA优化器的简化实现
                params = initial_params.copy()
                best_params = params.copy()
                best_energy = float('inf')
                
                learning_rate = options.get("SPSA", {}).get("learning_rate", 0.05)
                perturbation = options.get("SPSA", {}).get("perturbation", 0.05)
                
                for i in range(self.max_evals):
                    # 生成随机扰动方向
                    delta = np.random.choice([-1, 1], size=params.shape) * perturbation
                    
                    # 评估两个扰动点
                    params_plus = params + delta
                    params_minus = params - delta
                    
                    energy_plus = cost_function(params_plus)
                    energy_minus = cost_function(params_minus)
                    
                    # 估计梯度
                    gradient = (energy_plus - energy_minus) / (2 * delta)
                    
                    # 更新参数
                    params = params - learning_rate * gradient
                    
                    # 评估新参数
                    current_energy = cost_function(params)
                    
                    # 记录最佳参数
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_params = params.copy()
                    
                    # 调用回调函数
                    callback(params)
                    
                    # 检查收敛
                    if abs(current_energy - self.exact_energy) < self.accuracy_threshold:
                        break
                
                # 返回结果对象，模拟scipy.optimize.minimize的返回格式
                class Result:
                    def __init__(self, x, fun):
                        self.x = x
                        self.fun = fun
                        self.success = True
                
                return Result(best_params, best_energy)
            
            return spsa_optimizer
        
        elif optimizer_type == "L-BFGS-B":
            from scipy.optimize import minimize
            
            def lbfgs_optimizer(cost_function, initial_params, callback):
                return minimize(
                    fun=cost_function,
                    x0=initial_params,
                    method='L-BFGS-B',
                    options={'maxiter': self.max_evals, 'disp': False, **options.get("L-BFGS-B", {})},
                    callback=callback
                )
            
            return lbfgs_optimizer
        
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _callback(self, current_params):
        """
        性能监测的核心！
        
        这个函数会在优化器的每一步被调用。
        不同优化器库的回调函数签名可能不同，需要适配。
        """
        # 经典部分开始
        classic_start_time = time.perf_counter()
        
        # 调用成本函数，并计时"量子部分"
        quantum_start_time = time.perf_counter()
        energy = self.cost_function(current_params)
        quantum_end_time = time.perf_counter()
        
        # 记录数据
        self.eval_count += 1
        self.convergence_history.append(energy)
        quantum_time = quantum_end_time - quantum_start_time
        self.quantum_step_times.append(quantum_time)
        
        # 计算并记录"经典部分"时间
        classic_end_time = time.perf_counter()
        classic_time = (classic_end_time - classic_start_time) - quantum_time
        self.classic_step_times.append(classic_time)
        
        # 检查收敛
        if not self.converged and abs(energy - self.exact_energy) < self.accuracy_threshold:
            self.converged = True
            self.time_to_solution = time.perf_counter() - (self.start_time or 0)
            print(f"  在第{self.eval_count}次评估后收敛，能量: {energy:.6f}")
            raise StopVQE("达到收敛阈值")
    
    def run(self, initial_params: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        执行VQE运行
        
        Args:
            initial_params: 初始参数，如果为None则随机生成
            
        Returns:
            包含性能指标的字典
        """
        self.start_time = time.perf_counter()
        
        # 启动内存监控线程
        monitor = MemoryMonitor(os.getpid())
        monitor.start()
        
        try:
            # 生成初始参数
            if initial_params is None:
                initial_params = np.random.uniform(0, 2 * np.pi, self.get_param_count())
            
            # 执行优化，并将_callback"注入"进去
            result = self.optimizer(
                self.cost_function,
                initial_params,
                self._callback
            )
            
            final_energy = result.fun
            final_params = result.x
            
        except StopVQE as e:
            print(f"  优化提前停止: {e}")
            final_energy = self.convergence_history[-1] if self.convergence_history else None
            final_params = None
        except Exception as e:
            print(f"  优化过程中发生错误: {e}")
            final_energy = self.convergence_history[-1] if self.convergence_history else None
            final_params = None
        finally:
            # 停止内存监控并收集结果
            monitor.stop()
            peak_memory = monitor.get_peak_mb()
        
        total_time = time.perf_counter() - self.start_time
        if self.converged and self.time_to_solution is None:
            self.time_to_solution = total_time
        
        # 整理并返回所有性能指标
        return {
            "final_energy": final_energy,
            "final_params": final_params,
            "total_time": total_time,
            "time_to_solution": self.time_to_solution,
            "peak_memory": peak_memory,
            "convergence_history": self.convergence_history.copy(),
            "eval_count": self.eval_count,
            "converged": self.converged,
            "avg_quantum_time": np.mean(self.quantum_step_times) if self.quantum_step_times else 0,
            "avg_classic_time": np.mean(self.classic_step_times) if self.classic_step_times else 0,
            "memory_exceeded": monitor.is_memory_exceeded(),
            "final_error": abs((final_energy - self.exact_energy) / self.exact_energy) if final_energy is not None else None
        }
    
    def get_param_count(self) -> int:
        """
        获取参数数量
        
        Returns:
            参数数量
        """
        # 这里应该从成本函数中推断参数数量
        # 由于不同框架的实现方式不同，这里返回一个默认值
        # 实际使用时，应该从具体的Ansatz中获取
        return 10  # 默认值，实际使用时会被覆盖

# --- 5. BenchmarkController控制器 ---

class BenchmarkController:
    """
    VQE基准测试控制器，协调整个测试流程
    
    这个类使用FrameworkWrapper和VQERunner组件来执行用户定义的整个实验
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基准测试控制器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.results = {fw: {} for fw in config["frameworks_to_test"]}
        self.start_time = None
        
        # 创建框架适配器
        self.wrappers = {}
        self._create_wrappers()
    
    def _create_wrappers(self):
        """创建框架适配器"""
        backend_config = self.config.get("backend_details", {})
        
        for framework_name in self.config["frameworks_to_test"]:
            if framework_name == "Qiskit":
                self.wrappers[framework_name] = QiskitWrapper(backend_config)
            elif framework_name == "PennyLane":
                self.wrappers[framework_name] = PennyLaneWrapper(backend_config)
            elif framework_name == "Qibo":
                self.wrappers[framework_name] = QiboWrapper(backend_config)
            else:
                raise ValueError(f"不支持的框架: {framework_name}")
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        运行所有基准测试
        
        Returns:
            包含所有结果的字典
        """
        self.start_time = time.perf_counter()
        print("开始VQE框架性能基准测试")
        print(f"配置: {self.config}")
        print("=" * 60)
        
        # 遍历所有量子比特数和框架
        for n_qubits in self.config["n_qubits_range"]:
            print(f"\n===== 测试 {n_qubits} 量子比特 =====")
            
            for framework_name in self.config["frameworks_to_test"]:
                print(f"\n--- 测试框架: {framework_name} ---")
                
                # 运行框架测试
                framework_results = self._run_framework_tests(framework_name, n_qubits)
                
                # 计算统计数据
                if len(framework_results["time_to_solution"]) > 0:
                    avg_time_to_solution = np.mean(framework_results["time_to_solution"])
                    std_time_to_solution = np.std(framework_results["time_to_solution"])
                else:
                    avg_time_to_solution = std_time_to_solution = None
                
                if len(framework_results["final_error"]) > 0:
                    avg_final_error = np.mean(framework_results["final_error"])
                    std_final_error = np.std(framework_results["final_error"])
                else:
                    avg_final_error = std_final_error = None
                
                # 保存结果
                self.results[framework_name][n_qubits] = {
                    "avg_time_to_solution": avg_time_to_solution,
                    "std_time_to_solution": std_time_to_solution,
                    "avg_total_time": np.mean(framework_results["total_time"]) if len(framework_results["total_time"]) > 0 else 0,
                    "std_total_time": np.std(framework_results["total_time"]) if len(framework_results["total_time"]) > 1 else 0,
                    "avg_peak_memory": np.mean(framework_results["peak_memory"]) if len(framework_results["peak_memory"]) > 0 else 0,
                    "std_peak_memory": np.std(framework_results["peak_memory"]) if len(framework_results["peak_memory"]) > 1 else 0,
                    "avg_total_evals": np.mean(framework_results["total_evals"]) if len(framework_results["total_evals"]) > 0 else 0,
                    "std_total_evals": np.std(framework_results["total_evals"]) if len(framework_results["total_evals"]) > 1 else 0,
                    "avg_final_error": avg_final_error,
                    "std_final_error": std_final_error,
                    "avg_quantum_time": np.mean(framework_results["avg_quantum_time"]) if len(framework_results["avg_quantum_time"]) > 0 else 0,
                    "std_quantum_time": np.std(framework_results["avg_quantum_time"]) if len(framework_results["avg_quantum_time"]) > 1 else 0,
                    "avg_classic_time": np.mean(framework_results["avg_classic_time"]) if len(framework_results["avg_classic_time"]) > 0 else 0,
                    "std_classic_time": np.std(framework_results["avg_classic_time"]) if len(framework_results["avg_classic_time"]) > 1 else 0,
                    "convergence_rate": framework_results["converged_count"] / self.config["n_runs"],
                    "energy_histories": framework_results["energy_histories"],
                    "errors": framework_results["errors"]
                }
                
                # 打印摘要
                print(f"  收敛率: {self.results[framework_name][n_qubits]['convergence_rate']:.1%}")
                if avg_time_to_solution is not None:
                    print(f"  平均求解时间: {avg_time_to_solution:.3f} ± {std_time_to_solution:.3f} 秒")
                if avg_final_error is not None:
                    print(f"  平均最终误差: {avg_final_error:.2e} ± {std_final_error:.2e}")
                print(f"  平均内存使用: {self.results[framework_name][n_qubits]['avg_peak_memory']:.1f} MB")
                
                if framework_results["errors"]:
                    print(f"  错误数量: {len(framework_results['errors'])}")
        
        total_time = time.perf_counter() - self.start_time
        print(f"\n基准测试完成，总耗时: {total_time:.2f} 秒")
        
        return self.results
    
    def _run_framework_tests(self, framework_name: str, n_qubits: int) -> Dict[str, List[Any]]:
        """
        运行指定框架和量子比特数的所有测试
        
        Args:
            framework_name: 框架名称
            n_qubits: 量子比特数
            
        Returns:
            框架测试结果
        """
        framework_results = {
            "time_to_solution": [],
            "total_time": [],
            "peak_memory": [],
            "total_evals": [],
            "final_error": [],
            "avg_quantum_time": [],
            "avg_classic_time": [],
            "converged_count": 0,
            "energy_histories": [],
            "errors": []
        }
        
        # 获取框架适配器
        wrapper = self.wrappers[framework_name]
        
        # 构建问题
        try:
            # 构建哈密顿量
            problem_config = self.config.get("problem", {})
            framework_hamiltonian = wrapper.build_hamiltonian(problem_config, n_qubits)
            
            # 获取精确基态能量
            # 使用PennyLane的哈密顿量矩阵计算精确能量
            j_coupling = problem_config.get("j_coupling", 1.0)
            h_field = problem_config.get("h_field", 1.0)
            
            # 构建哈密顿量矩阵
            try:
                import pennylane as qml
                import numpy as np
                
                # 使用PennyLane构建哈密顿量
                hamiltonian = qml.spin.transverse_ising(
                    lattice="chain",
                    n_cells=[n_qubits],
                    coupling=j_coupling,
                    h=h_field
                )
                
                # 转换为矩阵
                hamiltonian_matrix = qml.matrix(hamiltonian)
                
                # 计算精确基态能量
                eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
                exact_energy = eigenvalues[0]  # 最小特征值
                
            except Exception as e:
                print(f"  计算精确能量失败: {e}")
                # 使用近似值
                exact_energy = -n_qubits * (j_coupling + h_field)
            
            # 构建Ansatz
            ansatz_config = self.config.get("ansatz_details", {})
            ansatz_config["ansatz_type"] = self.config.get("ansatz_type", "HardwareEfficient")
            ansatz = wrapper.build_ansatz(ansatz_config, n_qubits)
            
            # 获取成本函数
            cost_function = wrapper.get_cost_function(framework_hamiltonian, ansatz,n_qubits)
            
            # 获取参数数量
            if framework_name == "PennyLane":
                param_count = wrapper.get_param_count(ansatz_config, n_qubits)
            elif framework_name == "Qibo":
                param_count = wrapper.get_param_count(ansatz_config, n_qubits)
            else:
                param_count = wrapper.get_param_count(ansatz, n_qubits)
            
        except Exception as e:
            print(f"  构建问题时出错: {e}")
            return framework_results
        
        # 运行多次测试
        for run_id in range(self.config["n_runs"]):
            print(f"  运行 #{run_id+1}: {framework_name} with {n_qubits} qubits")
            
            try:
                # 创建VQE执行引擎
                optimizer_config = self.config.get("optimizer_details", {})
                optimizer_config["optimizer"] = self.config.get("optimizer", "COBYLA")
                convergence_config = self.config.get("optimizer_details", {})
                
                vqe_runner = VQERunner(
                    cost_function=cost_function,
                    optimizer_config=optimizer_config,
                    convergence_config=convergence_config,
                    exact_energy=exact_energy
                )
                
                # 设置参数数量
                vqe_runner.get_param_count = lambda: param_count
                
                # 运行VQE
                result = vqe_runner.run()
                
                # 添加运行ID
                result["run_id"] = run_id
                
                # 收集结果
                if result["time_to_solution"] is not None:
                    framework_results["time_to_solution"].append(result["time_to_solution"])
                
                framework_results["total_time"].append(result["total_time"])
                framework_results["peak_memory"].append(result["peak_memory"])
                framework_results["total_evals"].append(result["eval_count"])
                
                if result["final_error"] is not None:
                    framework_results["final_error"].append(result["final_error"])
                
                framework_results["avg_quantum_time"].append(result["avg_quantum_time"])
                framework_results["avg_classic_time"].append(result["avg_classic_time"])
                
                if result["converged"]:
                    framework_results["converged_count"] += 1
                
                if result["convergence_history"]:
                    framework_results["energy_histories"].append(result["convergence_history"])
                
                # 检查资源限制
                if result["memory_exceeded"]:
                    print(f"    警告：内存使用超过限制")
                
                if result["total_time"] > self.config.get("system", {}).get("max_time_seconds", 1800):
                    print(f"    警告：运行时间超过限制")
                
            except Exception as e:
                print(f"    错误：{e}")
                framework_results["errors"].append(str(e))
        
        return framework_results

# --- 6. 可视化仪表盘 ---

class VQEBenchmarkVisualizer:
    """VQE基准测试结果可视化器"""
    
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any]):
        self.results = results
        self.config = config
        self.frameworks = config["frameworks_to_test"]
        self.n_qubits_range = config["n_qubits_range"]
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_dashboard(self, output_dir: str = None) -> None:
        """生成并显示包含六个核心图表的仪表盘"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 24))
        fig.suptitle("VQE框架性能基准测试仪表盘", fontsize=20)
        
        # 图 1: 总求解时间 vs. 量子比特数
        self._plot_time_to_solution(axes[0, 0])
        
        # 图 2: 峰值内存使用 vs. 量子比特数
        self._plot_peak_memory(axes[0, 1])
        
        # 图 3: 收敛轨迹 (以最大比特数为例)
        self._plot_convergence_trajectories(axes[1, 0])
        
        # 图 4: 总求值次数 vs. 量子比特数
        self._plot_total_evaluations(axes[1, 1])
        
        # 图 5: 最终求解精度 vs. 量子比特数
        self._plot_final_accuracy(axes[2, 0])
        
        # 图 6: 单步耗时分解 vs. 量子比特数
        self._plot_time_breakdown(axes[2, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图片
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vqe_benchmark_dashboard_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"仪表盘已保存到: {filepath}")
        
        #plt.show()
    
    def _plot_time_to_solution(self, ax):
        """绘制总求解时间 vs. 量子比特数"""
        for fw in self.frameworks:
            times = []
            stds = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    avg_time = self.results[fw][n_qubits]["avg_time_to_solution"]
                    std_time = self.results[fw][n_qubits]["std_time_to_solution"]
                    if avg_time is not None:
                        times.append(avg_time)
                        stds.append(std_time)
                    else:
                        times.append(None)
                        stds.append(None)
                else:
                    times.append(None)
                    stds.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, t in enumerate(times) if t is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_times = [times[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            
            if valid_times:
                ax.errorbar(valid_qubits, valid_times, yerr=valid_stds, 
                           marker='o', linestyle='-', label=fw, capsize=5)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("求解时间 (秒, 对数尺度)")
        ax.set_title("I. 核心性能: 求解时间")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--")
    
    def _plot_peak_memory(self, ax):
        """绘制峰值内存使用 vs. 量子比特数"""
        for fw in self.frameworks:
            mems = []
            stds = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    mems.append(self.results[fw][n_qubits]["avg_peak_memory"])
                    stds.append(self.results[fw][n_qubits]["std_peak_memory"])
                else:
                    mems.append(None)
                    stds.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, m in enumerate(mems) if m is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_mems = [mems[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            
            if valid_mems:
                ax.errorbar(valid_qubits, valid_mems, yerr=valid_stds,
                           marker='o', linestyle='-', label=fw, capsize=5)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("峰值内存使用 (MB, 对数尺度)")
        ax.set_title("I. 核心性能: 内存扩展性")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--")
    
    def _plot_convergence_trajectories(self, ax):
        """绘制收敛轨迹 (以最大比特数为例)"""
        max_qubits = max(self.n_qubits_range)
        
        # 获取精确能量
        exact_energy = None
        for fw in self.frameworks:
            if fw in self.results and max_qubits in self.results[fw]:
                # 从第一个历史记录中获取精确能量
                histories = self.results[fw][max_qubits]["energy_histories"]
                if histories and len(histories) > 0:
                    # 创建临时VQE执行器来获取精确能量
                    temp_runner = self._create_temp_runner(fw, max_qubits)
                    if temp_runner:
                        exact_energy = temp_runner.exact_energy
                        break
        
        if exact_energy is None:
            # 如果无法获取精确能量，使用估计值
            exact_energy = -1.27 * max_qubits
        
        for fw in self.frameworks:
            if fw in self.results and max_qubits in self.results[fw]:
                histories = self.results[fw][max_qubits]["energy_histories"]
                if histories:
                    # 计算平均历史轨迹
                    max_len = max(len(h) for h in histories)
                    padded_histories = []
                    for h in histories:
                        if len(h) < max_len:
                            # 用最后一个值填充
                            padded = h + [h[-1]] * (max_len - len(h))
                        else:
                            padded = h
                        padded_histories.append(padded)
                    
                    avg_history = np.mean(padded_histories, axis=0)
                    std_history = np.std(padded_histories, axis=0)
                    
                    evals = range(len(avg_history))
                    ax.plot(evals, avg_history, label=f"{fw} (平均 {len(avg_history)} 次评估)")
                    ax.fill_between(evals, 
                                   avg_history - std_history,
                                   avg_history + std_history,
                                   alpha=0.2)
        
        ax.axhline(exact_energy, color='r', linestyle='--', label='精确能量')
        ax.axhline(exact_energy + self.config.get("optimizer_details", {}).get("accuracy_threshold", 1e-4), 
                  color='g', linestyle=':', label='收敛阈值')
        ax.set_xlabel("成本函数评估次数")
        ax.set_ylabel("能量")
        ax.set_title(f"II. 优化动力学: 收敛轨迹 (N={max_qubits})")
        ax.legend()
        ax.grid(True, ls="--")
    
    def _plot_total_evaluations(self, ax):
        """绘制总求值次数 vs. 量子比特数"""
        for fw in self.frameworks:
            evals = []
            stds = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    evals.append(self.results[fw][n_qubits]["avg_total_evals"])
                    stds.append(self.results[fw][n_qubits]["std_total_evals"])
                else:
                    evals.append(None)
                    stds.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, e in enumerate(evals) if e is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_evals = [evals[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            
            if valid_evals:
                ax.errorbar(valid_qubits, valid_evals, yerr=valid_stds,
                           marker='o', linestyle='-', label=fw, capsize=5)
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("总函数评估次数")
        ax.set_title("II. 优化动力学: 评估次数")
        ax.legend()
        ax.grid(True, ls="--")
    
    def _plot_final_accuracy(self, ax):
        """绘制最终求解精度 vs. 量子比特数"""
        for fw in self.frameworks:
            errors = []
            stds = []
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    avg_error = self.results[fw][n_qubits]["avg_final_error"]
                    std_error = self.results[fw][n_qubits]["std_final_error"]
                    if avg_error is not None:
                        errors.append(avg_error)
                        stds.append(std_error)
                    else:
                        errors.append(None)
                        stds.append(None)
                else:
                    errors.append(None)
                    stds.append(None)
            
            # 过滤掉None值
            valid_indices = [i for i, e in enumerate(errors) if e is not None]
            valid_qubits = [self.n_qubits_range[i] for i in valid_indices]
            valid_errors = [errors[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            
            if valid_errors:
                ax.errorbar(valid_qubits, valid_errors, yerr=valid_stds,
                           marker='o', linestyle='-', label=fw, capsize=5)
        
        ax.axhline(self.config.get("optimizer_details", {}).get("accuracy_threshold", 1e-4), 
                  color='r', linestyle='--', label='目标阈值')
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("最终相对误差 (对数尺度)")
        ax.set_title("III. 诊断: 最终精度验证")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--")
    
    def _plot_time_breakdown(self, ax):
        """绘制单步耗时分解 vs. 量子比特数"""
        width = 0.2
        x = np.arange(len(self.n_qubits_range))
        
        for i, fw in enumerate(self.frameworks):
            q_times = []
            c_times = []
            
            for n_qubits in self.n_qubits_range:
                if fw in self.results and n_qubits in self.results[fw]:
                    q_times.append(self.results[fw][n_qubits]["avg_quantum_time"] * 1000)  # 转换为毫秒
                    c_times.append(self.results[fw][n_qubits]["avg_classic_time"] * 1000)
                else:
                    q_times.append(0)
                    c_times.append(0)
            
            # 绘制堆叠条形图
            ax.bar(x + i*width, q_times, width, label=f'{fw} 量子部分', hatch='//')
            ax.bar(x + i*width, c_times, width, bottom=q_times, 
                  label=f'{fw} 经典部分', hatch='..')
        
        ax.set_xlabel("量子比特数")
        ax.set_ylabel("平均单步时间 (毫秒, 对数尺度)")
        ax.set_title("III. 诊断: 时间分解")
        ax.set_xticks(x + width, self.n_qubits_range)
        ax.set_yscale('log')
        ax.legend(fontsize='small')
        ax.grid(True, which="both", ls="--")
    
    def _create_temp_runner(self, framework: str, n_qubits: int):
        """创建临时VQE运行器以获取精确能量"""
        try:
            # 获取问题参数
            problem_config = self.config.get("problem", {})
            j_coupling = problem_config.get("j_coupling", 1.0)
            h_field = problem_config.get("h_field", 1.0)
            
            # 构建哈密顿量矩阵
            try:
                import pennylane as qml
                import numpy as np
                
                # 使用PennyLane构建哈密顿量
                hamiltonian = qml.spin.transverse_ising(
                    lattice="chain",
                    n_cells=[n_qubits],
                    coupling=j_coupling,
                    h=h_field
                )
                
                # 转换为矩阵
                hamiltonian_matrix = qml.matrix(hamiltonian)
                
                # 计算精确基态能量
                eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
                exact_energy = eigenvalues[0]  # 最小特征值
                
            except Exception as e:
                print(f"  计算精确能量失败: {e}")
                # 使用近似值
                exact_energy = -n_qubits * (j_coupling + h_field)
            
            # 创建一个简单的对象来存储精确能量
            class TempRunner:
                def __init__(self, exact_energy):
                    self.exact_energy = exact_energy
            
            return TempRunner(exact_energy)
        except:
            return None

# --- 7. 主函数 ---

def main():
    """主函数"""
    from vqe_config import merge_configs
    
    # 获取配置
    config = merge_configs()
    
    # 创建并运行基准测试
    controller = BenchmarkController(config)
    results = controller.run_all_benchmarks()
    
    # 生成可视化
    print("\n生成可视化仪表盘...")
    visualizer = VQEBenchmarkVisualizer(results, config)
    output_dir = config.get("system", {}).get("output_dir", "./results/")
    visualizer.plot_dashboard(output_dir)
    
    # 打印最终摘要
    print("\n" + "=" * 60)
    print("基准测试摘要")
    print("=" * 60)
    
    for framework in config["frameworks_to_test"]:
        print(f"\n{framework} 框架:")
        for n_qubits in config["n_qubits_range"]:
            if framework in results and n_qubits in results[framework]:
                data = results[framework][n_qubits]
                print(f"  {n_qubits} 量子比特:")
                print(f"    收敛率: {data['convergence_rate']:.1%}")
                if data['avg_time_to_solution'] is not None:
                    print(f"    求解时间: {data['avg_time_to_solution']:.3f} ± {data['std_time_to_solution']:.3f} 秒")
                print(f"    内存使用: {data['avg_peak_memory']:.1f} ± {data['std_peak_memory']:.1f} MB")
                if data['avg_final_error'] is not None:
                    print(f"    最终误差: {data['avg_final_error']:.2e}")
                print(f"    总评估次数: {data['avg_total_evals']:.1f} ± {data['std_total_evals']:.1f}")
    
    print(f"\n测试完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()