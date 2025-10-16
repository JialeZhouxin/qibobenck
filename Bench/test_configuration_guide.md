# VQE集成测试配置指南

## 1. 测试配置结构

### 1.1 测试配置概述

本指南提供了VQE基准测试框架集成测试所需的配置结构和模拟数据定义。测试配置分为几个层次：

1. **基础测试配置**: 用于快速验证功能的简单配置
2. **标准测试配置**: 用于常规集成测试的完整配置
3. **压力测试配置**: 用于验证系统性能和稳定性的配置
4. **错误场景配置**: 用于测试错误处理机制的配置

### 1.2 配置文件组织

```
Bench/
├── test_configs/
│   ├── quick_test_config.py      # 快速测试配置
│   ├── standard_test_config.py   # 标准测试配置
│   ├── stress_test_config.py     # 压力测试配置
│   └── error_test_config.py      # 错误场景配置
├── test_data/
│   ├── mock_results.py           # 模拟测试结果
│   ├── reference_values.py       # 参考值和预期结果
│   └── test_hamiltonians.py      # 测试用哈密顿量数据
└── test_utils/
    ├── test_helpers.py           # 测试辅助函数
    ├── assertions.py             # 自定义断言
    └── fixtures.py               # 测试fixtures
```

## 2. 基础测试配置

### 2.1 快速测试配置

```python
# test_configs/quick_test_config.py

QUICK_TEST_CONFIG = {
    # 核心用户层配置
    "n_qubits_range": [4],  # 最小规模，快速执行
    "frameworks_to_test": ["Qiskit"],  # 单框架测试
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 1,  # 单次运行
    "experiment_name": "Quick_Integration_Test",
    
    # 高级研究层配置
    "problem": {
        "model_type": "TFIM_1D",
        "boundary_conditions": "periodic",
        "j_coupling": 1.0,
        "h_field": 1.0,
        "disorder_strength": 0.0,
    },
    
    "ansatz_details": {
        "n_layers": 1,  # 最小层数
        "entanglement_style": "linear",
    },
    
    "optimizer_details": {
        "options": {
            "COBYLA": {"tol": 1e-3, "rhobeg": 1.0}
        },
        "max_evaluations": 10,  # 最少评估次数
        "accuracy_threshold": 1e-3,  # 宽松精度要求
    },
    
    "backend_details": {
        "simulation_mode": "statevector",
        "n_shots": 1024,  # 最少采样次数
        "framework_backends": {
            "Qiskit": "aer_simulator",
            "PennyLane": "default.qubit",  # 使用默认后端，避免依赖问题
            "Qibo": {"backend": "numpy", "platform": "numpy"}
        }
    },
    
    "system": {
        "seed": 42,  # 固定随机种子
        "save_results": False,  # 不保存结果，节省时间
        "output_dir": "./test_results/quick/",
        "verbose": False,  # 减少输出
        "max_memory_mb": 1024,  # 低内存限制
        "max_time_seconds": 60,  # 短时间限制
    }
}

# 快速测试配置的不同变体
QUICK_TEST_PENNYLANE_CONFIG = QUICK_TEST_CONFIG.copy()
QUICK_TEST_PENNYLANE_CONFIG["frameworks_to_test"] = ["PennyLane"]

QUICK_TEST_QIBO_CONFIG = QUICK_TEST_CONFIG.copy()
QUICK_TEST_QIBO_CONFIG["frameworks_to_test"] = ["Qibo"]
```

### 2.2 单组件测试配置

```python
# test_configs/component_test_config.py

# 框架适配器测试配置
FRAMEWORK_ADAPTER_TEST_CONFIG = {
    "backend_config": {
        "framework_backends": {
            "Qiskit": "aer_simulator",
            "PennyLane": "default.qubit",
            "Qibo": {"backend": "numpy", "platform": "numpy"}
        }
    },
    "problem_config": {
        "model_type": "TFIM_1D",
        "j_coupling": 1.0,
        "h_field": 1.0
    },
    "ansatz_config": {
        "ansatz_type": "HardwareEfficient",
        "n_layers": 1,
        "entanglement_style": "linear"
    },
    "n_qubits": 4
}

# VQE执行引擎测试配置
VQE_RUNNER_TEST_CONFIG = {
    "optimizer_config": {
        "optimizer": "COBYLA",
        "options": {
            "COBYLA": {"tol": 1e-3, "rhobeg": 1.0}
        }
    },
    "convergence_config": {
        "max_evaluations": 20,
        "accuracy_threshold": 1e-3
    },
    "exact_energy": -4.0,  # 4量子比特TFIM的近似基态能量
    "param_count": 8  # 4量子比特 * 2层 * 2种旋转门
}
```

## 3. 标准测试配置

### 3.1 多框架对比配置

```python
# test_configs/standard_test_config.py

STANDARD_TEST_CONFIG = {
    # 核心用户层配置
    "n_qubits_range": [4, 6],  # 中等规模
    "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 2,  # 多次运行以验证统计一致性
    "experiment_name": "Standard_Integration_Test",
    
    # 高级研究层配置
    "problem": {
        "model_type": "TFIM_1D",
        "boundary_conditions": "periodic",
        "j_coupling": 1.0,
        "h_field": 1.0,
        "disorder_strength": 0.0,
    },
    
    "ansatz_details": {
        "n_layers": 2,
        "entanglement_style": "linear",
    },
    
    "optimizer_details": {
        "options": {
            "COBYLA": {"tol": 1e-4, "rhobeg": 1.0}
        },
        "max_evaluations": 50,
        "accuracy_threshold": 1e-4,
    },
    
    "backend_details": {
        "simulation_mode": "statevector",
        "n_shots": 4096,
        "framework_backends": {
            "Qiskit": "aer_simulator",
            "PennyLane": "lightning.qubit",
            "Qibo": {"backend": "qibojit", "platform": "numba"}
        }
    },
    
    "system": {
        "seed": 42,
        "save_results": True,
        "output_dir": "./test_results/standard/",
        "verbose": True,
        "max_memory_mb": 2048,
        "max_time_seconds": 300,
    }
}
```

### 3.2 多算法对比配置

```python
# 多算法对比配置
ALGORITHM_COMPARISON_CONFIG = STANDARD_TEST_CONFIG.copy()
ALGORITHM_COMPARISON_CONFIG.update({
    "experiment_name": "Algorithm_Comparison_Test",
    "n_qubits_range": [4],  # 固定规模，专注算法对比
    "frameworks_to_test": ["Qiskit"],  # 单框架，减少变量
})

# 为不同算法创建变体
COBYLA_CONFIG = ALGORITHM_COMPARISON_CONFIG.copy()
COBYLA_CONFIG["optimizer"] = "COBYLA"

SPSA_CONFIG = ALGORITHM_COMPARISON_CONFIG.copy()
SPSA_CONFIG["optimizer"] = "SPSA"
SPSA_CONFIG["optimizer_details"]["options"]["SPSA"] = {
    "learning_rate": 0.05,
    "perturbation": 0.05
}

LBFGS_CONFIG = ALGORITHM_COMPARISON_CONFIG.copy()
LBFGS_CONFIG["optimizer"] = "L-BFGS-B"
LBFGS_CONFIG["optimizer_details"]["options"]["L-BFGS-B"] = {
    "ftol": 1e-7,
    "gtol": 1e-5
}
```

## 4. 压力测试配置

### 4.1 大规模测试配置

```python
# test_configs/stress_test_config.py

STRESS_TEST_CONFIG = {
    # 核心用户层配置
    "n_qubits_range": [8, 10, 12],  # 大规模
    "frameworks_to_test": ["Qiskit", "PennyLane"],  # 排除可能不稳定的框架
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 3,  # 更多运行次数
    "experiment_name": "Stress_Integration_Test",
    
    # 高级研究层配置
    "problem": {
        "model_type": "TFIM_1D",
        "boundary_conditions": "periodic",
        "j_coupling": 1.0,
        "h_field": 1.0,
        "disorder_strength": 0.0,
    },
    
    "ansatz_details": {
        "n_layers": 3,  # 更深电路
        "entanglement_style": "full",  # 全连接，增加复杂度
    },
    
    "optimizer_details": {
        "options": {
            "COBYLA": {"tol": 1e-5, "rhobeg": 1.0}
        },
        "max_evaluations": 200,  # 更多评估次数
        "accuracy_threshold": 1e-5,  # 更高精度要求
    },
    
    "backend_details": {
        "simulation_mode": "statevector",
        "n_shots": 8192,
        "framework_backends": {
            "Qiskit": "aer_simulator",
            "PennyLane": "lightning.qubit",
            # Qibo在大规模时可能不稳定，暂时排除
        }
    },
    
    "system": {
        "seed": 42,
        "save_results": True,
        "output_dir": "./test_results/stress/",
        "verbose": True,
        "max_memory_mb": 8192,  # 高内存限制
        "max_time_seconds": 1800,  # 长时间限制
    }
}
```

### 4.2 资源限制测试配置

```python
# 资源限制测试配置
RESOURCE_LIMIT_CONFIG = STANDARD_TEST_CONFIG.copy()
RESOURCE_LIMIT_CONFIG.update({
    "experiment_name": "Resource_Limit_Test",
    "n_qubits_range": [6, 8],  # 中等规模
    "frameworks_to_test": ["Qiskit"],
    "system": {
        "seed": 42,
        "save_results": False,  # 不保存结果，节省资源
        "output_dir": "./test_results/resource_limit/",
        "verbose": False,
        "max_memory_mb": 512,  # 低内存限制
        "max_time_seconds": 120,  # 短时间限制
    }
})
```

## 5. 错误场景配置

### 5.1 无效参数配置

```python
# test_configs/error_test_config.py

# 无效量子比特数配置
INVALID_QUBITS_CONFIG = {
    "n_qubits_range": [0, -1, 1.5],  # 无效值
    "frameworks_to_test": ["Qiskit"],
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 1,
    "experiment_name": "Invalid_Qubits_Test"
}

# 无效框架配置
INVALID_FRAMEWORK_CONFIG = {
    "n_qubits_range": [4],
    "frameworks_to_test": ["InvalidFramework"],  # 不存在的框架
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 1,
    "experiment_name": "Invalid_Framework_Test"
}

# 无效优化器配置
INVALID_OPTIMIZER_CONFIG = {
    "n_qubits_range": [4],
    "frameworks_to_test": ["Qiskit"],
    "ansatz_type": "HardwareEfficient",
    "optimizer": "InvalidOptimizer",  # 不存在的优化器
    "n_runs": 1,
    "experiment_name": "Invalid_Optimizer_Test"
}

# 冲突参数配置
CONFLICTING_CONFIG = {
    "n_qubits_range": [4],
    "frameworks_to_test": ["Qiskit"],
    "ansatz_type": "HardwareEfficient",
    "optimizer": "COBYLA",
    "n_runs": 1,
    "experiment_name": "Conflicting_Parameters_Test",
    "optimizer_details": {
        "max_evaluations": 10,  # 很少评估次数
        "accuracy_threshold": 1e-10  # 极高精度要求，可能导致无法收敛
    }
}
```

### 5.2 资源耗尽配置

```python
# 内存耗尽配置
MEMORY_EXHAUSTION_CONFIG = STANDARD_TEST_CONFIG.copy()
MEMORY_EXHAUSTION_CONFIG.update({
    "experiment_name": "Memory_Exhaustion_Test",
    "n_qubits_range": [12, 14],  # 大规模
    "frameworks_to_test": ["Qiskit"],
    "ansatz_details": {
        "n_layers": 5,  # 深电路
        "entanglement_style": "full"
    },
    "system": {
        "seed": 42,
        "save_results": False,
        "output_dir": "./test_results/memory_exhaustion/",
        "verbose": False,
        "max_memory_mb": 256,  # 极低内存限制
        "max_time_seconds": 300,
    }
})

# 时间耗尽配置
TIME_EXHAUSTION_CONFIG = STANDARD_TEST_CONFIG.copy()
TIME_EXHAUSTION_CONFIG.update({
    "experiment_name": "Time_Exhaustion_Test",
    "n_qubits_range": [8, 10],
    "frameworks_to_test": ["Qiskit"],
    "optimizer_details": {
        "max_evaluations": 10000,  # 极多评估次数
        "accuracy_threshold": 1e-10  # 极高精度
    },
    "system": {
        "seed": 42,
        "save_results": False,
        "output_dir": "./test_results/time_exhaustion/",
        "verbose": False,
        "max_memory_mb": 2048,
        "max_time_seconds": 30,  # 极短时间限制
    }
})
```

## 6. 模拟数据定义

### 6.1 参考能量值

```python
# test_data/reference_values.py

# TFIM模型的精确基态能量（周期性边界条件）
TFIM_EXACT_ENERGIES = {
    # (n_qubits, j_coupling, h_field): energy
    (4, 1.0, 1.0): -4.0,
    (6, 1.0, 1.0): -6.0,
    (8, 1.0, 1.0): -8.0,
    (10, 1.0, 1.0): -10.0,
    (12, 1.0, 1.0): -12.0,
    
    # 不同参数组合
    (4, 0.5, 1.0): -2.618,
    (4, 1.0, 0.5): -3.366,
    (4, 0.8, 0.8): -3.200,
}

# 收敛阈值参考值
CONVERGENCE_THRESHOLDS = {
    "loose": 1e-3,
    "standard": 1e-4,
    "strict": 1e-5,
    "very_strict": 1e-6
}

# 性能基准值
PERFORMANCE_BENCHMARKS = {
    "Qiskit": {
        4: {"max_time": 5.0, "max_memory": 100},
        6: {"max_time": 15.0, "max_memory": 200},
        8: {"max_time": 60.0, "max_memory": 500},
    },
    "PennyLane": {
        4: {"max_time": 3.0, "max_memory": 80},
        6: {"max_time": 10.0, "max_memory": 150},
        8: {"max_time": 40.0, "max_memory": 400},
    },
    "Qibo": {
        4: {"max_time": 2.0, "max_memory": 60},
        6: {"max_time": 8.0, "max_memory": 120},
        8: {"max_time": 30.0, "max_memory": 300},
    }
}
```

### 6.2 模拟测试结果

```python
# test_data/mock_results.py

import numpy as np

def generate_mock_convergence_history(n_qubits, n_layers, converged=True):
    """生成模拟收敛历史"""
    n_params = n_qubits * n_layers * 2  # RY和RZ门
    max_evals = 50 if n_qubits <= 6 else 100
    
    if converged:
        # 生成收敛的历史
        energies = np.linspace(-n_qubits * 0.5, -n_qubits * 0.95, max_evals//2)
        energies = np.append(energies, [-n_qubits * 0.99] * (max_evals - len(energies)))
        # 添加一些噪声
        energies += np.random.normal(0, 0.01, len(energies))
    else:
        # 生成不收敛的历史
        energies = np.linspace(-n_qubits * 0.5, -n_qubits * 0.7, max_evals)
        energies += np.random.normal(0, 0.05, len(energies))
    
    return energies.tolist()

def generate_mock_framework_results(framework, n_qubits_range, n_runs=2):
    """生成模拟框架测试结果"""
    results = {}
    
    for n_qubits in n_qubits_range:
        # 基础性能值（根据框架和量子比特数调整）
        base_time = {
            "Qiskit": 0.5 * (1.2 ** n_qubits),
            "PennyLane": 0.3 * (1.15 ** n_qubits),
            "Qibo": 0.2 * (1.1 ** n_qubits)
        }.get(framework, 1.0)
        
        base_memory = {
            "Qiskit": 20 * (1.5 ** n_qubits),
            "PennyLane": 15 * (1.4 ** n_qubits),
            "Qibo": 10 * (1.3 ** n_qubits)
        }.get(framework, 50.0)
        
        # 生成多次运行的结果
        time_to_solutions = []
        total_times = []
        peak_memories = []
        total_evals = []
        final_errors = []
        quantum_times = []
        classic_times = []
        energy_histories = []
        converged_count = 0
        
        for run in range(n_runs):
            # 是否收敛（80%概率收敛）
            converged = np.random.random() < 0.8
            
            if converged:
                converged_count += 1
                time_to_solution = base_time * (0.8 + 0.4 * np.random.random())
                final_error = 10 ** (-4 - 2 * np.random.random())
            else:
                time_to_solution = None
                final_error = 10 ** (-2 - np.random.random())
            
            time_to_solutions.append(time_to_solution)
            total_times.append(base_time * (1.2 + 0.6 * np.random.random()))
            peak_memories.append(base_memory * (0.8 + 0.4 * np.random.random()))
            total_evals.append(np.random.randint(20, 100))
            final_errors.append(final_error)
            quantum_times.append(base_time * 0.6 * (0.8 + 0.4 * np.random.random()))
            classic_times.append(base_time * 0.2 * (0.8 + 0.4 * np.random.random()))
            
            # 生成收敛历史
            history = generate_mock_convergence_history(n_qubits, 2, converged)
            energy_histories.append(history)
        
        # 计算统计值
        results[n_qubits] = {
            "avg_time_to_solution": np.mean([t for t in time_to_solutions if t is not None]) if time_to_solutions else None,
            "std_time_to_solution": np.std([t for t in time_to_solutions if t is not None]) if time_to_solutions and len([t for t in time_to_solutions if t is not None]) > 1 else None,
            "avg_total_time": np.mean(total_times),
            "std_total_time": np.std(total_times) if len(total_times) > 1 else 0,
            "avg_peak_memory": np.mean(peak_memories),
            "std_peak_memory": np.std(peak_memories) if len(peak_memories) > 1 else 0,
            "avg_total_evals": np.mean(total_evals),
            "std_total_evals": np.std(total_evals) if len(total_evals) > 1 else 0,
            "avg_final_error": np.mean(final_errors),
            "std_final_error": np.std(final_errors) if len(final_errors) > 1 else 0,
            "avg_quantum_time": np.mean(quantum_times),
            "std_quantum_time": np.std(quantum_times) if len(quantum_times) > 1 else 0,
            "avg_classic_time": np.mean(classic_times),
            "std_classic_time": np.std(classic_times) if len(classic_times) > 1 else 0,
            "convergence_rate": converged_count / n_runs,
            "energy_histories": energy_histories,
            "errors": []  # 无错误
        }
    
    return results

# 预生成的模拟结果
MOCK_RESULTS = {
    framework: generate_mock_framework_results(framework, [4, 6, 8], 3)
    for framework in ["Qiskit", "PennyLane", "Qibo"]
}
```

## 7. 测试辅助函数

### 7.1 测试工具函数

```python
# test_utils/test_helpers.py

import os
import tempfile
import shutil
from typing import Dict, Any, List

def create_test_environment():
    """创建临时测试环境"""
    temp_dir = tempfile.mkdtemp(prefix="vqe_test_")
    return temp_dir

def cleanup_test_environment(temp_dir):
    """清理测试环境"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def validate_results_structure(results: Dict[str, Any]) -> bool:
    """验证结果结构是否正确"""
    required_keys = [
        "avg_time_to_solution", "std_time_to_solution",
        "avg_total_time", "std_total_time",
        "avg_peak_memory", "std_peak_memory",
        "avg_total_evals", "std_total_evals",
        "avg_final_error", "std_final_error",
        "avg_quantum_time", "std_quantum_time",
        "avg_classic_time", "std_classic_time",
        "convergence_rate", "energy_histories", "errors"
    ]
    
    for framework_data in results.values():
        for n_qubits_data in framework_data.values():
            for key in required_keys:
                if key not in n_qubits_data:
                    return False
    return True

def compare_with_reference(results: Dict[str, Any], reference: Dict[str, Any], tolerance: float = 0.1) -> bool:
    """将结果与参考值比较"""
    for framework, framework_data in results.items():
        if framework not in reference:
            continue
            
        for n_qubits, data in framework_data.items():
            if n_qubits not in reference[framework]:
                continue
                
            ref_data = reference[framework][n_qubits]
            
            # 比较关键指标
            if abs(data["avg_final_error"] - ref_data["avg_final_error"]) > tolerance:
                return False
                
            if abs(data["avg_total_time"] - ref_data["avg_total_time"]) > tolerance * ref_data["avg_total_time"]:
                return False
    
    return True

def check_resource_usage(results: Dict[str, Any], limits: Dict[str, Any]) -> List[str]:
    """检查资源使用是否超限"""
    violations = []
    
    for framework, framework_data in results.items():
        if framework not in limits:
            continue
            
        for n_qubits, data in framework_data.items():
            if n_qubits not in limits[framework]:
                continue
                
            limits_data = limits[framework][n_qubits]
            
            # 检查时间限制
            if data["avg_total_time"] > limits_data["max_time"]:
                violations.append(f"{framework} {n_qubits} qubits: time exceeded ({data['avg_total_time']:.2f} > {limits_data['max_time']})")
            
            # 检查内存限制
            if data["avg_peak_memory"] > limits_data["max_memory"]:
                violations.append(f"{framework} {n_qubits} qubits: memory exceeded ({data['avg_peak_memory']:.2f} > {limits_data['max_memory']})")
    
    return violations
```

### 7.2 自定义断言

```python
# test_utils/assertions.py

def assert_valid_benchmark_results(results: Dict[str, Any]):
    """断言基准测试结果有效"""
    assert isinstance(results, dict), "Results should be a dictionary"
    assert len(results) > 0, "Results should not be empty"
    
    for framework, framework_data in results.items():
        assert isinstance(framework, str), f"Framework name should be string, got {type(framework)}"
        assert isinstance(framework_data, dict), f"Framework data should be dictionary, got {type(framework_data)}"
        
        for n_qubits, data in framework_data.items():
            assert isinstance(n_qubits, int), f"Qubit count should be int, got {type(n_qubits)}"
            assert isinstance(data, dict), f"Qubit data should be dictionary, got {type(data)}"
            
            # 检查必需字段
            required_fields = [
                "avg_time_to_solution", "avg_total_time", "avg_peak_memory",
                "avg_total_evals", "avg_final_error", "convergence_rate"
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
                assert data[field] is not None or field == "avg_time_to_solution", f"Field {field} should not be None"

def assert_convergence_achieved(results: Dict[str, Any], min_rate: float = 0.5):
    """断言达到最小收敛率"""
    for framework, framework_data in results.items():
        for n_qubits, data in framework_data.items():
            convergence_rate = data["convergence_rate"]
            assert convergence_rate >= min_rate, f"{framework} {n_qubits} qubits: convergence rate {convergence_rate} < {min_rate}"

def assert_performance_within_limits(results: Dict[str, Any], limits: Dict[str, Any]):
    """断言性能在限制范围内"""
    violations = check_resource_usage(results, limits)
    assert len(violations) == 0, f"Resource limit violations: {violations}"
```

## 8. 测试执行指南

### 8.1 测试运行命令

```bash
# 运行快速测试
python -m pytest tests/test_integration.py::test_quick_integration -v

# 运行标准测试
python -m pytest tests/test_integration.py::test_standard_integration -v

# 运行压力测试
python -m pytest tests/test_integration.py::test_stress_integration -v

# 运行错误场景测试
python -m pytest tests/test_integration.py::test_error_scenarios -v

# 运行所有集成测试
python -m pytest tests/test_integration.py -v

# 生成覆盖率报告
python -m pytest tests/test_integration.py --cov=vqe_bench_new --cov-report=html
```

### 8.2 测试环境设置

```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-mock pytest-html

# 设置环境变量
export VQE_TEST_MODE=1
export VQE_TEST_DATA_DIR=./test_data/
export VQE_TEST_OUTPUT_DIR=./test_output/
```

### 8.3 持续集成配置

```yaml
# .github/workflows/integration_tests.yml
name: VQE Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock
    
    - name: Run integration tests
      run: |
        pytest tests/test_integration.py --cov=vqe_bench_new --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

这个配置指南提供了VQE集成测试所需的完整配置结构和模拟数据，为后续的测试实现提供了坚实的基础。