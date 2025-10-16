# VQE基准测试分层配置系统

## 概述

本配置系统实现了分层设计理念，将VQE基准测试的配置分为两个层次：

1. **核心用户层 (CONFIG)**: 最常用且易于理解的参数，让新用户能在30秒内看懂并运行基准测试
2. **高级研究层 (ADVANCED_CONFIG)**: 专家级设置，用于深入、特定的基准测试

这种设计既保留了工具的全部潜力，又为普通用户提供了一条平缓的学习曲线，完美地平衡了灵活性与复杂性。

## 文件结构

```
Bench/
├── vqe_config.py              # 分层配置系统核心文件
├── test_config_compatibility.py  # 兼容性测试脚本
├── example_usage.py           # 使用示例脚本
├── README_vqe_config.md       # 本文档
└── vqe_bench.py              # 原始基准测试脚本
```

## 快速开始

### 1. 基本使用

```python
from vqe_config import merge_configs, validate_config
from vqe_bench import VQEBenchmarkRunner

# 获取默认配置
config = merge_configs()

# 验证配置
is_valid, errors = validate_config(config)
if not is_valid:
    print(f"配置错误: {errors}")
    return

# 转换为兼容格式
from vqe_config import get_legacy_config
legacy_config = get_legacy_config()

# 运行基准测试
runner = VQEBenchmarkRunner(legacy_config)
results = runner.run_all_benchmarks()
```

### 2. 快速开始配置

```python
from vqe_config import get_quick_start_config

# 获取快速开始配置（适合新用户）
config = get_quick_start_config()
print(f"量子比特数: {config['n_qubits_range']}")
print(f"测试框架: {config['frameworks_to_test']}")
print(f"运行次数: {config['n_runs']}")
```

### 3. 自定义核心配置

```python
from vqe_config import merge_configs

# 自定义核心参数
custom_core = {
    "n_qubits_range": [4, 6, 8],
    "frameworks_to_test": ["Qiskit", "Qibo"],
    "ansatz_type": "QAOA",
    "optimizer": "SPSA",
    "n_runs": 5
}

# 合并配置（使用默认高级配置）
config = merge_configs(core_config=custom_core)
```

### 4. 自定义高级配置

```python
from vqe_config import merge_configs

# 自定义高级参数
custom_advanced = {
    "problem": {
        "boundary_conditions": "open",
        "j_coupling": 0.8,
        "h_field": 1.2
    },
    "ansatz_details": {
        "n_layers": 3,
        "entanglement_style": "circular"
    },
    "optimizer_details": {
        "max_evaluations": 1000,
        "accuracy_threshold": 1e-5
    }
}

# 合并配置（使用默认核心配置）
config = merge_configs(advanced_config=custom_advanced)
```

## 配置参数详解

### 核心用户层参数

| 参数 | 类型 | 描述 | 可选值 |
|------|------|------|--------|
| `n_qubits_range` | List[int] | 量子比特数范围 | 如 [4, 6, 8] |
| `frameworks_to_test` | List[str] | 要测试的框架列表 | ["Qiskit", "PennyLane", "Qibo"] |
| `ansatz_type` | str | 算法思路 | "HardwareEfficient", "QAOA" |
| `optimizer` | str | 经典优化器 | "COBYLA", "SPSA", "L-BFGS-B" |
| `n_runs` | int | 运行次数 | 正整数 |
| `experiment_name` | str | 实验名称（可选） | 任意字符串 |

### 高级研究层参数

#### 1. 物理问题细节 (problem)

| 参数 | 类型 | 描述 | 可选值 |
|------|------|------|--------|
| `model_type` | str | 物理模型类型 | "TFIM_1D" |
| `boundary_conditions` | str | 边界条件 | "periodic", "open" |
| `j_coupling` | float | 相互作用强度 | 非负数 |
| `h_field` | float | 横向场强度 | 非负数 |
| `disorder_strength` | float | 无序强度 | 非负数 |

#### 2. Ansatz电路细节 (ansatz_details)

| 参数 | 类型 | 描述 | 可选值 |
|------|------|------|--------|
| `n_layers` | int | Ansatz层数 | 正整数 |
| `entanglement_style` | str | 纠缠样式 | "linear", "circular", "full" |

#### 3. 优化器与收敛细节 (optimizer_details)

| 参数 | 类型 | 描述 | 可选值 |
|------|------|------|--------|
| `options` | Dict | 优化器特定选项 | 详见下表 |
| `max_evaluations` | int | 最大评估次数 | 正整数 |
| `accuracy_threshold` | float | 收敛精度阈值 | 正数 |

##### 优化器选项

| 优化器 | 选项 | 类型 | 描述 |
|--------|------|------|------|
| COBYLA | tol | float | 收敛容差 |
|        | rhobeg | float | 初始步长 |
| SPSA | learning_rate | float | 学习率 |
|        | perturbation | float | 扰动参数 |
| L-BFGS-B | ftol | float | 函数收敛容差 |
|          | gtol | float | 梯度收敛容差 |

#### 4. 模拟器后端细节 (backend_details)

| 参数 | 类型 | 描述 | 可选值 |
|------|------|------|--------|
| `simulation_mode` | str | 模拟方式 | "statevector", "shot_based" |
| `n_shots` | int | 采样次数 | 正整数 |
| `framework_backends` | Dict | 框架后端配置 | 详见下表 |

##### 框架后端配置

| 框架 | 后端配置 |
|------|----------|
| Qiskit | "aer_simulator" |
| PennyLane | "lightning.qubit" |
| Qibo | {"backend": "qibojit", "platform": "numba"} |

#### 5. 系统与I/O控制 (system)

| 参数 | 类型 | 描述 | 可选值 |
|------|------|------|--------|
| `seed` | int | 随机种子 | 整数 |
| `save_results` | bool | 是否保存结果 | True, False |
| `output_dir` | str | 输出目录 | 路径字符串 |
| `verbose` | bool | 详细输出 | True, False |
| `max_memory_mb` | int | 最大内存(MB) | 正数 |
| `max_time_seconds` | int | 最大时间(秒) | 正数 |

## API参考

### merge_configs()

```python
def merge_configs(core_config: Optional[Dict[str, Any]] = None, 
                 advanced_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    将高级配置合并到核心配置中
    
    Args:
        core_config: 核心用户层配置，如果为None则使用默认的CONFIG
        advanced_config: 高级研究层配置，如果为None则使用默认的ADVANCED_CONFIG
        
    Returns:
        合并后的完整配置字典
    """
```

### validate_config()

```python
def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置的有效性
    
    Args:
        config: 要验证的配置字典
        
    Returns:
        (is_valid, errors): 元组，其中is_valid表示配置是否有效，errors是错误消息列表
    """
```

### get_quick_start_config()

```python
def get_quick_start_config() -> Dict[str, Any]:
    """
    获取快速开始配置，适合新用户快速上手
    
    Returns:
        一个简化的配置字典，包含最常用的参数
    """
```

### get_performance_config()

```python
def get_performance_config() -> Dict[str, Any]:
    """
    获取高性能配置，适合详细的性能评估
    
    Returns:
        一个高性能配置字典，包含更多测试点和运行次数
    """
```

### get_legacy_config()

```python
def get_legacy_config() -> Dict[str, Any]:
    """
    获取与现有vqe_bench.py兼容的配置格式
    
    Returns:
        一个与vqe_bench.py中DEFAULT_CONFIG格式兼容的配置字典
    """
```

## 与现有代码集成

### 方法1: 直接替换DEFAULT_CONFIG

```python
# 在vqe_bench.py中，将原来的DEFAULT_CONFIG替换为:
from vqe_config import get_legacy_config
DEFAULT_CONFIG = get_legacy_config()
```

### 方法2: 使用新配置系统但保持兼容性

```python
# 在主函数中，使用新配置系统:
from vqe_config import merge_configs, get_legacy_config

# 获取新配置
new_config = merge_configs()

# 转换为兼容格式
config = get_legacy_config()
config.update({
    "n_qubits_range": new_config["n_qubits_range"],
    "frameworks_to_test": new_config["frameworks_to_test"],
    "n_runs": new_config["n_runs"]
})

# 创建并运行基准测试
runner = VQEBenchmarkRunner(config)
results = runner.run_all_benchmarks()
```

### 方法3: 完全使用新配置系统

```python
# 创建一个适配器类，将新配置格式转换为VQEBenchmarkRunner需要的格式
class ConfigAdapter:
    @staticmethod
    def adapt(new_config):
        # 将新配置格式转换为VQEBenchmarkRunner需要的格式
        return {
            "n_qubits_range": new_config["n_qubits_range"],
            "j_coupling": new_config["problem"]["j_coupling"],
            "h_field": new_config["problem"]["h_field"],
            "n_layers": new_config["ansatz_details"]["n_layers"],
            "optimizer": new_config["optimizer"],
            "max_evaluations": new_config["optimizer_details"]["max_evaluations"],
            "accuracy_threshold": new_config["optimizer_details"]["accuracy_threshold"],
            "n_runs": new_config["n_runs"],
            "frameworks_to_test": new_config["frameworks_to_test"],
            "seed": new_config["system"]["seed"],
            "max_memory_mb": new_config["system"]["max_memory_mb"],
            "max_time_seconds": new_config["system"]["max_time_seconds"],
        }

# 使用适配器
from vqe_config import merge_configs
new_config = merge_configs()
adapted_config = ConfigAdapter.adapt(new_config)
runner = VQEBenchmarkRunner(adapted_config)
```

## 测试和验证

### 运行兼容性测试

```bash
cd Bench
python test_config_compatibility.py
```

### 运行使用示例

```bash
cd Bench
python example_usage.py
```

### 测试配置系统

```bash
cd Bench
python vqe_config.py
```

## 设计理念

### 分层配置的优势

1. **易于上手 (Low Barrier to Entry)**: 新用户看到CONFIG时不会被海量选项淹没。他们可以专注于最重要的科学问题，而把实现细节交给工具的"明智默认值"。

2. **高灵活性 (High Ceiling)**: 当需要时，研究人员拥有完全的控制权。他们可以深入ADVANCED_CONFIG，精确地构建他们想要的实验环境。

3. **代码更清晰 (Cleaner Code)**: 底层代码的逻辑也变得更清晰。主循环处理核心参数的迭代，而构建哈密顿量、ansatz和优化器的函数则从ADVANCED_CONFIG中读取详细配置。

4. **可读性与可复现性 (Readability & Reproducibility)**: 当别人看到你的配置文件时，他们能立刻明白你是在进行一个"标准测试"还是一个"高度定制化的研究"。

### 参数选择原则

**核心参数的选择标准**:
- 对实验结果有决定性影响的参数
- 用户最可能需要调整的参数
- 容易理解和解释的参数

**高级参数的选择标准**:
- 提供默认值就能正常工作的参数
- 只有特定研究需求才需要调整的参数
- 需要专业知识才能正确设置的参数

## 常见问题

### Q1: 如何选择合适的优化器？

**A1**: 
- **COBYLA**: 适用于无梯度或梯度计算困难的问题，适合参数空间较大的情况
- **SPSA**: 适用于噪声环境或大规模问题，对梯度估计不敏感
- **L-BFGS-B**: 适用于光滑问题，需要精确梯度信息，收敛速度快

### Q2: 如何设置合适的量子比特数范围？

**A2**: 
- 从小范围开始，如[4, 6, 8]，确保系统能正常运行
- 根据计算资源和时间限制逐步扩展
- 注意内存需求随量子比特数指数增长

### Q3: 如何选择合适的Ansatz类型？

**A3**: 
- **HardwareEfficient**: 适用于一般量子问题，实现简单
- **QAOA**: 适用于组合优化问题，需要根据具体问题设计

### Q4: 如何调整收敛条件？

**A4**: 
- **accuracy_threshold**: 控制收敛精度，数值越小精度要求越高
- **max_evaluations**: 控制最大计算资源，数值越大可能找到更好的解但也更耗时

## 更新日志

### v1.0.0 (2025-10-16)
- 初始版本发布
- 实现分层配置系统
- 提供完整的API和文档
- 确保与现有vqe_bench.py兼容

## 贡献指南

欢迎提交问题报告和改进建议！在提交代码前，请确保：

1. 所有测试通过
2. 代码符合项目风格
3. 添加适当的文档和注释
4. 更新相关文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。