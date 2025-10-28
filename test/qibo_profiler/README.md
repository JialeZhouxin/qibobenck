# Qibo 量子电路性能分析器 (改进版)

`qibo_profiler_improved.py` 是一个企业级的量子电路性能分析工具，专为测量和评估量子电路的执行效率、资源使用情况以及计算保真度而设计。该工具采用模块化架构，提供了线程安全、高精度测量和全面的错误处理机制。

## 🚀 核心特性

### 多后端支持
- **numpy**: 标准NumPy后端
- **qibojit (numba)**: 高性能JIT编译后端
- **qibotn (qutensornet)**: 张量网络后端
- **qiboml (jax/pytorch/tensorflow)**: 机器学习框架后端
- **qulacs**: 高性能量子计算模拟器

### 企业级架构
- **线程安全缓存系统**: 避免重复计算，提高分析效率
- **精确性能测量**: 高精度时间和内存测量
- **安全后端管理**: 自动后端切换和错误恢复
- **统一日志管理**: 结构化日志记录和错误追踪
- **输入验证**: 全面的参数验证和错误处理

### 丰富的性能指标
- 运行时间统计（平均值、标准差、每次运行详情）
- CPU利用率（系统级、进程级、每核利用率）
- 内存使用情况（平均使用量、峰值、增量）
- 量子态保真度（与基准状态对比）
- 详细的环境信息收集

## 🎯 快速开始

### 基础使用示例

```python
from qibo.models import Circuit
from qibo import gates
import sys
import os

# 添加路径
sys.path.append('E:/qiboenv/test/test_function')
from qibo_profiler_improved import profile_circuit, generate_markdown_report

# 创建一个简单的量子电路
circuit = Circuit(nqubits=5)
circuit.add(gates.H(0))  # Hadamard门
circuit.add(gates.CNOT(0, 1))  # CNOT门
circuit.add(gates.CNOT(1, 2))  # CNOT门
circuit.add(gates.M(0, 1, 2))  # 测量门

# 分析电路性能
report = profile_circuit(
    circuit,
    n_runs=5,                    # 运行5次取平均值
    mode='basic',                # 基础分析模式
    calculate_fidelity=True,      # 计算保真度
    timeout_seconds=60.0         # 60秒超时
)

# 生成Markdown报告
report_path = generate_markdown_report(report)
print(f"报告已生成: {report_path}")
```

### 快速命令行分析

```python
# 创建更复杂的电路进行测试
def create_ghz_circuit(n_qubits):
    """创建GHZ态电路"""
    circuit = Circuit(n_qubits)
    circuit.add(gates.H(0))
    for i in range(1, n_qubits):
        circuit.add(gates.CNOT(0, i))
    return circuit

# 分析不同规模的GHZ电路
for n_qubits in [5, 10, 15]:
    circuit = create_ghz_circuit(n_qubits)
    report = profile_circuit(
        circuit,
        n_runs=3,
        mode='detailed',  # 详细分析模式
        calculate_fidelity=True
    )
    
    # 生成带时间戳的报告
    report_path = generate_markdown_report(
        report, 
        output_path=f'ghz_analysis_{n_qubits}qubits.md'
    )
    print(f"{n_qubits}量子比特GHZ电路分析完成: {report_path}")
```

## 📊 使用案例

### 案例1: 多后端性能对比

```python
import qibo
from qibo_profiler_improved import profile_circuit

def create_test_circuit(n_qubits=8):
    """创建测试电路"""
    circuit = Circuit(n_qubits)
    # 添加随机门
    circuit.add(gates.H(0))
    for i in range(n_qubits - 1):
        circuit.add(gates.CNOT(i, i + 1))
    for i in range(n_qubits):
        circuit.add(gates.RX(i, theta=0.5))
    return circuit

# 测试电路
test_circuit = create_test_circuit(8)

# 测试不同后端
backends = [
    ("numpy", None),
    ("qibojit", "numba"),
]

results = {}
for backend_name, platform in backends:
    try:
        # 切换后端
        qibo.set_backend(backend_name, platform=platform)
        
        # 执行分析
        report = profile_circuit(
            test_circuit,
            n_runs=10,
            mode='comprehensive',  # 最全面的分析模式
            calculate_fidelity=True
        )
        
        results[f"{backend_name}_{platform or 'default'}"] = report
        print(f"✅ {backend_name} ({platform}) 分析完成")
        
    except Exception as e:
        print(f"❌ {backend_name} ({platform}) 分析失败: {e}")

# 生成对比报告
for backend, report in results.items():
    generate_markdown_report(report, f"backend_comparison_{backend}.md")
```

### 案例2: 大规模电路性能分析

```python
def analyze_large_circuit():
    """分析大规模电路性能"""
    
    # 创建大规模量子傅里叶变换电路
    def create_qft_circuit(n_qubits):
        circuit = Circuit(n_qubits)
        for j in range(n_qubits):
            circuit.add(gates.H(j))
            for k in range(j + 1, n_qubits):
                circuit.add(gates.CU1(k, j, theta=np.pi / 2**(k - j)))
        return circuit
    
    # 测试不同规模
    sizes = [8, 12, 16]
    
    for size in sizes:
        print(f"开始分析 {size} 量子比特QFT电路...")
        
        try:
            circuit = create_qft_circuit(size)
            
            # 使用详细模式进行深入分析
            report = profile_circuit(
                circuit,
                n_runs=5,
                mode='comprehensive',
                calculate_fidelity=True,
                timeout_seconds=300.0  # 5分钟超时
            )
            
            # 提取关键指标
            runtime_avg = report['results']['summary']['runtime_avg']['value']
            memory_peak = report['results']['summary']['memory_usage_peak']['value']
            fidelity = report['results']['summary'].get('fidelity', {}).get('value')
            
            print(f"  运行时间: {runtime_avg:.3f}秒")
            print(f"  峰值内存: {memory_peak:.2f} MiB")
            print(f"  保真度: {fidelity:.6f}" if fidelity else "  保真度: 计算失败")
            
            # 生成报告
            generate_markdown_report(report, f"qft_analysis_{size}qubits.md")
            
        except Exception as e:
            print(f"  分析失败: {e}")

analyze_large_circuit()
```

### 案例3: 保真度验证分析

```python
def fidelity_analysis():
    """专门用于保真度验证的分析"""
    
    # 创建已知状态的电路
    def create_bell_state():
        circuit = Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
        return circuit
    
    def create_ghz_state(n_qubits):
        circuit = Circuit(n_qubits)
        circuit.add(gates.H(0))
        for i in range(1, n_qubits):
            circuit.add(gates.CNOT(0, i))
        return circuit
    
    circuits = [
        ("Bell态", create_bell_state()),
        ("GHZ态(3q)", create_ghz_state(3)),
        ("GHZ态(5q)", create_ghz_state(5)),
    ]
    
    for name, circuit in circuits:
        print(f"分析 {name} 的保真度...")
        
        # 多次运行以获得统计信息
        report = profile_circuit(
            circuit,
            n_runs=20,  # 更多运行次数以获得统计显著性
            mode='detailed',
            calculate_fidelity=True
        )
        
        # 提取保真度信息
        fidelity = report['results']['summary'].get('fidelity', {}).get('value')
        runtime_std = report['results']['summary']['runtime_std_dev']['value']
        
        print(f"  平均保真度: {fidelity:.6f}")
        print(f"  运行时间标准差: {runtime_std:.4f}秒")
        
        # 检查保真度是否在合理范围内
        if fidelity and fidelity > 0.99:
            print(f"  ✅ {name} 保真度优秀")
        elif fidelity and fidelity > 0.95:
            print(f"  ⚠️  {name} 保真度良好")
        else:
            print(f"  ❌ {name} 保真度需要改进")
        
        generate_markdown_report(report, f"fidelity_analysis_{name.replace('(', '').replace(')', '')}.md")

fidelity_analysis()
```

## ⚙️ 配置选项详解

### ProfilerConfig 参数

```python
from qibo_profiler_improved import ProfilerConfig, profile_circuit

# 创建自定义配置
config = ProfilerConfig(
    n_runs=10,              # 运行次数，影响统计准确性
    mode='comprehensive',   # 分析模式: 'basic', 'detailed', 'comprehensive'
    calculate_fidelity=True, # 是否计算保真度
    timeout_seconds=180.0,   # 超时时间（秒）
    version="1.0"           # 分析器版本
)

# 使用自定义配置
report = profile_circuit(circuit, config=config)
```

### 分析模式说明

- **basic**: 基础性能指标，适合快速评估
- **detailed**: 详细分析，包含更多统计信息
- **comprehensive**: 全面分析，最深度的性能洞察

### 支持的后端配置

```python
# 后端配置示例
SUPPORTED_BACKENDS = {
    "numpy": {"backend_name": "numpy", "platform_name": None},
    "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
    "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
    "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
    "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
    "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"},
    "qulacs": {"backend_name": "qulacs", "platform_name": None}
}
```

## 📈 输出报告格式

### 完整报告结构

```json
{
    "metadata": {
        "profiler_version": "1.0",
        "timestamp_utc": "2025-10-14T10:00:00Z"
    },
    "inputs": {
        "profiler_settings": {
            "n_runs": 5,
            "mode": "detailed",
            "fidelity_calculated": true
        },
        "circuit_properties": {
            "n_qubits": 8,
            "depth": 10,
            "total_gates": 15,
            "gate_counts": {"h": 1, "cnot": 7, "rx": 7},
            "qasm_hash_sha256": "abc123..."
        },
        "environment": {
            "qibo_backend": "qibojit(numba)",
            "qibo_version": "0.2.0",
            "python_version": "3.8.10",
            "cpu_model_friendly": "Intel Core i7-9700K",
            "cpu_cores_physical": 8,
            "total_memory": {"value": 16.0, "unit": "GiB"}
        }
    },
    "results": {
        "summary": {
            "runtime_avg": {"value": 0.125, "unit": "seconds"},
            "runtime_std_dev": {"value": 0.008, "unit": "seconds"},
            "cpu_utilization_avg": {"value": 85.5, "unit": "percent"},
            "cpu_utilization_psutil_avg": {"value": 82.3, "unit": "percent"},
            "cpu_utilization_normalized": {"value": 10.3, "unit": "percent"},
            "memory_usage_avg": {"value": 128.5, "unit": "MiB"},
            "memory_usage_peak": {"value": 156.2, "unit": "MiB"},
            "fidelity": {"value": 0.998765, "unit": null}
        },
        "raw_metrics": {
            "runtime_per_run": {
                "values": [0.120, 0.125, 0.130, 0.123, 0.127],
                "unit": "seconds"
            }
        }
    },
    "error": null
}
```

### 性能指标说明

| 指标 | 说明 | 单位 |
|------|------|------|
| runtime_avg | 平均运行时间 | 秒 |
| runtime_std_dev | 运行时间标准差 | 秒 |
| cpu_utilization_avg | CPU平均利用率 | 百分比 |
| cpu_utilization_normalized | 每核CPU利用率 | 百分比 |
| memory_usage_avg | 平均内存使用 | MiB |
| memory_usage_peak | 峰值内存使用 | MiB |
| fidelity | 量子态保真度 | 无量纲 |

## 🔧 高级功能

### 自定义初始状态

```python
import numpy as np

# 创建自定义初始状态
initial_state = np.zeros(2**n_qubits, dtype=complex)
initial_state[0] = 1.0  # |00...0⟩ 态

# 使用自定义初始状态进行分析
report = profile_circuit(
    circuit,
    n_runs=5,
    initial_state=initial_state,
    calculate_fidelity=True
)
```

### 错误处理和调试

```python
from qibo_profiler_improved import ProfilerError, BackendError

try:
    report = profile_circuit(circuit, n_runs=10, timeout_seconds=60.0)
except BackendError as e:
    print(f"后端错误: {e}")
except ProfilerError as e:
    print(f"分析器错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")

# 检查报告中的错误信息
if report.get('error'):
    print(f"分析过程中发生错误: {report['error']}")
    if 'error_context' in report:
        print(f"错误上下文: {report['error_context']}")
```

### 批量分析

```python
def batch_analysis(circuits, names):
    """批量分析多个电路"""
    results = {}
    
    for circuit, name in zip(circuits, names):
        try:
            print(f"正在分析: {name}")
            report = profile_circuit(
                circuit,
                n_runs=5,
                mode='detailed',
                calculate_fidelity=True
            )
            results[name] = report
            generate_markdown_report(report, f"batch_{name}.md")
            print(f"✅ {name} 分析完成")
            
        except Exception as e:
            print(f"❌ {name} 分析失败: {e}")
            results[name] = {"error": str(e)}
    
    return results

# 使用示例
circuits = [create_ghz_circuit(5), create_ghz_circuit(10), create_ghz_circuit(15)]
names = ["GHZ_5q", "GHZ_10q", "GHZ_15q"]
batch_results = batch_analysis(circuits, names)
```

## 🎯 最佳实践

### 性能优化建议

1. **选择合适的运行次数**：
   - 快速测试：`n_runs=3-5`
   - 正式分析：`n_runs=10-20`
   - 统计分析：`n_runs=50+`

2. **超时设置**：
   - 小规模电路：`timeout_seconds=30.0`
   - 中等规模电路：`timeout_seconds=120.0`
   - 大规模电路：`timeout_seconds=600.0`

3. **内存管理**：
   - 大规模电路分析前关闭不必要的程序
   - 使用`calculate_fidelity=False`减少内存使用

### 常见问题解决

1. **后端切换失败**：
   ```python
   # 检查后端是否可用
   try:
       qibo.set_backend("qibojit", platform="numba")
       print("后端切换成功")
   except Exception as e:
       print(f"后端不可用: {e}")
       qibo.set_backend("numpy")  # 回退到默认后端
   ```

2. **内存不足**：
   ```python
   # 减少运行次数或关闭保真度计算
   report = profile_circuit(
       circuit,
       n_runs=3,
       calculate_fidelity=False  # 减少内存使用
   )
   ```

3. **保真度计算失败**：
   ```python
   # 检查电路是否为空或配置是否正确
   if not circuit.queue:
       print("警告: 电路为空，无法计算保真度")
       calculate_fidelity = False
   ```

## 📝 更新日志

### v1.0 (改进版)
- ✅ 新增线程安全缓存系统
- ✅ 增强错误处理和恢复机制
- ✅ 支持更多后端配置
- ✅ 改进性能测量精度
- ✅ 添加详细的环境信息收集
- ✅ 优化内存使用统计
- ✅ 增加配置验证功能
- ✅ 改进日志管理系统

---

## 📞 技术支持

如遇到问题或需要技术支持，请：
1. 检查错误日志和报告中的错误信息
2. 确认所有依赖项已正确安装
3. 验证电路配置和参数设置
4. 参考本文档的最佳实践部分

**注意**: 本工具专为研究和开发环境设计，生产环境使用前请进行充分测试。
