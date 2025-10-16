# 命令行使用参考手册

本手册提供了`run_benchmarks.py`脚本的完整命令行语法规范和参数说明，涵盖了Qibo、Qiskit和PennyLane三个主要量子计算框架的多种后端选项，包括高性能CPU后端、GPU加速后端、机器学习后端和张量网络后端。

## 基本语法

```bash
python run_benchmarks.py [选项]
```

## 参数详解

### 1. `--circuits` 参数

**语法**: `--circuits <电路名称1> <电路名称2> ...`

**说明**: 指定要测试的基准电路列表

**可用选项**:
- `qft`: 量子傅里叶变换电路

**默认值**: `["qft"]`

**示例**:
```bash
# 测试QFT电路
python run_benchmarks.py --circuits qft

# 测试多个电路（当有多个电路可用时）
python run_benchmarks.py --circuits qft other_circuit
```

### 2. `--qubits` 参数

**语法**: `--qubits <量子比特数1> <量子比特数2> ...`

**说明**: 指定要测试的量子比特数列表

**类型**: 整数列表

**默认值**: `[2, 3, 4]`

**示例**:
```bash
# 测试2-6量子比特
python run_benchmarks.py --qubits 2 3 4 5 6

# 测试特定量子比特数
python run_benchmarks.py --qubits 4 8 12
```

### 3. `--simulators` 参数

**语法**: `--simulators <平台-后端1> <平台-后端2> ...`

**说明**: 指定要测试的模拟器列表，格式为`platform-backend`

**默认值**: `["qibo-qibojit"]`

#### Qibo后端选项

| 标识符 | 后端名称 | 描述 | 性能特点 | 安装要求 |
|--------|----------|------|----------|----------|
| `qibo-numpy` | NumPy | 标准NumPy后端 | 基准性能，兼容性最好 | 基础qibo包 |
| `qibo-qibojit` | QiboJIT | JIT编译后端（基于Numba） | 高性能，10-20倍加速 | qibojit包，numba |
| `qibo-qiboml-tensorflow` | QiboML TensorFlow | 机器学习后端（TensorFlow后端） | 适合量子机器学习任务 | qiboml包，tensorflow |
| `qibo-qiboml-pytorch` | QiboML PyTorch | 机器学习后端（PyTorch后端） | 适合量子机器学习任务 | qiboml包，pytorch |
| `qibo-qiboml-jax` | QiboML JAX | 机器学习后端（JAX后端） | 适合量子机器学习任务 | qiboml包，jax |
| `qibo-qibotn` | QiboTN | 张量网络后端 | 适合大规模量子系统模拟 | qibotn包 |
| `qibo-qulacs` | Qulacs | Qulacs高性能后端 | 极高性能，日本开发 | qibo-qulacs包，qulacs |

#### Qiskit后端选项

| 标识符 | 后端名称 | 描述 | 性能特点 | 安装要求 |
|--------|----------|------|----------|----------|
| `qiskit-aer_simulator` | Aer Simulator | Qiskit Aer高性能模拟器 | 高精度模拟，支持噪声模型 | qiskit-aer包 |

#### PennyLane后端选项

| 标识符 | 后端名称 | 描述 | 性能特点 | 安装要求 |
|--------|----------|------|----------|----------|
| `pennylane-default.qubit` | Default Qubit | 默认Python后端 | 基准性能，功能完整 | 基础pennylane包 |
| `pennylane-lightning.qubit` | Lightning Qubit | C++优化后端 | 高性能，2-5倍加速 | pennylane-lightning包 |
| `pennylane-lightning.gpu` | Lightning GPU | GPU加速后端 | 极高性能，适合大规模电路 | pennylane-lightning[gpu]包 |
| `pennylane-qiskit.aer` | Qiskit Aer | 通过Qiskit Aer后端 | 高精度，支持噪声模型 | pennylane-qiskit包 |
| `pennylane-qulacs` | Qulacs | Qulacs高性能后端 | 极高性能，日本开发 | pennylane-qulacs包 |

**示例**:
```bash
# 测试所有高性能后端
python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit qiskit-aer_simulator

# 测试多个后端进行比较
python run_benchmarks.py --simulators qibo-numpy qibo-qibojit qiskit-aer_simulator pennylane-default.qubit pennylane-lightning.qubit

# 测试Qibo机器学习后端
python run_benchmarks.py --simulators qibo-qiboml-tensorflow qibo-qiboml-pytorch qibo-qiboml-jax

# 测试GPU加速后端
python run_benchmarks.py --simulators pennylane-lightning.gpu qibo-qibojit

# 测试张量网络后端
python run_benchmarks.py --simulators qibo-qibotn qibo-qulacs pennylane-qulacs

# 测试单个后端
python run_benchmarks.py --simulators qibo-qibojit
```

### 4. `--golden-standard` 参数

**语法**: `--golden-standard <平台-后端>`

**说明**: 指定用于生成参考态的模拟器

**默认值**: `"qibo-numpy"`

**示例**:
```bash
# 使用qibo-numpy作为黄金标准
python run_benchmarks.py --golden-standard qibo-numpy

# 使用qibo-qibojit作为黄金标准
python run_benchmarks.py --golden-standard qibo-qibojit
```

### 5. `--output-dir` 参数

**语法**: `--output-dir <目录路径>`

**说明**: 指定结果输出目录

**默认值**: `"results"`

**示例**:
```bash
# 输出到默认目录
python run_benchmarks.py --output-dir results

# 输出到自定义目录
python run_benchmarks.py --output-dir my_benchmark_results
```

### 6. `--verbose` 参数

**语法**: `--verbose` (无参数)

**说明**: 启用详细输出，显示每个测试步骤的详细信息

**默认值**: 未启用

**示例**:
```bash
# 启用详细输出
python run_benchmarks.py --verbose
```

## 完整命令示例

### 基础测试

```bash
# 基本测试：使用默认设置
python run_benchmarks.py
```

### 高性能后端测试

```bash
# 测试所有高性能后端
python run_benchmarks.py \
  --simulators qibo-qibojit pennylane-lightning.qubit qiskit-aer_simulator \
  --qubits 2 3 4 5 6 \
  --verbose
```

### 全面性能比较

```bash
# 比较所有后端性能
python run_benchmarks.py \
  --simulators qibo-numpy qibo-qibojit qiskit-aer_simulator pennylane-default.qubit pennylane-lightning.qubit \
  --qubits 2 3 4 5 6 \
  --circuits qft \
  --golden-standard qibo-numpy \
  --output-dir comprehensive_comparison \
  --verbose
```

### 机器学习后端测试

```bash
# 测试QiboML各种机器学习后端
python run_benchmarks.py \
  --simulators qibo-qiboml-tensorflow qibo-qiboml-pytorch qibo-qiboml-jax \
  --qubits 2 3 4 5 \
  --golden-standard qibo-numpy \
  --output-dir ml_backends_comparison \
  --verbose
```

### GPU加速后端测试

```bash
# 测试GPU加速后端（需要GPU支持）
python run_benchmarks.py \
  --simulators pennylane-lightning.gpu qibo-qibojit \
  --qubits 8 10 12 14 \
  --golden-standard qibo-qibojit \
  --output-dir gpu_comparison \
  --verbose
```

### 张量网络后端测试

```bash
# 测试张量网络后端（适合大规模系统）
python run_benchmarks.py \
  --simulators qibo-qibotn qibo-qulacs pennylane-qulacs \
  --qubits 10 12 14 16 \
  --golden-standard qibo-numpy \
  --output-dir tn_backends_comparison \
  --verbose
```

### 大规模测试

```bash
# 大规模量子比特测试
python run_benchmarks.py \
  --simulators qibo-qibojit pennylane-lightning.qubit qiskit-aer_simulator \
  --qubits 8 10 12 14 \
  --verbose
```

### 快速验证

```bash
# 快速验证安装
python run_benchmarks.py \
  --simulators qibo-qibojit \
  --qubits 2 \
  --verbose
```

## 严格语法规范

### 1. 参数顺序

参数顺序不影响功能，但建议按以下顺序排列：
```bash
python run_benchmarks.py \
  --circuits <电路列表> \
  --qubits <量子比特数列表> \
  --simulators <模拟器列表> \
  --golden-standard <黄金标准> \
  --output-dir <输出目录> \
  --verbose
```

### 2. 空格和换行

- 参数名和参数值之间必须有空格
- 多个参数值之间用空格分隔
- 可以使用反斜杠(`\`)进行换行以提高可读性

### 3. 引号使用

- 参数值通常不需要引号
- 如果参数值包含特殊字符，可以使用引号：
  ```bash
  python run_benchmarks.py --output-dir "my results"
  ```

### 4. 大小写敏感

- 所有参数名都是小写
- 后端名称区分大小写，必须使用指定的格式

## 输出解读

### 控制台输出

启用`--verbose`参数后，控制台将显示：

```
Quantum Simulator Benchmark Runner
Output directory: results/benchmark_20251014_152045
Verbose mode: True

Initializing simulators...
Successfully created qibo simulator with backend qibojit
Successfully created pennylane simulator with backend lightning.qubit
Available simulators: ['qibo-qibojit', 'pennylane-lightning.qubit']

Initializing circuits...
Successfully created qft circuit
Available circuits: ['qft']

Running benchmarks...

Running qft with 2 qubits...
  Generating reference state using qibo-qibojit...
  Reference state generated successfully
  Running on qibo-qibojit...
    Completed in 0.0012s, fidelity: 1.0000
  Running on pennylane-lightning.qubit...
    Completed in 0.0023s, fidelity: 1.0000

Running qft with 3 qubits...
  ...
```

### 结果文件

测试完成后，将在输出目录中生成以下文件：

1. **raw_results.csv**: 原始测试数据
   ```csv
   simulator,backend,circuit_name,n_qubits,wall_time_sec,cpu_time_sec,peak_memory_mb,cpu_utilization_percent,state_fidelity
   qibo,qibojit,qft,2,0.0012,0.0011,0.0,85.5,1.0
   pennylane,lightning.qubit,qft,2,0.0023,0.0022,0.0,92.1,1.0
   ```

2. **summary_report.md**: 摘要报告，包含性能分析

3. **性能图表**:
   - `wall_time_scaling.png`: 墙上时间扩展性
   - `memory_scaling.png`: 内存使用扩展性
   - `fidelity.png`: 保真度对比

## 性能优化建议

### 1. 后端选择策略

#### 通用性能测试
- **基准测试**: 始终包含`qibo-numpy`作为参考基准
- **高性能CPU测试**: 使用`qibo-qibojit`（10-20倍加速）
- **跨框架比较**: 使用`pennylane-lightning.qubit`和`qiskit-aer_simulator`

#### 机器学习应用
- **TensorFlow生态**: 使用`qibo-qiboml-tensorflow`
- **PyTorch生态**: 使用`qibo-qiboml-pytorch`
- **JAX生态**: 使用`qibo-qiboml-jax`

#### 大规模量子系统
- **中等规模（8-14量子比特）**: 使用`qibo-qibojit`或`pennylane-lightning.qubit`
- **大规模（15+量子比特）**: 使用`qibo-qibotn`、`qibo-qulacs`或`pennylane-qulacs`张量网络后端
- **GPU加速**: 使用`pennylane-lightning.gpu`（需要GPU支持）

#### 高精度模拟
- **含噪声模拟**: 使用`qiskit-aer_simulator`
- **高精度验证**: 使用`pennylane-qiskit.aer`

### 2. 量子比特数策略

- **验证阶段**: 2-4量子比特（快速验证所有后端）
- **性能测试**: 5-8量子比特（平衡性能与资源消耗）
- **大规模测试**: 10-14量子比特（测试高性能后端）
- **极限测试**: 15+量子比特（仅张量网络后端，需要大量内存）

### 3. 资源管理

- 使用`--verbose`监控资源使用
- 大规模测试前关闭不必要应用
- 考虑使用SSD存储提高I/O性能
- GPU测试前确保CUDA环境正确配置
- 张量网络测试前确保有足够系统内存（建议32GB+）

## 故障排除

### 常见错误及解决方案

1. **Qibo后端不可用错误**
   ```
   Warning: Failed to create qibo simulator with backend qibojit: Backend 'qibojit' is not available
   ```
   **解决方案**:
   - 安装qibojit包: `pip install qibojit`
   - 安装numba: `pip install numba`
   - 检查Qibo版本兼容性

2. **QiboML后端错误**
   ```
   ImportError: No module named 'qiboml'
   ```
   **解决方案**:
   - 安装qiboml包: `pip install qiboml`
   - 安装对应的ML框架: `pip install tensorflow` 或 `pip install pytorch` 或 `pip install jax`

3. **QiboTN后端错误**
   ```
   ImportError: No module named 'qibotn'
   ```
   **解决方案**:
   - 安装qibotn包: `pip install qibotn`
   - 确保有足够系统内存（建议32GB+）

4. **QiboQulacs后端错误**
   ```
   ImportError: No module named 'qibo_qulacs'
   ```
   **解决方案**:
   - 安装qibo-qulacs包: `pip install qibo-qulacs`
   - 安装qulacs包: `pip install qulacs`
   - 确保qulacs已正确编译并安装

5. **Qiskit后端错误**
   ```
   ImportError: No module named 'qiskit_aer'
   ```
   **解决方案**:
   - 安装qiskit-aer包: `pip install qiskit-aer`
   - 或使用完整Qiskit安装: `pip install qiskit[all]`

5. **PennyLane后端错误**
   ```
   ImportError: No module named 'pennylane_lightning'
   ```
   **解决方案**:
   - 安装pennylane-lightning包: `pip install pennylane-lightning`
   - 对于GPU版本: `pip install pennylane-lightning[gpu]`

6. **GPU后端错误**
   ```
   RuntimeError: CUDA not available
   ```
   **解决方案**:
   - 检查CUDA安装: `nvidia-smi`
   - 安装CUDA版本的PyTorch/TensorFlow
   - 检查GPU内存是否足够

7. **内存不足错误**
   ```
   MemoryError: Unable to allocate array
   ```
   **解决方案**:
   - 减少量子比特数
   - 使用张量网络后端（qibo-qibotn, pennylane-qulacs）
   - 关闭其他应用程序释放内存

### 调试技巧

1. **使用最小配置测试**:
   ```bash
   python run_benchmarks.py --simulators qibo-numpy --qubits 2 --verbose
   ```

2. **逐个测试后端**:
   ```bash
   python run_benchmarks.py --simulators qibo-qibojit --qubits 2 --verbose
   python run_benchmarks.py --simulators qiskit-aer_simulator --qubits 2 --verbose
   python run_benchmarks.py --simulators pennylane-lightning.qubit --qubits 2 --verbose
   ```

3. **检查环境**:
   ```bash
   python verify_backends.py
   ```

4. **验证特定后端安装**:
   ```bash
   # 检查Qibo后端
   python -c "import qibo; qibo.set_backend('qibojit'); print('QiboJIT可用')"
   python -c "import qibo; qibo.set_backend('qulacs'); print('QiboQulacs可用')"
   
   # 检查PennyLane后端
   python -c "import pennylane as qml; dev = qml.device('lightning.qubit', wires=1); print('Lightning可用')"
   
   # 检查Qiskit后端
   python -c "from qiskit_aer import AerSimulator; print('Aer Simulator可用')"
   ```

5. **查看可用后端列表**:
   ```bash
   # Qibo后端
   python -c "import qibo; print(qibo.get_available_backends())"
   
   # PennyLane后端
   python -c "import pennylane as qml; print(qml.plugin_devices())"
   ```

## 总结

通过本参考手册，您应该能够：
1. 理解所有命令行参数的语法和用法
2. 了解所有可用的量子计算后端及其特点
3. 根据不同应用场景选择合适的后端组合
4. 构建适合不同测试场景的命令
5. 解读测试输出和结果文件
6. 解决常见的技术问题

### 后端选择快速参考

| 应用场景 | 推荐后端 | 备注 |
|---------|---------|------|
| 基准性能测试 | `qibo-numpy` | 作为参考基准 |
| 高性能CPU计算 | `qibo-qibojit` | 10-20倍加速 |
| 量子机器学习 | `qibo-qiboml-*` | 根据ML框架选择 |
| 大规模系统模拟 | `qibo-qibotn`, `qibo-qulacs`, `pennylane-qulacs` | 张量网络后端 |
| GPU加速计算 | `pennylane-lightning.gpu` | 需要GPU支持 |
| 高精度模拟 | `qiskit-aer_simulator` | 支持噪声模型 |
| 跨框架比较 | `qibo-qibojit`, `pennylane-lightning.qubit`, `qiskit-aer_simulator` | 多框架对比 |

记住，选择合适的后端组合可以显著提升测试速度和准确性。始终建议在性能测试中包含基准后端(`qibo-numpy`)以便进行准确的性能比较。