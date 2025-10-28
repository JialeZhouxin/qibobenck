# 多环境使用指南

本指南详细说明如何为不同的量子计算框架创建专用的conda环境，确保框架之间的隔离性，避免依赖冲突。

## 概述

量子计算框架（Qibo、Qiskit、PennyLane）可能有相互冲突的依赖项。为了确保每个框架都能在最佳环境中运行，我们建议为每个框架创建专用的conda环境。

## 环境配置

### 1. Qiskit专用环境

创建一个名为`qibo-benchmark-qiskit`的conda环境，专门用于运行Qiskit相关的基准测试。

#### 环境配置文件 (environment-qiskit.yml)

```yaml
name: qibo-benchmark-qiskit
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - numpy>=1.24.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - psutil>=5.9.0
  - pytest>=7.4.0
  - black>=23.0.0
  - flake8>=6.0.0
  - isort>=5.12.0
  - pip:
    - qibo>=0.2.21
    - qiskit>=1.4.4
    - qiskit-aer>=0.13.3
```

#### 创建和激活环境

```bash
# 创建环境
conda env create -f environment-qiskit.yml

# 激活环境
conda activate qibo-benchmark-qiskit

# 验证安装
python -c "import qiskit; print(f'Qiskit version: {qiskit.__version__}')"
python -c "import qibo; print(f'Qibo version: {qibo.__version__}')"
```

#### 运行Qiskit基准测试

```bash
# 确保已激活正确环境
conda activate qibo-benchmark-qiskit

# 运行包含Qiskit的基准测试
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator --qubits 2 3 4 --verbose
```

### 2. PennyLane专用环境

创建一个名为`qibo-benchmark-pennylane`的conda环境，专门用于运行PennyLane相关的基准测试。

#### 环境配置文件 (environment-pennylane.yml)

```yaml
name: qibo-benchmark-pennylane
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - numpy>=1.24.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - psutil>=5.9.0
  - pytest>=7.4.0
  - black>=23.0.0
  - flake8>=6.0.0
  - isort>=5.12.0
  - pip:
    - qibo>=0.2.21
    - pennylane>=0.33.0
```

#### 创建和激活环境

```bash
# 创建环境
conda env create -f environment-pennylane.yml

# 激活环境
conda activate qibo-benchmark-pennylane

# 验证安装
python -c "import pennylane as qml; print(f'PennyLane version: {qml.__version__}')"
python -c "import qibo; print(f'Qibo version: {qibo.__version__}')"
```

#### 运行PennyLane基准测试

```bash
# 确保已激活正确环境
conda activate qibo-benchmark-pennylane

# 运行包含PennyLane的基准测试
python run_benchmarks.py --simulators qibo-numpy pennylane-default.qubit --qubits 2 3 4 --verbose
```

### 3. 通用环境（仅Qibo）

如果您只需要测试Qibo框架，可以使用更轻量的环境。

#### 环境配置文件 (environment-qibo.yml)

```yaml
name: qibo-benchmark-qibo
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - numpy>=1.24.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - psutil>=5.9.0
  - pytest>=7.4.0
  - black>=23.0.0
  - flake8>=6.0.0
  - isort>=5.12.0
  - pip:
    - qibo>=0.2.21
```

#### 创建和激活环境

```bash
# 创建环境
conda env create -f environment-qibo.yml

# 激活环境
conda activate qibo-benchmark-qibo

# 验证安装
python -c "import qibo; print(f'Qibo version: {qibo.__version__}')"
```

#### 运行Qibo基准测试

```bash
# 确保已激活正确环境
conda activate qibo-benchmark-qibo

# 运行仅包含Qibo的基准测试
python run_benchmarks.py --simulators qibo-numpy qibo-qibojit --qubits 2 3 4 5 --verbose
```

## 环境管理

### 列出所有环境

```bash
conda env list
```

### 删除环境

```bash
conda env remove -n qibo-benchmark-qiskit
conda env remove -n qibo-benchmark-pennylane
conda env remove -n qibo-benchmark-qibo
```

### 导出环境配置

```bash
# 导出当前环境
conda env export > environment-backup.yml

# 从备份创建环境
conda env create -f environment-backup.yml
```

## 测试流程

### 1. 环境隔离测试

为了验证环境隔离是否正常工作，可以运行以下测试：

```bash
# 测试Qiskit环境是否包含预期包
conda activate qibo-benchmark-qiskit
python -c "import qiskit, qibo; print('Qiskit environment OK')" || echo "Qiskit environment failed"

# 测试PennyLane环境是否包含预期包
conda activate qibo-benchmark-pennylane
python -c "import pennylane, qibo; print('PennyLane environment OK')" || echo "PennyLane environment failed"
```
### 2. 跨框架一致性测试

在不同环境中运行相同的基准测试，比较结果：

```bash
# 在Qiskit环境中
conda activate qibo-benchmark-qiskit
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator --qubits 2 3 --output-dir results-qiskit

# 在PennyLane环境中
conda activate qibo-benchmark-pennylane
python run_benchmarks.py --simulators qibo-numpy pennylane-default.qubit --qubits 2 3 --output-dir results-pennylane
```

### 3. 结果比较

比较不同环境中的Qibo结果，确保一致性：

```python
import glob
import pandas as pd

# 读取不同环境的结果
qiskit_files = glob.glob("results-qiskit/benchmark_*/raw_results.csv")
pennylane_files = glob.glob("results-pennylane/benchmark_*/raw_results.csv")

df_qiskit = pd.concat([pd.read_csv(f) for f in qiskit_files])
df_pennylane = pd.concat([pd.read_csv(f) for f in pennylane_files])

# 比较Qibo的结果
qibo_qiskit = df_qiskit[df_qiskit['simulator'] == 'qibo']
qibo_pennylane = df_pennylane[df_pennylane['simulator'] == 'qibo']

# 检查一致性
print("Qibo results comparison:")
print(f"Qiskit env: {qibo_qiskit['wall_time_sec'].mean():.6f}s")
print(f"PennyLane env: {qibo_pennylane['wall_time_sec'].mean():.6f}s")
```

## 故障排除

### 常见问题

1. **依赖冲突**
   - 如果遇到依赖冲突，尝试更新conda：`conda update conda`
   - 使用`mamba`代替`conda`以获得更快的解析：`conda install mamba -n base -c conda-forge`

2. **内存不足**
   - 对于大型电路，可能需要增加系统虚拟内存
   - 减少并行测试的量子比特数

### 环境恢复

如果环境损坏，可以重新创建：

```bash
# 删除损坏的环境
conda env remove -n qibo-benchmark-qiskit

# 重新创建
conda env create -f environment-qiskit.yml
```

## 自动化脚本

为了简化环境管理，可以创建以下自动化脚本：

### 环境创建脚本 (setup-envs.sh)

```bash
#!/bin/bash
# 创建所有环境

echo "Creating Qiskit environment..."
conda env create -f environment-qiskit.yml

echo "Creating PennyLane environment..."
conda env create -f environment-pennylane.yml

echo "Creating Qibo-only environment..."
conda env create -f environment-qibo.yml

echo "All environments created successfully!"
```

### 测试脚本 (test-envs.sh)

```bash
#!/bin/bash
# 测试所有环境

echo "Testing Qiskit environment..."
conda run -n qibo-benchmark-qiskit python -c "import qiskit, qibo; print('Qiskit environment OK')"

echo "Testing PennyLane environment..."
conda run -n qibo-benchmark-pennylane python -c "import pennylane, qibo; print('PennyLane environment OK')"

echo "Testing Qibo-only environment..."
conda run -n qibo-benchmark-qibo python -c "import qibo; print('Qibo-only environment OK')"

echo "All environments tested successfully!"
```

## 最佳实践

1. **定期更新环境**：定期更新环境以获取最新的功能和安全修复
2. **版本锁定**：在生产环境中，考虑锁定特定版本的依赖
3. **文档记录**：记录每个环境的具体用途和特殊配置
4. **备份配置**：保存环境配置文件的备份，以便快速重建
5. **隔离测试**：在进行重大更改前，在隔离环境中测试

通过遵循本指南，您可以确保每个量子计算框架都在最佳环境中运行，避免依赖冲突，并获得可靠的基准测试结果。