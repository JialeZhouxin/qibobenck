# 快速开始指南

本指南提供快速设置和使用量子模拟器基准测试平台的步骤。

## 系统要求

- Python 3.12.0 或更高版本
- Conda 或 Miniconda
- 至少 4GB 可用内存
- 2GB 可用磁盘空间

## 快速安装（5分钟）

### 选项1：仅Qibo框架（推荐新手）

```bash
# 1. 克隆项目
git clone <repository-url>
cd Bench

# 2. 创建环境
conda create -n qibo-benchmark python=3.12 -y
conda activate qibo-benchmark

# 3. 安装依赖
pip install qibo numpy pandas matplotlib seaborn psutil pytest black flake8 isort

# 4. 运行基准测试
python run_benchmarks.py --qubits 2 3 4 --verbose
```

### 选项2：多框架环境（高级用户）

```bash
# 1. 克隆项目
git clone <repository-url>
cd Bench

# 2. 创建Qiskit环境
conda env create -f environment-qiskit.yml
conda activate qibo-benchmark-qiskit

# 3. 运行基准测试
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator --qubits 2 3 --verbose
```

## 验证安装

运行以下命令验证安装是否成功：

```bash
# 测试Qibo
python -c "import qibo; print(f'Qibo {qibo.__version__} installed successfully')"

# 运行简单测试
python run_benchmarks.py --qubits 2 --verbose
```

## 基本用法

### 运行简单基准测试

```bash
# 默认配置（Qibo numpy后端，2-3量子比特）
python run_benchmarks.py

# 指定量子比特数
python run_benchmarks.py --qubits 2 3 4 5

# 启用详细输出
python run_benchmarks.py --verbose
```

### 多框架比较

```bash
# 比较Qibo和Qiskit（需要Qiskit环境）
python run_benchmarks.py --simulators qibo-numpy qiskit-aer_simulator --qubits 2 3 4

# 比较Qibo和PennyLane（需要PennyLane环境）
python run_benchmarks.py --simulators qibo-numpy pennylane-default.qubit --qubits 2 3 4
```

## 查看结果

基准测试完成后，结果保存在`results/`目录下的时间戳文件夹中：

```bash
# 查看最新结果
ls results/$(ls -t results/ | head -1)

# 查看CSV数据
cat results/$(ls -t results/ | head -1)/raw_results.csv

# 查看摘要报告
cat results/$(ls -t results/ | head -1)/summary_report.md
```

## 常见问题

### 问题1：ImportError: No module named 'qibo'

**解决方案：**
```bash
# 确保已激活正确环境
conda activate qibo-benchmark

# 重新安装Qibo
pip install qibo
```

### 问题2：内存不足

**解决方案：**
```bash
# 减少量子比特数
python run_benchmarks.py --qubits 2 3

# 或使用更小的电路
python run_benchmarks.py --circuits qft --qubits 2
```

### 问题3：图表不显示

**解决方案：**
```bash
# 安装matplotlib后端
pip install matplotlib tkinter

# 或在Jupyter中运行
jupyter notebook
```

## 下一步

1. 阅读[完整文档](README.md)了解所有功能
2. 查看[多环境指南](MULTI_ENVIRONMENT_GUIDE.md)设置专业环境
3. 运行[测试套件](tests/)验证安装
4. 尝试不同的[电路和配置](benchmark_harness/circuits/)

## 获取帮助

- 查看完整文档：`README.md`
- 多环境设置：`MULTI_ENVIRONMENT_GUIDE.md`
- 提交问题：[GitHub Issues](link-to-issues)
- 示例结果：`results/`目录

## 性能提示

1. **小型测试**：从2-3量子比特开始
2. **并行测试**：使用多核CPU
3. **内存优化**：关闭不必要的应用程序
4. **GPU加速**：安装CUDA版本的Qibo

享受基准测试之旅！