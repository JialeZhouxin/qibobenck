# Qibo 后端测试环境

这是一个用于测试 Qibo 量子计算框架不同后端的项目环境。

## 项目结构

```
qiboenv/
├── qibovenv/                 # Python 虚拟环境
├── QASMBench/                # QASMBench 量子电路基准测试集
├── test/                     # 测试脚本和文档
│   ├── test_backends.py      # 后端测试脚本
│   ├── test_backends.ipynb   # Jupyter Notebook 测试文档
│   └── 测试说明书.ipynb       # 中文使用说明
├── run_qft_18.py             # QFT 电路运行脚本
├── run_qft_modified copy.py  # 修改后的 QFT 运行脚本
├── requirements.txt          # Python 依赖包
└── README.md                 # 项目说明文档
```

## 功能特性

- **多后端支持**: 测试 Qibo 的所有可用后端
- **QASMBench 集成**: 包含完整的量子电路基准测试集
- **性能比较**: 自动比较不同后端的执行性能
- **虚拟环境**: 独立的 Python 环境配置

## 支持的后端

根据测试结果，以下后端可以正常工作：

1. **numpy** - 基础后端 
2. **qibojit (numba)** - 使用 Numba 加速的后端
3. **qibotn (qutensornet)** - 张量网络后端
4. **qiboml (jax)** - 基于 JAX 的机器学习后端
5. **qiboml (pytorch)** - 基于 PyTorch 的机器学习后端
6. **qiboml (tensorflow)** - 基于 TensorFlow 的机器学习后端

## 快速开始

### 1. 激活虚拟环境
```bash
.\qibovenv\Scripts\Activate.ps1
```

### 2. 运行后端测试
```bash
python test\test_backends.py
```

### 3. 测试 QFT 电路
```bash
python run_qft_18.py
```

## 环境要求

- Python 3.8+
- Windows/Linux/macOS
- Git

## 安装依赖

```bash
pip install -r requirements.txt
```

## 测试结果示例

```
成功测试的后端 (按执行时间排序):
1. qibojit (numba): 0.1788秒
2. qiboml (pytorch): 2.9259秒
3. qibotn (qutensornet): 5.4387秒
4. qiboml (jax): 8.0038秒
5. numpy: 9.1884秒
6. qiboml (tensorflow): 17.6525秒
```

## 许可证

本项目基于 QASMBench 基准测试集，遵循相应的开源许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。