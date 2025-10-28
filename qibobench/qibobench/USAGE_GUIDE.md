# QASMBench通用基准测试工具使用指南

## 快速开始

### 1. 列出所有可用电路
```bash
cd qibobench
python qasmbench_runner.py --list
```

### 2. 运行特定电路的基准测试

**使用完整文件路径**
```bash
# 使用完整路径运行电路
python qasmbench_runner.py --circuit QASMBench/small/adder_n10/adder_n10_transpiled.qasm
```

## 针对您的问题：运行small/adder_n10_transpiled.qasm

### 步骤1：进入目录
```bash
cd qibobench
```

### 步骤2：运行基准测试
```bash

# 或者使用特定文件
python qasmbench_runner.py --circuit QASMBench/small/adder_n10/adder_n10_transpiled.qasm
```

### 步骤3：查看结果
工具会自动生成以下报告文件：
- `small_adder_n10_benchmark_report.csv` - CSV格式报告
- `small_adder_n10_benchmark_report.md` - Markdown格式报告  
- `small_adder_n10_benchmark_report.json` - JSON格式报告

## 测试的后端

工具会自动测试以下Qibo后端：
- **numpy** - 基准后端
- **qibojit (numba)** - JIT编译后端
- **qibotn (qutensornet)** - 张量网络后端
- **qiboml (jax)** - JAX机器学习后端
- **qiboml (pytorch)** - PyTorch后端
- **qiboml (tensorflow)** - TensorFlow后端

## 性能指标

每个后端都会测量：
- ✅ 执行时间（均值和标准差）
- ✅ 峰值内存使用
- ✅ 相对于numpy的加速比
- ✅ 计算结果正确性验证
- ✅ 门操作吞吐率
- ✅ JIT编译时间（如果适用）

## 示例输出

运行后会看到类似这样的输出：
```
🚀 开始QASMBench基准测试: adder_n10
电路文件: QASMBench/small/adder_n10/adder_n10_transpiled.qasm
================================================================================

预热运行 numpy...
正式测试运行 numpy (5次)...
运行 1/5: 0.1234秒
运行 2/5: 0.1187秒
...

✅ numpy 基准测试完成
   执行时间: 0.1201 ± 0.0023 秒
   峰值内存: 45.2 MB
   正确性: Passed

📊 基准测试总结
================================================================================
成功测试的后端 (按执行时间排序):
1. qibojit (numba): 0.0456秒 (2.63x)
2. numpy: 0.1201秒
3. qiboml (jax): 0.1567秒 (0.77x)

报告文件已生成:
  - small_adder_n10_benchmark_report.csv
  - small_adder_n10_benchmark_report.md
  - small_adder_n10_benchmark_report.json

🎯 基准测试完成!
```

## 故障排除

### 如果遇到"找不到电路"错误：
1. 检查QASMBench目录结构是否正确
2. 使用`--list`参数确认电路是否存在
3. 确保电路文件有.qasm扩展名

### 如果遇到导入错误：
1. 确保已安装Qibo：`pip install qibo`
2. 检查Python版本兼容性

### 如果测试失败：
1. 检查电路文件是否包含Qibo不支持的指令（如barrier）
2. 查看错误信息中的具体原因

## 高级用法

### 自定义配置
您可以修改`qasmbench_runner.py`中的`QASMBenchConfig`类来自定义：
- 运行次数 (`num_runs`)
- 预热次数 (`warmup_runs`) 
- 输出格式 (`output_formats`)
- 基准后端 (`baseline_backend`)

### 批量测试
可以编写脚本批量测试多个电路：
```python
from qasmbench_runner import run_benchmark_for_circuit

circuits = ['small/adder_n10', 'medium/qft_n18', 'large/ghz_n127']
for circuit in circuits:
    run_benchmark_for_circuit(circuit)
```

现在您可以尝试运行small/adder_n10电路了！