# run_benchmarks.py 测试脚本使用说明

## 概述

`test_run_benchmarks.py` 是为 `run_benchmarks.py` 设计的完整测试套件，验证基准测试运行器的所有主要功能。

## 测试内容

测试脚本包含以下测试用例：

1. **命令行参数解析测试**
   - `test_parse_arguments_default`: 测试默认参数解析
   - `test_parse_arguments_custom`: 测试自定义参数解析

2. **模拟器实例创建测试**
   - `test_create_simulator_instances_qibo`: 测试Qibo模拟器实例创建
   - `test_create_simulator_instances_invalid`: 测试无效模拟器配置处理

3. **电路实例创建测试**
   - `test_create_circuit_instances_qft`: 测试QFT电路实例创建
   - `test_create_circuit_instances_unsupported`: 测试不支持的电路类型处理

4. **基准测试核心功能测试**
   - `test_run_benchmarks_basic`: 测试基本基准测试流程
   - `test_run_benchmarks_multiple_simulators`: 测试多模拟器比较

5. **缓存功能测试**
   - `test_cache_configuration_memory`: 测试内存缓存配置
   - `test_cache_configuration_disk`: 测试磁盘缓存配置

6. **结果后处理测试**
   - `test_result_analysis`: 测试结果分析功能

7. **完整流程集成测试**
   - `test_main_basic_flow`: 测试main函数基本流程
   - `test_main_error_handling`: 测试main函数错误处理

## 运行测试

### 运行所有测试
```bash
cd Bench/tests
python test_run_benchmarks.py
```

### 运行特定测试
```bash
cd Bench/tests
python test_run_benchmarks.py TestRunBenchmarks.test_parse_arguments_default
```

### 使用unittest模块运行
```bash
cd Bench/tests
python -m unittest test_run_benchmarks.py
```

### 运行特定测试并显示详细输出
```bash
cd Bench/tests
python -m unittest test_run_benchmarks.TestRunBenchmarks.test_parse_arguments_default -v
```

## 测试特点

1. **资源优化**: 使用少量量子比特（2个）以减少计算资源使用
2. **自动清理**: 使用临时目录，测试后自动清理
3. **错误处理**: 包含对各种错误情况的测试
4. **跳过机制**: 当Qibo不可用时自动跳过相关测试

## 测试结果示例

```
Running Quantum Fourier Transform with 2 qubits...
  Getting reference state using cache...
[Qibo 0.2.21|INFO|2025-10-15 19:39:41]: Using numpy backend on /CPU:0
Computed reference state for qft(2 qubits) in 0.0449s
Saved reference state to disk in 0.0010s
  Reference state obtained from cache
  Running on qibo-numpy...
    Completed in 0.0005s, fidelity: 1.0000
.
----------------------------------------------------------------------
Ran 13 tests in 8.993s

OK (skipped=1)
```

## 依赖要求

- Python 3.6+
- unittest (标准库)
- numpy
- pandas
- qibo
- 其他 benchmark_harness 模块

## 注意事项

1. 测试前确保已安装Qibo和相关后端
2. 某些测试可能会因为环境问题被跳过
3. 测试过程中会创建临时文件和目录
4. 测试可能会花费几秒钟时间

## 故障排除

如果遇到导入错误，请确保：
1. 在正确的目录中运行测试（Bench/tests）
2. Python路径设置正确
3. 所有依赖模块已安装

如果测试失败，请检查：
1. Qibo和相关后端是否正确安装
2. 是否有足够的系统资源
3. 临时目录是否有写入权限