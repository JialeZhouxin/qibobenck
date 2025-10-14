# 量子模拟器基准测试平台 - 实施进度跟踪

## 阶段 1: 核心架构与抽象层定义 [未开始]
- [ ] 创建项目结构: `benchmark_harness/`, `tests/`, `results/`, `run_benchmarks.py`
- [ ] 在 `benchmark_harness/abstractions.py` 中定义抽象类和数据类
- [ ] 创建 `tests/test_abstractions.py` 并编写测试
- [ ] 验证抽象类无法被直接实例化

## 阶段 2: 可扩展的指标收集器模块 [未开始]
- [ ] 在 `benchmark_harness/metrics.py` 中实现 `MetricsCollector` 类
- [ ] 创建 `tests/test_metrics.py` 并编写单元测试
- [ ] 验证所有指标准确性

## 阶段 3: 第一个具体实现 (Qibo封装与QFT电路) [未开始]
- [ ] 在 `benchmark_harness/simulators/qibo_wrapper.py` 中创建 `QiboWrapper` 类
- [ ] 在 `benchmark_harness/circuits/qft.py` 中创建 `QFTCircuit` 类
- [ ] 编写集成测试验证协同工作

## 阶段 4: 扩展至Qiskit和PennyLane [未开始]
- [ ] 在 `benchmark_harness/simulators/qiskit_wrapper.py` 中创建 `QiskitWrapper` 类
- [ ] 在 `benchmark_harness/simulators/pennylane_wrapper.py` 中创建 `PennyLaneWrapper` 类
- [ ] 扩展 `QFTCircuit.build` 方法支持多平台

## 阶段 5: 运行器与命令行接口 [未开始]
- [ ] 在 `run_benchmarks.py` 中添加CLI参数解析
- [ ] 实现核心运行器算法（两阶段执行）
- [ ] 添加黄金标准参考态生成与分发逻辑

## 阶段 6: 结果后处理与可视化 [未开始]
- [ ] 在 `benchmark_harness/post_processing.py` 中创建 `analyze_results` 函数
- [ ] 生成CSV文件和可视化图表
- [ ] 创建项目README.md文档

## 预提交验证检查清单 [每个阶段完成后执行]
- [ ] 运行 `pytest` 确保所有测试通过
- [ ] 运行 `black .` 格式化代码
- [ ] 运行 `flake8 .` 检查代码质量
- [ ] 对照DefinitionOfDone验证阶段产出
