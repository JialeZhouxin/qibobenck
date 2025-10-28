# 重复运行功能设计总结

## 项目概述

本文档总结了为量子模拟器基准测试平台添加重复运行功能的完整设计方案。该功能旨在提高基准测试结果的可靠性，减少单次运行的偶然性，并使缓存机制更加有效。

## 设计目标

1. **提高测量可靠性**：通过多次运行减少偶然性影响
2. **提供统计分析**：计算均值、标准差、置信区间等统计指标
3. **增强可视化**：新增稳定性分析和置信区间图表
4. **保持兼容性**：确保与现有系统的向后兼容性
5. **优化缓存效果**：使缓存机制在重复运行中更加有效

## 核心功能设计

### 1. 命令行接口扩展

新增三个命令行参数：

```bash
--repeat N              # 重复运行次数（默认：1）
--warmup-runs N         # 预热运行次数（默认：0）
--statistical-analysis  # 启用统计分析
```

### 2. 数据结构扩展

扩展 [`BenchmarkResult`](Bench/benchmark_harness/abstractions.py:15) 类，新增字段：

- `run_id`: 运行标识
- `wall_time_mean/std/min/max`: 执行时间统计
- `cpu_time_mean/std`: CPU时间统计
- `memory_mean/std`: 内存使用统计
- `fidelity_mean/std`: 保真度统计
- `confidence_interval`: 置信区间

### 3. 接口变更

更新 [`SimulatorInterface.execute`](Bench/benchmark_harness/abstractions.py:51) 方法：

```python
def execute(
    self, 
    circuit: Any, 
    n_qubits: int, 
    reference_state: Optional[np.ndarray] = None,
    repeat: int = 1,
    warmup_runs: int = 0
) -> List[BenchmarkResult]:
```

## 实施计划

### 阶段1：核心数据结构和接口修改（1天）
- 修改 [`BenchmarkResult`](Bench/benchmark_harness/abstractions.py:15) 类
- 更新 [`SimulatorInterface`](Bench/benchmark_harness/abstractions.py:39) 接口

### 阶段2：模拟器包装器实现（2天）
- 更新 [`QiboWrapper`](Bench/benchmark_harness/simulators/qibo_wrapper.py)
- 更新 [`QiskitWrapper`](Bench/benchmark_harness/simulators/qiskit_wrapper.py)
- 更新 [`PennyLaneWrapper`](Bench/benchmark_harness/simulators/pennylane_wrapper.py)

### 阶段3：主运行脚本更新（1天）
- 添加命令行参数解析
- 更新执行逻辑

### 阶段4：后处理和可视化更新（2天）
- 修改结果分析函数
- 添加新的可视化图表
- 更新报告生成

### 阶段5：测试和验证（2天）
- 实现单元测试
- 实现集成测试
- 性能测试和验证

**总预计时间：8天**

## 输出变化

### 1. 数据文件

- **raw_results.csv**: 包含统计列（均值、标准差等）
- **detailed_runs.csv**: 新增文件，包含每次运行的详细数据

### 2. 可视化图表

- 现有图表增加误差条显示
- **execution_stability.png**: 执行稳定性图表
- **confidence_intervals.png**: 置信区间图表

### 3. 报告内容

- 重复运行次数信息
- 稳定性分析（变异系数）
- 统计显著性分析

## 使用示例

```bash
# 基本重复运行
python run_benchmarks.py --repeat 5 --circuits qft --qubits 2 3 4

# 带预热运行的重复测试
python run_benchmarks.py --repeat 10 --warmup-runs 2 --simulators qibo-numpy qiskit-aer_simulator

# 启用统计分析
python run_benchmarks.py --repeat 5 --statistical-analysis --verbose
```

## 影响分析

### 代码影响

1. **高影响**：
   - [`BenchmarkResult`](Bench/benchmark_harness/abstractions.py:15) 类扩展
   - [`SimulatorInterface`](Bench/benchmark_harness/abstractions.py:39) 接口变更
   - [`run_benchmarks.py`](Bench/run_benchmarks.py) 主逻辑

2. **中等影响**：
   - 后处理模块
   - 模拟器包装器

3. **低影响**：
   - 缓存系统（无需修改）

### 兼容性

- **向后兼容**：所有新参数都有默认值
- **接口兼容**：保持现有功能不变
- **数据兼容**：现有输出格式保持不变

## 质量保证

### 测试策略

1. **单元测试**：验证各组件功能正确性
2. **集成测试**：验证端到端流程
3. **性能测试**：评估性能影响
4. **兼容性测试**：确保向后兼容

### 测试覆盖率目标

- 单元测试覆盖率：≥ 90%
- 集成测试覆盖率：≥ 80%
- 整体测试覆盖率：≥ 85%

## 风险评估

### 高风险项

1. **接口变更导致的兼容性问题**
   - 缓解策略：保持向后兼容，提供默认值

2. **统计计算的准确性**
   - 缓解策略：使用成熟的统计库，充分测试

### 中等风险项

1. **性能影响**
   - 缓解策略：优化重复运行逻辑，使用预热运行

2. **数据处理的复杂性**
   - 缓解策略：模块化设计，清晰的代码结构

## 预期收益

1. **提高测量可靠性**：减少偶然性影响，提供更可信的基准测试结果
2. **增强分析能力**：通过统计分析提供更深入的洞察
3. **改善用户体验**：提供更详细的报告和可视化
4. **优化缓存效果**：重复运行使缓存机制更加有效

## 后续优化方向

1. **高级统计分析**：异常值检测、更复杂的统计模型
2. **性能优化**：并行化重复运行、内存优化
3. **用户体验**：进度条显示、详细错误信息
4. **扩展功能**：自适应重复次数、性能回归检测

## 结论

重复运行功能的设计方案全面考虑了功能需求、技术实现、兼容性和质量保证等方面。通过分阶段实施和充分测试，可以确保功能的可靠性和稳定性。该功能将显著提高基准测试平台的价值，为用户提供更准确、更可靠的性能评估结果。

## 相关文档

- [实施计划详情](Bench/plan/repeat_runs_implementation_plan.md)
- [测试设计详情](Bench/plan/repeat_runs_test_design.md)
- [项目架构文档](Bench/plan/project_architecture.md)
- [缓存系统设计](Bench/plan/reference_state_caching_design.md)