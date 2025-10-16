# 重复运行功能实施计划

## 概述

本文档详细描述了为量子模拟器基准测试平台添加重复运行功能的实施计划。该功能将提高测量结果的可靠性，减少单次运行的偶然性，并使缓存机制更加有效。

## 功能目标

1. 允许用户指定每个电路的重复运行次数
2. 提供预热运行选项以减少初始化影响
3. 计算并展示统计分析结果（均值、标准差、置信区间等）
4. 生成稳定性分析和新的可视化图表
5. 保持与现有系统的兼容性

## 实施步骤

### 第一阶段：核心数据结构和接口修改

#### 1.1 修改 BenchmarkResult 数据结构
**文件**: `benchmark_harness/abstractions.py`
**任务**: 扩展 BenchmarkResult 类以支持多次运行统计
**影响**: 所有使用 BenchmarkResult 的代码

```python
# 新增字段
run_id: int = 1
wall_time_mean: Optional[float] = None
wall_time_std: Optional[float] = None
wall_time_min: Optional[float] = None
wall_time_max: Optional[float] = None
cpu_time_mean: Optional[float] = None
cpu_time_std: Optional[float] = None
memory_mean: Optional[float] = None
memory_std: Optional[float] = None
fidelity_mean: Optional[float] = None
fidelity_std: Optional[float] = None
confidence_interval: Optional[Tuple[float, float]] = None
```

#### 1.2 修改 SimulatorInterface 接口
**文件**: `benchmark_harness/abstractions.py`
**任务**: 更新 execute 方法签名以支持重复运行参数
**影响**: 所有模拟器包装器实现

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

### 第二阶段：模拟器包装器实现

#### 2.1 更新 QiboWrapper
**文件**: `benchmark_harness/simulators/qibo_wrapper.py`
**任务**: 实现重复运行逻辑和统计计算
**影响**: Qibo 模拟器的基准测试

#### 2.2 更新 QiskitWrapper
**文件**: `benchmark_harness/simulators/qiskit_wrapper.py`
**任务**: 实现重复运行逻辑和统计计算
**影响**: Qiskit 模拟器的基准测试

#### 2.3 更新 PennyLaneWrapper
**文件**: `benchmark_harness/simulators/pennylane_wrapper.py`
**任务**: 实现重复运行逻辑和统计计算
**影响**: PennyLane 模拟器的基准测试

### 第三阶段：主运行脚本更新

#### 3.1 添加命令行参数
**文件**: `run_benchmarks.py`
**任务**: 在 parse_arguments 函数中添加新参数
**影响**: 命令行接口

```python
--repeat: 重复运行次数（默认：1）
--warmup-runs: 预热运行次数（默认：0）
--statistical-analysis: 启用统计分析
```

#### 3.2 更新执行逻辑
**文件**: `run_benchmarks.py`
**任务**: 修改 run_benchmarks 函数以传递重复运行参数
**影响**: 基准测试执行流程

### 第四阶段：后处理和可视化更新

#### 4.1 修改结果分析函数
**文件**: `benchmark_harness/post_processing.py`
**任务**: 更新 analyze_results 函数以处理多次运行数据
**影响**: 结果分析和可视化

#### 4.2 添加新的可视化图表
**文件**: `benchmark_harness/post_processing.py`
**任务**: 实现稳定性和置信区间图表
**影响**: 可视化输出

#### 4.3 更新报告生成
**文件**: `benchmark_harness/post_processing.py`
**任务**: 修改 generate_summary_report 函数以包含统计信息
**影响**: 摘要报告内容

### 第五阶段：测试和验证

#### 5.1 单元测试
**文件**: `tests/test_repeat_runs.py`
**任务**: 创建测试用例验证重复运行功能
**影响**: 测试覆盖率

#### 5.2 集成测试
**文件**: `tests/test_integration_repeat.py`
**任务**: 创建端到端测试验证完整流程
**影响**: 系统稳定性

## 代码修改影响分析

### 高影响修改

1. **BenchmarkResult 类扩展**
   - 影响范围：所有使用该类的代码
   - 兼容性：向后兼容，新字段有默认值
   - 风险：低

2. **SimulatorInterface 接口变更**
   - 影响范围：所有模拟器包装器
   - 兼容性：需要更新所有实现
   - 风险：中等

3. **run_benchmarks.py 主逻辑**
   - 影响范围：基准测试执行流程
   - 兼容性：向后兼容，新参数有默认值
   - 风险：中等

### 中等影响修改

1. **后处理模块**
   - 影响范围：结果分析和可视化
   - 兼容性：向后兼容
   - 风险：低

2. **模拟器包装器**
   - 影响范围：特定模拟器的基准测试
   - 兼容性：向后兼容
   - 风险：低

### 低影响修改

1. **缓存系统**
   - 影响范围：参考态缓存
   - 兼容性：无需修改
   - 风险：无

## 测试策略

### 单元测试

1. **BenchmarkResult 扩展测试**
   - 验证新字段的正确初始化
   - 验证统计计算的正确性

2. **模拟器包装器测试**
   - 验证重复运行逻辑
   - 验证预热运行功能
   - 验证统计计算准确性

3. **后处理模块测试**
   - 验证多次运行数据的处理
   - 验证新图表的生成

### 集成测试

1. **端到端流程测试**
   - 验证完整的重复运行流程
   - 验证不同参数组合的正确性

2. **兼容性测试**
   - 验证与现有功能的兼容性
   - 验证默认行为的一致性

### 性能测试

1. **重复运行性能测试**
   - 验证重复运行对性能的影响
   - 验证缓存机制的有效性

## 风险评估和缓解策略

### 高风险项

1. **接口变更导致的兼容性问题**
   - 缓解策略：保持向后兼容，提供默认值

2. **统计计算的准确性**
   - 缓解策略：使用成熟的统计库，编写充分的测试

### 中等风险项

1. **性能影响**
   - 缓解策略：优化重复运行逻辑，使用预热运行

2. **数据处理的复杂性**
   - 缓解策略：模块化设计，清晰的代码结构

### 低风险项

1. **可视化图表的渲染**
   - 缓解策略：使用成熟的可视化库

2. **报告格式的变更**
   - 缓解策略：保持现有格式，添加新内容

## 实施时间表

| 阶段 | 任务 | 预计时间 | 依赖关系 |
|------|------|----------|----------|
| 1 | 数据结构和接口修改 | 1天 | 无 |
| 2 | 模拟器包装器实现 | 2天 | 阶段1 |
| 3 | 主运行脚本更新 | 1天 | 阶段1 |
| 4 | 后处理和可视化更新 | 2天 | 阶段1,2,3 |
| 5 | 测试和验证 | 2天 | 阶段1,2,3,4 |
| **总计** | | **8天** | |

## 验收标准

1. **功能完整性**
   - [ ] 所有新参数正确工作
   - [ ] 重复运行逻辑正确执行
   - [ ] 统计计算准确无误

2. **兼容性**
   - [ ] 现有功能不受影响
   - [ ] 默认行为保持一致
   - [ ] 向后兼容性良好

3. **质量保证**
   - [ ] 所有测试通过
   - [ ] 代码覆盖率达标
   - [ ] 性能影响可接受

4. **文档完整性**
   - [ ] 用户文档更新
   - [ ] API文档更新
   - [ ] 示例代码提供

## 后续优化建议

1. **高级统计分析**
   - 添加异常值检测和处理
   - 实现更复杂的统计模型

2. **性能优化**
   - 并行化重复运行
   - 优化内存使用

3. **用户体验**
   - 添加进度条显示
   - 提供更详细的错误信息

4. **扩展功能**
   - 支持自适应重复次数
   - 添加性能回归检测