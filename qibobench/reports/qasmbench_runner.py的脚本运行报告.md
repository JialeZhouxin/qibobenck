# qasmbench_runner.py 的脚本运行报告

本文档介绍 qibobench/qasmbench_runner.py 脚本的功能、使用方法，并以 QASMBench 中的小规模电路 adder_n10_transpiled 为例展示输入与输出样例，帮助你快速完成基准测试与结果解读。

---

## 1. 脚本主要功能

该脚本用于对 QASMBench 数据集中的任意 QASM 电路进行多后端性能基准测试，自动生成标准化报告（CSV、Markdown、JSON），核心能力包括：
- 自动加载指定的 QASM 电路（会自动去除 barrier 语句，兼容 Qibo 后端）
- 针对多个后端进行多次运行、预热与统计（均值、标准差、峰值内存、吞吐率等）
- 基于 baseline 后端（默认 numpy）计算速度提升比（speedup）
- 自动将报告按电路名称保存至独立目录：qibobench/reports/<电路名>/benchmark_report.*
- 列出所有可用电路名称（按 small/medium/large 分类）供参考

后端覆盖示例（随安装环境不同会有变化）：
- numpy
- qibojit (numba)
- qibotn (qutensornet)
- qiboml (jax)
- qiboml (pytorch)
- qiboml (tensorflow)

说明：实际可用后端取决于你的环境是否已正确安装相应依赖。

---

## 2. 如何运行脚本（步骤）

准备条件：
- 已在 Python 环境中安装 qibo 及可选后端依赖
- 当前工作目录为项目根目录 e:/qiboenv（或根据你的实际路径调整命令）
- adder_n10_transpiled.qasm 在数据集路径中存在（QASMBench/small/adder_n10/）

常用命令：
1) 列出所有可用电路（用于确认电路名称与可用性）
```
python qibobench/qasmbench_runner.py --list
```

2) 对某个电路执行基准测试（通过 QASM 文件完整路径）
```
python qibobench/qasmbench_runner.py --circuit QASMBench/small/adder_n10/adder_n10_transpiled.qasm
```

运行完成后，脚本会：
- 在控制台打印各后端的执行时间、内存、正确性等摘要
- 在 qibobench/reports/adder_n10_transpiled/ 下生成三种报告：
  - benchmark_report.csv
  - benchmark_report.md
  - benchmark_report.json

提示：
- 当前脚本命令行参数支持 --list 与 --circuit。若你希望“通过电路名称直接运行”，可先使用 --list 查到路径，再用 --circuit 传入该 QASM 文件的完整路径。

---

## 3. 示例：以 adder_n10_transpiled 为例

示例电路路径：
```
QASMBench/small/adder_n10/adder_n10_transpiled.qasm
```

运行命令：
```
python qibobench/qasmbench_runner.py --circuit QASMBench/small/adder_n10/adder_n10_transpiled.qasm
```

控制台输出（示意，不同环境会略有差异）：
```
🚀 开始QASMBench基准测试: adder_n10_transpiled
电路文件: QASMBench/small/adder_n10/adder_n10_transpiled.qasm
================================================================================
预热运行 numpy...
正式测试运行 numpy (5次)...
运行 1/5: 0.0021秒
...
✅ numpy 基准测试完成
   执行时间: 0.0022 ± 0.0003 秒
   峰值内存: 5.10 MB
   正确性: Passed

...（其他已安装可用后端按序测试与打印）

📊 基准测试总结
================================================================================
成功测试的后端 (按执行时间排序):
1. numpy: 0.0022秒
...（其他后端）
```

生成报告文件（保存位置固定到独立目录，按电路名清晰标识）：
- qibobench/reports/adder_n10_transpiled/benchmark_report.csv
- qibobench/reports/adder_n10_transpiled/benchmark_report.md
- qibobench/reports/adder_n10_transpiled/benchmark_report.json

报告内容要点：
- CSV：便于表格化分析与聚合（均值、标准差、内存、吞吐率、加速比等）
- Markdown：便于人读的综合报告，包含环境信息、参数表格与性能分析
- JSON：结构化数据，适合进一步程序化处理/可视化

---

## 4. 常见参数配置说明

脚本内置的关键配置（类 QASMBenchConfig）：
- num_runs：每个后端的正式运行次数（默认 5）
- warmup_runs：预热运行次数（默认 1）
- output_formats：输出格式列表（默认 ['csv', 'markdown', 'json']）
- baseline_backend：计算 speedup 的基准后端（默认 "numpy"）
- qasm_directory：QASMBench 根目录（默认 "../QASMBench"；如工作目录不同可手动调整或使用绝对路径）

运行时参数（命令行）：
- --list：列出可用电路（small/medium/large 分类）
- --circuit <文件路径>：指定 QASM 电路文件的完整路径执行基准测试

建议与约定：
- 在 QASMBench 数据集中，优先选择文件名包含 “transpiled” 的 QASM 文件，以避免后端不支持原始门集导致的失败。
- 若你希望变更运行次数或输出格式，可在脚本中调整 QASMBenchConfig 的默认值（或扩展脚本接受更多命令行参数）。

---

## 5. 可能遇到的错误及解决方法

1) 找不到文件或路径错误
- 现象：提示“错误: 电路文件不存在”
- 解决：检查路径是否为项目根目录的相对路径（或改用绝对路径）；可先运行 `--list` 确认电路所在目录。

2) 后端不支持/依赖未安装
- 现象：某些后端测试失败，如 qiboml (jax)、qiboml (pytorch) 等
- 解决：安装相应依赖，或注释/移除该后端配置。至少保证 numpy 与 qibojit (numba) 可用即可完成基本评测。

3) 电路包含 barrier 或不兼容门
- 现象：构建/执行电路时报错
- 解决：脚本已自动过滤 barrier 行；若仍报错，优先使用 “transpiled” 版本（如 adder_n10_transpiled.qasm）。

4) 内存不足/运行缓慢（尤其 large 规模）
- 现象：执行时间过长或内存占用高
- 解决：先从 small/medium 规模电路开始，或减少 num_runs；必要时切换更适合的大规模模拟后端（例如张量网络类后端）并确保依赖安装完整。

5) 报告路径与打印路径不一致
- 现象：控制台摘要行中的简短文件名与实际保存位置不一致
- 说明：实际报告始终保存在 qibobench/reports/<电路名>/benchmark_report.*。以该目录为准进行查找。

---

## 6. 应用场景与性能解读要点

- 比较不同后端对同一电路的性能差异，为后续选型与优化提供依据
- 针对算法/电路的规模扩展性做快照分析（执行时间、吞吐率、内存趋势）
- 通过 JSON/CSV 结果与外部工具联动，形成自动化基准体系或可视化仪表板

解读建议：
- 以 baseline（numpy）为参照，观察其他后端的 speedup 与吞吐率提升
- 结合电路参数（qubits/depth/ngates）考虑规模与后端的适配性
- 注意环境差异（CPU/GPU、后端版本）会带来较明显的性能变化

---

## 7. 附：快速复现清单

- 列表电路：
```
python qibobench/qasmbench_runner.py --list
```

- 运行 adder_n10_transpiled：
```
python qibobench/qasmbench_runner.py --circuit QASMBench/small/adder_n10/adder_n10_transpiled.qasm
```

- 查看报告：
```
qibobench/reports/adder_n10_transpiled/benchmark_report.md
qibobench/reports/adder_n10_transpiled/benchmark_report.csv
qibobench/reports/adder_n10_transpiled/benchmark_report.json
```

如需扩展更多命令行参数（例如支持按“电路名称”直接运行、或修改 num_runs 等），可在当前脚本的 argparse 与 QASMBenchConfig 处增加相应选项。