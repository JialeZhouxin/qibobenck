import numpy as np
# switch backend to "tensorflow" through the Qiboml provider
import qibo
qibo.set_backend(backend="qiboml", platform="tensorflow")
from qibo import Circuit, gates
import sys
import os
# 直接使用绝对路径
sys.path.append('E:/qiboenv/test/test_function')
from qibo_profiler import profile_circuit  # 导入用于分析电路性能的函数
from qibo_profiler import generate_markdown_report  # 导入用于生成Markdown格式报告的函数

circuit = Circuit(2)
circuit.add(gates.X(0))
circuit.add(gates.X(1))
circuit.add(gates.CU1(0, 1, 0.1234))
circuit.compile()

for i in range(3):
    init_state = np.ones(4) / 2.0 + i
    circuit(init_state)
    report = profile_circuit(circuit, init_state)
    print(report)
    generate_markdown_report(report, f"report_{i}.md")