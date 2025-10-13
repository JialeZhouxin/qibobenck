from qibo import Circuit, gates  # 导入Qibo库中的Circuit和gates模块

circuit = Circuit(5)  # 创建一个包含5个量子比特的电路
circuit.add(gates.X(0))  # 在第0个量子比特上添加X门（翻转量子态）
circuit.add(gates.X(4))  # 在第4个量子比特上添加X门（翻转量子态）
circuit.add(gates.M(0, 1, register_name="A"))  # 测量第0和第1个量子比特，并将结果存储在名为"A"的寄存器中
circuit.add(gates.M(3, 4, register_name="B"))  # 测量第3和第4个量子比特，并将结果存储在名为"B"的寄存器中
result = circuit(nshots=100)  # 执行电路，进行100次测量
#print(result.samples(binary=True))  # 打印二进制格式的测量样本
#print(result.samples(binary=False))  # 打印十进制格式的测量样本
#print(result.frequencies(binary=True))  # 打印二进制格式的测量频率
#print(result.frequencies(binary=False))  # 打印十进制格式的测量频率
#print(result.probabilities())  # 打印测量的概率分布
#print(result.state())  # 打印量子态的最终状态
print(result.samples(binary=False, registers=True))  # 打印十进制格式的测量样本，并按寄存器分组
print(result.frequencies(binary=True, registers=True))  # 打印二进制格式的测量频率，并按寄存器分组
