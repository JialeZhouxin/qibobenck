from qibo import Circuit, gates

# 创建一个包含3个量子比特的电路
circuit = Circuit(3)

# 在第一个量子比特上添加Hadamard门
circuit.add(gates.H(0))

# 在第二个量子比特上添加Hadamard门
circuit.add(gates.H(1))

# 在第一个和第三个量子比特之间添加CNOT门
circuit.add(gates.CNOT(0, 2))

# 在第二个和第三个量子比特之间添加CNOT门
circuit.add(gates.CNOT(1, 2))

# 在第三个量子比特上添加Hadamard门
circuit.add(gates.H(2))

# 在第一个、第二个和第三个量子比特之间添加TOFFOLI门
circuit.add(gates.TOFFOLI(0, 1, 2))

# 打印电路的摘要信息
print(circuit.summary())
# 打印结果示例：
# '''
# Circuit depth = 5
# Total number of gates = 6
# Number of qubits = 3
# Most common gates:
# h: 3
# cx: 2
# ccx: 1
# '''

# 获取电路中最常见的门及其出现次数
common_gates = circuit.gate_names.most_common()
print(common_gates)
# 返回最常见的门列表，例如：[("h", 3), ("cx", 2), ("ccx", 1)]

# 获取最常见的门的名称
most_common_gate = common_gates[0][0]
# 返回最常见门的名称，例如："h"

# 获取所有Hadamard门及其在电路中的位置
all_h_gates = circuit.gates_of_type(gates.H)
# 返回所有Hadamard门的列表，例如：[(0, ref to H(0)), (1, ref to H(1)), (4, ref to H(2))]
print(all_h_gates)