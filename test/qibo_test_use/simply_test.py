import numpy as np
from qibo import Circuit, gates, set_backend
set_backend("qibojit")

# Construct the circuit
circuit = Circuit(2)
# Add some gates
circuit.add(gates.H(0))
circuit.add(gates.H(1))
# Define an initial state (optional - default initial state is |00>)
initial_state = np.ones(4) / 2.0
# Execute the circuit and obtain the final state
result = circuit(initial_state) # circuit.execute(initial_state) also works
print(result.state())
# should print `tf.Tensor([1, 0, 0, 0])`
print(result.state())
# should print `np.array([1, 0, 0, 0])`