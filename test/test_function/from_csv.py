from qibo_profiler import profile_circuit
from qibo import Circuit

def from_csv(circuit: Circuit, n_runs=1, mode='basic', calculate_fidelity=True):
    