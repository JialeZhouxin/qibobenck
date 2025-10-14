"""Manual test script for abstractions."""

import sys
import traceback
from benchmark_harness.abstractions import BenchmarkResult, SimulatorInterface, BenchmarkCircuit
import numpy as np


def test_benchmark_result():
    """Test BenchmarkResult creation."""
    try:
        final_state = np.array([1.0, 0.0])
        result = BenchmarkResult(
            simulator="qibo",
            backend="numpy", 
            circuit_name="test_circuit",
            n_qubits=1,
            wall_time_sec=0.1,
            cpu_time_sec=0.05,
            peak_memory_mb=10.0,
            cpu_utilization_percent=50.0,
            state_fidelity=1.0,
            final_state=final_state
        )
        
        assert result.simulator == "qibo"
        assert result.backend == "numpy"
        assert result.circuit_name == "test_circuit"
        assert result.n_qubits == 1
        assert result.wall_time_sec == 0.1
        assert result.cpu_time_sec == 0.05
        assert result.peak_memory_mb == 10.0
        assert result.cpu_utilization_percent == 50.0
        assert result.state_fidelity == 1.0
        np.testing.assert_array_equal(result.final_state, final_state)
        
        print("✓ BenchmarkResult creation test passed")
        return True
    except Exception as e:
        print(f"✗ BenchmarkResult creation test failed: {e}")
        traceback.print_exc()
        return False


def test_simulator_interface_abstract():
    """Test that SimulatorInterface cannot be instantiated."""
    try:
        interface = SimulatorInterface()
        print("✗ SimulatorInterface instantiation test failed - should raise TypeError")
        return False
    except TypeError:
        print("✓ SimulatorInterface instantiation test passed - correctly raises TypeError")
        return True
    except Exception as e:
        print(f"✗ SimulatorInterface instantiation test failed with unexpected error: {e}")
        return False


def test_benchmark_circuit_abstract():
    """Test that BenchmarkCircuit cannot be instantiated."""
    try:
        circuit = BenchmarkCircuit()
        print("✗ BenchmarkCircuit instantiation test failed - should raise TypeError")
        return False
    except TypeError:
        print("✓ BenchmarkCircuit instantiation test passed - correctly raises TypeError")
        return True
    except Exception as e:
        print(f"✗ BenchmarkCircuit instantiation test failed with unexpected error: {e}")
        return False


def main():
    """Run all tests."""
    print("Running manual tests for abstractions...")
    
    tests = [
        test_benchmark_result,
        test_simulator_interface_abstract,
        test_benchmark_circuit_abstract
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Stage 1 is complete.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
