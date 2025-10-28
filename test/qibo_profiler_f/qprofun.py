# qibo_profiler.py (版本 2)

import time
import psutil
import platform
import cProfile
import io
import pstats
from datetime import datetime
import hashlib
import numpy as np

import qibo
from qibo.models import Circuit

backend_configs = {
    "numpy": {"backend_name": "numpy", "platform_name": None},
    "qibojit (numba)": {"backend_name": "qibojit", "platform_name": "numba"},
    "qibotn (qutensornet)": {"backend_name": "qibotn", "platform_name": "qutensornet"},
    "qiboml (jax)": {"backend_name": "qiboml", "platform_name": "jax"},
    "qiboml (pytorch)": {"backend_name": "qiboml", "platform_name": "pytorch"},
    "qiboml (tensorflow)": {"backend_name": "qiboml", "platform_name": "tensorflow"},
    "qulacs": {"backend_name": "qulacs", "platform_name": None}
}
# --- 新增：全局缓存用于存储基准态矢量 ---
_BENCHMARK_STATE_CACHE = {}

def _get_benchmark_state(circuit: Circuit):
    """
    获取电路的基准态矢量。如果已缓存则直接返回，否则使用qibojit计算并缓存。
    """
    # 1. 序列化电路并计算哈希值作为唯一ID
    circuit_qasm = circuit.to_qasm()
    circuit_hash = hashlib.sha256(circuit_qasm.encode()).hexdigest()

    # 2. 检查缓存
    if circuit_hash in _BENCHMARK_STATE_CACHE:
        return _BENCHMARK_STATE_CACHE[circuit_hash]

    # 3. 如果不在缓存中，则计算
    print(" (首次计算该电路的基准态矢量，使用 qibojit...)")
    
    # 获取当前后端配置
    current_backend = qibo.get_backend()
    current_backend_name = current_backend.name
    current_platform = getattr(current_backend, 'platform', None)
    
    # 查找当前后端的配置
    original_backend_config = None
    for key, config in backend_configs.items():
        if config['backend_name'] == current_backend_name and config.get('platform_name') == current_platform:
            original_backend_config = config
            break
    
    if not original_backend_config:
        raise ValueError(f"未知的后端配置: {current_backend_name} with platform {current_platform}")
    
    if current_backend_name == "qibojit":
        # 如果当前就是基准后端，直接计算即可
        state = circuit(nshots=1).state()
        _BENCHMARK_STATE_CACHE[circuit_hash] = state
        return state
        
    try:
        # 安全地切换到基准后端
        qibo.set_backend("qibojit")
        state = circuit(nshots=1).state()
        _BENCHMARK_STATE_CACHE[circuit_hash] = state
        return state
    finally:
        # 无论成功或失败，都确保切换回原始后端
        qibo.set_backend(original_backend_config['backend_name'], platform=original_backend_config.get('platform_name'))

def _get_system_info():
    """Helper function: Collect system and environment information"""
    process = psutil.Process()
    return {
        "analysis_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "qibo_backend": qibo.get_backend(),
        "qibo_version": qibo.__version__,
        "python_version": platform.python_version(),
        "operating_system": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "cpu": platform.processor(),
        "cpu_core_counts_physical_logical": f"{psutil.cpu_count(logical=False)}/{psutil.cpu_count(logical=True)}",
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }

def _get_circuit_properties(circuit: Circuit):
    """Helper function: Analyze basic properties of the circuit"""
    gate_counts = {}
    for gate in circuit.queue:
        gate_name = gate.name.lower()  # Convert gate name to lowercase for consistency
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    return {
        "number_of_qubits": circuit.nqubits,
        "circuit_depth": circuit.depth,
        "total_number_of_gates": len(circuit.queue),
        "gate_type_statistics": gate_counts
    }

def profile_circuit(circuit: Circuit, n_runs: int = 1, mode: str = 'basic', calculate_fidelity: bool = False):
    """
    Analyze the performance of a Qibo circuit.

    Args:
        circuit (qibo.models.Circuit): The Qibo circuit object to be analyzed.
        n_runs (int): Number of times to run the circuit for calculating average performance.
        mode (str): Analysis mode, 'basic' or 'detailed'.
        calculate_fidelity (bool): Whether to calculate fidelity with the qibojit benchmark.

    Returns:
        dict: A dictionary containing a detailed performance report.
    """
    # 初始化报告字典，包含电路属性、系统信息和详细指标
    report = {
        "overview": {},
        "circuit_properties": _get_circuit_properties(circuit),
        "environment_snapshot": _get_system_info(),
        "detailed_metrics": {"runtime_list_seconds": []},
    }

    # 初始化总运行时间和峰值内存使用
    total_wall_time = 0
    peak_mem_usage = 0
    
    # 获取当前进程信息
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)  # 运行前的内存使用量（MB）
    cpu_start_times = process.cpu_times()  # 运行前的 CPU 时间

    for _ in range(n_runs):
        start_time = time.perf_counter()  # 记录开始时间
        result = circuit(nshots=1)  # 执行电路
        end_time = time.perf_counter()  # 记录结束时间

        run_time = end_time - start_time  # 计算单次运行时间
        total_wall_time += run_time  # 累加总运行时间
        report["detailed_metrics"]["runtime_list_seconds"].append(run_time)  # 记录每次运行时间

    # 获取运行后的 CPU 时间和内存使用量
    cpu_end_times = process.cpu_times()
    mem_after = process.memory_info().rss / (1024**2)  # 运行后的内存使用量（MB）
    peak_mem_usage = mem_after  # 峰值内存使用量

    # 计算平均运行时间
    avg_wall_time = total_wall_time / n_runs
    report["overview"]["runtime_avg_seconds"] = avg_wall_time
    report["overview"]["peak_memory_mb"] = peak_mem_usage  # 记录峰值内存使用量
        
    # 记录 CPU 时间的详细信息
    report["detailed_metrics"]["cpu_time_seconds"] = {
        "user": cpu_end_times.user - cpu_start_times.user,
        "system": cpu_end_times.system - cpu_start_times.system,
    }
    report["detailed_metrics"]["memory_details"] = {
        "initial_memory_mb": mem_before,
        "final_memory_mb": mem_after,
        "memory_increase_mb": mem_after - mem_before
    }
    # 计算总 CPU 时间并计算平均 CPU 利用率
    total_cpu_time = (cpu_end_times.user - cpu_start_times.user) + (cpu_end_times.system - cpu_start_times.system)
    if total_wall_time > 0:
        report["overview"]["average_cpu_utilization_percent"] = (total_cpu_time / total_wall_time) * 100 / psutil.cpu_count()

    # 如果需要计算保真度
    if calculate_fidelity:
        current_backend = qibo.get_backend()
        if current_backend == "qibojit":
            fidelity = 1.0
            report["overview"]["notes"] = "The current backend is the benchmark backend (qibojit)."
        else:
            benchmark_state_vector = _get_benchmark_state(circuit)
            fidelity = np.abs(np.vdot(benchmark_state_vector, result.state()))**2
        
        report["overview"]["fidelity"] = float(fidelity)

    if mode == 'detailed':  # 如果模式为 'detailed'，则启用详细的性能分析
        pr = cProfile.Profile()  # 创建一个性能分析器实例
        pr.enable()  # 启用性能分析器
        circuit(nshots=1)  # 执行电路模拟，nshots=1 表示执行一次
        pr.disable()  # 禁用性能分析器
        s = io.StringIO()  # 创建一个字符串IO对象，用于存储分析结果
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')  # 创建性能统计对象，并按累计时间排序
        ps.print_stats(10)  # 打印前10个最耗时的函数
        report["deep_profile"] = {"cprofile_hot_functions": s.getvalue().splitlines()}  # 将分析结果存储到报告中

    return report
    

# ... (format_report_for_print 函数保持不变) ...
def format_report_for_print(report: dict):
    """将报告字典格式化为易于阅读的字符串"""
    output = ""
    for section, data in report.items():
        output += f"\n--- {section} ---\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                     output += f"{key}: {value}\n"
                elif isinstance(value, dict):
                    output += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        output += f"  - {sub_key}: {sub_value}\n"
                else:
                    output += f"{key}: {value}\n"
        else:
             output += f"{data}\n"
    return output
def generate_markdown_report(report):
    """将性能报告转换为Markdown格式的字符串"""
    output = []

    def append_section(title, content):
        output.append(f"## {title}\n")
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, dict):
                    append_section(key, value)
                else:
                    output.append(f"**{key}**: {value}\n")
        elif isinstance(content, list):
            for item in content:
                output.append(f"- {item}\n")
        else:
            output.append(f"{content}\n")
        output.append("\n")
    
    append_section("性能分析报告", report)
    
    return "".join(output)
def save_report_to_markdown(report, filename):
    """将性能报告保存为Markdown文件"""
    markdown_content = generate_markdown_report(report)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"Markdown报告已保存到: {filename}")

if __name__ == '__main__':
    from qibo import gates
    n_qubits = 12
    c = Circuit(n_qubits)
    for q in range(n_qubits):
        c.add(gates.H(q))
    for q in range(0, n_qubits - 1, 2):
        c.add(gates.CNOT(q, q+1))
    c.add(gates.M(*range(n_qubits)))

    # --- 使用示例 ---
    # 假设我们想测试 numpy 后端的性能和准确度
    qibo.set_backend("qiboml", platform="pytorch")
    print(f"当前后端: {qibo.get_backend()}")

    # 第一次调用，会计算并缓存基准
    report1 = profile_circuit(c, n_runs=1, calculate_fidelity=True)
    print(format_report_for_print(report1))

    # 第二次调用同一个电路，将直接使用缓存，速度更快
    print("\n--- 第二次分析同一个电路 ---")
    report2 = profile_circuit(c, n_runs=1, calculate_fidelity=True)
    print(format_report_for_print(report2)) # 注意这次将不会打印 "首次计算..."