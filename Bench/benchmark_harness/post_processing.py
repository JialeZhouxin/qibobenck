"""
结果后处理与可视化模块

这个模块提供了基准测试结果的分析和可视化功能。
"""

import os
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_results(results_list: List[Any], output_dir: str):
    """分析基准测试结果并生成可视化图表"""
    if not results_list:
        print("No results to analyze")
        return

    # 创建DataFrame
    data = []
    for result in results_list:
        data.append(
            {
                "simulator": result.simulator,
                "backend": result.backend,
                "circuit_name": result.circuit_name,
                "n_qubits": result.n_qubits,
                "wall_time_sec": result.wall_time_sec,
                "cpu_time_sec": result.cpu_time_sec,
                "peak_memory_mb": result.peak_memory_mb,
                "cpu_utilization_percent": result.cpu_utilization_percent,
                "state_fidelity": result.state_fidelity,
            }
        )

    df = pd.DataFrame(data)
    df["runner_id"] = df["simulator"] + "-" + df["backend"]

    # 保存原始数据
    csv_path = os.path.join(output_dir, "raw_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw results saved to {csv_path}")

    # 设置图表样式
    plt.style.use("default")
    sns.set_palette("husl")

    # 生成图表
    _plot_fidelity_check(df, output_dir)
    _plot_wall_time_scaling(df, output_dir)
    _plot_memory_scaling(df, output_dir)
    _plot_cpu_time_scaling(df, output_dir)
    _plot_cpu_utilization(df, output_dir)

    print(f"All plots saved to {output_dir}")


def _plot_fidelity_check(df: pd.DataFrame, output_dir: str):
    """绘制保真度检查图表"""
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="runner_id", y="state_fidelity", hue="n_qubits")
    plt.title("State Fidelity vs Golden Standard")
    plt.ylabel("Fidelity")
    plt.xlabel("Simulator")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "fidelity.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_wall_time_scaling(df: pd.DataFrame, output_dir: str):
    """绘制墙上时间扩展图表"""
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="n_qubits", y="wall_time_sec", hue="runner_id", marker="o")
    plt.title("Wall Clock Time vs Number of Qubits")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # 使用对数刻度以便更好地显示扩展性
    plt.tight_layout()

    output_path = os.path.join(output_dir, "wall_time_scaling.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_memory_scaling(df: pd.DataFrame, output_dir: str):
    """绘制内存扩展图表"""
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="n_qubits", y="peak_memory_mb", hue="runner_id", marker="o")
    plt.title("Peak Memory Usage vs Number of Qubits")
    plt.ylabel("Memory (MB)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # 使用对数刻度以便更好地显示扩展性
    plt.tight_layout()

    output_path = os.path.join(output_dir, "memory_scaling.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cpu_time_scaling(df: pd.DataFrame, output_dir: str):
    """绘制CPU时间扩展图表"""
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="n_qubits", y="cpu_time_sec", hue="runner_id", marker="o")
    plt.title("CPU Time vs Number of Qubits")
    plt.ylabel("CPU Time (seconds)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # 使用对数刻度以便更好地显示扩展性
    plt.tight_layout()

    output_path = os.path.join(output_dir, "cpu_time_scaling.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cpu_utilization(df: pd.DataFrame, output_dir: str):
    """绘制CPU利用率图表"""
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="runner_id", y="cpu_utilization_percent", hue="n_qubits")
    plt.title("CPU Utilization")
    plt.ylabel("CPU %")
    plt.xlabel("Simulator")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "cpu_utilization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """生成摘要报告"""
    report_path = os.path.join(output_dir, "summary_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 量子模拟器基准测试报告\n\n")
        f.write(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 基本统计
        f.write("## 基本统计\n\n")
        f.write(f"- 总测试次数: {len(df)}\n")
        f.write(f"- 测试的模拟器: {', '.join(df['runner_id'].unique())}\n")
        f.write(f"- 测试的电路: {', '.join(df['circuit_name'].unique())}\n")
        f.write(
            f"- 量子比特数范围: {df['n_qubits'].min()} - {df['n_qubits'].max()}\n\n"
        )

        # 性能指标
        f.write("## 性能指标\n\n")

        # 最快的模拟器
        fastest = df.loc[df["wall_time_sec"].idxmin()]
        f.write(f"### 最快执行\n")
        f.write(f"- 模拟器: {fastest['runner_id']}\n")
        f.write(f"- 电路: {fastest['circuit_name']} ({fastest['n_qubits']} qubits)\n")
        f.write(f"- 时间: {fastest['wall_time_sec']:.4f} 秒\n\n")

        # 内存使用最少的模拟器
        min_memory = df.loc[df["peak_memory_mb"].idxmin()]
        f.write(f"### 内存使用最少\n")
        f.write(f"- 模拟器: {min_memory['runner_id']}\n")
        f.write(
            f"- 电路: {min_memory['circuit_name']} ({min_memory['n_qubits']} qubits)\n"
        )
        f.write(f"- 内存: {min_memory['peak_memory_mb']:.2f} MB\n\n")

        # 平均保真度
        avg_fidelity = (
            df.groupby("runner_id")["state_fidelity"]
            .mean()
            .sort_values(ascending=False)
        )
        f.write("### 平均保真度排名\n")
        for runner_id, fidelity in avg_fidelity.items():
            f.write(f"- {runner_id}: {fidelity:.4f}\n")
        f.write("\n")

        # 量子比特数扩展性分析
        f.write("## 扩展性分析\n\n")
        scalability = (
            df.groupby(["runner_id", "n_qubits"])["wall_time_sec"].mean().unstack()
        )
        f.write("### 执行时间随量子比特数的变化\n")
        f.write("```\n")
        f.write(scalability.to_string())
        f.write("\n```\n\n")

        # 建议
        f.write("## 建议\n\n")
        f.write("基于以上结果，建议:\n")
        f.write("1. 对于小型量子电路，选择执行时间最短的模拟器\n")
        f.write("2. 对于大型量子电路，优先考虑内存使用效率\n")
        f.write("3. 在需要高精度计算时，选择保真度最高的模拟器\n")

    print(f"Summary report saved to {report_path}")
