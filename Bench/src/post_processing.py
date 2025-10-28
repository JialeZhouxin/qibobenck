"""
结果后处理与可视化模块

这个模块提供了基准测试结果的分析和可视化功能。
"""

import os
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_results(results_list: List[Any], output_dir: str, repeat: int = 1):
    """分析基准测试结果并生成可视化图表"""
    if not results_list:
        print("No results to analyze")
        return

    # 创建DataFrame
    data = []
    for result in results_list:
        # 对于多次运行，只使用汇总结果（第一个结果）
        if repeat > 1 and result.run_id > 1:
            continue
            
        data.append({
            "simulator": result.simulator,
            "backend": result.backend,
            "circuit_name": result.circuit_name,
            "n_qubits": result.n_qubits,
            "wall_time_sec": result.wall_time_mean if result.wall_time_mean else result.wall_time_sec,
            "wall_time_std": result.wall_time_std if result.wall_time_std else 0.0,
            "cpu_time_sec": result.cpu_time_mean if result.cpu_time_mean else result.cpu_time_sec,
            "cpu_time_std": result.cpu_time_std if result.cpu_time_std else 0.0,
            "peak_memory_mb": result.memory_mean if result.memory_mean else result.peak_memory_mb,
            "memory_std": result.memory_std if result.memory_std else 0.0,
            "cpu_utilization_percent": result.cpu_utilization_percent,
            "state_fidelity": result.fidelity_mean if result.fidelity_mean else result.state_fidelity,
            "fidelity_std": result.fidelity_std if result.fidelity_std else 0.0,
            "repeat": repeat,
            # 添加电路信息字段
            "circuit_depth": getattr(result, 'circuit_info', {}).get("circuit_depth", None),
            "total_gates": getattr(result, 'circuit_info', {}).get("total_gates", None),
            "circuit_summary": getattr(result, 'circuit_info', {}).get("circuit_summary", ""),
        })

    df = pd.DataFrame(data)
    df["runner_id"] = df["simulator"] + "-" + df["backend"]

    # 保存原始数据
    csv_path = os.path.join(output_dir, "raw_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw results saved to {csv_path}")
    
    # 如果多次运行，保存详细运行数据
    if repeat > 1:
        detailed_data = []
        for result in results_list:
            detailed_data.append({
                "simulator": result.simulator,
                "backend": result.backend,
                "circuit_name": result.circuit_name,
                "n_qubits": result.n_qubits,
                "run_id": result.run_id,
                "wall_time_sec": result.wall_time_sec,
                "cpu_time_sec": result.cpu_time_sec,
                "peak_memory_mb": result.peak_memory_mb,
                "cpu_utilization_percent": result.cpu_utilization_percent,
                "state_fidelity": result.state_fidelity,
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = os.path.join(output_dir, "detailed_runs.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed run data saved to {detailed_csv_path}")

    # 设置图表样式
    plt.style.use("default")
    sns.set_palette("husl")

    # 生成图表
    _plot_fidelity_check(df, output_dir, repeat)
    _plot_wall_time_scaling(df, output_dir, repeat)
    _plot_memory_scaling(df, output_dir, repeat)
    _plot_cpu_time_scaling(df, output_dir, repeat)
    _plot_cpu_utilization(df, output_dir, repeat)
    
    # 如果多次运行，生成额外的统计图表
    if repeat > 1:
        _plot_run_stability(df, output_dir)
        _plot_confidence_intervals(df, output_dir)

    print(f"All plots saved to {output_dir}")


def _plot_fidelity_check(df: pd.DataFrame, output_dir: str, repeat: int = 1):
    """绘制保真度检查图表"""
    plt.figure(figsize=(12, 7))
    
    if repeat > 1:
        # 多次运行时，使用误差条显示标准差
        for runner_id in df["runner_id"].unique():
            runner_data = df[df["runner_id"] == runner_id]
            plt.errorbar(
                runner_data["n_qubits"],
                runner_data["state_fidelity"],
                yerr=runner_data["fidelity_std"],
                marker="o",
                label=runner_id,
                capsize=5
            )
    else:
        sns.barplot(data=df, x="runner_id", y="state_fidelity", hue="n_qubits")
    
    plt.title("State Fidelity vs Golden Standard")
    plt.ylabel("Fidelity")
    plt.xlabel("Simulator" if repeat == 1 else "Number of Qubits")
    
    if repeat == 1:
        plt.xticks(rotation=45)
    else:
        plt.legend()
    
    plt.tight_layout()

    output_path = os.path.join(output_dir, "fidelity.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_wall_time_scaling(df: pd.DataFrame, output_dir: str, repeat: int = 1):
    """绘制墙上时间扩展图表"""
    plt.figure(figsize=(12, 7))
    
    if repeat > 1:
        # 多次运行时，使用误差条显示标准差
        for runner_id in df["runner_id"].unique():
            runner_data = df[df["runner_id"] == runner_id]
            plt.errorbar(
                runner_data["n_qubits"],
                runner_data["wall_time_sec"],
                yerr=runner_data["wall_time_std"],
                marker="o",
                label=runner_id,
                capsize=5
            )
    else:
        sns.lineplot(data=df, x="n_qubits", y="wall_time_sec", hue="runner_id", marker="o")
    
    plt.title("Wall Clock Time vs Number of Qubits")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # 使用对数刻度以便更好地显示扩展性
    
    if repeat > 1:
        plt.legend()
    
    plt.tight_layout()

    output_path = os.path.join(output_dir, "wall_time_scaling.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_memory_scaling(df: pd.DataFrame, output_dir: str, repeat: int = 1):
    """绘制内存扩展图表"""
    plt.figure(figsize=(12, 7))
    
    if repeat > 1:
        # 多次运行时，使用误差条显示标准差
        for runner_id in df["runner_id"].unique():
            runner_data = df[df["runner_id"] == runner_id]
            plt.errorbar(
                runner_data["n_qubits"],
                runner_data["peak_memory_mb"],
                yerr=runner_data["memory_std"],
                marker="o",
                label=runner_id,
                capsize=5
            )
    else:
        sns.lineplot(data=df, x="n_qubits", y="peak_memory_mb", hue="runner_id", marker="o")
    
    plt.title("Peak Memory Usage vs Number of Qubits")
    plt.ylabel("Memory (MB)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # 使用对数刻度以便更好地显示扩展性
    
    if repeat > 1:
        plt.legend()
    
    plt.tight_layout()

    output_path = os.path.join(output_dir, "memory_scaling.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cpu_time_scaling(df: pd.DataFrame, output_dir: str, repeat: int = 1):
    """绘制CPU时间扩展图表"""
    plt.figure(figsize=(12, 7))
    
    if repeat > 1:
        # 多次运行时，使用误差条显示标准差
        for runner_id in df["runner_id"].unique():
            runner_data = df[df["runner_id"] == runner_id]
            plt.errorbar(
                runner_data["n_qubits"],
                runner_data["cpu_time_sec"],
                yerr=runner_data["cpu_time_std"],
                marker="o",
                label=runner_id,
                capsize=5
            )
    else:
        sns.lineplot(data=df, x="n_qubits", y="cpu_time_sec", hue="runner_id", marker="o")
    
    plt.title("CPU Time vs Number of Qubits")
    plt.ylabel("CPU Time (seconds)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # 使用对数刻度以便更好地显示扩展性
    
    if repeat > 1:
        plt.legend()
    
    plt.tight_layout()

    output_path = os.path.join(output_dir, "cpu_time_scaling.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cpu_utilization(df: pd.DataFrame, output_dir: str, repeat: int = 1):
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


def _plot_run_stability(df: pd.DataFrame, output_dir: str):
    """绘制运行稳定性图表（标准差与均值的关系）"""
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x="wall_time_sec", y="wall_time_std", hue="runner_id", size="n_qubits")
    plt.title("Execution Time Stability (Standard Deviation vs Mean)")
    plt.ylabel("Standard Deviation (seconds)")
    plt.xlabel("Mean Execution Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "execution_stability.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_confidence_intervals(df: pd.DataFrame, output_dir: str):
    """绘制置信区间图表"""
    plt.figure(figsize=(12, 7))
    
    # 为每个模拟器创建误差条图
    for runner_id in df["runner_id"].unique():
        runner_data = df[df["runner_id"] == runner_id]
        plt.errorbar(
            runner_data["n_qubits"],
            runner_data["wall_time_sec"],
            yerr=runner_data["wall_time_std"],
            marker="o",
            label=runner_id,
            capsize=5
        )
    
    plt.title("Execution Time with Confidence Intervals")
    plt.ylabel("Execution Time (seconds)")
    plt.xlabel("Number of Qubits")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "confidence_intervals.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: str, repeat: int = 1):
    """生成摘要报告"""
    report_path = os.path.join(output_dir, "summary_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 量子模拟器基准测试报告\n\n")
        f.write(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"重复运行次数: {repeat}\n\n")

        # 基本统计
        f.write("## 基本统计\n\n")
        f.write(f"- 总测试次数: {len(df)}\n")
        f.write(f"- 测试的模拟器: {', '.join(df['runner_id'].unique())}\n")
        f.write(f"- 测试的电路: {', '.join(df['circuit_name'].unique())}\n")
        f.write(
            f"- 量子比特数范围: {df['n_qubits'].min()} - {df['n_qubits'].max()}\n\n"
        )

        # 电路信息
        if "circuit_depth" in df.columns and not df["circuit_depth"].isna().all():
            f.write("## 电路信息\n\n")
            
            # 按电路和量子比特数分组显示电路信息
            circuit_info = df.groupby(["circuit_name", "n_qubits"]).agg({
                "circuit_depth": "first",
                "total_gates": "first"
            }).reset_index()
            
            f.write("### 电路复杂度\n\n")
            f.write("| 电路名称 | 量子比特数 | 电路深度 | 门总数 |\n")
            f.write("|---------|-----------|---------|--------|\n")
            
            for _, row in circuit_info.iterrows():
                f.write(f"| {row['circuit_name']} | {row['n_qubits']} | {row['circuit_depth']} | {row['total_gates']} |\n")
            
            f.write("\n")
            
            # 如果有电路摘要，添加一个示例
            if "circuit_summary" in df.columns:
                sample_summary = df["circuit_summary"].dropna().iloc[0] if not df["circuit_summary"].dropna().empty else ""
                if sample_summary:
                    f.write("### 电路摘要示例\n\n")
                    f.write("```\n")
                    f.write(sample_summary)
                    f.write("\n```\n\n")

        # 性能指标
        f.write("## 性能指标\n\n")

        # 最快的模拟器
        fastest = df.loc[df["wall_time_sec"].idxmin()]
        f.write(f"### 最快执行\n")
        f.write(f"- 模拟器: {fastest['runner_id']}\n")
        f.write(f"- 电路: {fastest['circuit_name']} ({fastest['n_qubits']} qubits)\n")
        if repeat > 1:
            f.write(f"- 平均时间: {fastest['wall_time_sec']:.4f} ± {fastest['wall_time_std']:.4f} 秒\n")
        else:
            f.write(f"- 时间: {fastest['wall_time_sec']:.4f} 秒\n")
        f.write("\n")

        # 内存使用最少的模拟器
        min_memory = df.loc[df["peak_memory_mb"].idxmin()]
        f.write(f"### 内存使用最少\n")
        f.write(f"- 模拟器: {min_memory['runner_id']}\n")
        f.write(
            f"- 电路: {min_memory['circuit_name']} ({min_memory['n_qubits']} qubits)\n"
        )
        if repeat > 1:
            f.write(f"- 平均内存: {min_memory['peak_memory_mb']:.2f} ± {min_memory['memory_std']:.2f} MB\n")
        else:
            f.write(f"- 内存: {min_memory['peak_memory_mb']:.2f} MB\n")
        f.write("\n")

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

        # 如果多次运行，添加稳定性分析
        if repeat > 1:
            f.write("## 稳定性分析\n\n")
            
            # 计算变异系数
            df["cv"] = df["wall_time_std"] / df["wall_time_sec"]
            most_stable = df.loc[df["cv"].idxmin()]
            least_stable = df.loc[df["cv"].idxmax()]
            
            f.write(f"### 最稳定执行\n")
            f.write(f"- 模拟器: {most_stable['runner_id']}\n")
            f.write(f"- 电路: {most_stable['circuit_name']} ({most_stable['n_qubits']} qubits)\n")
            f.write(f"- 变异系数: {most_stable['cv']:.4f}\n\n")
            
            f.write(f"### 最不稳定执行\n")
            f.write(f"- 模拟器: {least_stable['runner_id']}\n")
            f.write(f"- 电路: {least_stable['circuit_name']} ({least_stable['n_qubits']} qubits)\n")
            f.write(f"- 变异系数: {least_stable['cv']:.4f}\n\n")

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
        if repeat > 1:
            f.write("4. 对于需要稳定性能的应用，选择变异系数最小的模拟器\n")

    print(f"Summary report saved to {report_path}")
