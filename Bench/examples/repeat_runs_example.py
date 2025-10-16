#!/usr/bin/env python3
"""
重复运行功能使用示例

这个脚本演示了如何使用新的重复运行功能来提高基准测试的可靠性。
"""

import os
import sys
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """演示重复运行功能的使用"""
    
    print("重复运行功能使用示例")
    print("=" * 50)
    
    # 示例1：基本重复运行
    print("\n1. 基本重复运行（5次）:")
    print("python run_benchmarks.py --repeat 5 --circuits qft --qubits 2 3 4")
    
    # 示例2：带预热运行的重复测试
    print("\n2. 带预热运行的重复测试:")
    print("python run_benchmarks.py --repeat 10 --warmup-runs 2 --simulators qibo-numpy qiskit-aer_simulator")
    
    # 示例3：启用统计分析
    print("\n3. 启用统计分析:")
    print("python run_benchmarks.py --repeat 5 --statistical-analysis --verbose")
    
    # 示例4：完整测试示例
    print("\n4. 完整测试示例:")
    print("python run_benchmarks.py \\")
    print("  --circuits qft \\")
    print("  --qubits 2 3 4 5 \\")
    print("  --simulators qibo-numpy qiskit-aer_simulator \\")
    print("  --repeat 5 \\")
    print("  --warmup-runs 2 \\")
    print("  --output-dir my_results \\")
    print("  --verbose")
    
    print("\n" + "=" * 50)
    print("新功能说明:")
    print("- --repeat N: 指定每个电路的重复运行次数")
    print("- --warmup-runs N: 指定预热运行次数（不计入统计）")
    print("- --statistical-analysis: 启用统计分析")
    print("\n输出变化:")
    print("- raw_results.csv: 包含统计列（均值、标准差等）")
    print("- detailed_runs.csv: 新增文件，包含每次运行的详细数据")
    print("- 新增图表: 执行稳定性图、置信区间图")
    print("- 报告增强: 包含稳定性分析和统计信息")
    
    # 演示实际运行（如果环境允许）
    print("\n" + "=" * 50)
    print("是否要运行一个简单的演示？(y/n): ", end="")
    
    try:
        choice = input().lower().strip()
        if choice == 'y':
            run_demo()
        else:
            print("演示已取消。")
    except KeyboardInterrupt:
        print("\n演示已取消。")
    except Exception as e:
        print(f"运行演示时出错: {e}")


def run_demo():
    """运行一个简单的演示"""
    print("\n开始运行简单演示...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python", "run_benchmarks.py",
            "--circuits", "qft",
            "--qubits", "2",
            "--simulators", "qibo-qibojit",
            "--repeat", "3",
            "--warmup-runs", "1",
            "--output-dir", temp_dir,
            "--verbose"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("这可能需要一些时间...")
        
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ 演示运行成功！")
                
                # 检查输出文件
                results_dirs = [d for d in os.listdir(temp_dir) if d.startswith("benchmark_")]
                if results_dirs:
                    benchmark_dir = os.path.join(temp_dir, results_dirs[0])
                    print(f"结果保存在: {benchmark_dir}")
                    
                    # 列出生成的文件
                    files = os.listdir(benchmark_dir)
                    print("生成的文件:")
                    for file in files:
                        print(f"  - {file}")
                    
                    # 显示部分结果
                    if "raw_results.csv" in files:
                        import pandas as pd
                        df = pd.read_csv(os.path.join(benchmark_dir, "raw_results.csv"))
                        print("\n结果摘要:")
                        print(df[['runner_id', 'wall_time_sec', 'wall_time_std', 'state_fidelity']].to_string())
                    
                    if "summary_report.md" in files:
                        print("\n报告摘要:")
                        with open(os.path.join(benchmark_dir, "summary_report.md"), 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines[:20]:  # 显示前20行
                                print(line.rstrip())
                            if len(lines) > 20:
                                print("...")
                
            else:
                print("❌ 演示运行失败:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("❌ 演示运行超时")
        except Exception as e:
            print(f"❌ 运行演示时出错: {e}")


if __name__ == "__main__":
    main()