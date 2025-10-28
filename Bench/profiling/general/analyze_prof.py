#!/usr/bin/env python3
"""
分析 prof 文件的脚本
"""

import pstats
import sys
from collections import defaultdict

def analyze_prof_file(prof_file_path):
    """分析 prof 文件并返回性能数据"""
    try:
        # 创建 Stats 对象
        stats = pstats.Stats(prof_file_path)

        # 获取所有函数的性能数据
        stats.sort_stats('cumulative')  # 按累计时间排序

        print(f"成功加载性能分析文件: {prof_file_path}")
        print(f"总函数调用数: {len(stats.stats)}")

        return stats
    except Exception as e:
        print(f"无法分析文件 {prof_file_path}: {e}")
        return None

def get_top_functions(stats, limit=20):
    """获取最耗时的函数列表"""
    top_functions = []

    for func_info in stats.stats[:limit]:
        func_name = func_info[2]  # 函数名
        cc = func_info[3]        # 累计调用次数
        nc = func_info[0]        # 调用次数
        tt = func_info[2]        # 总时间
        ct = func_info[4]        # 调用时间（累积）

        top_functions.append({
            'function': func_name,
            'cumulative_time': cc,
            'call_count': nc,
            'total_time': tt,
            'cumulative_time_per_call': cc / nc if nc > 0 else 0,
            'total_time_per_call': ct / nc if nc > 0 else 0
        })

    return top_functions

def analyze_performance_bottlenecks(stats):
    """分析性能瓶颈"""
    print("\n" + "="*80)
    print("性能瓶颈分析")
    print("="*80)

    # 按累计时间排序的函数
    stats.sort_stats('cumulative')

    print("\n最耗时的函数 (按累计时间排序):")
    print("-" * 80)
    print(f"{'排名':<5} {'函数名':<50} {'累计时间':<12} {'调用次数':<10} {'每次调用时间':<15}")
    print("-" * 80)

    for i, (func, (cc, nc, tt, ct, callers)) in enumerate(stats.stats[:15], 1):
        func_name = func if isinstance(func, str) else str(func)
        if len(func_name) > 45:
            func_name = func_name[:42] + "..."

        print(f"{i:<5} {func_name:<50} {cc:<12.4f} {nc:<10} {ct/nc if nc > 0 else 0:<15.6f}")

def analyze_by_module(stats):
    """按模块分析性能"""
    print("\n" + "="*80)
    print("按模块分析性能")
    print("="*80)

    module_stats = defaultdict(lambda: {'time': 0, 'calls': 0, 'functions': []})

    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        module_name = func[0] if isinstance(func, tuple) else 'unknown'
        module_stats[module_name]['time'] += cc
        module_stats[module_name]['calls'] += nc
        module_stats[module_name]['functions'].append((func, cc))

    # 按时间排序模块
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['time'], reverse=True)

    print("\n各模块性能排名:")
    print("-" * 80)
    print(f"{'模块':<60} {'累计时间':<12} {'调用次数':<10} {'函数数':<10}")
    print("-" * 80)

    for i, (module, stats) in enumerate(sorted_modules[:10], 1):
        print(f"{i:<5} {module:<60} {stats['time']:<12.4f} {stats['calls']:<10} {len(stats['functions']):<10}")

def identify_optimization_opportunities(stats):
    """识别优化机会"""
    print("\n" + "="*80)
    print("优化机会分析")
    print("="*80)

    # 按单次调用时间排序
    stats.sort_stats('time')

    print("\n单次调用最耗时的函数:")
    print("-" * 80)
    print(f"{'排名':<5} {'函数名':<50} {'单次时间':<12} {'调用次数':<10}")
    print("-" * 80)

    optimization_targets = []
    for i, (func, (cc, nc, tt, ct, callers)) in enumerate(stats.stats[:10], 1):
        func_name = func if isinstance(func, str) else str(func)
        if len(func_name) > 45:
            func_name = func_name[:42] + "..."

        time_per_call = ct / nc if nc > 0 else 0
        print(f"{i:<5} {func_name:<50} {time_per_call:<12.6f} {nc:<10}")

        if time_per_call > 0.001:  # 超过1ms的函数
            optimization_targets.append({
                'function': func_name,
                'time_per_call': time_per_call,
                'call_count': nc,
                'potential_savings': time_per_call * nc
            })

    return optimization_targets

def generate_optimization_suggestions(optimization_targets):
    """生成优化建议"""
    print("\n" + "="*80)
    print("具体优化建议")
    print("="*80)

    if not optimization_targets:
        print("没有发现明显的优化机会")
        return

    print("\n高优先级优化目标:")
    for i, target in enumerate(optimization_targets[:5], 1):
        func_name = target['function']
        time_per_call = target['time_per_call']
        call_count = target['call_count']
        potential_savings = target['potential_savings']

        print(f"\n{i}. {func_name}")
        print(f"   单次调用时间: {time_per_call:.4f}s")
        print(f"   调用次数: {call_count}")
        print(f"   潜在节省时间: {potential_savings:.2f}s")

        # 根据函数名提供具体建议
        if 'numpy' in func_name.lower():
            print("   建议: 检查是否有不必要的数组复制或低效的 numpy 操作")
            print("   优化: 使用向量化操作，避免循环中的数组操作")
        elif 'matplotlib' in func_name.lower():
            print("   建议: 检查是否有不必要的绘图操作或复杂的渲染")
            print("   优化: 简化图形，使用更高效的绘图后端")
        elif 'qibo' in func_name.lower():
            print("   建议: 检查量子电路的复杂度和模拟器配置")
            print("   优化: 考虑使用 qibojit 后端，简化电路结构")
        elif 'pandas' in func_name.lower():
            print("   建议: 检查数据处理逻辑，避免不必要的操作")
            print("  优化: 使用更高效的数据处理方法，避免逐行操作")
        else:
            print("   建议: 分析函数内部逻辑，寻找性能瓶颈")
            print("   优化: 考虑算法优化、缓存结果或使用更高效的库")

def main():
    prof_file_path = r"E:\qiboenv\Bench\profiling\general\my_programg20.prof"

    print("开始分析性能分析文件...")
    print(f"文件路径: {prof_file_path}")

    # 分析性能数据
    stats = analyze_prof_file(prof_file_path)

    if not stats:
        print("分析失败，退出")
        return

    # 获取基本信息
    print(f"\n总执行时间: {stats.total_tt:.4f}s")
    print(f"原文件执行时间: {stats.total_tt:.4f}s")

    # 分析性能瓶颈
    analyze_performance_bottlenecks(stats)

    # 按模块分析
    analyze_by_module(stats)

    # 识别优化机会
    optimization_targets = identify_optimization_opportunities(stats)

    # 生成优化建议
    generate_optimization_suggestions(optimization_targets)

    # 保存分析结果
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)

if __name__ == "__main__":
    main()