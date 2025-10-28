#!/usr/bin/env python3
"""
分析 prof 文件的脚本 - 修正版本
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

def print_performance_summary(stats):
    """打印性能摘要"""
    print(f"\n总执行时间: {stats.total_tt:.4f}s")
    print(f"原文件执行时间: {stats.total_tt:.4f}s")
    print(f"函数调用总数: {len(stats.stats)}")

def analyze_performance_bottlenecks(stats):
    """分析性能瓶颈"""
    print("\n" + "="*80)
    print("性能瓶颈分析")
    print("="*80)

    # 按累计时间排序的函数
    stats.sort_stats('cumulative')

    print("\n最耗时的函数 (按累计时间排序):")
    print("-" * 80)
    print(f"{'排名':<5} {'函数名':<60} {'累计时间':<12} {'调用次数':<10} {'单次时间':<12}")
    print("-" * 80)

    # 获取统计数据
    func_stats = []
    for func_info in stats.stats:
        if len(func_info) >= 6:
            func_name = str(func_info[2])  # 函数名
            cc = func_info[3]  # 累计时间
            nc = func_info[0]  # 调用次数
            ct = func_info[4]  # 总时间

            func_stats.append({
                'function': func_name,
                'cumulative_time': cc,
                'call_count': nc,
                'total_time': ct,
                'time_per_call': ct / nc if nc > 0 else 0
            })

    # 按累计时间排序
    func_stats.sort(key=lambda x: x['cumulative_time'], reverse=True)

    for i, func_data in enumerate(func_stats[:15], 1):
        func_name = func_data['function']
        if len(func_name) > 55:
            func_name = func_name[:52] + "..."

        print(f"{i:<5} {func_name:<60} {func_data['cumulative_time']:<12.4f} {func_data['call_count']:<10} {func_data['time_per_call']:<12.6f}")

    return func_stats

def analyze_by_module(stats):
    """按模块分析性能"""
    print("\n" + "="*80)
    print("按模块分析性能")
    print("="*80)

    module_stats = defaultdict(lambda: {'time': 0, 'calls': 0, 'functions': []})

    for func_info in stats.stats:
        if len(func_info) >= 2:
            func_name = str(func_info[2])
            # 提取模块名
            if '.' in func_name:
                module_name = func_name.split('.')[0] + '.' + func_name.split('.')[1]
            else:
                module_name = 'unknown'

            if len(func_info) >= 6:
                cc = func_info[3]  # 累计时间
                nc = func_info[0]  # 调用次数
                module_stats[module_name]['time'] += cc
                module_stats[module_name]['calls'] += nc
                module_stats[module_name]['functions'].append(func_info)

    # 按时间排序模块
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['time'], reverse=True)

    print("\n各模块性能排名:")
    print("-" * 80)
    print(f"{'模块':<60} {'累计时间':<12} {'调用次数':<10} {'函数数':<10}")
    print("-" * 80)

    for i, (module, stats_data) in enumerate(sorted_modules[:10], 1):
        print(f"{i:<5} {module:<60} {stats_data['time']:<12.4f} {stats_data['calls']:<10} {len(stats_data['functions']):<10}")

def identify_optimization_opportunities(func_stats):
    """识别优化机会"""
    print("\n" + "="*80)
    print("优化机会分析")
    print("="*80)

    # 按单次调用时间排序
    func_stats_sorted = sorted(func_stats, key=lambda x: x['time_per_call'], reverse=True)

    print("\n单次调用最耗时的函数:")
    print("-" * 80)
    print(f"{'排名':<5} {'函数名':<60} {'单次时间':<12} {'调用次数':<10}")
    print("-" * 80)

    optimization_targets = []
    for i, func_data in enumerate(func_stats_sorted[:10], 1):
        func_name = func_data['function']
        if len(func_name) > 55:
            func_name = func_name[:52] + "..."

        time_per_call = func_data['time_per_call']
        call_count = func_data['call_count']
        potential_savings = time_per_call * call_count

        print(f"{i:<5} {func_name:<60} {time_per_call:<12.6f} {call_count:<10}")

        # 如果单次调用时间超过阈值，认为是优化目标
        if time_per_call > 0.001:  # 超过1ms的函数
            optimization_targets.append({
                'function': func_name,
                'time_per_call': time_per_call,
                'call_count': call_count,
                'potential_savings': potential_savings
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
        func_lower = func_name.lower()
        suggestions = []

        if 'numpy' in func_lower:
            suggestions = [
                "检查是否有不必要的数组复制操作",
                "使用向量化操作替代循环",
                "考虑使用更高效的numpy函数",
                "避免不必要的数据类型转换"
            ]
        elif 'matplotlib' in func_lower:
            suggestions = [
                "检查是否有不必要的图形更新",
                "简化图形复杂度",
                "使用更高效的绘图后端",
                "批量处理绘图操作"
            ]
        elif 'qibo' in func_lower:
            suggestions = [
                "检查量子电路的复杂度",
                "考虑使用 qibojit 后端",
                "简化电路结构",
                "优化门操作序列"
            ]
        elif 'pennylane' in func_lower:
            suggestions = [
                "检查量子电路的模拟配置",
                "优化梯度计算",
                "使用更高效的设备后端",
                "简化量子电路结构"
            ]
        elif 'scipy' in func_lower:
            suggestions = [
                "检查数值计算的精度要求",
                "使用更高效的算法",
                "避免不必要的函数调用",
                "考虑缓存计算结果"
            ]
        else:
            suggestions = [
                "分析函数内部逻辑",
                "寻找性能瓶颈",
                "考虑算法优化",
                "使用缓存或预计算"
            ]

        print("   建议:")
        for suggestion in suggestions:
            print(f"     - {suggestion}")

def main():
    prof_file_path = r"E:\qiboenv\Bench\profiling\general\my_programg20.prof"

    print("开始分析性能分析文件...")
    print(f"文件路径: {prof_file_path}")

    # 分析性能数据
    stats = analyze_prof_file(prof_file_path)

    if not stats:
        print("分析失败，退出")
        return

    # 打印性能摘要
    print_performance_summary(stats)

    # 分析性能瓶颈并获取函数统计
    func_stats = analyze_performance_bottlenecks(stats)

    # 按模块分析
    analyze_by_module(stats)

    # 识别优化机会
    optimization_targets = identify_optimization_opportunities(func_stats)

    # 生成优化建议
    generate_optimization_suggestions(optimization_targets)

    # 找出最耗时的函数
    if func_stats:
        most_expensive = func_stats[0]
        print(f"\n" + "="*80)
        print("最耗时的函数分析")
        print("="*80)
        print(f"函数: {most_expensive['function']}")
        print(f"累计时间: {most_expensive['cumulative_time']:.4f}s")
        print(f"调用次数: {most_expensive['call_count']}")
        print(f"总时间: {most_expensive['total_time']:.4f}s")
        print(f"单次调用时间: {most_expensive['time_per_call']:.6f}s")

    # 保存分析结果
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)

if __name__ == "__main__":
    main()