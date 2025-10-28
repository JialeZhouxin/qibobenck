#!/usr/bin/env python3
"""
Analyze my_programg21.prof file and compare with my_programg20.prof
"""

import pstats
import sys
from collections import defaultdict

def analyze_prof_file(prof_file_path):
    """Analyze prof file and get detailed performance data"""
    try:
        # Create Stats object
        stats = pstats.Stats(prof_file_path)

        # Sort by cumulative time
        stats.sort_stats('cumulative')

        print(f"Profiling file: {prof_file_path}")
        print(f"Total execution time: {stats.total_tt:.4f}s")
        print(f"Total function calls: {len(stats.stats)}")
        print(f"Primitive calls: {stats.prim_calls}")
        print()

        # Extract function data
        function_list = []
        for func_key, func_data in stats.stats.items():
            filename, line, func_name = func_key
            cc, nc, tt, ct, callers = func_data

            # Calculate time per call
            time_per_call = ct / nc if nc > 0 else 0

            function_list.append({
                'filename': filename,
                'line': line,
                'function': func_name,
                'cumulative_time': ct,
                'total_time': tt,
                'calls': nc,
                'time_per_call': time_per_call
            })

        # Sort by cumulative time
        function_list.sort(key=lambda x: x['cumulative_time'], reverse=True)

        return {
            'stats': stats,
            'functions': function_list,
            'total_time': stats.total_tt,
            'total_calls': len(stats.stats),
            'primitive_calls': stats.prim_calls
        }

    except Exception as e:
        print(f"Error analyzing file {prof_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_top_functions(function_list, total_time, limit=20):
    """Print top time-consuming functions"""
    print("="*90)
    print(f"TOP {limit} MOST TIME-CONSUMING FUNCTIONS (by cumulative time)")
    print("="*90)
    print(f"{'Rank':<5} {'Cumulative Time':<15} {'Time/Call':<12} {'Calls':<10} {'Function':<45}")
    print("-" * 90)

    for i, func in enumerate(function_list[:limit], 1):
        # Truncate long function names
        display_func = func['function']
        if len(display_func) > 40:
            display_func = display_func[:37] + "..."

        print(f"{i:<5} {func['cumulative_time']:<15.4f} {func['time_per_call']:<12.6f} {func['calls']:<10} {display_func:<45}")

def print_top_by_time_per_call(function_list, limit=10):
    """Print top functions by time per call"""
    print()
    print("="*90)
    print(f"TOP {limit} FUNCTIONS BY TIME PER CALL")
    print("="*90)
    print(f"{'Rank':<5} {'Time/Call':<15} {'Cumulative':<15} {'Calls':<10} {'Function':<45}")
    print("-" * 90)

    # Sort by time per call
    function_list_by_time_per_call = sorted(function_list, key=lambda x: x['time_per_call'], reverse=True)

    for i, func in enumerate(function_list_by_time_per_call[:limit], 1):
        # Truncate long function names
        display_func = func['function']
        if len(display_func) > 40:
            display_func = display_func[:37] + "..."

        print(f"{i:<5} {func['time_per_call']:<15.6f} {func['cumulative_time']:<15.4f} {func['calls']:<10} {display_func:<45}")

def analyze_module_breakdown(function_list):
    """Analyze performance by module"""
    print()
    print("="*90)
    print("MODULE BREAKDOWN")
    print("="*90)

    # Group by module
    module_stats = {}
    for func in function_list:
        # Extract module name from filename
        if '\\' in func['filename']:
            parts = func['filename'].split('\\')
            if len(parts) >= 2:
                module = parts[-2] + '/' + parts[-1]
            else:
                module = parts[-1]
        else:
            module = func['filename']

        if module not in module_stats:
            module_stats[module] = {
                'total_time': 0,
                'total_calls': 0,
                'functions': []
            }

        module_stats[module]['total_time'] += func['cumulative_time']
        module_stats[module]['total_calls'] += func['calls']
        module_stats[module]['functions'].append(func)

    # Sort modules by total time
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)

    print(f"{'Module':<60} {'Total Time':<12} {'Calls':<10} {'Functions':<10}")
    print("-" * 92)

    for i, (module, stats) in enumerate(sorted_modules[:10], 1):
        print(f"{module:<60} {stats['total_time']:<12.4f} {stats['total_calls']:<10} {len(stats['functions']):<10}")

    return sorted_modules

def compare_profiling_data(g20_data, g21_data):
    """Compare performance between two profiling runs"""
    print()
    print("="*90)
    print("PERFORMANCE COMPARISON: G20 vs G21")
    print("="*90)

    # Basic comparison
    print(f"{'Metric':<30} {'G20':<15} {'G21':<15} {'Change':<15}")
    print("-" * 75)
    print(f"{'Total Execution Time (s)':<30} {g20_data['total_time']:<15.2f} {g21_data['total_time']:<15.2f} {g21_data['total_time'] - g20_data['total_time']:+15.2f}")
    print(f"{'Total Function Calls':<30} {g20_data['total_calls']:<15} {g21_data['total_calls']:<15} {g21_data['total_calls'] - g20_data['total_calls']:+15}")
    print(f"{'Primitive Calls':<30} {g20_data['primitive_calls']:<15} {g21_data['primitive_calls']:<15} {g21_data['primitive_calls'] - g20_data['primitive_calls']:+15}")

    # Calculate performance change percentage
    time_change_pct = ((g21_data['total_time'] - g20_data['total_time']) / g20_data['total_time']) * 100
    print(f"{'Performance Change':<30} {'':<15} {'':<15} {time_change_pct:+15.1f}%")

    # Compare top functions
    print()
    print("TOP FUNCTIONS COMPARISON:")
    print("-" * 90)
    print(f"{'Function':<35} {'G20 Time':<12} {'G21 Time':<12} {'Change':<12} {'Change %':<10}")
    print("-" * 90)

    # Create function name maps for both
    g20_functions = {func['function']: func for func in g20_data['functions']}
    g21_functions = {func['function']: func for func in g21_data['functions']}

    # Get union of top 20 functions from both
    all_top_functions = set()
    for func in g20_data['functions'][:15]:
        all_top_functions.add(func['function'])
    for func in g21_data['functions'][:15]:
        all_top_functions.add(func['function'])

    # Compare each function
    function_changes = []
    for func_name in all_top_functions:
        g20_func = g20_functions.get(func_name)
        g21_func = g21_functions.get(func_name)

        if g20_func and g21_func:
            g20_time = g20_func['cumulative_time']
            g21_time = g21_func['cumulative_time']
            change = g21_time - g20_time
            change_pct = (change / g20_time) * 100 if g20_time > 0 else 0

            # Truncate long function names
            display_name = func_name
            if len(display_name) > 32:
                display_name = display_name[:29] + "..."

            print(f"{display_name:<35} {g20_time:<12.4f} {g21_time:<12.4f} {change:+12.4f} {change_pct:+10.1f}%")
            function_changes.append((func_name, change, change_pct))
        elif g20_func:
            print(f"{func_name[:35]:<35} {g20_func['cumulative_time']:<12.4f} {'N/A':<12} {'REMOVED':<12} {'N/A':<10}")
        elif g21_func:
            print(f"{func_name[:35]:<35} {'N/A':<12} {g21_func['cumulative_time']:<12.4f} {'NEW':<12} {'N/A':<10}")

    # Sort by absolute change
    function_changes.sort(key=lambda x: abs(x[1]), reverse=True)

    return function_changes

def main():
    """Main analysis function"""
    g20_file = r"E:\qiboenv\Bench\profiling\general\my_programg20.prof"
    g21_file = r"E:\qiboenv\Bench\profiling\general\my_programg21.prof"

    print("Analyzing G20 profiling data...")
    g20_data = analyze_prof_file(g20_file)

    print("\n" + "="*90)
    print("G20 ANALYSIS RESULTS")
    print("="*90)
    if g20_data:
        print_top_functions(g20_data['functions'], g20_data['total_time'])
        print_top_by_time_per_call(g20_data['functions'])
        g20_modules = analyze_module_breakdown(g20_data['functions'])

    print("\n" + "="*90)
    print("ANALYZING G21 PROFILING DATA...")
    print("="*90)
    print("Analyzing G21 profiling data...")
    g21_data = analyze_prof_file(g21_file)

    print("\n" + "="*90)
    print("G21 ANALYSIS RESULTS")
    print("="*90)
    if g21_data:
        print_top_functions(g21_data['functions'], g21_data['total_time'])
        print_top_by_time_per_call(g21_data['functions'])
        g21_modules = analyze_module_breakdown(g21_data['functions'])

    # Comparison
    if g20_data and g21_data:
        function_changes = compare_profiling_data(g20_data, g21_data)

        # Summary of significant changes
        print()
        print("="*90)
        print("SIGNIFICANT PERFORMANCE CHANGES")
        print("="*90)

        significant_changes = [change for change in function_changes if abs(change[2]) > 20]  # > 20% change

        if significant_changes:
            print("Functions with >20% performance change:")
            for func_name, change, change_pct in significant_changes:
                direction = "IMPROVED" if change < 0 else "DEGRADED"
                print(f"  {func_name}: {change:+.4f}s ({change_pct:+.1f}%) - {direction}")
        else:
            print("No significant performance changes (>20%) detected in top functions.")

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)

if __name__ == "__main__":
    main()