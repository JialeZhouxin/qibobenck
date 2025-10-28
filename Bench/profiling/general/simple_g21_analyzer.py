#!/usr/bin/env python3
"""
Simple analysis of my_programg21.prof file
"""

import pstats

def analyze_prof_file(prof_file_path):
    """Analyze prof file and get performance data"""
    try:
        stats = pstats.Stats(prof_file_path)
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
        return None

def print_top_functions(function_list, total_time, limit=20):
    """Print top time-consuming functions"""
    print("="*90)
    print(f"TOP {limit} MOST TIME-CONSUMING FUNCTIONS")
    print("="*90)
    print(f"{'Rank':<5} {'Cumulative Time':<15} {'Time/Call':<12} {'Calls':<10} {'Function':<45}")
    print("-" * 90)

    for i, func in enumerate(function_list[:limit], 1):
        display_func = func['function']
        if len(display_func) > 40:
            display_func = display_func[:37] + "..."

        print(f"{i:<5} {func['cumulative_time']:<15.4f} {func['time_per_call']:<12.6f} {func['calls']:<10} {display_func:<45}")

def analyze_module_breakdown(function_list):
    """Analyze performance by module"""
    print()
    print("="*90)
    print("MODULE BREAKDOWN")
    print("="*90)

    module_stats = {}
    for func in function_list:
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

    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)

    print(f"{'Module':<60} {'Total Time':<12} {'Calls':<10} {'Functions':<10}")
    print("-" * 92)

    for i, (module, stats) in enumerate(sorted_modules[:10], 1):
        print(f"{module:<60} {stats['total_time']:<12.4f} {stats['total_calls']:<10} {len(stats['functions']):<10}")

    return sorted_modules

def compare_with_g20(g21_data):
    """Compare G21 data with previously analyzed G20 data"""
    print()
    print("="*90)
    print("COMPARISON WITH G20 (previous analysis)")
    print("="*90)

    # G20 data from previous analysis
    g20_total_time = 964.4348
    g20_total_calls = 20198
    g20_primitive_calls = 607463520

    print(f"{'Metric':<30} {'G20':<15} {'G21':<15} {'Change':<15}")
    print("-" * 75)
    print(f"{'Total Execution Time (s)':<30} {g20_total_time:<15.2f} {g21_data['total_time']:<15.2f} {g21_data['total_time'] - g20_total_time:+15.2f}")
    print(f"{'Total Function Calls':<30} {g20_total_calls:<15} {g21_data['total_calls']:<15} {g21_data['total_calls'] - g20_total_calls:+15}")
    print(f"{'Primitive Calls':<30} {g20_primitive_calls:<15} {g21_data['primitive_calls']:<15} {g21_data['primitive_calls'] - g20_primitive_calls:+15}")

    # Calculate performance change percentage
    time_change_pct = ((g21_data['total_time'] - g20_total_time) / g20_total_time) * 100
    print(f"{'Performance Change':<30} {'':<15} {'':<15} {time_change_pct:+15.1f}%")

    # Identify top function
    if g21_data['functions']:
        top_function = g21_data['functions'][0]
        print()
        print("MOST TIME-CONSUMING FUNCTION IN G21:")
        print(f"Function: {top_function['function']}")
        print(f"File: {top_function['filename']}:{top_function['line']}")
        print(f"Cumulative Time: {top_function['cumulative_time']:.4f}s")
        print(f"Number of Calls: {top_function['calls']}")
        print(f"Time per Call: {top_function['time_per_call']:.6f}s")

def main():
    """Main analysis function"""
    g21_file = r"E:\qiboenv\Bench\profiling\general\my_programg21.prof"

    print("Analyzing G21 profiling data...")
    g21_data = analyze_prof_file(g21_file)

    if g21_data:
        print_top_functions(g21_data['functions'], g21_data['total_time'])
        analyze_module_breakdown(g21_data['functions'])
        compare_with_g20(g21_data)

        # Additional analysis
        print()
        print("="*90)
        print("PERFORMANCE INSIGHTS")
        print("="*90)

        # Calculate function time distribution
        total_time = g21_data['total_time']
        top_5_time = sum(func['cumulative_time'] for func in g21_data['functions'][:5])
        top_10_time = sum(func['cumulative_time'] for func in g21_data['functions'][:10])

        print(f"Top 5 functions consume: {top_5_time:.2f}s ({top_5_time/total_time*100:.1f}% of total)")
        print(f"Top 10 functions consume: {top_10_time:.2f}s ({top_10_time/total_time*100:.1f}% of total)")

        # Identify functions with high time per call
        high_time_per_call = [func for func in g21_data['functions'] if func['time_per_call'] > 10]
        if high_time_per_call:
            print(f"\nFunctions with high time per call (>10s):")
            for func in high_time_per_call[:5]:
                print(f"  {func['function']}: {func['time_per_call']:.2f}s/call ({func['calls']} calls)")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()