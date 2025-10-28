#!/usr/bin/env python3
"""
Simple prof file analyzer to extract top time-consuming functions
"""

import pstats
import sys

def analyze_prof_file(prof_file_path):
    """Analyze prof file and get top functions"""
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

        # Get top 20 functions by cumulative time
        print("="*80)
        print("TOP 20 MOST TIME-CONSUMING FUNCTIONS (by cumulative time)")
        print("="*80)
        print(f"{'Rank':<5} {'Cumulative Time':<15} {'Time/Call':<12} {'Calls':<8} {'Function':<50}")
        print("-" * 80)

        # stats.stats is a dictionary mapping (filename, line, function) -> (cc, nc, tt, ct, callers)
        # where: cc=cumulative calls, nc=actual calls, tt=total time, ct=cumulative time

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

        # Display top 20
        for i, func in enumerate(function_list[:20], 1):
            # Truncate long function names
            display_func = func['function']
            if len(display_func) > 45:
                display_func = display_func[:42] + "..."

            print(f"{i:<5} {func['cumulative_time']:<15.4f} {func['time_per_call']:<12.6f} {func['calls']:<8} {display_func:<50}")

        print()
        print("="*80)
        print("TOP 10 FUNCTIONS BY TIME PER CALL")
        print("="*80)
        print(f"{'Rank':<5} {'Time/Call':<15} {'Cumulative':<15} {'Calls':<8} {'Function':<50}")
        print("-" * 80)

        # Sort by time per call
        function_list_by_time_per_call = sorted(function_list, key=lambda x: x['time_per_call'], reverse=True)

        for i, func in enumerate(function_list_by_time_per_call[:10], 1):
            # Truncate long function names
            display_func = func['function']
            if len(display_func) > 45:
                display_func = display_func[:42] + "..."

            print(f"{i:<5} {func['time_per_call']:<15.6f} {func['cumulative_time']:<15.4f} {func['calls']:<8} {display_func:<50}")

        print()
        print("="*80)
        print("MODULE BREAKDOWN")
        print("="*80)

        # Group by module (top-level directory/package)
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
        print("-" * 82)

        for i, (module, stats) in enumerate(sorted_modules[:10], 1):
            print(f"{module:<60} {stats['total_time']:<12.4f} {stats['total_calls']:<10} {len(stats['functions']):<10}")

        # Identify the most time-consuming function
        most_expensive = function_list[0]
        print()
        print("="*80)
        print("MOST TIME-CONSUMING FUNCTION")
        print("="*80)
        print(f"Function: {most_expensive['function']}")
        print(f"File: {most_expensive['filename']}:{most_expensive['line']}")
        print(f"Cumulative Time: {most_expensive['cumulative_time']:.4f}s ({most_expensive['cumulative_time']/stats.total_tt*100:.1f}% of total)")
        print(f"Total Time: {most_expensive['total_time']:.4f}s")
        print(f"Number of Calls: {most_expensive['calls']}")
        print(f"Time per Call: {most_expensive['time_per_call']:.6f}s")

        return function_list, module_stats, most_expensive

    except Exception as e:
        print(f"Error analyzing file {prof_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    prof_file_path = r"E:\qiboenv\Bench\profiling\general\my_programg20.prof"
    functions, modules, top_function = analyze_prof_file(prof_file_path)