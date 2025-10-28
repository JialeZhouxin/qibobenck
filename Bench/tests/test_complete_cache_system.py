#!/usr/bin/env python3
import sys
import os

# 添加Bench目录到Python路径
bench_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bench_dir)
"""
完整缓存系统测试脚本

这个脚本测试所有三种缓存类型的功能，包括：
1. 内存缓存
2. 磁盘缓存
3. 混合缓存

以及它们与run_benchmarks.py的集成。

注意：这个脚本设计为在tests目录中运行，会自动调整路径以正确调用run_benchmarks.py。
"""

import os
import subprocess
import sys
from pathlib import Path

# 添加父目录到Python路径，以便导入benchmark_harness模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd):
    """运行命令并返回结果"""
    print(f"执行命令: {' '.join(cmd)}")
    try:
        # 在父目录(Bench)中运行命令
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            print(f"命令失败，返回码: {result.returncode}")
            print(f"错误输出: {result.stderr}")
            return False
        print("命令执行成功")
        return True
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False

def test_memory_cache():
    """测试内存缓存"""
    print("\n" + "="*60)
    print("测试内存缓存")
    print("="*60)
    
    cmd = [
        sys.executable, "run_benchmarks.py",
        "--cache-type", "memory",
        "--memory-cache-size", "2",
        "--qubits", "2", "3",
        "--cache-stats",
        "--verbose",
        "--clear-cache"
    ]
    
    return run_command(cmd)

def test_disk_cache():
    """测试磁盘缓存"""
    print("\n" + "="*60)
    print("测试磁盘缓存")
    print("="*60)
    
    cmd = [
        sys.executable, "run_benchmarks.py",
        "--cache-type", "disk",
        "--memory-cache-size", "0",
        "--qubits", "2", "3",
        "--cache-stats",
        "--verbose"
    ]
    
    return run_command(cmd)

def test_hybrid_cache():
    """测试混合缓存"""
    print("\n" + "="*60)
    print("测试混合缓存")
    print("="*60)
    
    cmd = [
        sys.executable, "run_benchmarks.py",
        "--cache-type", "hybrid",
        "--memory-cache-size", "2",
        "--qubits", "2", "3",
        "--cache-stats",
        "--verbose",
        "--clear-cache"
    ]
    
    return run_command(cmd)

def test_cache_persistence():
    """测试缓存持久性"""
    print("\n" + "="*60)
    print("测试缓存持久性")
    print("="*60)
    
    # 第一次运行，填充缓存
    print("第一次运行，填充缓存...")
    cmd1 = [
        sys.executable, "run_benchmarks.py",
        "--cache-type", "disk",
        "--qubits", "2", "3",
        "--verbose",
        "--clear-cache"
    ]
    
    if not run_command(cmd1):
        return False
    
    # 第二次运行，应该使用缓存
    print("第二次运行，应该使用缓存...")
    cmd2 = [
        sys.executable, "run_benchmarks.py",
        "--cache-type", "disk",
        "--qubits", "2", "3",
        "--cache-stats",
        "--verbose"
    ]
    
    return run_command(cmd2)

def test_cache_config_validation():
    """测试缓存配置验证"""
    print("\n" + "="*60)
    print("测试缓存配置验证")
    print("="*60)
    
    try:
        # 测试磁盘缓存可以使用0内存缓存
        from src.caching.cache_config import CacheConfig
        
        config = CacheConfig(cache_type="disk", memory_cache_size=0)
        print(f"✓ 磁盘缓存配置验证成功: cache_type={config.cache_type}, memory_cache_size={config.memory_cache_size}")
        
        # 测试内存缓存不能使用0内存缓存
        try:
            invalid_config = CacheConfig(cache_type="memory", memory_cache_size=0)
            print("✗ 内存缓存配置验证失败：应该抛出错误")
            return False
        except ValueError as e:
            print(f"✓ 内存缓存配置验证成功：{e}")
        
        # 测试混合缓存不能使用0内存缓存
        try:
            invalid_config = CacheConfig(cache_type="hybrid", memory_cache_size=0)
            print("✗ 混合缓存配置验证失败：应该抛出错误")
            return False
        except ValueError as e:
            print(f"✓ 混合缓存配置验证成功：{e}")
        
        return True
    except Exception as e:
        print(f"✗ 缓存配置验证失败: {e}")
        return False

def main():
    """主测试函数"""
    print("完整缓存系统测试")
    print("="*60)
    
    # 确保run_benchmarks.py存在
    bench_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_benchmarks_path = os.path.join(bench_dir, "run_benchmarks.py")
    if not Path(run_benchmarks_path).exists():
        print(f"错误: 未找到 {run_benchmarks_path}，请在正确的目录中运行此脚本")
        return 1
    
    tests = [
        ("缓存配置验证", test_cache_config_validation),
        ("内存缓存", test_memory_cache),
        ("磁盘缓存", test_disk_cache),
        ("混合缓存", test_hybrid_cache),
        ("缓存持久性", test_cache_persistence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 时出错: {e}")
            results.append((test_name, False))
    
    # 显示测试结果摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！缓存系统工作正常。")
        return 0
    else:
        print("❌ 部分测试失败，需要进一步检查。")
        return 1

if __name__ == "__main__":
    sys.exit(main())