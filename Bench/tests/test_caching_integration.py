#!/usr/bin/env python3
"""
缓存集成测试脚本

这个脚本用于测试混合缓存系统的集成和功能。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_harness.caching import CacheConfig, create_cache_instance
from benchmark_harness.caching.memory_cache import MemoryReferenceStateCache
from benchmark_harness.caching.disk_cache import PersistentReferenceStateCache
from benchmark_harness.caching.hybrid_cache import HybridReferenceStateCache


def test_memory_cache():
    """测试内存缓存"""
    print("Testing Memory Cache...")
    
    cache = MemoryReferenceStateCache(max_size=3)
    
    # 测试缓存未命中
    state1 = cache.get_reference_state("qft", 4, "qibojit")
    print(f"State 1 shape: {state1.shape}")
    
    # 测试缓存命中
    state1_again = cache.get_reference_state("qft", 4, "qibojit")
    print(f"State 1 from cache shape: {state1_again.shape}")
    
    # 添加更多条目测试LRU淘汰
    state2 = cache.get_reference_state("qft", 6, "qibojit")
    state3 = cache.get_reference_state("qft", 8, "qibojit")
    state4 = cache.get_reference_state("qft", 10, "qibojit")  # 应该淘汰state1
    
    # 获取缓存统计
    stats = cache.get_cache_stats()
    print(f"Memory cache stats: {stats}")
    
    print("Memory cache test completed.\n")


def test_disk_cache():
    """测试磁盘缓存"""
    print("Testing Disk Cache...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = PersistentReferenceStateCache(cache_dir=temp_dir, max_age_days=30)
        
        # 测试缓存未命中
        state1 = cache.get_reference_state("qft", 4, "qibojit")
        print(f"State 1 shape: {state1.shape}")
        
        # 测试缓存命中
        state1_again = cache.get_reference_state("qft", 4, "qibojit")
        print(f"State 1 from cache shape: {state1_again.shape}")
        
        # 获取缓存统计
        stats = cache.get_cache_stats()
        print(f"Disk cache stats: {stats}")
        
        # 测试缓存清理
        cache.clear_cache()
        print("Disk cache cleared")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    print("Disk cache test completed.\n")


def test_hybrid_cache():
    """测试混合缓存"""
    print("Testing Hybrid Cache...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = HybridReferenceStateCache(
            memory_cache_size=2,
            disk_cache_dir=temp_dir,
            max_age_days=30
        )
        
        # 测试缓存未命中
        state1 = cache.get_reference_state("qft", 4, "qibojit")
        print(f"State 1 shape: {state1.shape}")
        
        # 测试L1缓存命中
        state1_again = cache.get_reference_state("qft", 4, "qibojit")
        print(f"State 1 from L1 cache shape: {state1_again.shape}")
        
        # 添加更多条目测试内存到磁盘的提升
        state2 = cache.get_reference_state("qft", 6, "qibojit")
        state3 = cache.get_reference_state("qft", 8, "qibojit")  # 应该淘汰state1到磁盘
        state4 = cache.get_reference_state("qft", 10, "qibojit")  # 应该淘汰state2到磁盘
        
        # 测试L2缓存命中和提升
        state1_from_disk = cache.get_reference_state("qft", 4, "qibojit")
        print(f"State 1 from L2 cache shape: {state1_from_disk.shape}")
        
        # 获取缓存统计
        stats = cache.get_cache_stats()
        print(f"Hybrid cache stats: {stats}")
        
        # 测试缓存优化
        cache.optimize_cache_distribution()
        
        # 测试缓存清理
        cache.clear_cache()
        print("Hybrid cache cleared")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    print("Hybrid cache test completed.\n")


def test_cache_config():
    """测试缓存配置"""
    print("Testing Cache Configuration...")
    
    # 测试默认配置
    config = CacheConfig()
    print(f"Default config: {config.to_dict()}")
    
    # 测试内存配置
    memory_config = CacheConfig(cache_type="memory", memory_cache_size=128)
    print(f"Memory config: {memory_config.to_dict()}")
    
    # 测试磁盘配置
    disk_config = CacheConfig(cache_type="disk", disk_cache_dir="/tmp/test_cache")
    print(f"Disk config: {disk_config.to_dict()}")
    
    # 测试混合配置
    hybrid_config = CacheConfig(cache_type="hybrid", memory_cache_size=64, disk_cache_dir="/tmp/test_cache")
    print(f"Hybrid config: {hybrid_config.to_dict()}")
    
    print("Cache configuration test completed.\n")


def test_cache_factory():
    """测试缓存工厂函数"""
    print("Testing Cache Factory...")
    
    # 测试内存缓存创建
    memory_config = CacheConfig(cache_type="memory", memory_cache_size=32)
    memory_cache = create_cache_instance(memory_config)
    print(f"Created memory cache: {type(memory_cache).__name__}")
    
    # 测试磁盘缓存创建
    temp_dir = tempfile.mkdtemp()
    try:
        disk_config = CacheConfig(cache_type="disk", disk_cache_dir=temp_dir)
        disk_cache = create_cache_instance(disk_config)
        print(f"Created disk cache: {type(disk_cache).__name__}")
    finally:
        shutil.rmtree(temp_dir)
    
    # 测试混合缓存创建
    temp_dir = tempfile.mkdtemp()
    try:
        hybrid_config = CacheConfig(cache_type="hybrid", memory_cache_size=16, disk_cache_dir=temp_dir)
        hybrid_cache = create_cache_instance(hybrid_config)
        print(f"Created hybrid cache: {type(hybrid_cache).__name__}")
    finally:
        shutil.rmtree(temp_dir)
    
    print("Cache factory test completed.\n")


def main():
    """主测试函数"""
    print("="*60)
    print("Running Cache Integration Tests")
    print("="*60)
    
    try:
        test_cache_config()
        test_cache_factory()
        test_memory_cache()
        test_disk_cache()
        test_hybrid_cache()
        
        print("="*60)
        print("All cache integration tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())