#!/usr/bin/env python3
"""
测试缓存配置修复的脚本
"""

from benchmark_harness.caching.cache_config import CacheConfig, DISK_ONLY_CONFIG

def test_disk_only_config():
    """测试纯磁盘配置可以正常创建"""
    try:
        print(f"DISK_ONLY_CONFIG: cache_type={DISK_ONLY_CONFIG.cache_type}, memory_cache_size={DISK_ONLY_CONFIG.memory_cache_size}")
        print("✓ DISK_ONLY_CONFIG 创建成功")
        return True
    except Exception as e:
        print(f"✗ DISK_ONLY_CONFIG 创建失败: {e}")
        return False

def test_memory_cache_validation():
    """测试内存缓存验证逻辑"""
    try:
        # 对于内存缓存，memory_cache_size为0应该失败
        config = CacheConfig(cache_type='memory', memory_cache_size=0)
        print("✗ 内存缓存验证失败：应该抛出错误")
        return False
    except ValueError as e:
        print(f"✓ 内存缓存验证成功：{e}")
        return True
    except Exception as e:
        print(f"✗ 内存缓存验证失败：意外错误 {e}")
        return False

def test_hybrid_cache_validation():
    """测试混合缓存验证逻辑"""
    try:
        # 对于混合缓存，memory_cache_size为0应该失败
        config = CacheConfig(cache_type='hybrid', memory_cache_size=0)
        print("✗ 混合缓存验证失败：应该抛出错误")
        return False
    except ValueError as e:
        print(f"✓ 混合缓存验证成功：{e}")
        return True
    except Exception as e:
        print(f"✗ 混合缓存验证失败：意外错误 {e}")
        return False

def test_disk_cache_with_zero_memory():
    """测试磁盘缓存可以使用0内存缓存"""
    try:
        config = CacheConfig(cache_type='disk', memory_cache_size=0)
        print(f"✓ 磁盘缓存验证成功：cache_type={config.cache_type}, memory_cache_size={config.memory_cache_size}")
        return True
    except Exception as e:
        print(f"✗ 磁盘缓存验证失败：{e}")
        return False

if __name__ == "__main__":
    print("测试缓存配置修复...")
    print("=" * 50)
    
    results = [
        test_disk_only_config(),
        test_memory_cache_validation(),
        test_hybrid_cache_validation(),
        test_disk_cache_with_zero_memory()
    ]
    
    print("=" * 50)
    if all(results):
        print("✓ 所有测试通过！修复成功。")
    else:
        print("✗ 部分测试失败，需要进一步修复。")