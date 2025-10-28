"""
内存缓存实现模块

这个模块提供了基于内存的LRU缓存实现。
"""

import time
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from ..abstractions import BenchmarkCircuit
from .cache_config import CacheConfig
from .cache_utils import generate_cache_key


class MemoryReferenceStateCache:
    """基于内存的LRU缓存实现
    
    这个类使用Python内置的lru_cache装饰器实现内存缓存，
    提供快速的参考态访问和自动的LRU淘汰策略。
    """
    
    def __init__(self, max_size: int = 128):
        """初始化内存缓存
        
        Args:
            max_size: 缓存最大条目数
        """
        self.max_size = max_size
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'computations': 0,
            'total_requests': 0
        }
        
        # 使用装饰器创建LRU缓存
        self._cached_compute = lru_cache(maxsize=max_size)(self._compute_reference_state_uncached)
    
    def _compute_reference_state_uncached(self, circuit_name: str, n_qubits: int, backend: str) -> np.ndarray:
        """未缓存的参考态计算方法
        
        Args:
            circuit_name: 电路名称
            n_qubits: 量子比特数
            backend: 后端名称
            
        Returns:
            np.ndarray: 参考态
        """
        # 动态导入电路类
        module_name = f"benchmark_harness.circuits.{circuit_name}"
        module = __import__(module_name, fromlist=[circuit_name])
        
        # 获取电路类
        if circuit_name.lower() == "qft":
            circuit_class = getattr(module, "QFTCircuit")
        else:
            circuit_class = getattr(module, f"{circuit_name.title()}Circuit")
        
        # 创建电路实例
        circuit_instance = circuit_class()
        
        # 设置后端
        import qibo
        qibo.set_backend(backend)
        
        # 构建电路
        circuit = circuit_instance.build(platform="qibo", n_qubits=n_qubits)
        
        # 执行电路获取参考态
        start_time = time.time()
        result = circuit(nshots=1)
        computation_time = time.time() - start_time
        
        # 记录计算时间
        self.cache_stats['computations'] += 1
        
        if self.cache_stats['computations'] <= 5:  # 只显示前几次计算信息
            print(f"Computed reference state for {circuit_name}({n_qubits} qubits) in {computation_time:.4f}s")
        
        return result.state()
    
    def get_reference_state(self, circuit_name: str, n_qubits: int, backend: str,
                          circuit_instance: Optional[BenchmarkCircuit] = None) -> np.ndarray:
        """获取参考态，如果不存在则计算并缓存
        
        Args:
            circuit_name: 电路名称
            n_qubits: 量子比特数
            backend: 后端名称
            circuit_instance: 电路实例（可选，用于兼容性）
            
        Returns:
            np.ndarray: 参考态numpy数组
        """
        self.cache_stats['total_requests'] += 1
        
        # 获取缓存信息
        cache_info = self._cached_compute.cache_info()
        
        # 更新统计信息
        if cache_info.hits > self.cache_stats['hits']:
            self.cache_stats['hits'] = cache_info.hits
            if self.cache_stats['hits'] <= 10:  # 只显示前几次命中信息
                print(f"L1 cache hit for {circuit_name}({n_qubits} qubits)")
        
        if cache_info.misses > self.cache_stats['misses']:
            self.cache_stats['misses'] = cache_info.misses
        
        # 获取参考态（自动处理缓存）
        return self._cached_compute(circuit_name, n_qubits, backend)
    
    def clear_cache(self):
        """清空缓存"""
        self._cached_compute.cache_clear()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'computations': 0,
            'total_requests': 0
        }
        print("Memory cache cleared")
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息
        
        Returns:
            dict: 缓存统计信息
        """
        cache_info = self._cached_compute.cache_info()
        
        # 计算命中率
        total_requests = self.cache_stats['total_requests']
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = self.cache_stats['hits'] / total_requests
        
        # 估算内存使用
        estimated_memory_mb = self._estimate_memory_usage()
        
        return {
            'cache_type': 'memory',
            'cache_info': {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'maxsize': cache_info.maxsize,
                'currsize': cache_info.currsize,
                'hit_rate': hit_rate
            },
            'stats': self.cache_stats,
            'memory_usage_mb': estimated_memory_mb,
            'max_size': self.max_size
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）
        
        Returns:
            float: 估算的内存使用量（MB）
        """
        # 简单估算：假设每个缓存项平均占用10MB内存
        cache_info = self._cached_compute.cache_info()
        return cache_info.currsize * 10.0
    
    def get_cache_keys(self) -> list:
        """获取当前缓存中的所有键
        
        Returns:
            list: 缓存键列表
        """
        # 注意：lru_cache不直接提供访问缓存键的方法
        # 这里返回一个空列表，实际应用中可能需要使用其他数据结构
        return []
    
    def is_cache_enabled(self) -> bool:
        """检查缓存是否启用
        
        Returns:
            bool: 缓存是否启用
        """
        return self.max_size > 0
    
    def optimize_cache(self):
        """优化缓存性能"""
        # 对于内存缓存，优化主要是调整大小
        cache_info = self._cached_compute.cache_info()
        
        # 如果缓存接近满载，可以考虑增加大小
        if cache_info.maxsize is not None and cache_info.currsize >= cache_info.maxsize * 0.9:
            print(f"Memory cache is {cache_info.currsize}/{cache_info.maxsize} (90% full), consider increasing max_size")
        
        # 如果命中率很低，可能需要调整缓存策略
        total_requests = self.cache_stats['total_requests']
        if total_requests > 10:
            hit_rate = self.cache_stats['hits'] / total_requests
            if hit_rate < 0.3:
                print(f"Low hit rate ({hit_rate:.2f}), consider increasing cache size or checking access patterns")