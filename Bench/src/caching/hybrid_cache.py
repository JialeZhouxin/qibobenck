"""
混合缓存实现模块

这个模块提供了结合内存和磁盘的混合缓存实现。
"""

import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..abstractions import BenchmarkCircuit
from .cache_config import CacheConfig
from .cache_utils import (
    generate_cache_key, generate_cache_file_path, save_numpy_array, 
    load_numpy_array, save_metadata, load_metadata, is_cache_expired,
    get_file_size_mb, cleanup_expired_cache
)


class HybridReferenceStateCache:
    """混合式多级缓存实现
    
    这个类结合内存缓存和磁盘缓存的优势，提供高性能和大容量的缓存解决方案。
    """
    
    def __init__(self, memory_cache_size: int = 64, disk_cache_dir: str = ".benchmark_cache", 
                 max_age_days: int = 30):
        """初始化混合缓存系统
        
        Args:
            memory_cache_size: 内存缓存最大条目数
            disk_cache_dir: 磁盘缓存目录
            max_age_days: 缓存最大保存天数
        """
        # 内存缓存（L1缓存）- 使用OrderedDict实现LRU
        self.memory_cache_size = memory_cache_size
        self.memory_cache: OrderedDict = OrderedDict()
        
        # 磁盘缓存（L2缓存）
        self.disk_cache_dir = Path(disk_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.disk_cache_dir / "cache_metadata.pkl"
        self.disk_metadata = self._load_disk_metadata()
        
        # 统计信息
        self.cache_stats = {
            'l1_hits': 0,        # 内存缓存命中
            'l2_hits': 0,        # 磁盘缓存命中
            'misses': 0,         # 缓存未命中
            'computations': 0,   # 计算次数
            'l1_to_l2_promotes': 0,  # 内存到磁盘的提升
            'l2_to_l1_promotes': 0,  # 磁盘到内存的提升
            'l1_evictions': 0,   # 内存缓存淘汰
            'total_requests': 0
        }
        
        # 配置参数
        self.max_age_days = max_age_days
    
    def _load_disk_metadata(self) -> Dict[str, Dict]:
        """加载磁盘缓存元数据
        
        Returns:
            Dict[str, Dict]: 缓存元数据字典
        """
        metadata = load_metadata(self.metadata_file)
        return metadata if metadata is not None else {}
    
    def _save_disk_metadata(self):
        """保存磁盘缓存元数据"""
        save_metadata(self.metadata_file, self.disk_metadata)
    
    def _compute_reference_state(self, circuit_name: str, n_qubits: int, backend: str) -> np.ndarray:
        """计算参考态
        
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
    
    def _add_to_memory_cache(self, cache_key: str, reference_state: np.ndarray):
        """添加到内存缓存，实现LRU淘汰策略
        
        Args:
            cache_key: 缓存键
            reference_state: 参考态
        """
        # 如果缓存已满，淘汰最少使用的条目
        if len(self.memory_cache) >= self.memory_cache_size:
            # 获取最旧的条目（LRU）
            oldest_key, oldest_state = self.memory_cache.popitem(last=False)
            
            # 将淘汰的条目保存到磁盘缓存
            self._save_to_disk_cache(oldest_key, oldest_state)
            self.cache_stats['l1_evictions'] += 1
            self.cache_stats['l1_to_l2_promotes'] += 1
        
        # 添加新条目到内存缓存
        self.memory_cache[cache_key] = reference_state
    
    def _save_to_disk_cache(self, cache_key: str, reference_state: np.ndarray,
                           circuit_name: str = "unknown", n_qubits: int = 0, 
                           backend: str = "unknown"):
        """保存到磁盘缓存
        
        Args:
            cache_key: 缓存键
            reference_state: 参考态
            circuit_name: 电路名称
            n_qubits: 量子比特数
            backend: 后端名称
        """
        try:
            cache_path = generate_cache_file_path(str(self.disk_cache_dir), cache_key)
            success = save_numpy_array(cache_path, reference_state)
            
            if success:
                # 更新元数据
                if cache_key not in self.disk_metadata:
                    self.disk_metadata[cache_key] = {
                        'circuit_name': circuit_name,
                        'n_qubits': n_qubits,
                        'backend': backend,
                        'created_time': time.time(),
                        'last_access': time.time(),
                        'file_size': get_file_size_mb(cache_path),
                        'cache_version': 'v1',
                        'source': 'memory_eviction'
                    }
                
                self._save_disk_metadata()
            
        except Exception as e:
            print(f"Warning: Failed to save to disk cache: {e}")
    
    def get_reference_state(self, circuit_name: str, n_qubits: int, backend: str,
                          circuit_instance: Optional[BenchmarkCircuit] = None) -> np.ndarray:
        """获取参考态，采用多级缓存策略
        
        Args:
            circuit_name: 电路名称
            n_qubits: 量子比特数
            backend: 后端名称
            circuit_instance: 电路实例（可选，用于兼容性）
            
        Returns:
            np.ndarray: 参考态numpy数组
        """
        self.cache_stats['total_requests'] += 1
        
        # 生成缓存键
        cache_key = generate_cache_key(circuit_name, n_qubits, backend)
        
        # L1缓存：内存缓存检查
        if cache_key in self.memory_cache:
            # 移动到末尾（标记为最近使用）
            reference_state = self.memory_cache.pop(cache_key)
            self.memory_cache[cache_key] = reference_state
            
            self.cache_stats['l1_hits'] += 1
            
            if self.cache_stats['l1_hits'] <= 10:  # 只显示前几次命中信息
                print(f"L1 cache hit for {circuit_name}({n_qubits} qubits)")
            
            return reference_state
        
        # L2缓存：磁盘缓存检查
        disk_cache_path = generate_cache_file_path(str(self.disk_cache_dir), cache_key)
        if (disk_cache_path.exists() and 
            cache_key in self.disk_metadata and 
            not is_cache_expired(self.disk_metadata[cache_key], self.max_age_days)):
            
            try:
                # 从磁盘加载
                start_time = time.time()
                reference_state = load_numpy_array(disk_cache_path)
                load_time = time.time() - start_time
                
                if reference_state is not None:
                    # 提升到内存缓存
                    self._add_to_memory_cache(cache_key, reference_state)
                    
                    # 更新访问时间
                    self.disk_metadata[cache_key]['last_access'] = time.time()
                    self._save_disk_metadata()
                    
                    self.cache_stats['l2_hits'] += 1
                    self.cache_stats['l2_to_l1_promotes'] += 1
                    
                    if self.cache_stats['l2_hits'] <= 10:  # 只显示前几次命中信息
                        print(f"L2 cache hit for {circuit_name}({n_qubits} qubits), loaded in {load_time:.4f}s")
                    
                    return reference_state
                    
            except Exception as e:
                print(f"Warning: Failed to load from disk cache: {e}")
        
        # 缓存未命中，计算参考态
        self.cache_stats['misses'] += 1
        reference_state = self._compute_reference_state(circuit_name, n_qubits, backend)
        
        # 存储到多级缓存
        self._add_to_memory_cache(cache_key, reference_state)
        
        # 同时保存到磁盘缓存
        self._save_to_disk_cache(cache_key, reference_state, circuit_name, n_qubits, backend)
        
        return reference_state
    
    def clear_cache(self, level: str = "all"):
        """清空缓存
        
        Args:
            level: 缓存级别 ("l1", "l2", "all")
        """
        if level in ["l1", "all"]:
            # 清空内存缓存
            self.memory_cache.clear()
            print("Cleared L1 (memory) cache")
        
        if level in ["l2", "all"]:
            # 清空磁盘缓存
            cache_files = list(self.disk_cache_dir.glob("*.npy")) + list(self.disk_cache_dir.glob("*.npz"))
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete cache file {cache_file}: {e}")
            
            self.disk_metadata = {}
            self._save_disk_metadata()
            print(f"Cleared L2 (disk) cache: {len(cache_files)} files deleted")
        
        if level == "all":
            # 重置统计信息
            self.cache_stats = {
                'l1_hits': 0,
                'l2_hits': 0,
                'misses': 0,
                'computations': 0,
                'l1_to_l2_promotes': 0,
                'l2_to_l1_promotes': 0,
                'l1_evictions': 0,
                'total_requests': 0
            }
    
    def cleanup_expired_cache(self) -> int:
        """清理过期的缓存条目
        
        Returns:
            int: 清理的文件数量
        """
        # 使用工具函数清理过期文件
        cleaned_count = cleanup_expired_cache(str(self.disk_cache_dir), self.max_age_days)
        
        # 同时清理元数据中的过期条目
        expired_keys = []
        for cache_key, metadata in self.disk_metadata.items():
            if is_cache_expired(metadata, self.max_age_days):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            del self.disk_metadata[cache_key]
        
        if expired_keys:
            self._save_disk_metadata()
        
        total_cleaned = cleaned_count + len(expired_keys)
        if total_cleaned > 0:
            print(f"Cleaned up {total_cleaned} expired cache entries")
        
        return total_cleaned
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息
        
        Returns:
            dict: 缓存统计信息
        """
        # 计算总命中率
        total_requests = self.cache_stats['total_requests']
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self.cache_stats['l1_hits'] + self.cache_stats['l2_hits']) / total_requests
        
        # 计算磁盘缓存总大小
        total_disk_size = 0
        cache_files = list(self.disk_cache_dir.glob("*.npy")) + list(self.disk_cache_dir.glob("*.npz"))
        
        for cache_file in cache_files:
            total_disk_size += get_file_size_mb(cache_file)
        
        # 估算内存使用
        memory_usage_mb = len(self.memory_cache) * 10.0  # 简单估算
        
        return {
            'cache_type': 'hybrid',
            'stats': self.cache_stats,
            'performance': {
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'l1_hit_rate': self.cache_stats['l1_hits'] / total_requests if total_requests > 0 else 0,
                'l2_hit_rate': self.cache_stats['l2_hits'] / total_requests if total_requests > 0 else 0
            },
            'cache_info': {
                'l1_size': len(self.memory_cache),
                'l1_max_size': self.memory_cache_size,
                'l2_size': len(self.disk_metadata),
                'l2_total_size_mb': total_disk_size,
                'memory_usage_mb': memory_usage_mb
            }
        }
    
    def optimize_cache_distribution(self):
        """优化缓存分布，根据访问模式调整内存缓存大小"""
        # 分析访问模式
        l1_hit_rate = self.cache_stats['l1_hits'] / self.cache_stats['total_requests'] if self.cache_stats['total_requests'] > 0 else 0
        
        # 如果L1命中率太低，考虑增加内存缓存大小
        if l1_hit_rate < 0.5 and self.memory_cache_size < 128:
            print(f"L1 hit rate is low ({l1_hit_rate:.2f}), consider increasing memory cache size")
        
        # 如果L1淘汰率太高，考虑增加内存缓存大小
        eviction_rate = self.cache_stats['l1_evictions'] / self.cache_stats['computations'] if self.cache_stats['computations'] > 0 else 0
        if eviction_rate > 0.3:
            print(f"High eviction rate ({eviction_rate:.2f}), consider increasing memory cache size")
        
        # 清理过期缓存
        self.cleanup_expired_cache()
    
    def is_cache_enabled(self) -> bool:
        """检查缓存是否启用
        
        Returns:
            bool: 缓存是否启用
        """
        return self.memory_cache_size > 0 or self.disk_cache_dir is not None