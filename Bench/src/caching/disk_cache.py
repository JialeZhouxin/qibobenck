"""
磁盘缓存实现模块

这个模块提供了基于磁盘的持久化缓存实现。
"""

import os
import time
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


class PersistentReferenceStateCache:
    """基于磁盘的持久化缓存实现
    
    这个类将参考态以二进制格式存储在磁盘上，支持跨程序运行保持缓存。
    """
    
    def __init__(self, cache_dir: str = ".benchmark_cache", max_age_days: int = 30):
        """初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_age_days: 缓存最大保存天数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据文件
        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()
        
        # 统计信息
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'computations': 0,
            'disk_saves': 0,
            'disk_loads': 0,
            'total_requests': 0
        }
        
        # 配置参数
        self.max_age_days = max_age_days
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """加载缓存元数据
        
        Returns:
            Dict[str, Dict]: 缓存元数据字典
        """
        metadata = load_metadata(self.metadata_file)
        return metadata if metadata is not None else {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        save_metadata(self.metadata_file, self.metadata)
    
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
    
    def get_reference_state(self, circuit_name: str, n_qubits: int, backend: str,
                          circuit_instance: Optional[BenchmarkCircuit] = None) -> np.ndarray:
        """获取参考态，如果不存在则计算并缓存到磁盘
        
        Args:
            circuit_name: 电路名称
            n_qubits: 量子比特数
            backend: 后端名称
            circuit_instance: 电路实例（可选，用于兼容性）
            
        Returns:
            np.ndarray: 参考态numpy数组
        """
        self.cache_stats['total_requests'] += 1
        
        # 生成缓存键和文件路径
        cache_key = generate_cache_key(circuit_name, n_qubits, backend)
        cache_path = generate_cache_file_path(str(self.cache_dir), cache_key)
        
        # 检查缓存是否存在且未过期
        if cache_path.exists() and cache_key in self.metadata:
            # 检查缓存是否过期
            if not is_cache_expired(self.metadata[cache_key], self.max_age_days):
                try:
                    # 从磁盘加载缓存
                    start_time = time.time()
                    reference_state = load_numpy_array(cache_path)
                    load_time = time.time() - start_time
                    
                    if reference_state is not None:
                        # 更新统计信息
                        self.cache_stats['hits'] += 1
                        self.cache_stats['disk_loads'] += 1
                        
                        # 更新访问时间
                        self.metadata[cache_key]['last_access'] = time.time()
                        self._save_metadata()
                        
                        if self.cache_stats['hits'] <= 10:  # 只显示前几次命中信息
                            print(f"L2 cache hit for {circuit_name}({n_qubits} qubits), loaded in {load_time:.4f}s")
                        
                        return reference_state
                        
                except Exception as e:
                    print(f"Warning: Failed to load cached reference state: {e}")
        
        # 缓存未命中，计算参考态
        self.cache_stats['misses'] += 1
        reference_state = self._compute_reference_state(circuit_name, n_qubits, backend)
        
        # 保存到磁盘
        self._save_to_disk_cache(cache_key, cache_path, reference_state, 
                                circuit_name, n_qubits, backend)
        
        return reference_state
    
    def _save_to_disk_cache(self, cache_key: str, cache_path: Path, reference_state: np.ndarray,
                           circuit_name: str, n_qubits: int, backend: str):
        """保存到磁盘缓存
        
        Args:
            cache_key: 缓存键
            cache_path: 缓存文件路径
            reference_state: 参考态
            circuit_name: 电路名称
            n_qubits: 量子比特数
            backend: 后端名称
        """
        try:
            start_time = time.time()
            success = save_numpy_array(cache_path, reference_state)
            save_time = time.time() - start_time
            
            if success:
                # 更新元数据
                self.metadata[cache_key] = {
                    'circuit_name': circuit_name,
                    'n_qubits': n_qubits,
                    'backend': backend,
                    'created_time': time.time(),
                    'last_access': time.time(),
                    'file_size': get_file_size_mb(cache_path),
                    'cache_version': 'v1'
                }
                self._save_metadata()
                
                # 更新统计信息
                self.cache_stats['disk_saves'] += 1
                
                if self.cache_stats['disk_saves'] <= 5:  # 只显示前几次保存信息
                    print(f"Saved reference state to disk in {save_time:.4f}s")
            
        except Exception as e:
            print(f"Warning: Failed to cache reference state: {e}")
    
    def clear_cache(self):
        """清空所有缓存"""
        # 删除所有.npy文件
        cache_files = list(self.cache_dir.glob("*.npy")) + list(self.cache_dir.glob("*.npz"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete cache file {cache_file}: {e}")
        
        # 清空元数据
        self.metadata = {}
        self._save_metadata()
        
        # 重置统计信息
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'computations': 0,
            'disk_saves': 0,
            'disk_loads': 0,
            'total_requests': 0
        }
        
        print(f"Disk cache cleared: {len(cache_files)} files deleted")
    
    def cleanup_expired_cache(self) -> int:
        """清理过期的缓存条目
        
        Returns:
            int: 清理的文件数量
        """
        # 使用工具函数清理过期文件
        cleaned_count = cleanup_expired_cache(str(self.cache_dir), self.max_age_days)
        
        # 同时清理元数据中的过期条目
        current_time = time.time()
        expired_keys = []
        
        for cache_key, metadata in self.metadata.items():
            if is_cache_expired(metadata, self.max_age_days):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            del self.metadata[cache_key]
        
        if expired_keys:
            self._save_metadata()
        
        total_cleaned = cleaned_count + len(expired_keys)
        if total_cleaned > 0:
            print(f"Cleaned up {total_cleaned} expired cache entries")
        
        return total_cleaned
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息
        
        Returns:
            dict: 缓存统计信息
        """
        # 计算缓存文件总大小
        total_size = 0
        cache_files = list(self.cache_dir.glob("*.npy")) + list(self.cache_dir.glob("*.npz"))
        
        for cache_file in cache_files:
            total_size += get_file_size_mb(cache_file)
        
        # 计算命中率
        total_requests = self.cache_stats['total_requests']
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = self.cache_stats['hits'] / total_requests
        
        return {
            'cache_type': 'disk',
            'stats': self.cache_stats,
            'cache_entries': len(self.metadata),
            'total_cache_size_mb': total_size,
            'cache_dir': str(self.cache_dir),
            'max_age_days': self.max_age_days,
            'hit_rate': hit_rate
        }
    
    def get_cache_info(self) -> dict:
        """获取缓存详细信息
        
        Returns:
            dict: 缓存详细信息
        """
        cache_entries = []
        
        for cache_key, metadata in self.metadata.items():
            cache_path = generate_cache_file_path(str(self.cache_dir), cache_key)
            cache_entries.append({
                'cache_key': cache_key,
                'circuit_name': metadata.get('circuit_name', 'unknown'),
                'n_qubits': metadata.get('n_qubits', 0),
                'backend': metadata.get('backend', 'unknown'),
                'created_time': metadata.get('created_time', 0),
                'last_access': metadata.get('last_access', 0),
                'file_size_mb': metadata.get('file_size', 0),
                'file_exists': cache_path.exists()
            })
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_entries': len(cache_entries),
            'entries': cache_entries
        }
    
    def is_cache_enabled(self) -> bool:
        """检查缓存是否启用
        
        Returns:
            bool: 缓存是否启用
        """
        return True  # 磁盘缓存总是启用的
    
    def optimize_cache(self):
        """优化缓存性能"""
        # 清理过期缓存
        self.cleanup_expired_cache()
        
        # 检查缓存大小
        stats = self.get_cache_stats()
        if stats['total_cache_size_mb'] > 1000:  # 超过1GB
            print(f"Cache size is large ({stats['total_cache_size_mb']:.1f} MB), consider cleanup")
        
        # 检查命中率
        if stats['hit_rate'] < 0.3 and stats['stats']['total_requests'] > 10:
            print(f"Low hit rate ({stats['hit_rate']:.2f}), consider cache strategy review")