"""
缓存工具函数模块

这个模块提供了缓存系统的通用工具函数。
"""

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .cache_config import CacheConfig


def generate_cache_key(circuit_name: str, n_qubits: int, backend: str, 
                      cache_version: str = "v1") -> str:
    """生成唯一的缓存键
    
    Args:
        circuit_name: 电路名称
        n_qubits: 量子比特数
        backend: 后端名称
        cache_version: 缓存版本号
        
    Returns:
        str: 唯一的缓存键
    """
    # 标准化输入参数
    normalized_circuit = circuit_name.lower().strip()
    normalized_backend = backend.lower().strip()
    
    # 构建键字符串
    key_components = [
        normalized_circuit,
        str(n_qubits),
        normalized_backend,
        cache_version
    ]
    
    # 生成MD5哈希作为键
    key_string = "_".join(key_components)
    cache_key = hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    return cache_key


def generate_cache_file_path(cache_dir: str, cache_key: str, 
                           file_extension: str = ".npy") -> Path:
    """生成缓存文件路径
    
    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        file_extension: 文件扩展名
        
    Returns:
        Path: 缓存文件路径
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 使用哈希值作为文件名，避免文件名过长或包含特殊字符
    file_name = f"{cache_key}{file_extension}"
    return cache_dir_path / file_name


def save_numpy_array(file_path: Path, array: np.ndarray, 
                    compression: bool = False) -> bool:
    """保存NumPy数组到文件
    
    Args:
        file_path: 文件路径
        array: NumPy数组
        compression: 是否启用压缩
        
    Returns:
        bool: 保存是否成功
    """
    try:
        if compression:
            # 使用压缩格式保存
            np.savez_compressed(file_path, data=array)
        else:
            # 使用标准格式保存
            np.save(file_path, array)
        return True
    except Exception as e:
        print(f"Warning: Failed to save array to {file_path}: {e}")
        return False


def load_numpy_array(file_path: Path, compression: bool = False) -> Optional[np.ndarray]:
    """从文件加载NumPy数组
    
    Args:
        file_path: 文件路径
        compression: 是否启用压缩
        
    Returns:
        Optional[np.ndarray]: 加载的数组，失败时返回None
    """
    try:
        if compression and file_path.suffix == '.npz':
            # 从压缩文件加载
            with np.load(file_path) as data:
                return data['data']
        else:
            # 从标准文件加载
            return np.load(file_path)
    except Exception as e:
        print(f"Warning: Failed to load array from {file_path}: {e}")
        return None


def save_metadata(metadata_file: Path, metadata: Dict[str, Any]) -> bool:
    """保存元数据到文件
    
    Args:
        metadata_file: 元数据文件路径
        metadata: 元数据字典
        
    Returns:
        bool: 保存是否成功
    """
    try:
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        return True
    except Exception as e:
        print(f"Warning: Failed to save metadata to {metadata_file}: {e}")
        return False


def load_metadata(metadata_file: Path) -> Optional[Dict[str, Any]]:
    """从文件加载元数据
    
    Args:
        metadata_file: 元数据文件路径
        
    Returns:
        Optional[Dict[str, Any]]: 加载的元数据，失败时返回None
    """
    try:
        with open(metadata_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metadata from {metadata_file}: {e}")
        return None


def is_cache_expired(metadata: Dict[str, Any], max_age_days: int) -> bool:
    """检查缓存是否过期
    
    Args:
        metadata: 缓存元数据
        max_age_days: 最大保存天数
        
    Returns:
        bool: 是否过期
    """
    if 'created_time' not in metadata:
        return True
    
    created_time = metadata['created_time']
    current_time = time.time()
    age_seconds = current_time - created_time
    age_days = age_seconds / (24 * 3600)
    
    return age_days > max_age_days


def get_file_size_mb(file_path: Path) -> float:
    """获取文件大小（MB）
    
    Args:
        file_path: 文件路径
        
    Returns:
        float: 文件大小（MB）
    """
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def cleanup_expired_cache(cache_dir: str, max_age_days: int) -> int:
    """清理过期的缓存文件
    
    Args:
        cache_dir: 缓存目录
        max_age_days: 最大保存天数
        
    Returns:
        int: 清理的文件数量
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0
    
    cleaned_count = 0
    current_time = time.time()
    
    # 清理.npy文件
    for cache_file in cache_path.glob("*.npy"):
        try:
            file_age_seconds = current_time - cache_file.stat().st_mtime
            file_age_days = file_age_seconds / (24 * 3600)
            
            if file_age_days > max_age_days:
                cache_file.unlink()
                cleaned_count += 1
        except Exception as e:
            print(f"Warning: Failed to delete cache file {cache_file}: {e}")
    
    # 清理.npz文件（压缩格式）
    for cache_file in cache_path.glob("*.npz"):
        try:
            file_age_seconds = current_time - cache_file.stat().st_mtime
            file_age_days = file_age_seconds / (24 * 3600)
            
            if file_age_days > max_age_days:
                cache_file.unlink()
                cleaned_count += 1
        except Exception as e:
            print(f"Warning: Failed to delete cache file {cache_file}: {e}")
    
    return cleaned_count


def estimate_memory_usage(state: np.ndarray) -> float:
    """估算NumPy数组的内存使用量（MB）
    
    Args:
        state: NumPy数组
        
    Returns:
        float: 内存使用量（MB）
    """
    return state.nbytes / (1024 * 1024)


def validate_cache_integrity(cache_file: Path, expected_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """验证缓存文件的完整性
    
    Args:
        cache_file: 缓存文件路径
        expected_shape: 期望的数组形状
        
    Returns:
        bool: 文件是否完整
    """
    try:
        if not cache_file.exists():
            return False
        
        # 尝试加载数组
        if cache_file.suffix == '.npz':
            with np.load(cache_file) as data:
                array = data['data']
        else:
            array = np.load(cache_file)
        
        # 检查数组形状
        if expected_shape is not None and array.shape != expected_shape:
            return False
        
        # 检查数组是否包含NaN或Inf
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            return False
        
        return True
    except Exception:
        return False


def create_cache_instance(config: CacheConfig):
    """根据配置创建缓存实例
    
    Args:
        config: 缓存配置
        
    Returns:
        缓存实例
    """
    from .hybrid_cache import HybridReferenceStateCache
    from .memory_cache import MemoryReferenceStateCache
    from .disk_cache import PersistentReferenceStateCache
    
    if config.cache_type == "memory":
        return MemoryReferenceStateCache(max_size=config.memory_cache_size)
    elif config.cache_type == "disk":
        return PersistentReferenceStateCache(cache_dir=config.disk_cache_dir)
    elif config.cache_type == "hybrid":
        return HybridReferenceStateCache(
            memory_cache_size=config.memory_cache_size,
            disk_cache_dir=config.disk_cache_dir
        )
    else:
        raise ValueError(f"Unknown cache type: {config.cache_type}")