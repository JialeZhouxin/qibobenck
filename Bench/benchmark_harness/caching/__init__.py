"""
基准测试缓存模块

这个模块提供了用于缓存参考态的多种实现，包括内存缓存、磁盘缓存和混合缓存。
"""

from .cache_config import CacheConfig
from .hybrid_cache import HybridReferenceStateCache
from .memory_cache import MemoryReferenceStateCache
from .disk_cache import PersistentReferenceStateCache
from .cache_utils import create_cache_instance, generate_cache_key

__all__ = [
    "CacheConfig",
    "HybridReferenceStateCache",
    "MemoryReferenceStateCache",
    "PersistentReferenceStateCache",
    "create_cache_instance",
    "generate_cache_key"
]