"""
缓存配置模块

这个模块定义了缓存系统的配置类和相关常量。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheConfig:
    """缓存配置类
    
    这个类包含了缓存系统的所有配置参数，支持不同类型的缓存策略。
    
    Attributes:
        enable_cache: 是否启用缓存
        cache_type: 缓存类型 ("memory", "disk", "hybrid")
        memory_cache_size: 内存缓存最大条目数
        disk_cache_dir: 磁盘缓存目录路径
        auto_cleanup: 是否自动清理过期缓存
        max_cache_age_days: 缓存最大保存天数
        cache_version: 缓存版本号，用于缓存失效
        compression_enabled: 是否启用缓存压缩
        max_memory_usage_mb: 最大内存使用量(MB)
    """
    enable_cache: bool = True
    cache_type: str = "hybrid"
    memory_cache_size: int = 64
    disk_cache_dir: str = ".benchmark_cache"
    auto_cleanup: bool = True
    max_cache_age_days: int = 30
    cache_version: str = "v1"
    compression_enabled: bool = False
    max_memory_usage_mb: Optional[int] = None
    
    def __post_init__(self):
        """初始化后的验证和处理"""
        # 验证缓存类型
        valid_types = ["memory", "disk", "hybrid"]
        if self.cache_type not in valid_types:
            raise ValueError(f"Invalid cache_type: {self.cache_type}. Must be one of {valid_types}")
        
        # 验证内存缓存大小
        # 对于纯磁盘缓存，允许memory_cache_size为0
        if self.cache_type != "disk" and self.memory_cache_size <= 0:
            raise ValueError("memory_cache_size must be positive for non-disk cache types")
        
        # 处理磁盘缓存目录
        if not self.disk_cache_dir:
            self.disk_cache_dir = ".benchmark_cache"
        
        # 确保磁盘缓存目录是绝对路径或相对路径
        self.disk_cache_dir = os.path.expanduser(self.disk_cache_dir)
        
        # 验证缓存天数
        if self.max_cache_age_days <= 0:
            self.max_cache_age_days = 30
    
    @classmethod
    def from_args(cls, args) -> 'CacheConfig':
        """从命令行参数创建配置对象
        
        Args:
            args: 解析后的命令行参数对象
            
        Returns:
            CacheConfig: 配置对象
        """
        config = cls()
        
        # 从命令行参数更新配置
        if hasattr(args, 'enable_cache') and not args.enable_cache:
            config.enable_cache = False
            
        if hasattr(args, 'cache_type') and args.cache_type:
            config.cache_type = args.cache_type
            
        if hasattr(args, 'memory_cache_size') and args.memory_cache_size:
            config.memory_cache_size = args.memory_cache_size
            
        if hasattr(args, 'cache_dir') and args.cache_dir:
            config.disk_cache_dir = args.cache_dir
        
        return config
    
    def to_dict(self) -> dict:
        """转换为字典格式
        
        Returns:
            dict: 配置字典
        """
        return {
            'enable_cache': self.enable_cache,
            'cache_type': self.cache_type,
            'memory_cache_size': self.memory_cache_size,
            'disk_cache_dir': self.disk_cache_dir,
            'auto_cleanup': self.auto_cleanup,
            'max_cache_age_days': self.max_cache_age_days,
            'cache_version': self.cache_version,
            'compression_enabled': self.compression_enabled,
            'max_memory_usage_mb': self.max_memory_usage_mb
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'CacheConfig':
        """从字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            CacheConfig: 配置对象
        """
        return cls(**config_dict)


# 预定义的缓存配置
DEFAULT_CONFIG = CacheConfig()

MEMORY_ONLY_CONFIG = CacheConfig(
    cache_type="memory",
    memory_cache_size=128,
    disk_cache_dir=""  # 不使用磁盘缓存
)

DISK_ONLY_CONFIG = CacheConfig(
    cache_type="disk",
    memory_cache_size=0,  # 不使用内存缓存
    disk_cache_dir=".benchmark_cache"
)

HIGH_PERFORMANCE_CONFIG = CacheConfig(
    cache_type="hybrid",
    memory_cache_size=256,
    disk_cache_dir=".benchmark_cache",
    compression_enabled=True,
    max_memory_usage_mb=1024
)

LOW_MEMORY_CONFIG = CacheConfig(
    cache_type="disk",
    memory_cache_size=16,
    disk_cache_dir=".benchmark_cache",
    max_memory_usage_mb=256
)