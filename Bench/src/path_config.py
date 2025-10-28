"""
路径配置模块

统一管理Bench项目的所有路径配置，确保路径的一致性和可维护性。
"""

from pathlib import Path
import os

# 获取Bench项目的基础目录
BENCH_DIR = Path(__file__).parent.parent
BASE_DIR = BENCH_DIR.parent

# 输出目录配置
RESULTS_DIR = BENCH_DIR / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = BENCH_DIR / "logs"

# 缓存目录配置
CACHE_DIR = BENCH_DIR / ".benchmark_cache"
MEMORY_CACHE_SIZE = 64
DISK_CACHE_DIR = CACHE_DIR / "disk"
HYBRID_CACHE_DIR = CACHE_DIR / "hybrid"

# 实验数据目录
EXPERIMENTS_DIR = BENCH_DIR / "experiments"
VQE_RESULTS_DIR = EXPERIMENTS_DIR / "VQEtest" / "results"

# 配置目录
CONFIG_DIR = BENCH_DIR / "env_set"
DOCS_DIR = BENCH_DIR / "docs"

def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        RESULTS_DIR,
        BENCHMARKS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        DISK_CACHE_DIR,
        HYBRID_CACHE_DIR,
        VQE_RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"确保目录存在: {directory}")

def get_results_path(subdir: str = None, create: bool = True) -> Path:
    """获取结果目录路径"""
    if subdir:
        path = RESULTS_DIR / subdir
    else:
        path = RESULTS_DIR

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path

def get_cache_path(cache_type: str = "hybrid", create: bool = True) -> Path:
    """获取缓存目录路径"""
    if cache_type == "disk":
        path = DISK_CACHE_DIR
    elif cache_type == "hybrid":
        path = HYBRID_CACHE_DIR
    else:
        path = CACHE_DIR

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path

def get_benchmark_path(timestamp: str = None, create: bool = True) -> Path:
    """获取基准测试结果目录路径"""
    if timestamp:
        path = BENCHMARKS_DIR / f"benchmark_{timestamp}"
    else:
        path = BENCHMARKS_DIR

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path

# 初始化时确保目录存在
if __name__ == "__main__":
    ensure_directories()
    print("路径配置初始化完成！")
    print(f"Bench目录: {BENCH_DIR}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"缓存目录: {CACHE_DIR}")