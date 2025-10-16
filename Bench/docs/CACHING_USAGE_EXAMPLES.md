# 缓存系统使用示例

本文档展示了如何使用新集成的参考态缓存系统来优化基准测试性能。

## 1. 基本使用

### 1.1 默认混合缓存
```bash
# 使用默认混合缓存（推荐）
python run_benchmarks.py --circuits qft --qubits 2 3 4 --simulators qibo-numpy qibo-qibojit
```

### 1.2 内存缓存
```bash
# 使用纯内存缓存（适合小规模测试）
python run_benchmarks.py --cache-type memory --memory-cache-size 128 --circuits qft --qubits 2 3 4
```

### 1.3 磁盘缓存
```bash
# 使用纯磁盘缓存（适合大规模测试）
python run_benchmarks.py --cache-type disk --cache-dir ./my_cache --circuits qft --qubits 10 12 14
```

## 2. 高级配置

### 2.1 自定义缓存参数
```bash
# 自定义混合缓存配置
python run_benchmarks.py \
    --cache-type hybrid \
    --memory-cache-size 32 \
    --cache-dir ./benchmark_cache \
    --circuits qft \
    --qubits 8 10 12 \
    --simulators qibo-numpy qibo-qibojit \
    --verbose
```

### 2.2 禁用缓存
```bash
# 禁用缓存（用于对比测试）
python run_benchmarks.py --no-cache --circuits qft --qubits 2 3 4
```

### 2.3 清空缓存
```bash
# 清空现有缓存并运行测试
python run_benchmarks.py --clear-cache --circuits qft --qubits 2 3 4
```

## 3. 缓存统计信息

### 3.1 显示缓存统计
```bash
# 运行测试并显示缓存统计信息
python run_benchmarks.py --cache-stats --circuits qft --qubits 2 3 4
```

输出示例：
```
==================================================
Cache Statistics:
  cache_type: hybrid
  stats: {'l1_hits': 5, 'l2_hits': 2, 'misses': 3, 'computations': 3, ...}
  performance: {'total_requests': 10, 'hit_rate': 0.7, ...}
  cache_info: {'l1_size': 3, 'l1_max_size': 64, ...}
==================================================
```

## 4. 性能对比

### 4.1 缓存效果测试
```bash
# 第一次运行（冷缓存）
time python run_benchmarks.py --circuits qft --qubits 10 12 14 --simulators qibo-numpy qibo-qibojit

# 第二次运行（热缓存）
time python run_benchmarks.py --circuits qft --qubits 10 12 14 --simulators qibo-numpy qibo-qibojit
```

预期结果：第二次运行应该显著更快（50-80%的时间节省）。

### 4.2 不同缓存类型对比
```bash
# 内存缓存测试
time python run_benchmarks.py --cache-type memory --memory-cache-size 128 --circuits qft --qubits 8 10 12

# 磁盘缓存测试
time python run_benchmarks.py --cache-type disk --cache-dir ./disk_cache --circuits qft --qubits 8 10 12

# 混合缓存测试
time python run_benchmarks.py --cache-type hybrid --memory-cache-size 32 --cache-dir ./hybrid_cache --circuits qft --qubits 8 10 12
```

## 5. 缓存管理

### 5.1 查看缓存内容
缓存文件存储在指定的缓存目录中（默认为`.benchmark_cache`）：

```bash
# 查看缓存目录内容
ls -la .benchmark_cache/

# 查看缓存文件大小
du -sh .benchmark_cache/
```

### 5.2 手动清理缓存
```bash
# 清理特定缓存目录
rm -rf .benchmark_cache/

# 或者使用命令行选项
python run_benchmarks.py --clear-cache --circuits qft --qubits 2 3 4
```

## 6. 编程接口使用

### 6.1 直接使用缓存类
```python
from benchmark_harness.caching import CacheConfig, create_cache_instance

# 创建缓存配置
config = CacheConfig(
    cache_type="hybrid",
    memory_cache_size=64,
    disk_cache_dir="./my_cache"
)

# 创建缓存实例
cache = create_cache_instance(config)

# 获取参考态
reference_state = cache.get_reference_state(
    circuit_name="qft",
    n_qubits=10,
    backend="qibojit"
)

# 获取缓存统计
stats = cache.get_cache_stats()
print(f"Cache hit rate: {stats['performance']['hit_rate']:.2%}")
```

### 6.2 在自定义脚本中使用
```python
from benchmark_harness.caching import HybridReferenceStateCache

# 创建混合缓存实例
cache = HybridReferenceStateCache(
    memory_cache_size=32,
    disk_cache_dir="./custom_cache"
)

# 批量获取参考态
for n_qubits in [8, 10, 12, 14]:
    state = cache.get_reference_state("qft", n_qubits, "qibojit")
    print(f"QFT {n_qubits} qubits: state shape {state.shape}")

# 显示缓存统计
stats = cache.get_cache_stats()
print(f"L1 hits: {stats['stats']['l1_hits']}")
print(f"L2 hits: {stats['stats']['l2_hits']}")
print(f"Hit rate: {stats['performance']['hit_rate']:.2%}")
```

## 7. 故障排除

### 7.1 常见问题

#### 问题1：缓存初始化失败
```
Warning: Failed to initialize cache: [错误信息]
```

**解决方案**：
- 检查缓存目录权限
- 确保磁盘空间充足
- 尝试使用不同的缓存目录

#### 问题2：缓存命中率低
```
L1 hit rate is low (0.20), consider increasing memory cache size
```

**解决方案**：
- 增加内存缓存大小：`--memory-cache-size 128`
- 检查测试模式是否一致
- 考虑使用混合缓存

#### 问题3：磁盘空间不足
```
Warning: Failed to save to disk cache: [错误信息]
```

**解决方案**：
- 清理缓存：`--clear-cache`
- 使用更大的磁盘：`--cache-dir /tmp/large_cache`
- 减少内存缓存大小以增加磁盘缓存使用

### 7.2 调试技巧

#### 启用详细输出
```bash
python run_benchmarks.py --verbose --cache-stats --circuits qft --qubits 2 3 4
```

#### 测试缓存功能
```bash
# 运行缓存集成测试
python test_caching_integration.py
```

## 8. 最佳实践

### 8.1 缓存类型选择
- **小规模测试**（≤20量子比特）：使用内存缓存
- **中等规模测试**（20-25量子比特）：使用混合缓存
- **大规模测试**（>25量子比特）：使用磁盘缓存

### 8.2 缓存大小配置
- **内存缓存**：根据可用内存设置，通常32-128个条目
- **磁盘缓存**：确保有足够磁盘空间，每个条目约8-20MB

### 8.3 性能优化建议
1. **预热缓存**：先运行一次小规模测试来预热缓存
2. **合理设置大小**：根据测试规模调整缓存大小
3. **定期清理**：定期清理过期缓存以节省空间
4. **监控性能**：使用`--cache-stats`监控缓存效果

## 9. 示例工作流

### 9.1 开发阶段工作流
```bash
# 1. 清空缓存，确保干净环境
python run_benchmarks.py --clear-cache --circuits qft --qubits 2 3 4

# 2. 开发调试，使用内存缓存
python run_benchmarks.py --cache-type memory --memory-cache-size 32 --circuits qft --qubits 2 3 4 --verbose

# 3. 性能测试，使用混合缓存
python run_benchmarks.py --cache-stats --circuits qft --qubits 8 10 12 --simulators qibo-numpy qibo-qibojit
```

### 9.2 生产环境工作流
```bash
# 1. 大规模测试，使用磁盘缓存
python run_benchmarks.py --cache-type disk --cache-dir /production/cache --circuits qft --qubits 16 18 20 22 24

# 2. 回归测试，利用缓存加速
python run_benchmarks.py --cache-stats --cache-dir /production/cache --circuits qft --qubits 16 18 20 22 24

# 3. 生成报告
python run_benchmarks.py --cache-stats --cache-dir /production/cache --circuits qft --qubits 16 18 20 22 24 --output-dir /production/results
```

通过这些示例，您可以充分利用缓存系统来显著提高基准测试的效率。