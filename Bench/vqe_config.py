#!/usr/bin/env python3
"""
VQE基准测试分层配置系统

该模块实现了分层配置设计理念，将配置分为两个层次：
1. 核心用户层 (CONFIG): 最常用且易于理解的参数，让新用户能在30秒内看懂并运行基准测试
2. 高级研究层 (ADVANCED_CONFIG): 专家级设置，用于深入、特定的基准测试

使用示例:
    from vqe_config import merge_configs, validate_config
    
    # 获取默认配置
    config = merge_configs()
    
    # 验证配置
    is_valid, errors = validate_config(config)
    if not is_valid:
        print(f"配置错误: {errors}")
    
    # 使用配置运行基准测试
    from vqe_bench import VQEBenchmarkRunner
    runner = VQEBenchmarkRunner(config)
    results = runner.run_all_benchmarks()

作者：量子计算研究团队
版本：1.0.0
"""

from typing import Dict, List, Tuple, Any, Optional
import copy

# =============================================================================
# 核心用户层配置 (CORE USER CONFIGURATION)
# 
# 这些是最常用且最易于理解的参数，目标是让一个新用户在30秒内就能看懂并运行一个有意义的基准测试。
# =============================================================================

CONFIG = {
    # 1. 你想解决什么规模的问题？
    #    量子比特数范围，定义了测试问题的规模。建议从小范围开始，如[4, 6, 8]
    "n_qubits_range": [4, 6, 8],
    
    # 2. 你想对比哪些框架？
    #    支持的框架: "Qiskit", "PennyLane", "Qibo"
    #    可以选择一个或多个框架进行对比测试
    "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],
    
    # 3. 你想用哪种主流算法思路？
    #    'HardwareEfficient' - 通用硬件高效ansatz，适用于一般问题
    #    'QAOA' - 量子近似优化算法，适用于组合优化问题
    "ansatz_type": "HardwareEfficient",
    
    # 4. 你想用哪种经典优化器？
    #    'COBYLA' - 无梯度优化器，适合参数空间较大的问题
    #    'SPSA' - 模拟梯度优化器，适合噪声环境
    #    'L-BFGS-B' - 精确梯度优化器，适合光滑问题
    "optimizer": "COBYLA",
    
    # 5. 你想让结果多可靠？(运行次数)
    #    运行次数越多，统计结果越可靠，但运行时间越长
    #    建议快速测试时使用3-5次，正式测试时使用10次或更多
    "n_runs": 3,
    
    # (可选) 实验命名，用于结果保存
    #    如果不提供，系统将自动生成基于时间戳的名称
    "experiment_name": "Standard_TFIM_Benchmark_CPU"
}

# =============================================================================
# 高级研究层配置 (ADVANCED RESEARCH CONFIGURATION)
# 
# 这些是专家级设置，用于进行深入、特定的基准测试。
# 对于标准运行，您可以安全地忽略此部分，使用系统的默认值。
# =============================================================================

ADVANCED_CONFIG = {
    # --- 1. 物理问题细节 (Problem Details) ---
    #    定义要模拟的物理模型及其参数
    "problem": {
        # 要模拟的物理模型
        #    'TFIM_1D' - 一维横向场伊辛模型，目前支持的主要模型
        "model_type": "TFIM_1D",
        
        # 边界条件定义了一维自旋链两端的处理方式
        #    'periodic' - 周期性边界 (像一个环，最后一个自旋与第一个相互作用)
        #    'open'     - 开放边界 (像一条线，两端是自由的)
        "boundary_conditions": "periodic",
        
        # 哈密顿量中的相互作用强度 (J)
        #    控制自旋间相互作用的强度
        "j_coupling": 1.0,
        
        # 哈密顿量中的横向场强度 (h)
        #    控制横向磁场对自旋翻转的影响
        "h_field": 1.0,
        
        # 在J和h中引入的随机无序强度
        #    0.0: 一个完美的、纯净的系统
        #    > 0: 引入随机性，使问题更接近真实材料，也更难求解，用于测试优化器的鲁棒性
        "disorder_strength": 0.0,
    },
    
    # --- 2. Ansatz 电路细节 (Ansatz Details) ---
    #    定义参数化量子电路的结构和特性
    "ansatz_details": {
        # Ansatz中重复块的层数(P)
        #    层数越多，表达能力越强，但参数也越多，优化难度增加
        "n_layers": 2,
        
        # Ansatz中CNOT门的连接模式，影响其表达能力和深度
        #    'linear'   - 线性连接 (qubit_i 与 qubit_{i+1} 纠缠)
        #    'circular' - 环形连接 (线性连接 + 最后一个qubit与第一个纠缠)
        #    'full'     - 全连接 (每一对qubit之间都进行纠缠)
        "entanglement_style": "linear",
    },
    
    # --- 3. 优化器与收敛细节 (Optimizer & Convergence Details) ---
    #    控制优化过程的参数和收敛条件
    "optimizer_details": {
        # 为不同优化器提供专属的超参数，以进行精细调优
        "options": {
            # COBYLA优化器的参数
            #    tol: 收敛容差，数值越小精度要求越高
            #    rhobeg: 初始步长，影响搜索范围
            "COBYLA": {"tol": 1e-5, "rhobeg": 1.0},
            
            # SPSA优化器的参数
            #    learning_rate: 学习率，控制参数更新步长
            #    perturbation: 扰动参数，影响梯度估计
            "SPSA": {"learning_rate": 0.05, "perturbation": 0.05},
            
            # L-BFGS-B优化器的参数
            #    ftol: 函数收敛容差
            #    gtol: 梯度收敛容差
            "L-BFGS-B": {"ftol": 1e-7, "gtol": 1e-5},
        },
        
        # 优化器被允许调用的最大成本函数次数
        #    数值越大，优化时间可能越长，但可能找到更好的解
        "max_evaluations": 500,
        
        # 判断VQE是否收敛的能量精度阈值
        #    当能量与精确基态能量的差值小于此阈值时，认为已收敛
        "accuracy_threshold": 1e-4,
    },
    
    # --- 4. 模拟器后端细节 (Backend Details) ---
    #    控制量子模拟的方式和后端选择
    "backend_details": {
        # 模拟方式
        #    'statevector' - 理想化的、无噪声的状态向量模拟，精确但消耗大量内存
        #    'shot_based'  - 基于采样(shots)的模拟，会引入统计噪声，但内存需求较低
        "simulation_mode": "statevector",
        
        # 在 'shot_based' 模式下使用的测量次数
        #    数值越大，统计误差越小，但模拟时间越长
        "n_shots": 8192,
        
        # 为每个框架指定高性能后端
        #    留空或设为None将使用其默认后端
        "framework_backends": {
            # Qiskit的C++高性能模拟器
            "Qiskit": "aer_simulator",
            
            # PennyLane的C++高性能模拟器
            "PennyLane": "lightning.qubit",
            
            # Qibo的JIT编译后端
            #    backend: 使用的后端类型
            #    platform: 编译平台 (numba, cupy等)
            "Qibo": {"backend": "qibojit", "platform": "numba"}
        }
    },
    
    # --- 5. 系统与I/O控制 (System & I/O) ---
    #    控制实验的系统级别设置和输入输出
    "system": {
        # 随机种子，确保实验的可复现性
        #    设置为固定值可以确保每次运行结果一致
        "seed": 42,
        
        # 是否将结果数据和图表保存到文件
        "save_results": True,
        
        # 结果输出目录
        #    所有生成的图表、数据文件和报告将保存在此目录下
        "output_dir": "./benchmark_results_high_performance/",
        
        # 是否在终端打印详细的运行进度
        #    True: 显示每个测试步骤的详细信息
        #    False: 只显示关键摘要信息
        "verbose": True,
        
        # 资源限制设置
        #    max_memory_mb: 最大内存使用量 (MB)，超过此限制将发出警告
        #    max_time_seconds: 最大运行时间 (秒)，超过此限制将停止测试
        "max_memory_mb": 4096,
        "max_time_seconds": 1800,
    }
}

# =============================================================================
# 配置管理函数
# =============================================================================

def merge_configs(core_config: Optional[Dict[str, Any]] = None, 
                 advanced_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    将高级配置合并到核心配置中
    
    Args:
        core_config: 核心用户层配置，如果为None则使用默认的CONFIG
        advanced_config: 高级研究层配置，如果为None则使用默认的ADVANCED_CONFIG
        
    Returns:
        合并后的完整配置字典
        
    Example:
        # 使用默认配置
        config = merge_configs()
        
        # 使用自定义核心配置
        custom_core = {"n_qubits_range": [4, 6], "frameworks_to_test": ["Qiskit"]}
        config = merge_configs(core_config=custom_core)
        
        # 使用自定义高级配置
        custom_advanced = {"optimizer_details": {"max_evaluations": 1000}}
        config = merge_configs(advanced_config=custom_advanced)
    """
    # 使用默认配置如果未提供
    if core_config is None:
        core_config = copy.deepcopy(CONFIG)
    else:
        core_config = copy.deepcopy(core_config)
        
    if advanced_config is None:
        advanced_config = copy.deepcopy(ADVANCED_CONFIG)
    else:
        advanced_config = copy.deepcopy(advanced_config)
    
    # 创建合并后的配置
    merged_config = copy.deepcopy(core_config)
    
    # 深度合并高级配置到核心配置中
    for key, value in advanced_config.items():
        if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
            # 如果两边都是字典，递归合并
            merged_config[key].update(value)
        else:
            # 否则直接添加或覆盖
            merged_config[key] = value
    
    return merged_config

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置的有效性
    
    Args:
        config: 要验证的配置字典
        
    Returns:
        (is_valid, errors): 元组，其中is_valid表示配置是否有效，errors是错误消息列表
        
    Example:
        is_valid, errors = validate_config(config)
        if not is_valid:
            for error in errors:
                print(f"配置错误: {error}")
    """
    errors = []
    
    # 验证核心参数
    # 1. 验证n_qubits_range
    if "n_qubits_range" not in config:
        errors.append("缺少必需参数: n_qubits_range")
    else:
        n_qubits_range = config["n_qubits_range"]
        if not isinstance(n_qubits_range, list) or len(n_qubits_range) == 0:
            errors.append("n_qubits_range 必须是非空列表")
        else:
            for n in n_qubits_range:
                if not isinstance(n, int) or n < 1:
                    errors.append(f"n_qubits_range 中的值必须是正整数，发现: {n}")
    
    # 2. 验证frameworks_to_test
    if "frameworks_to_test" not in config:
        errors.append("缺少必需参数: frameworks_to_test")
    else:
        frameworks = config["frameworks_to_test"]
        if not isinstance(frameworks, list) or len(frameworks) == 0:
            errors.append("frameworks_to_test 必须是非空列表")
        else:
            valid_frameworks = ["Qiskit", "PennyLane", "Qibo"]
            for fw in frameworks:
                if fw not in valid_frameworks:
                    errors.append(f"不支持的框架: {fw}，支持的框架: {valid_frameworks}")
    
    # 3. 验证ansatz_type
    if "ansatz_type" not in config:
        errors.append("缺少必需参数: ansatz_type")
    else:
        ansatz_type = config["ansatz_type"]
        valid_ansatz_types = ["HardwareEfficient", "QAOA"]
        if ansatz_type not in valid_ansatz_types:
            errors.append(f"不支持的ansatz类型: {ansatz_type}，支持的类型: {valid_ansatz_types}")
    
    # 4. 验证optimizer
    if "optimizer" not in config:
        errors.append("缺少必需参数: optimizer")
    else:
        optimizer = config["optimizer"]
        valid_optimizers = ["COBYLA", "SPSA", "L-BFGS-B"]
        if optimizer not in valid_optimizers:
            errors.append(f"不支持的优化器: {optimizer}，支持的优化器: {valid_optimizers}")
    
    # 5. 验证n_runs
    if "n_runs" not in config:
        errors.append("缺少必需参数: n_runs")
    else:
        n_runs = config["n_runs"]
        if not isinstance(n_runs, int) or n_runs < 1:
            errors.append(f"n_runs 必须是正整数，发现: {n_runs}")
    
    # 验证高级参数（如果存在）
    # 1. 验证problem配置
    if "problem" in config:
        problem = config["problem"]
        
        # 验证model_type
        if "model_type" in problem:
            model_type = problem["model_type"]
            valid_models = ["TFIM_1D"]
            if model_type not in valid_models:
                errors.append(f"不支持的模型类型: {model_type}，支持的模型: {valid_models}")
        
        # 验证boundary_conditions
        if "boundary_conditions" in problem:
            bc = problem["boundary_conditions"]
            valid_bc = ["periodic", "open"]
            if bc not in valid_bc:
                errors.append(f"不支持的边界条件: {bc}，支持的边界条件: {valid_bc}")
        
        # 验证物理参数
        for param in ["j_coupling", "h_field", "disorder_strength"]:
            if param in problem:
                value = problem[param]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"{param} 必须是非负数，发现: {value}")
    
    # 2. 验证ansatz_details配置
    if "ansatz_details" in config:
        ansatz_details = config["ansatz_details"]
        
        # 验证n_layers
        if "n_layers" in ansatz_details:
            n_layers = ansatz_details["n_layers"]
            if not isinstance(n_layers, int) or n_layers < 1:
                errors.append(f"n_layers 必须是正整数，发现: {n_layers}")
        
        # 验证entanglement_style
        if "entanglement_style" in ansatz_details:
            style = ansatz_details["entanglement_style"]
            valid_styles = ["linear", "circular", "full"]
            if style not in valid_styles:
                errors.append(f"不支持的纠缠样式: {style}，支持的样式: {valid_styles}")
    
    # 3. 验证optimizer_details配置
    if "optimizer_details" in config:
        optimizer_details = config["optimizer_details"]
        
        # 验证max_evaluations
        if "max_evaluations" in optimizer_details:
            max_evals = optimizer_details["max_evaluations"]
            if not isinstance(max_evals, int) or max_evals < 1:
                errors.append(f"max_evaluations 必须是正整数，发现: {max_evals}")
        
        # 验证accuracy_threshold
        if "accuracy_threshold" in optimizer_details:
            threshold = optimizer_details["accuracy_threshold"]
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                errors.append(f"accuracy_threshold 必须是正数，发现: {threshold}")
    
    # 4. 验证backend_details配置
    if "backend_details" in config:
        backend_details = config["backend_details"]
        
        # 验证simulation_mode
        if "simulation_mode" in backend_details:
            mode = backend_details["simulation_mode"]
            valid_modes = ["statevector", "shot_based"]
            if mode not in valid_modes:
                errors.append(f"不支持的模拟模式: {mode}，支持的模式: {valid_modes}")
        
        # 验证n_shots
        if "n_shots" in backend_details:
            n_shots = backend_details["n_shots"]
            if not isinstance(n_shots, int) or n_shots < 1:
                errors.append(f"n_shots 必须是正整数，发现: {n_shots}")
    
    # 5. 验证system配置
    if "system" in config:
        system = config["system"]
        
        # 验证seed
        if "seed" in system:
            seed = system["seed"]
            if not isinstance(seed, int):
                errors.append(f"seed 必须是整数，发现: {seed}")
        
        # 验证资源限制
        for param in ["max_memory_mb", "max_time_seconds"]:
            if param in system:
                value = system[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"{param} 必须是正数，发现: {value}")
    
    return len(errors) == 0, errors

def get_quick_start_config() -> Dict[str, Any]:
    """
    获取快速开始配置，适合新用户快速上手
    
    Returns:
        一个简化的配置字典，包含最常用的参数
        
    Example:
        config = get_quick_start_config()
        runner = VQEBenchmarkRunner(config)
        results = runner.run_all_benchmarks()
    """
    return merge_configs(
        core_config={
            "n_qubits_range": [4, 6],
            "frameworks_to_test": ["Qiskit"],
            "ansatz_type": "HardwareEfficient",
            "optimizer": "COBYLA",
            "n_runs": 2,
            "experiment_name": "Quick_Start_Example"
        }
    )

def get_performance_config() -> Dict[str, Any]:
    """
    获取高性能配置，适合详细的性能评估
    
    Returns:
        一个高性能配置字典，包含更多测试点和运行次数
        
    Example:
        config = get_performance_config()
        runner = VQEBenchmarkRunner(config)
        results = runner.run_all_benchmarks()
    """
    return merge_configs(
        core_config={
            "n_qubits_range": [4, 6, 8, 10],
            "frameworks_to_test": ["Qiskit", "PennyLane", "Qibo"],
            "ansatz_type": "HardwareEfficient",
            "optimizer": "COBYLA",
            "n_runs": 10,
            "experiment_name": "Performance_Evaluation"
        },
        advanced_config={
            "optimizer_details": {
                "max_evaluations": 1000,
                "accuracy_threshold": 1e-5
            },
            "system": {
                "max_memory_mb": 8192,
                "max_time_seconds": 3600
            }
        }
    )

# =============================================================================
# 兼容性函数 - 确保与现有vqe_bench.py兼容
# =============================================================================

def get_legacy_config() -> Dict[str, Any]:
    """
    获取与现有vqe_bench.py兼容的配置格式
    
    Returns:
        一个与vqe_bench.py中DEFAULT_CONFIG格式兼容的配置字典
        
    Example:
        # 在vqe_bench.py中替换DEFAULT_CONFIG
        from vqe_config import get_legacy_config
        DEFAULT_CONFIG = get_legacy_config()
    """
    merged = merge_configs()
    
    # 转换为与vqe_bench.py兼容的格式
    legacy_config = {
        # 问题与扩展性设置
        "n_qubits_range": merged["n_qubits_range"],
        "j_coupling": merged["problem"]["j_coupling"],
        "h_field": merged["problem"]["h_field"],
        
        # Ansatz 定义
        "n_layers": merged["ansatz_details"]["n_layers"],
        
        # 优化器与收敛定义
        "optimizer": merged["optimizer"],
        "max_evaluations": merged["optimizer_details"]["max_evaluations"],
        "accuracy_threshold": merged["optimizer_details"]["accuracy_threshold"],
        
        # 实验控制
        "n_runs": merged["n_runs"],
        "frameworks_to_test": merged["frameworks_to_test"],
        "seed": merged["system"]["seed"],
        
        # 资源限制
        "max_memory_mb": merged["system"]["max_memory_mb"],
        "max_time_seconds": merged["system"]["max_time_seconds"],
    }
    
    return legacy_config

if __name__ == "__main__":
    # 示例用法
    print("VQE基准测试分层配置系统")
    print("=" * 50)
    
    # 获取默认配置
    config = merge_configs()
    print("默认配置:")
    for key, value in config.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # 验证配置
    is_valid, errors = validate_config(config)
    print(f"\n配置验证: {'通过' if is_valid else '失败'}")
    if not is_valid:
        print("错误:")
        for error in errors:
            print(f"  - {error}")
    
    # 显示快速开始配置
    print("\n快速开始配置示例:")
    quick_config = get_quick_start_config()
    print(f"  量子比特数: {quick_config['n_qubits_range']}")
    print(f"  框架: {quick_config['frameworks_to_test']}")
    print(f"  运行次数: {quick_config['n_runs']}")