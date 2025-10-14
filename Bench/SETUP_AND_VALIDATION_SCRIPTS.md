# 环境设置和验证脚本

本文档包含了设置和验证qibo的qibojit(numba)后端以及pennylane的lightning.qubit后端所需的所有脚本。

## 1. 环境设置脚本

### setup_advanced_env.sh (Linux/macOS)

```bash
#!/bin/bash
# 高级环境设置脚本
# 用于创建支持qibo-qibojit和pennylane-lightning.qubit的环境

set -e  # 遇到错误时退出

echo "=== 量子模拟器高级环境设置 ==="
echo ""

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: conda未找到。请确保已安装Anaconda或Miniconda。"
    exit 1
fi

echo "1. 创建高级环境..."

# 创建环境配置文件
cat > environment-advanced.yml << EOF
name: qibo-benchmark-advanced
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - numpy>=1.24.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - psutil>=5.9.0
  - pytest>=7.4.0
  - black>=23.0.0
  - flake8>=6.0.0
  - isort>=5.12.0
  - pip:
    - qibo>=0.2.21
    - qibojit>=0.1.12
    - pennylane>=0.33.0
    - pennylane-lightning>=0.33.0
    - numba>=0.58.0
EOF

# 创建环境
conda env create -f environment-advanced.yml

if [ $? -eq 0 ]; then
    echo "✅ 环境创建成功！"
else
    echo "❌ 环境创建失败！"
    exit 1
fi

echo ""
echo "2. 环境信息..."
echo "环境名称: qibo-benchmark-advanced"
echo "Python版本: 3.12"
echo "主要依赖: qibo, qibojit, pennylane, pennylane-lightning"

echo ""
echo "3. 激活环境..."
echo "运行以下命令激活环境："
echo "conda activate qibo-benchmark-advanced"

echo ""
echo "4. 验证安装..."
echo "运行验证脚本："
echo "python verify_backends.py"

echo ""
echo "=== 设置完成 ==="
echo "现在您可以运行基准测试："
echo "python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose"
```

### setup_advanced_env.bat (Windows)

```batch
@echo off
REM 高级环境设置脚本 (Windows)
REM 用于创建支持qibo-qibojit和pennylane-lightning.qubit的环境

echo === 量子模拟器高级环境设置 ===
echo.

REM 检查conda是否可用
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: conda未找到。请确保已安装Anaconda或Miniconda。
    pause
    exit /b 1
)

echo 1. 创建高级环境...

REM 创建环境配置文件
(
echo name: qibo-benchmark-advanced
echo channels:
echo   - conda-forge
echo   - defaults
echo dependencies:
echo   - python=3.12
echo   - pip
echo   - numpy^>=1.24.0
echo   - pandas^>=2.0.0
echo   - matplotlib^>=3.7.0
echo   - seaborn^>=0.12.0
echo   - psutil^>=5.9.0
echo   - pytest^>=7.4.0
echo   - black^>=23.0.0
echo   - flake8^>=6.0.0
echo   - isort^>=5.12.0
echo   - pip:
echo     - qibo^>=0.2.21
echo     - qibojit^>=0.1.12
echo     - pennylane^>=0.33.0
echo     - pennylane-lightning^>=0.33.0
echo     - numba^>=0.58.0
) > environment-advanced.yml

REM 创建环境
conda env create -f environment-advanced.yml

if %errorlevel% equ 0 (
    echo ✅ 环境创建成功！
) else (
    echo ❌ 环境创建失败！
    pause
    exit /b 1
)

echo.
echo 2. 环境信息...
echo 环境名称: qibo-benchmark-advanced
echo Python版本: 3.12
echo 主要依赖: qibo, qibojit, pennylane, pennylane-lightning

echo.
echo 3. 激活环境...
echo 运行以下命令激活环境：
echo conda activate qibo-benchmark-advanced

echo.
echo 4. 验证安装...
echo 运行验证脚本：
echo python verify_backends.py

echo.
echo === 设置完成 ===
echo 现在您可以运行基准测试：
echo python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose

pause
```

## 2. 验证脚本

### verify_backends.py

```python
#!/usr/bin/env python3
"""
验证所有后端是否可用
用于检查qibo-qibojit和pennylane-lightning.qubit后端的安装状态
"""

import sys
import traceback

def check_qibo_backends():
    """检查Qibo后端"""
    print("=== 检查Qibo后端 ===")
    
    # 检查基本Qibo
    try:
        import qibo
        print(f"✅ Qibo {qibo.__version__} 可用")
    except ImportError as e:
        print(f"❌ Qibo不可用: {e}")
        return False
    
    # 检查qibojit
    try:
        import qibojit
        print("✅ QiboJIT 可用")
    except ImportError as e:
        print(f"❌ QiboJIT不可用: {e}")
        return False
    
    # 测试numpy后端
    try:
        qibo.set_backend("numpy")
        print("✅ Qibo NumPy后端设置成功")
    except Exception as e:
        print(f"❌ Qibo NumPy后端设置失败: {e}")
        return False
    
    # 测试qibojit后端
    try:
        qibo.set_backend("qibojit")
        print("✅ QiboJIT后端设置成功")
    except Exception as e:
        print(f"❌ QiboJIT后端设置失败: {e}")
        return False
    
    return True

def check_pennylane_backends():
    """检查PennyLane后端"""
    print("\n=== 检查PennyLane后端 ===")
    
    # 检查基本PennyLane
    try:
        import pennylane as qml
        print(f"✅ PennyLane {qml.__version__} 可用")
    except ImportError as e:
        print(f"❌ PennyLane不可用: {e}")
        return False
    
    # 检查default.qubit
    try:
        dev = qml.device("default.qubit", wires=1)
        print("✅ default.qubit 可用")
    except Exception as e:
        print(f"❌ default.qubit 不可用: {e}")
        return False
    
    # 检查lightning.qubit
    try:
        dev = qml.device("lightning.qubit", wires=1)
        print("✅ lightning.qubit 可用")
    except Exception as e:
        print(f"❌ lightning.qubit 不可用: {e}")
        print("   提示: 尝试安装 pennylane-lightning: pip install pennylane-lightning>=0.33.0")
        return False
    
    return True

def check_system_requirements():
    """检查系统要求"""
    print("\n=== 检查系统要求 ===")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 12:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} (满足要求)")
    else:
        print(f"⚠️ Python {python_version.major}.{python_version.minor}.{python_version.micro} (建议使用3.12+)")
    
    # 检查numba
    try:
        import numba
        print(f"✅ Numba {numba.__version__} 可用")
    except ImportError:
        print("❌ Numba不可用 (QiboJIT需要)")
        return False
    
    # 检查numpy
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} 可用")
    except ImportError:
        print("❌ NumPy不可用")
        return False
    
    return True

def run_basic_test():
    """运行基本测试"""
    print("\n=== 运行基本测试 ===")
    
    try:
        import qibo
        import pennylane as qml
        
        # 测试Qibo简单电路
        qibo.set_backend("qibojit")
        c = qibo.models.Circuit(2)
        c.add(qibo.gates.H(0))
        c.add(qibo.gates.CNOT(0, 1))
        result = c()
        print("✅ QiboJIT基本电路测试通过")
        
        # 测试PennyLane简单电路
        dev = qml.device("lightning.qubit", wires=2)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.state()
        
        state = circuit()
        print("✅ PennyLane Lightning基本电路测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("量子模拟器后端验证工具")
    print("=" * 50)
    
    # 检查系统要求
    system_ok = check_system_requirements()
    
    # 检查Qibo后端
    qibo_ok = check_qibo_backends()
    
    # 检查PennyLane后端
    pennylane_ok = check_pennylane_backends()
    
    # 运行基本测试
    if qibo_ok and pennylane_ok:
        test_ok = run_basic_test()
    else:
        test_ok = False
    
    # 总结
    print("\n" + "=" * 50)
    print("验证总结:")
    print(f"系统要求: {'✅ 通过' if system_ok else '❌ 失败'}")
    print(f"Qibo后端: {'✅ 通过' if qibo_ok else '❌ 失败'}")
    print(f"PennyLane后端: {'✅ 通过' if pennylane_ok else '❌ 失败'}")
    print(f"基本测试: {'✅ 通过' if test_ok else '❌ 失败'}")
    
    if all([system_ok, qibo_ok, pennylane_ok, test_ok]):
        print("\n🎉 所有验证通过！您可以运行基准测试了。")
        print("\n建议的测试命令:")
        print("python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose")
        return 0
    else:
        print("\n⚠️ 验证失败！请检查上述错误并修复后重试。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## 3. 测试脚本

### test_advanced_backends.py

```python
#!/usr/bin/env python3
"""
高级后端性能测试脚本
用于快速验证qibo-qibojit和pennylane-lightning.qubit的性能
"""

import time
import numpy as np
from datetime import datetime

def test_qibo_qibojit():
    """测试QiboJIT后端性能"""
    print("=== 测试QiboJIT后端 ===")
    
    try:
        import qibo
        qibo.set_backend("qibojit")
        
        # 创建测试电路
        n_qubits = 6
        c = qibo.models.Circuit(n_qubits)
        
        # 添加QFT门
        for i in range(n_qubits):
            c.add(qibo.gates.H(i))
            for j in range(i+1, n_qubits):
                c.add(qibo.gates.CU1(i, j, np.pi/2**(j-i)))
        
        # 性能测试
        start_time = time.time()
        result = c()
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"✅ QiboJIT测试完成")
        print(f"   量子比特数: {n_qubits}")
        print(f"   执行时间: {execution_time:.4f}秒")
        print(f"   状态向量形状: {result.state().shape}")
        
        return execution_time
        
    except Exception as e:
        print(f"❌ QiboJIT测试失败: {e}")
        return None

def test_pennylane_lightning():
    """测试PennyLane Lightning后端性能"""
    print("\n=== 测试PennyLane Lightning后端 ===")
    
    try:
        import pennylane as qml
        
        # 创建设备
        n_qubits = 6
        dev = qml.device("lightning.qubit", wires=n_qubits)
        
        # 创建QFT电路
        @qml.qnode(dev)
        def qft_circuit():
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                for j in range(i+1, n_qubits):
                    qml.CRX(np.pi/2**(j-i), wires=[i, j])
            return qml.state()
        
        # 性能测试
        start_time = time.time()
        state = qft_circuit()
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"✅ PennyLane Lightning测试完成")
        print(f"   量子比特数: {n_qubits}")
        print(f"   执行时间: {execution_time:.4f}秒")
        print(f"   状态向量形状: {state.shape}")
        
        return execution_time
        
    except Exception as e:
        print(f"❌ PennyLane Lightning测试失败: {e}")
        return None

def compare_performance(qibo_time, pennylane_time):
    """比较性能"""
    print("\n=== 性能比较 ===")
    
    if qibo_time is not None and pennylane_time is not None:
        if qibo_time < pennylane_time:
            speedup = pennylane_time / qibo_time
            print(f"QiboJIT比PennyLane Lightning快 {speedup:.2f}倍")
        else:
            speedup = qibo_time / pennylane_time
            print(f"PennyLane Lightning比QiboJIT快 {speedup:.2f}倍")
        
        print(f"\n性能排名:")
        print(f"1. {'QiboJIT' if qibo_time < pennylane_time else 'PennyLane Lightning'}: {min(qibo_time, pennylane_time):.4f}秒")
        print(f"2. {'PennyLane Lightning' if qibo_time < pennylane_time else 'QiboJIT'}: {max(qibo_time, pennylane_time):.4f}秒")
    else:
        print("⚠️ 无法进行性能比较，请检查测试结果")

def main():
    """主函数"""
    print("高级后端性能测试")
    print("=" * 50)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 测试QiboJIT
    qibo_time = test_qibo_qibojit()
    
    # 测试PennyLane Lightning
    pennylane_time = test_pennylane_lightning()
    
    # 比较性能
    compare_performance(qibo_time, pennylane_time)
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    if qibo_time is not None and pennylane_time is not None:
        print("\n建议的基准测试命令:")
        print("python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose")

if __name__ == "__main__":
    main()
```

## 4. 快速启动脚本

### quick_start.sh (Linux/macOS)

```bash
#!/bin/bash
# 快速启动脚本 - 一键设置和测试

echo "=== 量子模拟器快速启动 ==="
echo ""

# 1. 设置环境
echo "1. 设置环境..."
./setup_advanced_env.sh

if [ $? -ne 0 ]; then
    echo "❌ 环境设置失败！"
    exit 1
fi

echo ""
echo "2. 验证安装..."
conda run -n qibo-benchmark-advanced python verify_backends.py

if [ $? -ne 0 ]; then
    echo "❌ 验证失败！"
    exit 1
fi

echo ""
echo "3. 运行性能测试..."
conda run -n qibo-benchmark-advanced python test_advanced_backends.py

echo ""
echo "4. 运行基准测试..."
echo "运行以下命令开始基准测试："
echo "conda activate qibo-benchmark-advanced"
echo "python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose"

echo ""
echo "=== 快速启动完成 ==="
```

### quick_start.bat (Windows)

```batch
@echo off
REM 快速启动脚本 - 一键设置和测试

echo === 量子模拟器快速启动 ===
echo.

REM 1. 设置环境
echo 1. 设置环境...
call setup_advanced_env.bat

if %errorlevel% neq 0 (
    echo ❌ 环境设置失败！
    pause
    exit /b 1
)

echo.
echo 2. 验证安装...
conda run -n qibo-benchmark-advanced python verify_backends.py

if %errorlevel% neq 0 (
    echo ❌ 验证失败！
    pause
    exit /b 1
)

echo.
echo 3. 运行性能测试...
conda run -n qibo-benchmark-advanced python test_advanced_backends.py

echo.
echo 4. 运行基准测试...
echo 运行以下命令开始基准测试：
echo conda activate qibo-benchmark-advanced
echo python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose

echo.
echo === 快速启动完成 ===
pause
```

## 5. 使用说明

### 步骤1: 环境设置

**Linux/macOS:**
```bash
chmod +x setup_advanced_env.sh
./setup_advanced_env.sh
```

**Windows:**
```cmd
setup_advanced_env.bat
```

### 步骤2: 验证安装

```bash
conda activate qibo-benchmark-advanced
python verify_backends.py
```

### 步骤3: 性能测试

```bash
python test_advanced_backends.py
```

### 步骤4: 运行基准测试

```bash
python run_benchmarks.py --simulators qibo-qibojit pennylane-lightning.qubit --qubits 2 3 4 5 6 --verbose
```

## 6. 故障排除

### 常见问题

1. **权限错误** (Linux/macOS)
   ```bash
   chmod +x *.sh
   ```

2. **conda命令不可用**
   - 确保已安装Anaconda或Miniconda
   - 检查PATH环境变量

3. **网络连接问题**
   - 使用国内镜像源：
   ```bash
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
   ```

4. **内存不足**
   - 减少量子比特数
   - 关闭其他应用程序

### 获取帮助

如果遇到问题，请检查：
1. Python版本是否为3.12+
2. 所有依赖包是否正确安装
3. 系统是否有足够的内存和磁盘空间

通过这些脚本，您应该能够轻松设置和验证高性能量子模拟器环境，并开始进行基准测试。