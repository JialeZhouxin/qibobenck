import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import time
from contextlib import contextmanager

# 确保 qibo 相关导入正确
import qibo
from qibo import Circuit, gates, set_backend
from qibo.quantum_info import infidelity

warnings.filterwarnings('ignore')

@contextmanager
def backend_context(backend_name, platform=None):
    """安全切换后端的上下文管理器"""
    original_backend = qibo.get_backend()
    try:
        set_backend(backend=backend_name, platform=platform)
        yield qibo.get_backend()
    finally:
        # 确保恢复原始后端
        set_backend(original_backend.name)

def create_variational_circuit(nqubits, params):
    """创建标准化的变分量子电路"""
    circuit = Circuit(nqubits)
    if nqubits >= 2:
        circuit.add(gates.RX(0, params[0]))
        circuit.add(gates.RY(1, params[1]))
    return circuit

def create_target_state(backend_framework, nqubits=2):
    """创建标准化的目标状态（均匀叠加态）"""
    dim = 2 ** nqubits
    if backend_framework == "tensorflow":
        import tensorflow as tf
        return tf.ones(dim, dtype=tf.complex128) / np.sqrt(dim)
    elif backend_framework == "pytorch":
        import torch
        return torch.ones(dim, dtype=torch.complex128) / np.sqrt(dim)
    else:
        return np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

def initialize_parameters(framework, nparams=2):
    """标准化参数初始化"""
    if framework == "tensorflow":
        import tensorflow as tf
        return tf.Variable(
            tf.random.uniform((nparams,), dtype=tf.float64, minval=0, maxval=2*np.pi)
        )
    elif framework == "pytorch":
        import torch
        return torch.rand(nparams, dtype=torch.float64, requires_grad=True)
    else:
        return np.random.uniform(0, 2*np.pi, nparams)

def train_vqa(backend_name, platform, nepochs=1000, learning_rate=0.01):
    """标准化的VQA训练函数"""
    try:
        with backend_context(backend_name, platform) as backend:
            # 获取框架特定的模块
            if platform == "tensorflow":
                tf = backend.tf
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                target_state = create_target_state("tensorflow")
                params = initialize_parameters("tensorflow")
                
                circuit = create_variational_circuit(2, params)
                loss_history = []
                
                print(f"  当前后端: {backend.name}")
                print(f"  TensorFlow 版本: {tf.__version__}")
                print(f"  目标状态范数: {tf.linalg.norm(target_state).numpy():.6f}")
                print(f"  初始参数: {params.numpy()}")
                
                for epoch in range(nepochs):
                    with tf.GradientTape() as tape:
                        circuit.set_parameters(params)
                        final_state = circuit().state()
                        loss = infidelity(final_state, target_state, backend=backend)
                    
                    grads = tape.gradient(loss, params)
                    optimizer.apply_gradients(zip([grads], [params]))
                    loss_history.append(loss.numpy())
                    
                    if (epoch + 1) % 200 == 0:
                        print(f"    Epoch {epoch+1:4d}: Loss = {loss.numpy():.6f}")
                
                # 计算最终结果
                final_loss = loss_history[-1]
                final_fidelity = 1 - final_loss
                final_state = circuit().state()
                probabilities = np.abs(final_state.numpy())**2
                
                print(f"  ✓ 训练完成！")
                print(f"    最终损失: {final_loss:.6f}")
                print(f"    最终保真度: {final_fidelity:.6f}")
                print(f"    最终参数: {params.numpy()}")
                print(f"    概率分布: {probabilities}")
                
                return {
                    'loss_history': [float(l) for l in loss_history],
                    'final_params': params.numpy().tolist(),
                    'final_loss': float(final_loss),
                    'final_fidelity': float(final_fidelity),
                    'probabilities': probabilities.tolist()
                }
                
            elif platform == "pytorch":
                import torch
                from qibo.quantum_info.metrics import infidelity
                
                optimizer = torch.optim.Adam
                target_state = create_target_state("pytorch")
                params = initialize_parameters("pytorch")
                optimizer = optimizer([params], lr=learning_rate)
                
                circuit = create_variational_circuit(2, params)
                loss_history = []
                
                print(f"  当前后端: qiboml (pytorch)")
                print(f"  PyTorch 版本: {torch.__version__}")
                print(f"  目标状态范数: {torch.norm(target_state).item():.6f}")
                print(f"  初始参数: {params.detach().numpy()}")
                
                for epoch in range(nepochs):
                    optimizer.zero_grad()
                    
                    circuit.set_parameters(params)
                    final_state = circuit().state()
                    loss = infidelity(final_state, target_state)
                    
                    loss.backward()
                    optimizer.step()
                    
                    loss_history.append(loss.item())
                    
                    if (epoch + 1) % 200 == 0:
                        print(f"    Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
                
                # 计算最终结果
                final_loss = loss_history[-1]
                final_fidelity = 1 - final_loss
                final_state = circuit().state()
                probabilities = np.abs(final_state.detach().numpy())**2
                
                print(f"  ✓ 训练完成！")
                print(f"    最终损失: {final_loss:.6f}")
                print(f"    最终保真度: {final_fidelity:.6f}")
                print(f"    最终参数: {params.detach().numpy()}")
                print(f"    概率分布: {probabilities}")
                
                return {
                    'loss_history': [float(l) for l in loss_history],
                    'final_params': params.detach().numpy().tolist(),
                    'final_loss': float(final_loss),
                    'final_fidelity': float(final_fidelity),
                    'probabilities': probabilities.tolist()
                }
                
    except Exception as e:
        print(f"  ❌ {platform} 训练失败: {e}")
        return None

def plot_results(results, platform, output_dir="results"):
    """绘制结果的安全函数"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        
        # 损失函数
        plt.subplot(1, 3, 1)
        color = 'blue' if platform == 'tensorflow' else 'orange'
        plt.plot(results['loss_history'], color=color)
        plt.title(f'{platform.capitalize()}: 损失函数变化')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.yscale('log')
        plt.grid(True)
        
        # 参数变化
        plt.subplot(1, 3, 2)
        plt.bar(['θ₀ (RX)', 'θ₁ (RY)'], results['final_params'], color=color)
        plt.title(f'{platform.capitalize()}: 最终参数')
        plt.ylabel('参数值 (弧度)')
        
        # 概率分布
        plt.subplot(1, 3, 3)
        states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        plt.bar(states, results['probabilities'], color=color)
        plt.title(f'{platform.capitalize()}: 概率分布')
        plt.ylabel('概率')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # 安全保存文件
        output_path = os.path.join(output_dir, f'{platform}_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    📊 {platform.capitalize()} 结果图已保存为 {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"    ⚠ 绘图失败: {e}")

def performance_comparison():
    """修复后的性能对比函数"""
    try:
        print("  进行性能对比测试...")
        
        def measure_tensorflow_time(n_epochs=100):
            with backend_context("qiboml", "tensorflow") as backend:
                tf = backend.tf
                
                params = tf.Variable(tf.random.uniform((2,), dtype=tf.float64))
                optimizer = tf.keras.optimizers.Adam()
                target_state = tf.ones(4, dtype=tf.complex128) / 2.0
                circuit = Circuit(2)
                circuit.add(gates.RX(0, params[0]))
                circuit.add(gates.RY(1, params[1]))
                
                start_time = time.time()
                for _ in range(n_epochs):
                    with tf.GradientTape() as tape:
                        circuit.set_parameters(params)
                        final_state = circuit().state()
                        loss = infidelity(final_state, target_state, backend=backend)
                    grads = tape.gradient(loss, params)
                    optimizer.apply_gradients(zip([grads], [params]))
                
                return time.time() - start_time
        
        def measure_pytorch_time(n_epochs=100):
            with backend_context("qiboml", "pytorch"):
                import torch
                from qibo.quantum_info.metrics import infidelity
                
                params = torch.rand(2, dtype=torch.float64, requires_grad=True)
                optimizer = torch.optim.Adam([params])
                target_state = torch.ones(4, dtype=torch.complex128) / 2.0
                circuit = Circuit(2)
                circuit.add(gates.RX(0, params[0]))
                circuit.add(gates.RY(1, params[1]))
                
                start_time = time.time()
                for _ in range(n_epochs):
                    optimizer.zero_grad()
                    circuit.set_parameters(params)
                    final_state = circuit().state()
                    loss = infidelity(final_state, target_state)
                    loss.backward()
                    optimizer.step()
                
                return time.time() - start_time
        
        # 测量时间
        tf_time = measure_tensorflow_time(100)
        torch_time = measure_pytorch_time(100)
        
        print(f"    TensorFlow 训练时间 (100轮): {tf_time:.4f} 秒")
        print(f"    PyTorch 训练时间 (100轮): {torch_time:.4f} 秒")
        print(f"    速度比 (TF/PyTorch): {tf_time/torch_time:.2f}")
        
    except Exception as e:
        print(f"  ❌ 性能对比失败: {e}")

def main():
    print("=" * 60)
    print("使用 Qibo 自动微分后端进行 VQA 算法演示（修复版本）")
    print("=" * 60)
    
    # 检查依赖
    try:
        from qibo import Circuit, gates, set_backend
        from qibo.quantum_info import infidelity
        import qibo
        print("✓ Qibo 导入成功")
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请安装 qibo: pip install qibo qiboml")
        return
    
    # 检查深度学习框架
    tensorflow_available = False
    pytorch_available = False
    
    try:
        import tensorflow as tf
        tensorflow_available = True
        print("✓ TensorFlow 可用")
    except ImportError:
        print("⚠ TensorFlow 不可用")
    
    try:
        import torch
        pytorch_available = True
        print("✓ PyTorch 可用")
    except ImportError:
        print("⚠ PyTorch 不可用")
    
    if not tensorflow_available and not pytorch_available:
        print("❌ 需要至少安装 TensorFlow 或 PyTorch")
        return
    
    print("\n" + "=" * 60)
    print("开始 VQA 算法演示")
    print("=" * 60)
    
    # 存储结果用于可能的后续分析
    results = {}
    
    # 运行演示
    if tensorflow_available:
        print("\n🔹 TensorFlow 后端演示")
        tf_results = train_vqa("qiboml", "tensorflow")
        if tf_results:
            results['tensorflow'] = tf_results
            plot_results(tf_results, 'tensorflow')
    
    if pytorch_available:
        print("\n🔹 PyTorch 后端演示")
        pytorch_results = train_vqa("qiboml", "pytorch")
        if pytorch_results:
            results['pytorch'] = pytorch_results
            plot_results(pytorch_results, 'pytorch')
    
    if tensorflow_available and pytorch_available:
        print("\n🔹 性能对比")
        performance_comparison()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()