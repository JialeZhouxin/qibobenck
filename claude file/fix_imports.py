#!/usr/bin/env python3
"""
修复项目中的导入问题脚本

这个脚本会自动修复所有错误的导入路径，特别是将
`from src.benchmark_harness.*` 修正为 `from src.*`
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """修复单个文件中的导入问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 修复模式：from src.caching -> from src.caching
        content = re.sub(
            r'from src\.benchmark_harness\.caching',
            'from src.caching',
            content
        )

        # 修复模式：from src.abstractions -> from src.abstractions
        content = re.sub(
            r'from src\.benchmark_harness\.abstractions',
            'from src.abstractions',
            content
        )

        # 修复模式：from src.simulators -> from src.simulators
        content = re.sub(
            r'from src\.benchmark_harness\.simulators',
            'from src.simulators',
            content
        )

        # 修复模式：from src.circuits -> from src.circuits
        content = re.sub(
            r'from src\.benchmark_harness\.circuits',
            'from src.circuits',
            content
        )

        # 修复模式：from src.metrics -> from src.metrics
        content = re.sub(
            r'from src\.benchmark_harness\.metrics',
            'from src.metrics',
            content
        )

        # 修复模式：from src.post_processing -> from src.post_processing
        content = re.sub(
            r'from src\.benchmark_harness\.post_processing',
            'from src.post_processing',
            content
        )

        # 如果内容发生了变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已修复: {file_path}")
            return True
        else:
            print(f"无需修复: {file_path}")
            return False

    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False

def find_python_files_with_bad_imports():
    """查找包含错误导入的Python文件"""
    files_to_fix = []

    # 在项目目录中查找所有Python文件
    for root, dirs, files in os.walk('.'):
        # 跳过虚拟环境和git目录
        if 'qibovenv' in root or '.git' in root:
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否包含错误的导入模式
                    if 'src.benchmark_harness' in content:
                        files_to_fix.append(file_path)

                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")

    return files_to_fix

def main():
    """主函数"""
    print("开始修复项目中的导入问题...")

    # 查找需要修复的文件
    files_to_fix = find_python_files_with_bad_imports()

    if not files_to_fix:
        print("没有发现需要修复的导入问题。")
        return

    print(f"发现 {len(files_to_fix)} 个文件需要修复:")
    for file_path in files_to_fix:
        print(f"  - {file_path}")

    print("\n开始修复...")

    fixed_count = 0
    for file_path in files_to_fix:
        if fix_imports_in_file(file_path):
            fixed_count += 1

    print(f"\n修复完成! 共修复了 {fixed_count} 个文件。")

if __name__ == "__main__":
    main()