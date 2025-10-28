#!/usr/bin/env python3
"""
系统性检查项目中所有Python文件的导入问题
"""

import ast
import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Set, Tuple

class ImportChecker:
    def __init__(self):
        self.issues = []
        self.checked_files = []
        self.failed_imports = set()

    def get_all_python_files(self) -> List[str]:
        """获取所有Python文件列表（排除虚拟环境目录）"""
        cmd = 'find . -name "*.py" -not -path "./qibovenv/*" -not -path "./.git/*" -not -path "./venv/*" -not -path "./env/*"'
        result = os.popen(cmd).read().strip().split('\n')
        return [f for f in result if f and f.endswith('.py')]

    def extract_imports(self, file_path: str) -> List[str]:
        """从Python文件中提取所有import语句"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.level == 0:
                            # 绝对导入
                            imports.append(node.module)
                        else:
                            # 相对导入，转换为绝对导入
                            module_path = self.resolve_relative_import(file_path, node.module, node.level)
                            if module_path:
                                imports.append(module_path)

            return imports
        except Exception as e:
            self.issues.append({
                'file': file_path,
                'error_type': 'ParseError',
                'error': f"无法解析文件: {str(e)}"
            })
            return []

    def resolve_relative_import(self, file_path: str, module: str, level: int) -> str:
        """解析相对导入路径"""
        try:
            file_dir = os.path.dirname(os.path.abspath(file_path))
            # 向上遍历level-1个目录
            for _ in range(level - 1):
                file_dir = os.path.dirname(file_dir)

            if module:
                return f"{os.path.basename(file_dir)}.{module}"
            else:
                return os.path.basename(file_dir)
        except:
            return None

    def check_import(self, module_name: str) -> Tuple[bool, str]:
        """检查模块是否可以正常导入"""
        try:
            # 特殊处理一些内置模块
            if module_name in sys.builtin_module_names:
                return True, ""

            # 尝试导入模块
            importlib.import_module(module_name)
            return True, ""
        except ImportError as e:
            return False, str(e)
        except ModuleNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def check_file_imports(self, file_path: str):
        """检查单个文件的所有导入"""
        imports = self.extract_imports(file_path)

        for imp in imports:
            # 跳过一些常见的内置模块和标准库
            if imp in ['os', 'sys', 'json', 'time', 'datetime', 'math', 'random',
                      'itertools', 'collections', 'functools', 'typing', 'pathlib']:
                continue

            # 检查是否是项目内部模块
            if self.is_project_module(imp):
                # 检查项目内部模块是否存在
                if not self.check_project_module_exists(imp):
                    self.issues.append({
                        'file': file_path,
                        'module': imp,
                        'error_type': 'ProjectModuleNotFound',
                        'error': f"项目内部模块不存在: {imp}"
                    })
            else:
                # 检查外部模块是否可以导入
                can_import, error_msg = self.check_import(imp)
                if not can_import:
                    if imp not in self.failed_imports:
                        self.failed_imports.add(imp)
                        self.issues.append({
                            'file': file_path,
                            'module': imp,
                            'error_type': 'ImportError',
                            'error': error_msg
                        })

    def is_project_module(self, module_name: str) -> bool:
        """判断是否是项目内部模块"""
        project_prefixes = ['Bench', 'QASMBench', 'qibobench', 'test']
        return any(module_name.startswith(prefix) for prefix in project_prefixes)

    def check_project_module_exists(self, module_name: str) -> bool:
        """检查项目内部模块是否存在"""
        # 将模块名转换为文件路径
        parts = module_name.split('.')
        possible_paths = []

        # 尝试不同的路径组合
        for i in range(len(parts)):
            path_part = '/'.join(parts[i:])
            possible_paths.extend([
                f"./{path_part}.py",
                f"./{path_part}/__init__.py"
            ])

        for path in possible_paths:
            if os.path.exists(path):
                return True
        return False

    def check_circular_imports(self) -> List[Dict]:
        """检查循环导入问题（简单版本）"""
        circular_issues = []
        # 这里可以实现更复杂的循环导入检测逻辑
        return circular_issues

    def run_check(self):
        """运行完整的导入检查"""
        print("Starting comprehensive import check for all Python files...")

        files = self.get_all_python_files()
        print(f"Found {len(files)} Python files")

        for file_path in files:
            print(f"Checking file: {file_path}")
            self.checked_files.append(file_path)
            self.check_file_imports(file_path)

        # 检查循环导入
        circular_issues = self.check_circular_imports()
        self.issues.extend(circular_issues)

        print(f"\nCheck completed! Total files checked: {len(self.checked_files)}")
        return self.issues

    def generate_report(self) -> str:
        """生成详细的检查报告"""
        report = []
        report.append("=" * 80)
        report.append("Python File Import Issues Check Report")
        report.append("=" * 80)

        report.append(f"\nCheck Statistics:")
        report.append(f"- Total files checked: {len(self.checked_files)}")
        report.append(f"- Total issues found: {len(self.issues)}")
        report.append(f"- Failed import modules: {len(self.failed_imports)}")

        if not self.issues:
            report.append("\n[SUCCESS] All file imports are working correctly!")
        else:
            report.append("\n[ISSUES FOUND] Details of discovered problems:")

            # 按错误类型分组
            issues_by_type = {}
            for issue in self.issues:
                error_type = issue['error_type']
                if error_type not in issues_by_type:
                    issues_by_type[error_type] = []
                issues_by_type[error_type].append(issue)

            for error_type, issues in issues_by_type.items():
                report.append(f"\n{error_type} ({len(issues)} issues):")
                for issue in issues:
                    report.append(f"  File: {issue['file']}")
                    if 'module' in issue:
                        report.append(f"  Module: {issue['module']}")
                    report.append(f"  Error: {issue['error']}")
                    report.append("")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

if __name__ == "__main__":
    checker = ImportChecker()
    issues = checker.run_check()
    report = checker.generate_report()

    print(report)

    # 将报告保存到文件
    with open("import_check_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存到: import_check_report.txt")