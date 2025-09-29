# 云端仓库上传指南

## 当前项目状态
- ✅ Git仓库已初始化
- ✅ 代码已提交到本地仓库
- ✅ README文档已添加
- ⚠️ 等待配置远程仓库

## 可选的云端仓库平台

### 1. GitHub
```bash
# 在GitHub上创建新仓库，然后执行：
git remote add origin https://github.com/用户名/仓库名.git
git branch -M main
git push -u origin main
```

### 2. GitLab
```bash
# 在GitLab上创建新仓库，然后执行：
git remote add origin https://gitlab.com/用户名/仓库名.git
git branch -M main
git push -u origin main
```

### 3. Gitee (码云)
```bash
# 在Gitee上创建新仓库，然后执行：
git remote add origin https://gitee.com/用户名/仓库名.git
git branch -M main
git push -u origin main
```

### 4. Azure DevOps
```bash
# 在Azure DevOps上创建项目，然后执行：
git remote add origin https://dev.azure.com/组织名/项目名/_git/仓库名
git branch -M main
git push -u origin main
```

## 项目特点说明

### 项目优势
1. **完整的Qibo测试环境** - 包含所有主要后端
2. **QASMBench基准测试** - 丰富的量子电路测试集
3. **性能比较工具** - 自动比较不同后端性能
4. **详细文档** - 中文和英文的使用说明

### 技术栈
- **框架**: Qibo 量子计算框架
- **语言**: Python 3.8+
- **测试**: QASMBench 基准测试集
- **环境**: 虚拟环境隔离

## 上传前注意事项

### 1. 敏感信息检查
确保没有包含：
- API密钥
- 密码文件
- 个人配置信息
- 大型二进制文件

### 2. 许可证检查
QASMBench 基准测试集有特定的许可证要求，请确保遵守。

### 3. 文件大小
检查是否有过大的文件需要排除。

## 快速上传脚本

创建 `upload.sh` 脚本：
```bash
#!/bin/bash
# 上传到GitHub
git remote add origin $1
git branch -M main
git push -u origin main
echo "项目已成功上传到: $1"
```

使用方法：
```bash
./upload.sh https://github.com/用户名/仓库名.git
```

## 后续维护

### 1. 定期更新
```bash
git add .
git commit -m "更新说明"
git push
```

### 2. 版本标签
```bash
git tag v1.0.0
git push origin v1.0.0
```

### 3. 分支管理
```bash
git checkout -b feature/new-feature
git push -u origin feature/new-feature
```

## 技术支持

如有上传问题，请参考：
- Git官方文档
- 对应平台的帮助文档
- 项目README中的联系方式