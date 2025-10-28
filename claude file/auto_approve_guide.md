# Claude 自动批准配置指南

## 🎯 目标
配置Claude Code自动批准工具请求，减少手动确认的步骤。

## 📋 配置方法

### 方法1: 修改项目配置 (推荐)

编辑 `C:\Users\Administrator\.claude.json` 文件：

```json
{
  "projects": {
    "E:\\qiboenv": {
      "allowedTools": ["*"],
      "hasTrustDialogAccepted": true,
      "ignorePatterns": []
    }
  }
}
```

### 方法2: 启动时参数

```bash
# 自动接受编辑
claude --permission-mode acceptEdits

# 跳过所有权限检查 (仅限受信任环境)
claude --dangerously-skip-permissions

# 允许特定工具
claude --allowed-tools "Read,Write,Edit,Bash,Grep"

# 启动时加载配置文件
claude --settings claude_settings.json
```

### 方法3: 环境变量

```bash
# 设置环境变量
export CLAUDE_PERMISSION_MODE=acceptEdits
export CLAUDE_AUTO_APPROVE=true
```

## 🔧 权限模式选项

| 模式 | 描述 | 安全性 |
|-----|------|--------|
| `default` | 默认模式，需要手动确认 | 🔒 高 |
| `acceptEdits` | 自动接受文件编辑 | 🔓 中 |
| `bypassPermissions` | 跳过所有权限检查 | ⚠️ 低 |
| `plan` | 计划模式，不执行实际操作 | 🔒 高 |

## ⚙️ 推荐配置

对于你的量子计算项目，建议使用：

```json
{
  "projects": {
    "E:\\qiboenv": {
      "allowedTools": ["*"],
      "hasTrustDialogAccepted": true,
      "ignorePatterns": ["qibovenv/*", ".git/*", "__pycache__/*"]
    }
  },
  "autoApproveEdits": true,
  "permissionMode": "acceptEdits"
}
```

## 🚨 安全注意事项

1. **仅在受信任的环境中使用自动批准**
2. **避免在包含敏感信息的项目中使用**
3. **定期检查Claude的操作历史**
4. **使用ignorePatterns排除重要文件**

## 🔄 生效方法

配置修改后：
1. 重启Claude Code会话
2. 或使用 `claude --continue` 继续当前会话

## 📝 验证配置

运行以下命令验证配置是否生效：

```bash
# 测试文件编辑权限
echo "test" > test_auto_approve.txt

# 如果文件自动创建，说明配置成功
rm test_auto_approve.txt
```