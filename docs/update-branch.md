# 如何更新分支

本文档说明如何拉取和更新 APD 项目的最新代码。

## 基本操作

1. 检查当前分支状态

```bash
git status
```

2. 拉取最新代码

```bash
# 拉取所有远程分支信息
git fetch origin

# 拉取并合并当前分支的更新
git pull origin ocr-apd-2.0
```

## 切换分支

如果需要切换到特定分支：

```bash
# 列出所有分支
git branch -a

# 切换到 ocr-apd-2.0 分支
git checkout ocr-apd-2.0
```

## 处理本地修改

如果有未提交的本地修改：

1. 保存修改

```bash
# 暂存当前修改
git stash save "我的本地修改"
```

2. 拉取更新

```bash
git pull origin ocr-apd-2.0
```

3. 恢复修改

```bash
git stash pop
```

## 注意事项

- 拉取更新前请确保本地修改已提交或暂存
- 如果遇到冲突，请仔细解决后再继续
- 建议定期拉取更新以保持代码同步
