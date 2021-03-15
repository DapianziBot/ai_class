## Git 常用命令/操作备忘录
大多IDE内有GUI操作界面，功能和流程上与在终端使用命令没有区别


### 0. 通过git协作举例
```
git pull # 获取远端的更新，同步其他同学的最新进度 REMOTE-01

# coding...
# done.

git commit # 保存本地做的修改。此时本地代码库会更新版本至 LOCAL-01

git pull # 非必须操作，但是在你写代码这段时间，可能有其他同学提交了更新 REMOTE-02，因此强烈建议先检查

# 分支1： git发现远端没有更新，提示当前已是最新版本
# 分支2： git发现远端有新版本 REMOTE-02，并且可以自动合并（auto-merge）生成本地版本 LOCAL-02
# 分支3： git发现远端有新版本 REMOTE-02，无法自动合并（通常是相同的文件的相同行被不同的同学修改了），需要手动合并代码，此时最好联系导致冲突的作者商量后再进行手动合并

git push # 将本地版本库的更新 LOCAL-02 推送到远端。此时其他同学使用 git pull 将获取到版本 LOCAL-02

[LOCAL-00] --> [REMOTE-01] --> [LOCAL-01] --> 合并[REMOTE-02] --> [LOCAL-02]
```


### 1. 获取代码库
```
git clone [git仓库地址] [本地名称]
```

### 2. 获取远端更新
```
git pull [远端仓库名称] [分支名称]
```
`git clone`下来的仓库会自动把远端克隆源命名`origin`,自动创建分支`master`，自动省略

### 3. 创建分支
```
git branch [分支名称]
git checkout -b [分支名称]
```
举个例子，对当前代码有一些想法但是改动可能需要好几天才能实现，这段时间项目仍然需要维护更新。这时可以通过`git branch new_ver`创建`new_ver`分支，在`master`分支上继续维护，`new_ver`分支进行新的想法验证。

### 4. 合并分支
```
git merge [分支名称] # 把目标分支的改动合并到当前分支
```
`new_ver`顺利完成验证可行，合并到`master`上一起发布：
```
git checkout master # 切换到master
git merge new_ver # 把new_ver的改动合并到master
```

### 5. 提交更新
```
git commit
```

### 6. 推送更新到远端
```
git push [远端仓库名称] [分支名称]
```

### 7. **合并冲突**
1. 冲突的文件被git标记，此时文件中包含本地和远端2个版本的片段，需要手动编辑进行修改
    + 明确知道本地的版本可以弃用，完全使用他人的版本的【小白】
    + 明确知道他人的版本可以弃用（慎重，容易引起纠纷），完全使用本地版本的【大牛】
    + 明确知道彼此的代码功能，进行人肉编辑合并代码的【大神】
2. 解决冲突后，把冲突文件重新添加到代码库中， 然后提交更新
```
git add [冲突的文件名]
git commit -m '合并了版本A和版本B，解决了冲突文件xxx...'
```

### 8. 其他操作
```
git branch              # 列出所有分支
git branch -v           # 同上，且显示各分支的最新提交版本
git branch -vv          # 同上，且显示各分支对应的远端

git remote              # 列出远端名称（如 origin）
git remote show [远端名] # 显示远端仓库的详情

git add [filename]      # 把指定文件添加到git库
git add -A              # 添加所有未被追踪的文件

```