# Transformer 从零开始教程

通过 5 个交互式 Jupyter Notebook，从零基础到理解 "Attention Is All You Need" 论文。

## 教程结构

**第一部分：预备知识**

| 编号 | 内容 | 前置要求 |
|------|------|----------|
| 01 | 线性代数基础（向量、矩阵、点积、Softmax） | 无 |
| 02 | 深度学习基础（神经网络、梯度下降、PyTorch） | 01 |
| 03 | 序列模型与 Transformer 的动机（RNN 的问题、注意力机制） | 01, 02 |

**第二部分：论文精读**

| 编号 | 内容 | 前置要求 |
|------|------|----------|
| 04 | 注意力机制详解（Q/K/V、多头注意力、掩码） | 01-03 |
| 05 | Transformer 完整架构（位置编码、编码器、解码器、完整模型） | 01-04 |

按编号顺序学习即可，每个 notebook 都有代码示例和练习。

---

## 如何运行这些教程

教程使用 **Jupyter Notebook**（`.ipynb` 文件）——一种可以边写代码边运行、边看结果的交互式文档。下面介绍几种打开方式，选一种你觉得方便的就行。

### 方式一：VS Code（推荐，最简单）

如果你已经装了 VS Code：

1. 安装 Python 扩展：打开 VS Code → 左侧扩展图标 → 搜索 "Python" → 安装微软官方的那个
2. 安装 Jupyter 扩展：同样搜索 "Jupyter" → 安装
3. 用 VS Code 打开这个文件夹：`File → Open Folder → 选择 transformer-tutorial`
4. 点击任意 `.ipynb` 文件，就能看到 notebook 界面
5. 点击每个代码块左边的 ▶ 按钮运行，从上到下依次运行

### 方式二：JupyterLab（经典方式）

```bash
# 1. 安装 JupyterLab（如果没装过）
pip install jupyterlab

# 2. 进入教程目录
cd transformer-tutorial

# 3. 启动 JupyterLab
jupyter lab
```

浏览器会自动打开，左侧文件列表中点击 notebook 文件即可。

### 方式三：Google Colab（无需本地安装）

1. 打开 https://colab.research.google.com
2. 点击 `文件 → 上传笔记本`
3. 上传 `.ipynb` 文件
4. 直接在浏览器中运行（Google 提供免费的运行环境）

---

## 环境准备

需要安装以下 Python 包：

```bash
pip install numpy matplotlib torch
```

如果你还没有 Python 环境，推荐安装 [pyenv](https://github.com/pyenv/pyenv)，然后：

```bash
# 安装 pyenv（macOS）
brew install pyenv

# 安装 Python 3.11 并设为当前目录使用的版本
pyenv install 3.11
pyenv local 3.11

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

pip install numpy matplotlib torch jupyterlab
```

## Notebook 使用技巧

- **从上到下运行**：每个 notebook 的代码块需要按顺序执行，因为后面的代码可能依赖前面的变量
- **Shift + Enter**：运行当前代码块并跳到下一个（最常用的快捷键）
- **修改代码**：随意改！改完重新运行那个代码块就行。这是学习的最好方式
- **重新开始**：如果运行出错了，可以 `Kernel → Restart` 然后从头重新运行
- **练习题**：每个 notebook 里有标记为 Exercise 的部分，先自己试，再看下方的 Solution
