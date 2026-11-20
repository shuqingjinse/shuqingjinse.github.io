---
layout: course-note
title: CS336 第 7 讲 · 并行训练基础：从集合通信到模型切分
description: 通过 all-reduce、数据并行、张量并行与流水线并行，理解多卡怎样协同计算同一个训练目标。
permalink: /course-notes/cs336-lecture-07/
course_id: cs336
lecture: 7
instructor: Percy Liang
original_lessons: Spring 2026 · Lecture 7 · Parallelism I · 2026-04-20
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_07
sections:
- id: state
  title: 先明确每张卡拥有什么
- id: collectives
  title: 集合通信的四个基本动作
- id: dp
  title: 数据并行：分数据，合梯度
- id: tp
  title: 张量并行：一层矩阵由多卡完成
- id: pp
  title: 流水线：把不同层排成生产线
- id: review
  title: 练习与自测
---
## 1. 先明确每张卡拥有什么 {#state}

多卡训练首先是一个状态归属问题：每张卡保存哪些参数，处理哪批数据，算出哪个局部结果，什么时候与其他卡同步？只画几块 GPU，不写清这四件事，很容易把「模型能放下」与「训练能加速」混为一谈。

讲义用小网络说明数据并行、张量并行和流水线并行，再对照完整计算结果。学习时也应先在小输入上检查数值，再扩到多机。网络拓扑与跨机带宽会改变实际收益；GPU 数量本身不是性能保证。

**讲义定位：`hardware`、`torch_distributed`、各并行示例。**以下通信算例与检查方法为补充。

## 2. 集合通信的四个基本动作 {#collectives}

| 操作 | 输入分布 | 输出分布 |
| --- | --- | --- |
| broadcast | 某个 rank 持有完整值 | 所有 rank 得到同一值 |
| all-gather | 每个 rank 持有一块 | 所有 rank 得到拼接结果 |
| reduce-scatter | 各 rank 提供可归约输入 | 归约后每人保留一块 |
| all-reduce | 每个 rank 持有局部值 | 所有 rank 得到归约结果 |

补充算例：两张卡分别持有 `[1,2]`、`[3,4]`，sum all-reduce 后两张卡都是 `[4,6]`；all-gather 则得到 `[1,2,3,4]`。一个是在相同位置求和，一个是在新维度收集，不能互换。

在匹配布局和归约约定时，all-reduce 可以用 reduce-scatter 后接 all-gather 表达。所有参与者必须匹配调用次序、数据类型和形状；一张卡在分支里跳过 collective，其他卡可能一直等待。

**讲义定位：`collective_operations`、`collective_operations_main`、`benchmarking`。**

## 3. 数据并行：分数据，合梯度 {#dp}

每个 rank 保存同样的模型，用不同数据计算梯度，再同步得到共同更新。若每张卡有相同有效样本量，平均局部平均梯度就对应全局平均梯度：

$$
g=\frac1P\sum_{r=1}^P g_r.
$$

若各卡有效 token 数不同，需要按 token 数加权，而不是机械地除以卡数。梯度累积还要明确何时同步：每个 micro-batch 都同步可能浪费通信，只在合适边界同步则需保证语义一致。

补充算例：4 卡、每卡每个 micro-batch 处理 2 条序列，累积 8 次，有效 batch 为 64 条；若每条有效长度不同，loss 仍应按目标 token 口径核算。增加卡数后若有效 batch 也增加，训练轨迹会改变，不能只比较每步耗时。

**讲义定位：`data_parallelism`、`data_parallelism_main`。**

## 4. 张量并行：一层矩阵由多卡完成 {#tp}

对 $$Y=XW$$，按 W 的输出维切开，可以让各卡计算不同输出列，必要时再拼接；按输入维切开，则每张卡算输出的部分和，需要归约。

$$
W=[W_1\;W_2]\Rightarrow Y=[XW_1\;XW_2],
$$

$$
X=[X_1\;X_2],\quad W=\begin{bmatrix}W_1\\W_2\end{bmatrix}
\Rightarrow Y=X_1W_1+X_2W_2.
$$

列切分与行切分经常配对，用来减少中间拼接。非线性的位置很关键：一般有 $$f(a+b)\ne f(a)+f(b)$$，因此不能随意把激活函数移到归约前后。

**讲义定位：`tensor_parallelism`、`tensor_parallelism_main`。**

## 5. 流水线：把不同层排成生产线 {#pp}

流水线并行把层分给不同 stage，再将 batch 切为 micro-batch，让不同 stage 交错处理。只把前半模型放卡 1、后半放卡 2，然后逐样本串行执行，会产生大量等待。

理想的简单前向流水线有 P 个 stage、M 个等耗时 micro-batch，其利用率近似为：

$$
U\approx\frac{M}{M+P-1}.
$$

这是理解填充和排空气泡的教学模型，不是所有训练调度的精确公式。实际还有反向、负载不均、激活保存和通信。micro-batch 太小会降低矩阵效率，太大则可能显存不足。

**讲义定位：`pipeline_parallelism`、`pipeline_parallelism_main`。**

## 6. 练习与自测 {#review}

补充练习：用两张卡的纸面模型完成一次线性层前向和反向，分别用数据切分、输出列切分、输入行切分表示，标注每次通信。

<details><summary>DP 为什么不能直接让一个过大的完整模型放进单卡？</summary><p>普通 DP 每卡复制模型状态，分的主要是数据。需要模型切分或状态分片才能降低每卡相关内存。</p></details>
<details><summary>all-reduce 返回的是平均值吗？</summary><p>取决于操作及框架约定。sum 操作得到和，平均可能由框架另做归一化；自己再除一次会改变梯度尺度。</p></details>
<details><summary>为什么两卡结果应先与单卡小例子核对？</summary><p>可以发现初始化不一致、梯度少除或多除、归约轴错误和数据重复等问题，避免把错误带入大规模训练。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_07) · [官方视频](https://www.youtube.com/watch?v=SzpOcwdIL0Y)。整理核对日期：2026-11-20。
