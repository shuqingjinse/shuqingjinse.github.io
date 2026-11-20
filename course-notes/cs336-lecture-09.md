---
layout: course-note
title: CS336 第 9 讲 · 扩展规律：用小实验决定模型与数据预算
description: 推导参数量、token 数与计算预算的关系，理解 IsoFLOP、Chinchilla、临界 batch 和外推边界。
permalink: /course-notes/cs336-lecture-09/
course_id: cs336
lecture: 9
instructor: Tatsunori Hashimoto
original_lessons: Spring 2026 · Lecture 9 · Scaling laws I · 2026-04-27
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf
sections:
- id: question
  title: 扩展规律回答什么
- id: power
  title: 幂律与不可约损失
- id: allocation
  title: 固定计算预算下怎样分配 N 与 D
- id: fit
  title: 三种拟合方法与可比性
- id: deployment
  title: 训练最优不等于产品最优
- id: review
  title: 练习与自测
---
## 1. 扩展规律回答什么 {#question}

大模型训练昂贵，不能把每个超参数都在最终规模上搜索。扩展规律用一组较小实验拟合「投入多少资源，能得到怎样的损失」，帮助选择规模与实验方向。

它是经验关系，前提包括数据分布、模型家族、训练方案和测量口径。曲线在某段范围平滑，不意味着换 tokenizer、换数据质量或换架构后仍有相同系数。预训练 loss 的可预测性，也不等于每项下游能力都同样平滑。

**讲义定位：第 2–27 页。**下文约束优化为补充展开。

## 2. 幂律与不可约损失 {#power}

简化数据扩展可以写作 $$L(D)=E+BD^{-\beta}$$；对参数和数据联合建模的一种常用形式为：

$$
L(N,D)=E+AN^{-\alpha}+BD^{-\beta},\qquad A,B,\alpha,\beta>0.
$$

$$E$$ 是拟合中的极限项，不应随意省掉后直接对原始 loss 取对数。模型参数 $$N$$ 与训练处理 token $$D$$ 在此各有一项，但真实数据重复、来源变化和优化不足可能让该形式失效。

数据翻倍会把**数据相关的超额损失项**乘以 $$2^{-\beta}$$，而不是把总 loss 减半。指数越小，同样的相对改善需要越多资源。有限数据重复训练不能永远等价于新增独立数据。

**讲义定位：第 14–27、43–49 页。**

## 3. 固定计算预算下怎样分配 N 与 D {#allocation}

对稠密模型，采用训练近似 $$C\approx6ND$$。固定 C 后，$$D=C/(6N)$$，所以：

$$
L(N)=E+AN^{-\alpha}+B\left(\frac{6N}{C}\right)^\beta.
$$

令导数为零，可得：

$$
N_*^{\alpha+\beta}=\frac{\alpha A}{\beta B}\left(\frac C6\right)^\beta,
\qquad
N_*\propto C^{\frac\beta{\alpha+\beta}},\quad
D_*\propto C^{\frac\alpha{\alpha+\beta}}.
$$

如果两指数相近，两者近似随计算量平方根增长。所谓「算力最优」依赖拟合与预算定义，不是所有模型必须遵守同一个 token / 参数比例。MoE、长上下文和大量重计算还需调整计算模型。

补充算例：假设 $$\alpha=\beta$$ 且其他条件保持不变，计算预算扩大 4 倍时，最优 N、D 各扩大约 2 倍；把 N 单独扩大 4 倍而保持 D 不变，是另一种预算分配。

## 4. 三种拟合方法与可比性 {#fit}

| 方法 | 怎么取信息 | 主要注意点 |
| --- | --- | --- |
| 训练曲线下包络 | 从多次运行中取预算处较优点 | 学习率日程、是否充分训练 |
| IsoFLOP | 固定预算，改变 N 与 D | 每条曲线的最小值要被覆盖 |
| 联合参数拟合 | 对多组 N、D 的最终 loss 拟合 | 函数形式、极限项和误差 |

不同论文的参数计数、训练日程、数据和拟合方法不同，得到的最优分配可能不同。拿未完成衰减的中间 checkpoint 与专门训练到终点的模型比较，会引入优化进度混杂。

**讲义定位：第 43–55 页。**

## 5. 训练最优不等于产品最优 {#deployment}

一个模型只训练一次，却可能生成大量 token。总成本可拆成训练成本与生命周期推理成本。更小模型用更多数据训练，可能牺牲一点训练预算上的最优性，换取长期更低的部署成本。

临界 batch 描述继续增大 batch 后的收益递减：更大的 batch 不必然按比例减少达到目标 loss 所需计算。学习率也与参数化、规模和 batch 有关，不能将小模型最优值直接当作大模型定值。

补充实践：拟合时预留一个较大规模点做外推检验，同时保留与业务相关的下游指标。每个点记录实际训练 FLOPs、唯一数据量、处理 token 数及模型计数口径。

**讲义定位：第 28–42、54–57 页。**

## 6. 练习与自测 {#review}

补充练习：画三条相同计算预算、不同 N 的训练终点曲线，解释为什么最小模型和最大模型都可能落后于中间模型。

<details><summary>「数据翻倍」为什么不代表总 loss 减半？</summary><p>幂律作用于相应超额损失项，指数也通常不等于 1；总 loss 还包含模型项与极限项。</p></details>
<details><summary>20 tokens / 参数是数学定律吗？</summary><p>不是，它来自特定研究与条件下的经验配比。不同数据、架构、预算和部署目标应重新验证。</p></details>
<details><summary>小模型曲线拟合得很好，为什么仍要留外推点？</summary><p>拟合只检验已观察区间。规模变化可能引入优化、系统或数据边界，预留点可以检验预测可靠性。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf) · [官方视频](https://www.youtube.com/watch?v=Q15rhEWZPQ4)。整理核对日期：2026-11-20。
