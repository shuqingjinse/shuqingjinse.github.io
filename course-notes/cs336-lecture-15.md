---
layout: course-note
title: CS336 第 15 讲 · 中期训练与对齐：SFT、RLHF 和 DPO
description: 从指令数据与偏好标注出发，理解监督微调、奖励模型、KL 约束及直接偏好优化的关系。
permalink: /course-notes/cs336-lecture-15/
course_id: cs336
lecture: 15
instructor: Tatsunori Hashimoto
original_lessons: 'Spring 2026 · Lecture 15 · Mid/post-training: SFT and RLHF · 2026-05-18'
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf
sections:
- id: stages
  title: 从续写文本到遵循指令
- id: sft
  title: SFT 的监督信号在哪些 token 上
- id: preferences
  title: 偏好数据不是天然客观
- id: rlhf
  title: RLHF 为什么加参考策略约束
- id: dpo
  title: DPO 如何直接使用偏好对
- id: review
  title: 练习与自测
---
## 1. 从续写文本到遵循指令 {#stages}

预训练让模型学习文本分布，并不自动保证它按用户要求回答。中期训练调整能力与数据重点；SFT 用示范告诉模型如何组织回答；偏好优化再引导它在多个可行回答中选择更符合目标的行为。

这些阶段相互影响，不是可以随意交换的固定流水线。数据风格、已有能力与目标场景决定实际效果。模型学会更像助手，也不说明它在事实、数学和安全上同步提升。

**讲义定位：第 2–6、28–33 页。**下文公式和算例为补充展开。

## 2. SFT 的监督信号在哪些 token 上 {#sft}

对提示 x、目标回答 y，SFT 最小化回答条件负对数似然：

$$
L_{SFT}=-\sum_{t\in\text{回答}}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

实际实现还要约定按 token、序列或 batch 平均。聊天模板、角色 token、EOS、padding 与回答 mask 必须一致，否则模型可能在错误位置学习。

示范数据同时传递知识、语气、长度、格式和拒绝行为。对模型未掌握的长尾事实，只靠少量回答做模仿可能鼓励无依据续写；需要根据任务选择检索、数据补充或训练，并单独评估事实正确性。

**讲义定位：第 7–30 页。**

## 3. 偏好数据不是天然客观 {#preferences}

常见标注是给同一提示提供两个回答，让人选更好者。偏好会受长度、表达风格、任务规则和标注人群影响。AI 反馈能扩充规模，也会继承裁判偏差；人类与 AI 的混合并不自动消除这些问题。

补充的数据检查表：两回答是否解决同一问题、是否只因篇幅获胜、有没有事实错误、难例是否覆盖、标注一致性如何。对分歧样本保留原因，通常比强行制造一个绝对标签更有用。

奖励模型常用 Bradley–Terry 形式：

$$
P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l)).
$$

它只约束相对差异，奖励绝对零点不一定可识别。奖励分数上升不等于真实任务质量无限上升。

**讲义定位：第 34–48 页。**

## 4. RLHF 为什么加参考策略约束 {#rlhf}

一种常见目标是提高奖励，同时限制策略偏离参考模型：

$$
\max_\pi\ \mathbb E_{x,y\sim\pi}[r(x,y)]-\beta\mathbb E_x[D_{KL}(\pi(\cdot|x)\Vert\pi_{ref}(\cdot|x))].
$$

$$\beta$$ 控制奖励与约束的权衡。PPO 类流程要采样、评分、估计优势，再更新策略；参考策略与用于重要性比率的旧策略承担不同角色，不能混用。

过度追逐不完美奖励可能学会钻评分漏洞，也可能降低多样性或改变校准。应同时监控独立评估、KL、回答长度和失败案例，而不只盯着训练 reward。

**讲义定位：第 49–53、61–65 页。**

## 5. DPO 如何直接使用偏好对 {#dpo}

在相应建模假设下，DPO 将 KL 正则最优策略与隐式奖励联系起来，得到偏好对损失：

$$
L_{DPO}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right).
$$

序列 log probability 通常是所选回答 token 的 log probability 之和，需保持模板和 mask 一致。DPO 省去显式奖励模型与在线 PPO 流程，但仍依赖偏好覆盖、参考模型和训练稳定性；它并不是不再优化策略。

补充直觉：DPO 鼓励 chosen 相对于 rejected、并相对于 reference 获得更大优势，不要求每次更新都令 chosen 的绝对概率上升。不同长度还会影响序列概率，因此应观察长度与质量。

**讲义定位：第 54–60 页。**

## 6. 练习与自测 {#review}

补充练习：设计 5 个固定提示，记录 SFT 前后、偏好优化前后的回答，并分别评价格式、事实、拒绝和篇幅，避免用一个偏好票覆盖所有属性。

<details><summary>SFT 的 loss 是否应总在提示词上计算？</summary><p>取决于训练设计。常见助手微调只监督目标回答；模板和 mask 必须明确，不能把 padding 误当真实 token。</p></details>
<details><summary>DPO 中参考模型与 chosen 的作用一样吗？</summary><p>不同。chosen 是偏好标签中的较优回答，reference 提供概率基准与约束背景，两者不可互换。</p></details>
<details><summary>奖励持续提高但人类评价下降说明什么？</summary><p>可能出现奖励过优化或分布偏移。应看独立任务结果、长度、偏好偏差和具体错误，而非继续只追训练奖励。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf) · [官方视频](https://www.youtube.com/watch?v=2oH6PWPrYFo)。整理核对日期：2026-11-20。
