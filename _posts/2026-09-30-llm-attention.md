---
layout: post
title: "LLM 基础（01）：Attention 篇"
description: "从缩放点积注意力出发，理解 Q、K、V、方差与 softmax 梯度饱和。"
categories: [LLM基础]
tags: [LLM, Attention, Transformer, 深度学习]
permalink: /AI/2026/09/30/llm-attention/
---

## 缩放点积注意力

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

核心思想：将每个词“想要寻找的内容”（Query）与所有词“所能提供的特征”（Key）进行匹配，然后根据匹配结果，从最相关的词中提取“实际信息”（Value）。

## 为什么除以 $\sqrt{d_k}$

- 矩阵相乘时，行与列的点积数值可能过大，使 softmax 产生极端输出（接近 0 或 1），导致梯度变小、模型难以训练。缩放可以将数值维持在合理范围内。
- 假设 Q 和 K 的每个分量都是独立同分布的随机变量，均值为 0、方差为 1，则未缩放点积的方差等于键向量维度 $d_k$，缩放后的方差约为 1。
- 因此，希望 softmax 的输入 logits 不要一开始就产生过于极端的概率分布。当输出接近 one-hot 时，softmax 对 logits 的导数通常会接近 0，反向传播的梯度会变弱。

总结：缩放点积注意力通过控制 softmax 输入 logits 的方差，避免注意力权重过早变得极端，从而减轻 softmax 梯度饱和问题。

## 点积方差为什么会随 $d_k$ 增大

两个向量的点积，是每对对应元素相乘后再求和：

$$
q^T k=q_1k_1+q_2k_2+\cdots+q_{d_k}k_{d_k}
$$

$d_k$ 越大，参与求和的项越多，总和的波动幅度也越大。

假设 query 和 key 向量的每个元素都满足均值为 0、方差为 1，则单个乘积项 $q_i k_i$ 的方差约为 1。点积是 $d_k$ 个独立乘积项之和，因此：

$$
\operatorname{Var}(q^T k)\approx d_k
$$

随机变量除以常数 $c$ 后，方差会除以 $c^2$。所以：

$$
\operatorname{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right)
=\frac{\operatorname{Var}(q^T k)}{d_k}
\approx 1
$$

## 过大的点积对 softmax 的影响

例如，对 $[50,10,5]$ 应用 softmax，结果约为 $[1,0,0]$。点积得分过大且未经缩放时，softmax 输出几乎退化为独热编码，模型表现得好像只有一个词重要，而忽略了其他位置。

相比之下，对 $[5,1,0.5]$ 应用 softmax，结果约为 $[0.971,0.018,0.011]$。虽然第一个值获得了最高注意力，但其他位置并未完全归零，模型仍然可以从所有位置学习。

需要注意的是，缩放并不意味着注意力分布永远不能变得尖锐。它主要是避免 logits 随 $d_k$ 增大而在训练早期天然过大，使 softmax 过早进入饱和状态。

## 背景：softmax 函数

$$
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

softmax 会将多分类输出转换为取值在 $[0,1]$ 且总和为 1 的概率分布。在注意力机制中，它通常沿着每一行施加，因此每一行的注意力权重之和都是 1。

## 常见问题

### Q1：Q、K、V 的权重矩阵是人工指定的吗？

不是。$W_Q$、$W_K$、$W_V$ 都是在训练过程中自动学习出来的参数。

### Q2：除以 $\sqrt{d_k}$ 会改变哪个词获得最高关注度吗？

不会。对同一行 logits 乘以同一个正数不会改变大小关系，因此通常不会改变最大值所在的位置。缩放的作用是控制数值尺度和 softmax 的温度，防止 softmax 输入过大而饱和，从而保持更有用的梯度。

### Q3：为什么每行的注意力权重加和一定是 1？

因为 softmax 是按行施加的归一化函数。

## 学习来源

- [The Math Behind Attention (QKV)](https://outcomeschool.com/blog/math-behind-attention-qkv)
- [Scaling Dot-Product Attention](https://outcomeschool.com/blog/scaling-dot-product-attention)
