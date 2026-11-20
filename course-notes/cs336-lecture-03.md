---
layout: course-note
title: CS336 第 3 讲 · 架构与超参数：现代 Transformer 的设计取舍
description: 从 Pre-Norm、RMSNorm、SwiGLU、RoPE 到 GQA，理解模型结构、参数预算与数值稳定性。
permalink: /course-notes/cs336-lecture-03/
course_id: cs336
lecture: 3
instructor: Tatsunori Hashimoto
original_lessons: Spring 2026 · Lecture 3 · Architectures, hyperparameters · 2026-04-06
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf
sections:
- id: baseline
  title: 先搭出可比较的基线
- id: norm
  title: 残差通路与归一化
- id: mlp
  title: SwiGLU 为什么改变中间层宽度
- id: position
  title: RoPE、头维度与 KV 头数
- id: stability
  title: 稳定性与公平实验
- id: review
  title: 练习与自测
---
## 1. 先搭出可比较的基线 {#baseline}

Transformer 的设计不是把论文里的新组件全部装上去，而是在固定数据、计算量和评估方法下，判断某个改动是否值得。一个 decoder-only 模型依次包含 token embedding、若干残差块、最终归一化和词表投影；每个块通常含因果注意力和前馈网络。

设输入为 $$X\in\mathbb R^{B\times T\times d}$$，多头注意力先将表示投影为 Q、K、V，再对每个位置能看到的历史做加权组合：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

$$M$$ 把未来位置设为负无穷。没有因果 mask，训练时模型可能偷看要预测的答案；验证 loss 再好也不能说明自回归生成能力。比较架构时，应保持 tokenizer、训练 token、上下文长度、计算预算与验证集一致。

**讲义定位：第 3–9 页。**本页公式展开、参数算例和练习为补充解释。

## 2. 残差通路与归一化 {#norm}

Pre-Norm 先归一化再进入子层，残差主通路保留加法形式：

$$
H=X+\operatorname{Attention}(\operatorname{Norm}(X)),\qquad
Y=H+\operatorname{MLP}(\operatorname{Norm}(H)).
$$

Post-Norm 则在相加后归一化。二者函数表达能力可能相近，梯度传播和训练稳定性却不同；不能只移动一行代码，再照搬全部学习率配置。

LayerNorm 去均值并除以标准差；RMSNorm 只按均方根缩放，常用可学习增益 $$g$$：

$$
\operatorname{RMSNorm}(x)_i=\frac{x_i}{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}g_i.
$$

少做运算不必然等比例加速，是否融合、内存读写和 kernel 实现都影响结果。模型中移除部分 bias 也是简化设计，但应核对具体结构，不能把「现代模型」理解为所有层都同一个模板。

**讲义定位：第 10–19 页。**

## 3. SwiGLU 为什么改变中间层宽度 {#mlp}

普通两矩阵 MLP 的主要参数约为 $$2dd_{\mathrm{ff}}$$。门控 MLP 使用三矩阵，例如：

$$
\operatorname{SwiGLU}(x)=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d.
$$

它不是挑选一个专家，而是逐元素调节信息，主要参数约为 $$3dd_{\mathrm{ff}}$$。为保持与普通 $$d_{\mathrm{ff}}=4d$$ MLP 相近的参数预算，可取门控宽度约 $$8d/3$$，实际通常再对齐硬件友好的整数。

**补充算例：**$$d=768$$ 时，普通中间宽度 3072，门控中间宽度 2048，两者的主要矩阵参数都是 4,718,592。若门控结构仍用 3072，就增加了预算，不能把全部收益归因于激活函数。

**讲义定位：第 20–29、36–41 页。**

## 4. RoPE、头维度与 KV 头数 {#position}

注意力本身不识别次序。RoPE 按位置旋转 Q、K 的二维分量，使二者内积包含相对位置信息；它不直接旋转 V。对二维旋转矩阵 $$R_i$$：

$$
(R_iq)^\top(R_jk)=q^\top R_{j-i}k.
$$

这说明相对距离怎样进入打分，不说明模型可以无代价地外推到任意长度。训练长度、频率设置和位置分布仍然重要。

标准多头注意力为每个 query 头提供独立 KV 头；MQA 共享一个 KV 头，GQA 由多组 query 头共享较少 KV 头。减少 KV 头主要降低 KV cache 和相关访存，不能把它写成「所有注意力参数减少同样倍数」。滑动窗口进一步限制可见历史；混合局部与全局层可以兼顾成本和远距离信息，但改变了模型的信息路径。

**讲义定位：第 30–35、42–47、57–66 页。**

## 5. 稳定性与公平实验 {#stability}

softmax 的指数运算对数值范围敏感。减去最大值可避免指数溢出；QK norm、logit soft-capping 和 z-loss 则分别从不同位置约束数值，它们不是同一方法的三个名字。优化技巧要连同训练精度、学习率、初始化一起考察。

| 设计问题 | 先记录什么 | 怎样验证 |
| --- | --- | --- |
| 更深还是更宽 | 参数量、步时、激活内存 | 同预算验证 loss 与稳定性 |
| 更大的词表 | 序列压缩与输出层成本 | 同一原始文本上的指标 |
| 更少的 KV 头 | cache、decode 延迟 | 长上下文与任务质量 |
| dropout、weight decay | 数据规模、重复程度 | 训练与验证曲线、消融实验 |

**讲义定位：第 44–56 页。**表格为补充实验模板。

## 6. 练习与自测 {#review}

补充练习：给一个 6 层小模型写配置表，列出 $$d$$、头数、KV 头数、MLP 宽度和位置方案。只改一个组件，并报告实际参数量与每步 token 数。

<details><summary>为什么门控 MLP 的中间宽度通常比 4d 小？</summary><p>它有三个主要权重矩阵。预算对齐时 3dd_ff 约等于 8d²，得到 d_ff 约为 8d/3；并不是中间宽度越小越好。</p></details>
<details><summary>用了 RoPE，是否就支持无限上下文？</summary><p>不能。相对位置的数学形式不等于任意距离上的训练覆盖和可靠泛化，还要考虑频率、数据和计算成本。</p></details>
<details><summary>比较 MHA 和 GQA 时应测什么？</summary><p>同时报告质量、KV cache、prefill 和 decode 性能，并固定 batch、长度、精度、硬件及测试条件。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) · [官方视频](https://www.youtube.com/watch?v=lVynu4bo1rY)。整理核对日期：2026-11-20。
