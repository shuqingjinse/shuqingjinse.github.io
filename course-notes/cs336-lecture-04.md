---
layout: course-note
title: CS336 第 4 讲 · 注意力替代方案与 MoE：减少计算，扩大容量
description: 理解线性注意力、状态递推、混合架构、专家路由、负载均衡及 MoE 的通信成本。
permalink: /course-notes/cs336-lecture-04/
course_id: cs336
lecture: 4
instructor: Tatsunori Hashimoto
original_lessons: Spring 2026 · Lecture 4 · Attention alternatives and mixture of experts · 2026-04-08
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf
sections:
- id: motivation
  title: 长上下文为什么需要替代方案
- id: linear
  title: 从矩阵重排到递推状态
- id: experts
  title: MoE 把容量与每 token 计算分开
- id: routing
  title: 路由学习与负载均衡
- id: systems
  title: 为什么 MoE 更难部署
- id: review
  title: 练习与自测
---
## 1. 长上下文为什么需要替代方案 {#motivation}

标准注意力在每层建立位置之间的关联，注意力核心计算随序列长度呈平方增长。FlashAttention 改善数据搬运，并没有把精确全注意力的计算量变成线性。因此可以从两个不同方向降低成本：减少需要访问的位置，或改变用于保存历史的信息形式。

| 路线 | 保留历史的方式 | 主要取舍 |
| --- | --- | --- |
| 局部或稀疏注意力 | 只访问部分历史 token | 依赖窗口、索引和选择质量 |
| 线性注意力 / 状态递推 | 将历史压缩为固定或受控状态 | 长度成本低，精确检索可能困难 |
| 混合架构 | 交替使用状态层和全注意力 | 配比、系统实现和任务表现需共同验证 |

**讲义定位：第 2–13 页。**下文递推算式与算例为补充推导。

## 2. 从矩阵重排到递推状态 {#linear}

如果去掉 softmax，矩阵乘法可以重排为 $$Q(K^\top V)$$，先计算一个与历史长度无关的状态。带因果约束时，可用：

$$
S_t=S_{t-1}+k_tv_t^\top,\qquad y_t=q_t^\top S_t.
$$

这里 $$S_t\in\mathbb R^{d_k\times d_v}$$，不再保存全部过去 token。加入特征映射和归一化可以得到其他线性注意力形式；上述最简递推不等于 softmax 注意力，也省略了稳定与归一化设计。

遗忘因子、选择性更新和 delta rule 进一步让状态能够保留、衰减或修正信息。关键问题始终是：什么信息值得进入状态，状态容量是否足够，能否同时高效训练和推理？不要只凭「时间线性」判断性能，它还依赖矩阵规模、并行扫描和硬件 kernel。

**讲义定位：第 4–11 页。**

## 3. MoE 把容量与每 token 计算分开 {#experts}

典型 MoE 用多个前馈专家替代稠密 MLP，由路由器为每个 token 选择少量专家。简化为：

$$
y=\sum_{e\in\operatorname{TopK}(r(x))}p_e(x)E_e(x).
$$

$$E_e$$ 是专家网络，$$p_e$$ 是对应权重。权重是否在选中的专家内重新归一化，取决于具体实现。所有 token 仍通常经过注意力等共享组件，因此必须同时区分**总参数、激活参数和共享计算**。

补充算例：8 个同样大小的专家，每 token 选 2 个，只能说本层执行了约四分之一的专家网络；不能说整个模型训练成本必然只有四分之一。路由、共享层、通信、负载不均和实际 batch 都要计入。

**讲义定位：第 14–35 页。**

## 4. 路由学习与负载均衡 {#routing}

稀疏选择带来一个矛盾：希望每个 token 到最合适的专家，又希望专家都被充分利用。若多数 token 集中到少数专家，其他设备空闲，拥挤专家还可能超过容量。

常见手段包括带噪路由、均衡辅助损失、调整专家选择偏置、设置容量与调度策略。token-choice 让每个 token 选择专家；expert-choice 让专家选择 token，两者的负载约束不同。共享专家可处理通用模式，更细粒度专家则改变容量分配，但效果应通过消融验证。

路由器输出过大也会造成数值问题；使用较高精度路由和 z-loss 是可能的稳定措施。均衡并非只看训练 loss，实际还要记录每个专家分配量、丢弃率与最慢专家耗时。

**讲义定位：第 26–49 页。**

## 5. 为什么 MoE 更难部署 {#systems}

专家分散在多卡时，系统要先把 token 派发到目标设备，再把结果送回来，常涉及 all-to-all。总参数多意味着权重存储仍大；每 token 激活少，也不自动意味着单请求延迟低。

| 观察到的问题 | 可能的原因 | 补充排查方法 |
| --- | --- | --- |
| FLOPs 少但步时长 | 通信或负载不均 | 看路由分布与 all-to-all 时间线 |
| 吞吐随 batch 波动 | 每个专家拿到的矩阵过小 | 比较专家有效 batch 与 kernel 效率 |
| 小数据微调退化 | 专家利用变化、过拟合 | 同时监控通用评估与专家分布 |
| 同输入结果略有变化 | 浮点顺序、容量或调度 | 固定 batch 与路由配置后复现 |

upcycling 可由已有稠密模型初始化专家；MLA 压缩注意力缓存，MTP 增加预测目标，它们与 MoE 可组合，却不是 MoE 的定义。

**讲义定位：第 44–60 页。**

## 6. 练习与自测 {#review}

补充练习：画出 4 个专家、每 token 选 2 个的派发图。给 6 个 token 指定路由，数出各专家负载，再构造一个最拥挤专家决定总耗时的例子。

<details><summary>线性注意力是否只是更快的同一个注意力函数？</summary><p>通常改变了模型函数和历史表示。FlashAttention 才是在保持精确注意力数学目标的前提下优化 IO 的例子，浮点结果仍可能略有差异。</p></details>
<details><summary>MoE 的总参数为什么不能直接代入稠密 6ND？</summary><p>稠密估算假定主要参数每 token 都参与。MoE 需分别核算激活专家、共享层和路由，再考虑通信等非 FLOPs 开销。</p></details>
<details><summary>为什么负载均衡是模型与系统共同的问题？</summary><p>路由影响学习分工，也影响设备忙闲、容量和步时。完全均匀不等于任务质量最优，完全自由又可能导致拥塞。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf) · [官方视频](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)。整理核对日期：2026-11-20。
