---
layout: course-note
title: CS336 第 10 讲 · 推理系统：KV Cache、批处理与推测解码
description: 分清 prefill 与 decode，估算 KV cache，理解量化、连续批处理、PagedAttention 与精确推测采样。
permalink: /course-notes/cs336-lecture-10/
course_id: cs336
lecture: 10
instructor: Percy Liang
original_lessons: Spring 2026 · Lecture 10 · Inference · 2026-04-29
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_10
sections:
- id: stages
  title: 推理是两个不同阶段
- id: cache
  title: KV Cache 保存什么
- id: quantization
  title: 量化省什么，付出什么
- id: batching
  title: 批处理与分页管理
- id: speculative
  title: 推测采样怎样保持目标分布
- id: review
  title: 练习与自测
---
## 1. 推理是两个不同阶段 {#stages}

prefill 一次处理提示词的多个位置，构建各层历史状态；decode 根据已有历史逐步生成新 token。大矩阵有较好的权重复用，小 batch 的单步 decode 则可能花大量时间搬运权重和缓存。

不能把「推理」统一归成带宽受限：batch、长度、模型、精度和并行都会改变瓶颈。用户感受到的首 token 时间 TTFT、后续 token 间隔和系统总吞吐也不是同一个指标。

**讲义定位：`review_transformer`、`review_of_arithmetic_intensity`、`arithmetic_intensity_of_inference`、`throughput_and_latency`。**

## 2. KV Cache 保存什么 {#cache}

因果注意力中，过去位置的 K、V 在后续生成时不变。缓存它们，可避免每一步重新对整段历史做全部投影。Q 通常只需当前生成位置的表示；cache 不是把历史注意力概率矩阵永久保存下来。

设层数 L、batch B、历史长度 T、KV 头数 $$h_{kv}$$、每头维度 $$d_h$$、每元素 s bytes，则：

$$
M_{KV}\approx2LBT h_{kv}d_hs.
$$

补充算例：L=32、B=1、T=4096、$$h_{kv}=8$$、$$d_h=128$$、BF16，两份 K、V 共约 512 MiB。它不含参数、临时量和内存碎片。batch 或长度翻倍，该配置的 cache 也翻倍。

GQA / MQA 减少 KV 头；滑动窗口限制部分层的历史；MLA 等方法用不同表示压缩缓存。每种方法都需核对模型训练与推理实现。

**讲义定位：`compute_transformer_performance_stats`、`reduce_kv_cache_size`。**

## 3. 量化省什么，付出什么 {#quantization}

权重量化减少权重字节，可能改善带宽与容量；cache 量化针对另一笔内存；激活量化又有不同数值要求。不能从「4 bit 权重」直接推断所有中间量都占半字节。

简单对称量化可表示为 $$q=\operatorname{round}(x/s)$$，反量化近似 $$\hat x=sq$$，并受可表示范围限制。缩放粒度、异常值、累积精度和硬件解码都会影响误差与速度。

更小文件不等于更低延迟。若没有合适 kernel，转换和布局开销可能抵消收益。剪枝同样需要实际结构和硬件利用，稀疏权重不自动变成更少的运行时间。

**讲义定位：`quantization`、`model_pruning`。**

## 4. 批处理与分页管理 {#batching}

静态 batch 要等待同一批请求处理，短请求结束后可能留下空位。连续批处理在迭代边界移出完成请求、加入新请求，提高服务利用率，但需要处理长度不齐与抢占。

PagedAttention 把逻辑上连续的 KV 序列映射到物理块，减少预留长缓冲的浪费，并支持更灵活的管理。它改变缓存组织，不是让注意力本身变为近似算法，也不能解决无限长度缓存的总容量问题。

补充服务指标：分别记录不同提示长度和输出长度下的 TTFT、每 token 延迟、总 tokens/s、并发、P50 / P95，以及显存。只报最大吞吐会掩盖排队时间。

**讲义定位：`continuous_batching`、`paged_attention`。**

## 5. 推测采样怎样保持目标分布 {#speculative}

小草稿模型先提出多个 token，大模型并行验证。若草稿分布为 q、目标分布为 p，对草稿 token x 采用接受率：

$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
$$

被拒绝时从归一化后的 $$\max(p-q,0)$$ 残差分布采样，而不是简单地选大模型概率最高的 token。相应条件下，这能保持目标采样分布。temperature、截断等处理必须纳入 p、q 的实际定义。

收益取决于草稿成本、接受率、验证成本和硬件利用；草稿越小不一定越好，越接近目标通常接受率更高却也更贵。贪心验证是另一种场景，不能把它与概率采样证明混用。

**讲义定位：`speculative_sampling`；接受与拒绝机制为补充解释。**

## 6. 练习与自测 {#review}

补充练习：比较 1 个长请求和 8 个短请求时的缓存预算，分别讨论如何调 batch、何时可能牺牲首 token 延迟换取吞吐。

<details><summary>缓存 K、V 后，decode 是否不再依赖历史长度？</summary><p>不是。当前 query 仍需访问历史的 K、V，缓存节省的是重复投影等工作，不是所有历史读取和注意力计算。</p></details>
<details><summary>为何 PagedAttention 不等于减少了每个 token 的 KV 内容？</summary><p>它主要优化物理分配、碎片与共享。压缩 KV 表示是另一个维度，可以组合但不能混淆。</p></details>
<details><summary>推测采样直接拒绝后贪心补一个 token 可以吗？</summary><p>若目标是保持随机采样分布，不可以，必须使用正确的残差分布与验证规则。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_10) · [官方视频](https://www.youtube.com/watch?v=EfM546A79aM)。整理核对日期：2026-11-20。
