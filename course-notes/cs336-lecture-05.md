---
layout: course-note
title: CS336 第 5 讲 · GPU 与 TPU：用硬件视角理解训练效率
description: 从线程、内存层级与 Tensor Core 出发，分析低精度、融合、访存合并、分块和 FlashAttention。
permalink: /course-notes/cs336-lecture-05/
course_id: cs336
lecture: 5
instructor: Tatsunori Hashimoto
original_lessons: Spring 2026 · Lecture 5 · GPUs, TPUs · 2026-04-13
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf
sections:
- id: hardware
  title: GPU 优化的是吞吐
- id: memory
  title: 显存够大与带宽够高是两件事
- id: techniques
  title: 五类优化怎样起作用
- id: tiling
  title: 分块不是越大越好
- id: flash
  title: FlashAttention：不把整个注意力矩阵写回显存
- id: review
  title: 练习与自测
---
## 1. GPU 优化的是吞吐 {#hardware}

CPU 擅长复杂控制和少量低延迟任务；GPU 用大量并行执行单元处理规则的数据计算。CUDA 的 thread 组成 block，多个 block 构成 grid；block 被调度到 SM，线程按 warp 执行。线程是编程模型，不能把「一个线程」直接理解为独占一个物理核心。

TPU 同样围绕大规模矩阵运算、片上存储与高速互联设计，但它的编程、执行与通信组织不同。讨论哪种硬件合适，应先说清工作负载，而不是只比较某个峰值数字。

**讲义定位：第 5–19 页。**本页算例和实验建议为补充。

## 2. 显存够大与带宽够高是两件事 {#memory}

| 存储层 | 特点 | 写程序时关心什么 |
| --- | --- | --- |
| 寄存器 | 靠近计算、每线程资源有限 | 使用过多可能降低并发或溢出 |
| shared memory / 片上缓存 | 可让 block 内复用数据 | 容量、同步和布局 |
| HBM | 容量较大，搬运成本更高 | 总流量、连续访问和复用 |
| 主机内存与存储 | 更远，受互联限制 | 传输、数据供给和预取 |

GPU 的算力增长与内存带宽增长并不同步。Roofline 用算术强度 $$I=C/Q$$ 连接二者：

$$
\text{有效吞吐}\leq\min(P,\beta I).
$$

$$P$$ 是计算峰值，$$\beta$$ 是所分析层级的带宽。容量决定能否放下；带宽和流量决定取数多快。一个张量放得下，并不意味着计算单元就能持续吃满。

**讲义定位：第 9–12、18–21 页。**

## 3. 五类优化怎样起作用 {#techniques}

低精度减少每元素字节数，并可能使用更高吞吐的矩阵硬件，但累积、归约和敏感算子常需更高精度。BF16、FP16、FP8 的范围、精度和硬件支持不同，不能只改 dtype 后假设训练稳定。

融合把多个相邻算子放进一个 kernel，减少中间张量落回 HBM 和重复启动。重计算在反向时重建中间量，以额外 FLOPs 换取更少存储和读取；这也是为什么「多算几次」有时反而更快。

合并访存让相邻线程访问相邻地址，减少浪费的事务。分块则把矩阵的一小片装到近处，在多次乘加中反复使用。控制流分歧会让同一 warp 的不同分支难以同时有效执行，应结合数据布局和实际执行行为分析。

**讲义定位：第 22–44 页。**

## 4. 分块不是越大越好 {#tiling}

矩阵乘法 $$C=AB$$ 中，一个输出 tile 沿归约轴逐块读取 A、B，并累积部分和。较大 tile 增加复用，却也占更多寄存器或片上空间，可能减少同时驻留的 block。

补充思考：把边长从 64 增到 128，输出元素量从 4096 增到 16384。即使复用变好，输出累加器资源也增长了 4 倍。若矩阵维度不能整除 tile，还要做掩码或填充；边缘 tile 可能浪费计算。最后一波 block 数量不足时，也会出现设备利用率的台阶变化。

因此测速曲线不一定平滑。尺寸对齐、wave quantization、kernel 选择和编译策略都可能导致相邻形状耗时突变。

**讲义定位：第 40–49 页。**

## 5. FlashAttention：不把整个注意力矩阵写回显存 {#flash}

朴素注意力会产生 $$T\times T$$ 分数和概率中间量。FlashAttention 分块处理 Q、K、V，使用在线 softmax 合并块结果，不长期物化完整概率矩阵。

合并两块统计量时，需要同时跟踪最大值 $$m$$、归一化和 $$\ell$$ 及加权值和 $$u$$。若新块最大值为 $$m_b$$，则：

$$
m'=\max(m,m_b),\quad
\ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'},\quad
u'=e^{m-m'}u+\sum_j e^{s_j-m'}v_j.
$$

最后输出 $$u'/\ell'$$。旧统计量要重新缩放，否则不同块使用不同指数基准，结果会错。该方法优化 IO 与中间存储，并不意味着全注意力计算不再是平方量级。

**讲义定位：第 50–55 页；在线合并写法为补充展开。**

## 6. 练习与自测 {#review}

补充练习：给逐元素激活、矩阵乘法、注意力各画一条「HBM → 片上 → 计算 → HBM」路径，标记可融合或复用的中间量。

<details><summary>更低精度一定能按字节比例加速吗？</summary><p>不能。格式转换、缩放、归约、实际 kernel 及原有瓶颈都会影响效果；还必须检查数值与任务质量。</p></details>
<details><summary>FlashAttention 为什么要保留每行最大值？</summary><p>在线 softmax 需要稳定的指数基准，并在合并新块时对旧累积值重新缩放，才能与全行归一化对应。</p></details>
<details><summary>FLOPs 减少但时间没变，应先查什么？</summary><p>看是否仍受 HBM、启动或通信限制，并确认测量已预热、同步且没有把编译时间混进稳态结果。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf) · [官方视频](https://www.youtube.com/watch?v=izZba4UA7iY)。整理核对日期：2026-11-20。
