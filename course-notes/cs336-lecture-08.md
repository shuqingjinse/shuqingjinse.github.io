---
layout: course-note
title: CS336 第 8 讲 · 大规模并行：ZeRO、FSDP 与多维组合
description: 核算分片收益，理解通信拓扑、流水线气泡、序列与专家并行，并选择适合模型和集群的组合。
permalink: /course-notes/cs336-lecture-08/
course_id: cs336
lecture: 8
instructor: Tatsunori Hashimoto
original_lessons: Spring 2026 · Lecture 8 · Parallelism II · 2026-04-22
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf
sections:
- id: network
  title: 多机训练为什么需要理解网络
- id: zero
  title: ZeRO 分掉哪些状态
- id: model-parallel
  title: TP 与 PP 的通信位置不同
- id: other-axes
  title: 激活、长序列与专家还要分别切
- id: recipe
  title: 怎样设计一个组合
- id: review
  title: 练习与自测
---
## 1. 多机训练为什么需要理解网络 {#network}

集群不是一块无限大的 GPU。机内与机间连接的带宽、延迟和拓扑不同，collective 的实现也影响真实成本。某种切分如果每层都需要通信，通常比每个大阶段只通信一次更依赖高速互联。

用 $$\alpha$$ 表示一次消息的启动延迟、$$\beta$$ 表示带宽，简化通信模型为 $$t\approx n_{\mathrm{msg}}\alpha+V/\beta$$。大量小消息可能受延迟限制，大消息则更看吞吐；这只是数量级估计，还没有包含拥塞和重叠。

**讲义定位：第 4–13 页。**通信公式与后文算例为补充。

## 2. ZeRO 分掉哪些状态 {#zero}

| 方式 | 参数 | 梯度 | 优化器状态 |
| --- | --- | --- | --- |
| 普通 DP | 复制 | 复制 | 复制 |
| ZeRO-1 | 复制 | 复制 | 分片 |
| ZeRO-2 | 复制 | 分片 | 分片 |
| ZeRO-3 / 全分片思路 | 分片 | 分片 | 分片 |

分片解决的是状态冗余，执行某层时仍需拿到参与计算的权重。FSDP 类实现按单元 all-gather 参数，反向归约并分片梯度；何时释放完整权重、是否预取、是否重计算，都影响峰值内存与通信。

补充算例：按 BF16 参数 2N、BF16 梯度 2N、FP32 Adam 状态 8N 的特定配置，4 卡普通 DP 每卡静态状态为 12N bytes；理想 ZeRO-1 为 6N，ZeRO-2 为 4.5N，全部分片平均为 3N。最后一项还要加临时聚合权重、激活和通信缓冲，不能把平均值当作真实峰值。

**讲义定位：第 15–30 页。**

## 3. TP 与 PP 的通信位置不同 {#model-parallel}

TP 切分层内矩阵，频繁同步部分结果，适合高速互联域。PP 切分层，主要跨 stage 传激活及其梯度，单次通信频率和内容不同，但需要处理气泡与 stage 负载。

PP 的阶段划分不能只按层数平均：embedding、输出词表投影、不同序列长度和不同层结构都可能造成不均。调整 micro-batch 数量可以减少气泡，也可能提高调度与激活管理复杂度。更复杂的调度试图安排反向中的不同工作，但依赖实际计算图和实现支持。

**讲义定位：第 31–43 页。**

## 4. 激活、长序列与专家还要分别切 {#other-axes}

状态放下以后，激活仍可能占满显存。序列并行在部分运算中沿序列维分配激活；上下文并行让长上下文的注意力工作跨卡协作。两者相关，但不能只凭名字认为所有实现含义一致。

专家并行把不同专家放到不同 rank，用 token 派发与合并完成 MoE。注意力层和专家层可以采用不同设备分组，不能简单将所有并行度相乘就认为全部 GPU 都有有效工作。

**讲义定位：第 44–56 页。**

## 5. 怎样设计一个组合 {#recipe}

先按单卡显存核算参数、梯度、优化器、激活和临时量，再根据网络选择通信密集部分的切分范围。最后用真实任务测吞吐和稳定性，保留故障恢复与检查点成本。

| 当前瓶颈 | 可以优先试验 | 必须再检查 |
| --- | --- | --- |
| 状态内存过大 | 优化器 / 梯度 / 参数分片 | 聚合峰值与通信 |
| 单层过大 | TP | 每层归约成本 |
| 总层数很多 | PP | 气泡、stage 均衡 |
| 长上下文激活 | checkpoint、序列 / 上下文切分 | 重计算和注意力通信 |
| MoE 专家权重 | EP | 路由不均与 all-to-all |

补充算例：64 卡可以尝试 8 路 TP × 8 路 DP，但是否合理取决于机内规模、模型大小和 batch。保持全局训练 token 与更新口径一致，才能比较速度。论文中的配置是特定条件下的实例，不能当成固定处方。

**讲义定位：第 57–73 页。**

## 6. 练习与自测 {#review}

补充练习：为一个假设 7B 模型设计 8 卡训练方案，先写每卡静态内存，再列出两种可能导致真实峰值超预算的情况。

<details><summary>为什么 ZeRO-3 的峰值不等于总状态除以卡数？</summary><p>计算时会临时聚合参数，同时还有激活、缓冲、预取和分配器开销；需按运行时间线核算。</p></details>
<details><summary>为什么 TP 常安排在机内？</summary><p>它往往每层进行通信，对低延迟和高带宽要求高。机内快互联更容易摊薄这部分成本，但具体部署仍需实测。</p></details>
<details><summary>有更多 GPU 后吞吐没有线性提升，一定是模型实现差吗？</summary><p>也可能是 batch 太小、通信占比增加、流水线气泡、数据供给或设备负载不均。应先用时间线定位。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf) · [官方视频](https://www.youtube.com/watch?v=6-cXp-aOmdg)。整理核对日期：2026-11-20。
