---
layout: post
title: "LLM 基础（02）：KV Cache 篇"
description: "理解 KV Cache 的工作原理、显存开销，以及量化、Token 剔除、GQA 和 MLA 等压缩方法。"
categories: [LLM基础]
tags: [LLM, KV Cache, Attention, 推理优化, GQA, MLA]
permalink: /AI/2026/09/30/llm-kv-cache/
---

# KV Cache 介绍

KV Cache 是 LLM 内部的一种缓存机制，用于保存早期 Token 已完成的计算工作，从而使模型在生成每个新 Token 时无需重复进行相同的计算。

## LLM 如何生成文本

每次模型预测一个新的 Token 时，它都需要查看之前所有的 Token 来决定接下来生成什么。

这种逐 Token 生成的过程分为两个阶段：预填充（Prefill）和解码（Decode），而 KV Cache 是连接两者的桥梁。

## 模型内部发生了什么

模型包含一个叫作注意力层（Attention Layer）的组件。它帮助模型判断哪些先前的 Token 对预测下一个 Token 是重要的。

注意力层内部有三个矩阵：Query、Key、Value。

当前 Token 使用它的 Query 与之前所有 Token 的 Key 进行对比，得到注意力得分（Attention Scores）。这些数值告诉模型应该给每个先前的 Token 赋予多少关注度，然后模型使用这些得分作为权重，与 Value 一起完成计算。

## 问题：重复计算

每次模型预测下一个 Token 时，都会为序列中的所有 Token 计算 Key、Value 和 Query，而不仅仅是针对最新的 Token。

## 什么是 KV Cache？

KV Cache 是一种内存缓存，用于保存已经处理过的每个 Token 的 Key 和 Value，从而使模型无需再次计算它们。

## 为什么只缓存 Key 和 Value，而不缓存 Query？

Query 只对当前 Token 有用，也就是当前正在生成的那个 Token。当前 Token 使用它的 Query 与之前所有 Token 的 Key 进行对比，以找出哪些内容相关。一旦这一步预测完成，该 Query 就不再需要了。

但是，每个过去 Token 的 Key 和 Value 在未来每一步都需要被使用，因为每个新 Token 都必须查看所有先前 Token 来做出预测。

所以，只需要存储 Key 和 Value。

## 能快多少？

举个具体例子：如果模型要生成一个包含 100 个 Token 的序列：

- 不使用 KV Cache，跨所有步骤的 Key 和 Value 计算总次数为 $2 + 3 + 4 + \cdots + 100 = 5{,}049$ 次。
- 使用 KV Cache，计算次数为 $2 + 1 + 1 + \cdots + 1 = 101$ 次。

这大约减少了 50 倍的计算量。序列越长，节省的计算量就越多。

这就是 KV Cache 在加速文本生成方面如此有效的原因。

## 权衡：速度 vs. 显存/内存

KV Cache 提高了生成速度，但也带来了权衡（Trade-off）：它需要额外的内存或显存来存储迄今为止生成的每个 Token 的所有 Key 和 Value 信息。

对于包含数千个 Token 的超长序列，缓存可能会消耗大量显存。

常见方法之一是仅保留最近窗口内的 Token 以及最初的几个注意力汇聚 Token（Attention Sinks），从而使缓存保持固定大小。

因此，KV Cache 是一种用更多内存换取计算时间的权衡。当同时为许多用户提供 LLM 服务时，高效管理 KV Cache 显存就变得至关重要。

## 常见问题

- **Q1：使用 KV Cache 时，模型还会为新 Token 进行计算吗？**
  - 答：会。模型仍然会计算新 Token 的 Key、Value 和 Query，只是跳过之前 Token 的重复计算，直接从缓存中读取它们的 Key 和 Value。然后，它会将新 Token 的 Key 和 Value 保存到缓存中以备后用。
- **Q2：KV Cache 可以在不同的请求之间复用吗？**
  - 答：可以。在共享相同起始文本（如系统提示词或 Prompt）的多个请求之间，可以复用已经缓存的 Key 和 Value。这被称为 Prompt Caching（提示词缓存）。
- **Q3：如何防止长对话中的 KV Cache 无限增长？**
  - 答：一种常见的解决办法是仅保留最近窗口内的 Token 以及最初的几个注意力汇聚 Token（Attention Sinks），让缓存保持固定大小。这是多种 KV Cache 压缩技术之一。
- **Q4：KV Cache 解决了 LLM 的显存/内存问题吗？**
  - 答：没有。KV Cache 节省的是计算时间，但会消耗额外显存，并且缓存会随着序列变长而增大。在多用户并发服务时，Paged Attention 才是专门解决 KV Cache 内存管理问题的技术。

# KV Cache 压缩

KV Cache 压缩是一组旨在减少模型生成回复时用于记录上下文的内存技术，让模型能够在消耗更少内存的情况下处理更长的文本。

## 什么是 LLM？它如何生成文本

模型不是一次性生成整段回复，而是一小块一小块地生成。

每生成一个新的 Token，模型都需要“回头看”之前生成的所有 Token。这个“回头看”的过程，就是由注意力机制（Attention）完成的。

## 什么是 Attention

Attention 是模型用来判断之前哪些 Token 对生成下一个 Token 更重要的机制。简单来说，Attention 就是“回顾并挑选出关键信息”。

具体来说，Attention 通过 Q、K、V 实现：

- **Query（Q）**：当前 Token 正在寻找的内容。
- **Key（K）**：每个历史 Token 所能提供的特征标识，类似标签。
- **Value（V）**：每个历史 Token 所承载的实际信息。

新 Token 的 Query 会与所有历史 Token 的 Key 进行比对，根据匹配度，将对应的 Value 加权组合后向下传递，用于帮助生成下一个 Token。

此外，模型并非只执行一次 Attention，而是拥有多个依次叠加的层（Layers），每一层又包含多个注意力头（Attention Heads）。每个 Attention Head 都像是一个侧重点不同的读者：有的负责看语法，有的负责看人名，有的负责寻找代词的指代关系。

## 什么是 KV Cache

KV Cache = Key + Value + Cache（缓存）。

缓存是一种将计算结果暂存以便复用的内存机制。KV Cache 就是模型用来存储所有历史 Token 的 Key 和 Value 的内存区域，以便生成后续 Token 时直接复用。

## 为什么 KV Cache 会变得非常庞大

以 Llama 2 模型为例：

- 32 层；
- 每层 32 个注意力头；
- 每个 Head 的 Key 向量维度是 128，Value 向量维度是 128；
- 每个数值采用 16 位（即 2 字节）存储。

单个 Token 在缓存中占用的内存大小：

$$
2 \times (128+128) \times 32 \times 32 = 524288\text{ 字节} \approx 0.5\text{ MB}
$$

一段 4000 Token 的对话：

$$
4000 \times 0.5\text{ MB}=2\text{ GB}
$$

一段 100000 Token 的长文本：

$$
100{,}000 \times 0.5\text{ MB}=50\text{ GB}
$$

大模型运行在专门擅长矩阵计算的 GPU 上。一块 GPU 通常拥有 24 GB 至 80 GB 的显存，且大部分显存已被模型权重占用。

如果有 100 个用户同时在一块 GPU 上与模型对话，显存需求将远超 GPU 的承受极限。

总结：KV Cache 的体积随文本长度和并发用户数成正比增长，会极大消耗 GPU 显存空间。同时，每生成一个新 Token，模型都要从显存中读取完整缓存；缓存越大，读取耗时越长，生成速度就越慢。

## 什么是 KV Cache 压缩

KV Cache 压缩是指在基本不牺牲模型输出质量的前提下，降低 KV Cache 所占显存空间的一系列技术。

实现这一目标的方法很多：部分方法可以直接应用于**现有**的预训练模型，另一部分则需要在模型**训练前**融入架构设计。

## 方法一：量化

量化是指使用更少的位数（Bits）来存储缓存中的每个数值。

在 KV Cache 中，默认每个数值使用 16 位（FP16/BF16）存储。通过量化技术，可以将其降至 8 位（INT8）甚至 4 位（INT4）：

- 16 位降至 8 位：缓存体积减少至 $1/2$。
- 16 位降至 4 位：缓存体积减少至 $1/4$。
- 优点：容易集成部署；无需剔除任何 Token；精度损失通常很小。
- 缺点：低于 4 位后，舍入误差增大，模型输出质量可能显著下降，因此压缩存在下限。

由于 Key 和 Value 的数值分布特性不同（Key 的某些位置常出现极大的离群值），像 **KIVI** 这样的量化技术会对 Key 和 Value 分别采用差异化的量化策略，以尽量减小误差。

## 方法二：Token 剔除

Token 剔除是指直接从缓存中移除不重要 Token 的 Key 和 Value。

在模型计算 Attention 时，绝大部分注意力往往集中在少数关键 Token 上，其余大多数 Token 分配到的注意力接近于零。通过在文本生成过程中累计每个 Token 获得的注意力得分，得分最高的 Token 即为“重度注意 Token”（Heavy Hitters，如 **H2O** 技术所述）。

一种常见的剔除规则是：

- 设定固定的缓存上限，例如 1000 个 Token；
- 始终保留最新的 Token，因为近期上下文几乎总是重要的；
- 在历史 Token 中仅保留 Heavy Hitters；
- 剔除其余 Token。

采用这种方法后，无论对话延长到多少字，缓存体积都将固定在设定的预算上限内。

### 注意力汇聚现象：Attention Sinks

研究表明，无论文本开头的几个 Token 具体内容是什么，它们总会获得较高的注意力。

这是因为 Attention 强行要求权重总和为 100%。当没有合适的历史 Token 可关注时，模型倾向于将多余的注意力“倾倒”在开头 Token 上，这就是 StreamingLLM 发现的 Attention Sinks 现象。

因此，通常不能剔除开头的几个 Token，否则模型输出可能退化为无意义的乱码。

- 优点：缓存体积完全固定，解决长文本下缓存无限增长的问题。
- 缺点：Token 一旦被剔除便永久丢失。如果用户后续提问涉及被剔除的具体细节，模型将无法精确检索。它适合对话、摘要等任务，但不适合精确回忆任务。

## 方法三：跨 Head 共享 Key 与 Value

不同于前述模型训练后的处理手段，本方法需要在模型设计与训练阶段就对架构进行调整。

在一个拥有 32 个 Attention Head 的层中，默认配置会产生 32 套独立的 Key 和 Value。但研究表明，很多 Head 可以共享相同的一套 Key 和 Value，仅保留各自独立的 Query。

主要共享架构模式：

- **Multi-Query Attention（MQA）**：同一层中的所有 Head 共享一套 Key 和 Value，缓存体积理论上可缩小至约 $1/32$。
- **Grouped-Query Attention（GQA）**：将 Head 分组，组内共享一套 Key 和 Value。例如 32 个 Head 分为 8 组，缓存体积缩小至约 $1/4$。

GQA 在大幅节省显存的同时通常不会明显牺牲模型质量，因此已被 Llama 3、Mistral、Gemma 等许多主流开源模型采用。

- 优点：不丢弃 Token 信息，没有量化精度损失，内建于模型架构中，推理时无额外的缓存压缩开销。
- 缺点：必须在模型训练前确定，无法直接应用于已有预训练模型；缓存依然会随着文本长度增加，只是增长速率放缓。

## 方法四：低秩压缩

低秩压缩是指在缓存中仅保存 Key 和 Value 的低维压缩表示，在需要计算时再恢复展开。

与其保存每个 Head 庞大的原始 Key 和 Value，模型可以将它们压缩存储为一个低维隐向量（Latent Vector），需要计算 Attention 时，再通过训练好的小型线性变换将其还原。

DeepSeek 在 DeepSeek-V2 和 DeepSeek-V3 中提出的 MLA（Multi-Head Latent Attention）技术就是这种思路的典型代表。

例如，原本每层需要存储 $128 \times 32 \times 2=8192$ 个数值，通过 MLA 压缩后，每层每 Token 仅需存储一个 512 维的隐向量，体积可以缩小 10 倍以上。由于模型从训练之初就适应了这种压缩表达，它保留的是核心有效信息，通常可以保持较好的质量。

- 优点：显存压缩率高，不丢弃 Token，模型输出质量好。
- 缺点：必须在训练前构建架构；还原计算过程会引入少量额外计算开销。

## 各种方法综合对比

上述方法并非互斥，而是可以叠加组合使用。例如，使用 MLA 架构的模型可以对隐向量进行量化，并在超长文本生成中进一步叠加 Token 剔除。

| 对比维度 | 量化 | Token 剔除 | 跨 Head 共享（GQA/MQA） | 低秩压缩 MLA |
| --- | --- | --- | --- | --- |
| 节省原理 | 降低每个数值的存储位数 | 减少存储的 Token 数量 | 减少每层 Key/Value 副本数 | 压缩单个 Token 的向量维度 |
| 需要重新训练 | 否 | 否 | 是 | 是 |
| 丢弃 Token | 否 | 是 | 否 | 否 |
| 典型节省比例 | 2x-4x | 固定缓存上限 | 4x-8x | 10x 或更高 |
| 质量影响 | 极小 | 视任务而定 | 极小 | 通常较小 |
| 代表技术 | KIVI | H2O、StreamingLLM | GQA、MQA | MLA（DeepSeek） |

## 不同场景选用建议

- 现有预训练模型快速优化显存：首选量化（Quantization）。
- 无限流长文本或连续对话场景：结合 Attention Sink 的 Token 剔除。
- 需要对长文档进行精确细节检索：避免使用 Token 剔除，选择量化，或选用原生支持 GQA/MLA 的模型。
- 从零训练新模型：优先在架构层面集成 GQA 或 MLA。
- 追求极限显存压缩：组合使用，例如基于 MLA 的模型加 KV 显存量化和动态剔除。

## 常见问题解答

- **Q1：缩小 KV Cache 是否也能提升文本生成速度？**
  - 答：是的。生成每个新 Token 时，模型都需要从显存读取完整缓存。缓存体积更小意味着显存带宽压力更低、读取速度更快，从而直接提升文本生成吞吐量。
- **Q2：Paged Attention（分页注意力）属于 KV Cache 压缩技术吗？**
  - 答：不属于。Paged Attention（如 vLLM 中所用）解决的是显存碎片化与动态分配浪费的问题，旨在高效利用既有显存；KV Cache 压缩则是直接将缓存内容本身变小。两者属于互补关系。
- **Q3：可以同时结合多种 KV Cache 压缩技术吗？**
  - 答：可以。例如使用 MLA 架构的模型，可以对其生成的隐向量实施量化存储，并在处理超长上下文时引入动态 Token 剔除。
- **Q4：为什么通常不能剔除开头的几个 Token？**
  - 答：因为首批 Token 可能承担注意力汇聚（Attention Sink）的作用，模型习惯将多余的注意力权重倾倒在开头。一旦将其剥离，Attention 的权重分布可能被破坏，导致模型输出质量下降。
- **Q5：能否直接将 GQA 应用于已经训练好的 MHA 模型上？**
  - 答：不能直接无损套用。跨 Head 的 Key/Value 共享机制（如 GQA、MQA）以及低秩压缩（如 MLA）改变了模型的参数矩阵结构，通常需要在预训练阶段确立，或经过额外转换与再训练。

## 来源

- [KV Cache in LLMs](https://outcomeschool.com/blog/kv-cache-in-llms)
- [KV Cache Compression](https://outcomeschool.com/blog/kv-cache-compression)
- [知乎：KV Cache 相关介绍](https://zhuanlan.zhihu.com/p/1990116555888035411)
