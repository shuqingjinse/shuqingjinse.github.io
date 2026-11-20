---
layout: course-note
title: CS336 第 17 讲 · 多模态模型：从图文对齐到视觉语言与生成
description: 梳理 CLIP、SigLIP、LLaVA、Qwen-VL 与离散图像 token，理解分辨率、位置、数据和训练阶段。
permalink: /course-notes/cs336-lecture-17/
course_id: cs336
lecture: 17
instructor: Percy Liang
original_lessons: Spring 2026 · Lecture 17 · Alignment and multimodality · 2026-05-27
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_17
sections:
- id: tokens
  title: 把非文本信息接入 Transformer
- id: contrastive
  title: CLIP 与 SigLIP 的对齐目标
- id: vlm
  title: LLaVA：视觉编码器、投影器和语言模型
- id: resolution
  title: 分辨率与视频 token 预算
- id: generation
  title: 理解与生成对表示的要求不同
- id: review
  title: 练习与自测
---
## 1. 把非文本信息接入 Transformer {#tokens}

文本通过分词器变成离散 token，图像、视频和音频也需要某种表示才能进入 Transformer。表示可以是连续向量，也可以是离散码；选择取决于要做理解、检索还是生成。

图像分类只需抓住主要语义；OCR 需要保留小字，定位需要空间关系，生成还要重建细节。一个擅长图文检索的编码器，不自动保留所有像素信息。

**讲义定位：`main`。**下文公式、预算算例和练习为补充。

## 2. CLIP 与 SigLIP 的对齐目标 {#contrastive}

CLIP 在一批匹配图文中编码图像和文本，用相似度让正确配对高于错误配对。单向的一个简化损失为：

$$
L_{i\to t}=-\frac1B\sum_i\log\frac{e^{s(i_i,t_i)/\tau}}{\sum_j e^{s(i_i,t_j)/\tau}}.
$$

通常还加入文本到图像的方向。批内其他配对充当负例，温度、batch 和数据质量都影响学习。

SigLIP 将图文配对作为二分类问题使用 sigmoid 损失，减少对全批 softmax 归一化的依赖，但仍需选择正负配对与训练策略。两者学到的是图文语义对齐，不等于直接完成多轮对话。

**讲义定位：`clip`、`siglip`。**

## 3. LLaVA：视觉编码器、投影器和语言模型 {#vlm}

典型 VLM 将图像编码成 patch 特征，再经 projector 映射到语言模型的隐藏维度，与文本序列共同输入。投影器不仅要形状匹配，还需通过训练建立有效接口。

LLaVA 的阶段化思路先训练接口对齐，再使用视觉指令数据训练回答能力。其他模型可能选择不同冻结策略，因此不能把某一版的训练阶段套给所有 VLM。

合成问答能扩大视觉指令数据，但生成器可能只看到图注或检测结果，而非完整图像；它产生的问答需与视觉证据核对。幻觉常来自数据缺口和语言先验，而非只是缺一条提示词。

**讲义定位：`llava`、`llava_onevision`。**

## 4. 分辨率与视频 token 预算 {#resolution}

固定裁剪容易丢失边缘或小字。多块切图、动态分辨率和更灵活的位置编码，可以保留更多信息，也会增加序列长度和计算。

补充算例：336×336 图像、14×14 patch，在不含额外 token 且不做压缩时，得到 24×24=576 个 patch。将两个方向分辨率翻倍，patch 数变为 2304，约 4 倍；若直接做全注意力，相关成对交互成本可能约 16 倍。实际架构有压缩和不同注意力设计，需按模型核算。

视频还引入时间：采样帧率、帧数、空间分辨率与时间位置要联合考虑。稀疏采样可能漏掉短动作，堆更多帧则占满上下文。MRoPE 等设计把时间和二维空间引入位置，但不是给每一帧简单加一个文本序号。

**讲义定位：`llava_onevision`、`qwen_vl`、`qwen2_vl`、`qwen3_vl`。**

## 5. 理解与生成对表示的要求不同 {#generation}

连续视觉编码器可以把语义送入语言模型，但要输出图像，还需要生成路径，例如解码器、扩散模型或离散图像 tokenizer。Chameleon 路线把图像映射成离散码，与文本统一建模，使自回归生成图像 token 成为可能。

多模态混合训练还要控制不同模态的长度与 loss 权重。视频序列很长，简单按全部 token 求和可能让少量长视频样本占据大量梯度；归一化与采样改变各任务权重，应在文本和视觉任务上分别检查。

| 目标 | 重点保留什么 | 应验证的失败 |
| --- | --- | --- |
| 图文检索 | 全局语义 | 相似但不匹配的配对 |
| OCR / 图表 | 细节、空间、数字 | 小字漏读与单位错误 |
| 视频理解 | 时间关系 | 帧间顺序与瞬时事件 |
| 图像生成 | 可重建的视觉信息 | 细节、布局与文本一致性 |

**讲义定位：`qwen3_vl`、`chameleon`；表格为补充整理。**

## 6. 练习与自测 {#review}

补充练习：为含小字的截图设计两种输入方案，估算 token 数，再用可读性和成本共同选择。对视频则写出采样间隔及可能漏掉的事件。

<details><summary>CLIP 能对齐图文，为什么还需要 VLM 指令训练？</summary><p>对齐相似度与生成可用回答是不同目标，后者需要把视觉表示接入语言模型并学习任务与交互格式。</p></details>
<details><summary>把图片分辨率翻倍，成本为什么可能远超 2 倍？</summary><p>两个空间方向同时扩大使 patch 数约增至 4 倍；全注意力对序列的平方项可能约增至 16 倍，实际还取决于压缩和架构。</p></details>
<details><summary>连续图像特征能直接作为词表类别采样吗？</summary><p>通常不能。需要定义生成机制，或先建立离散图像码，再配合相应解码器还原图像。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_17) · [官方视频](https://www.youtube.com/watch?v=26FtD08ZpOU)。整理核对日期：2026-11-20。
