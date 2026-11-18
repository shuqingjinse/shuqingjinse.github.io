---
layout: course
title: "大语言模型的后训练 · Post-training of LLMs"
description: "DeepLearning.AI《Post-training of LLMs》完整中文学习笔记，7 讲掌握 SFT、DPO、PPO 与 GRPO 的原理、数据与训练实践。"
permalink: /course-notes/post-training/
course_id: post-training
---

## 如何阅读这套笔记

这门短课程由 **Banghua Zhu（朱邦华）**主讲，围绕三种后训练方法展开。这里把原课的 9 段视频重组为 **7 讲中文笔记**：导言合并进全景介绍，结课内容并入最后一讲；三组「原理 → 实践」保持独立。建议按顺序学习，复习时可直接看第 7 讲的对照表与自测。

| 本站讲次 | 对应官方视频 | 学完能回答 |
| --- | --- | --- |
| 1 | Introduction + Introduction to Post-training | 为什么后训练，什么时候值得训练？ |
| 2–3 | Basics of SFT + SFT in Practice | 如何用示范数据教会模型按要求回答？ |
| 4–5 | Basics of DPO + DPO in Practice | 如何用偏好对改变回答习惯？ |
| 6–7 | Basics of Online RL + Online RL in Practice + Conclusion | 如何采样、评分并通过 GRPO 改善表现？ |

**资料说明：**课程主题、视频结构与三个实验目标依据[官方课程介绍](https://www.deeplearning.ai/courses/post-training-of-llms)；SFT 实践同时整合了已有中文讲义。数据字段与模型名称已对照[讲师公开的 Hugging Face 资源](https://huggingface.co/banghua)核对。视频页本次未能直接读取，因此这套内容是按课程大纲组织的学习笔记，补充推导、例子和代码均明确标注，不能视为视频逐句译稿。

**实践约定：**示例代码用于解释训练流程，按 TRL **0.19.1** 文档接口组织，超参数为教学起点；不声称已经完成模型训练或达到某个分数。真正复现实验请使用课程环境，或固定依赖、模型及数据版本，并在独立测试集评估。文中的自测是原创复习题。
