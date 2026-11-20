---
layout: course-note
title: CS336 第 12 讲 · 模型评估：从困惑度到真实任务
description: 区分概率建模、考试、聊天、智能体和安全评估，理解有效性、真实性、污染与实验口径。
permalink: /course-notes/cs336-lecture-12/
course_id: cs336
lecture: 12
instructor: Percy Liang
original_lessons: Spring 2026 · Lecture 12 · Evaluation · 2026-05-06
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_12
sections:
- id: construct
  title: 先定义想测的能力
- id: perplexity
  title: 困惑度衡量概率分配
- id: benchmarks
  title: 考试、聊天与裁判偏差
- id: agents
  title: 智能体、推理与安全要测完整过程
- id: validity
  title: 有效性、真实性与污染
- id: review
  title: 练习与自测
---
## 1. 先定义想测的能力 {#construct}

「好模型」可能意味着文本概率高、解题正确、用户喜欢、任务完成率高，或成本低。把抽象目标转换成可计算指标时，总会丢掉一些信息。评估不是训练完成后的机械打分，它决定数据与系统最终朝什么方向优化。

比较对象也要写清：是基础模型、带提示策略的方法，还是含工具、重试和检索的完整智能体？给一方更多采样、工具或推理时间，所得分数就不只是模型参数本身的差异。

**讲义定位：`what_is_good`、`how_to_think_about_evaluation`。**以下计算与评估卡片为补充。

## 2. 困惑度衡量概率分配 {#perplexity}

对 N 个目标 token，平均负对数似然为：

$$
L=-\frac1N\sum_{t=1}^N\log p(x_t\mid x_{<t}),\qquad \operatorname{PPL}=e^L.
$$

使用自然对数时，loss=2 对应 PPL 约 7.39。越低表示给真实序列更高概率，但它不能单独说明回答正确、对齐或工具使用可靠。

不同 tokenizer 会改变 token 数和预测单位，直接比较每 token PPL 可能不公平。对同一原始文本可考虑按字节等统一单位归一化，同时仍要对齐文本处理与上下文。条件困惑度只对指定回答区域计分，和对整个提示加回答计分不同。

**讲义定位：`perplexity`。**

## 3. 考试、聊天与裁判偏差 {#benchmarks}

考试型基准易于自动判分，可以控制知识领域和难度，却不能覆盖开放式真实使用。选择题还受到选项设置、提示格式与答案抽取规则影响。

聊天偏好更贴近开放回答，但会混入长度、格式、语气和用户群体偏好。模型裁判能够扩大规模，也可能有位置偏差、偏爱某种文风或与被评模型共享盲点。

补充设计：匿名、交换候选顺序、约束比较口径，并对一部分样本做人类复核。长度变化应单独报告，不能把回答变长后赢得更多偏好票简单解释为事实正确性提高。

**讲义定位：`exam_benchmarks`、`chat_benchmarks`。**

## 4. 智能体、推理与安全要测完整过程 {#agents}

智能体评估包含环境、工具接口、行动预算、任务状态和验收规则。只判断最终自然语言有没有说「已完成」，无法证明文件确实修改或测试确实通过。

推理任务应区分 pass@1、重复采样后至少一次成功和由验证器挑选的结果；安全评估则需要同时关注有害请求的处理、正常请求被误拒及攻击场景。不同风险类型很难压成一个数。

补充算例：若每次独立成功概率为 p，k 次至少一次成功概率是 $$1-(1-p)^k$$；真实样本常相关，且实际产品未必知道哪次成功，所以它不等于有一个可靠选择器的用户体验。

**讲义定位：`agentic_benchmarks`、`pure_reasoning_benchmarks`、`safety_benchmarks`。**

## 5. 有效性、真实性与污染 {#validity}

有效性问「这个分数真的反映想测的东西吗」，真实性问「它和实际使用有多接近」。非常真实的环境可能难以稳定复现，封闭题库则可能离产品任务很远。

训练数据与评测重叠会破坏解释，简单文本去重也未必能移除改写、答案或近似题。应记录评测版本、数据时间和去污染策略，避免在反复查看同一测试集后把它变成隐性开发集。

| 一张评估卡片 | 需要写清 |
| --- | --- |
| 目标与样本 | 用户、任务、覆盖范围与样本量 |
| 规则 | 提示、工具、次数、token 与时间预算 |
| 计分 | 答案抽取、裁判、失败与超时处理 |
| 质量与成本 | 成功率、波动、延迟和消耗 |
| 局限 | 污染、分布变化和未覆盖场景 |

**讲义定位：`realism`、`validity`；表格为补充模板。**

## 6. 练习与自测 {#review}

补充练习：为代码修改助手设计 20 个样本，明确仓库初始状态、允许工具、隐藏测试和超时条件，并记录失败类别。

<details><summary>更低困惑度为什么不保证偏好评估更好？</summary><p>概率建模和偏好只部分相关。风格、安全、任务目标以及后训练会改变回答方式，必须分别测量。</p></details>
<details><summary>pass@10 能直接与另一个模型的 pass@1 比吗？</summary><p>不能据此直接判断同预算能力，采样次数、推理成本与选择条件不同，需要对齐规则。</p></details>
<details><summary>为什么要保存失败实例，而非只保存总分？</summary><p>总分掩盖了工具错误、知识不足、格式错误或验收漏洞，实例能指导真正有效的数据和系统改进。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_12) · [官方视频](https://www.youtube.com/watch?v=JpAxdTWQJxM)。整理核对日期：2026-11-20。
