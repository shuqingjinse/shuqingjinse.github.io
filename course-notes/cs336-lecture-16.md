---
layout: course-note
title: CS336 第 16 讲 · 可验证奖励强化学习：PPO、GRPO 与推理模型
description: 推导策略比率和组内优势，分析 RLVR、长度偏差、验证器漏洞、蒸馏与训练基础设施。
permalink: /course-notes/cs336-lecture-16/
course_id: cs336
lecture: 16
instructor: Tatsunori Hashimoto
original_lessons: 'Spring 2026 · Lecture 16 · Post-training: RLVR · 2026-05-20'
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf
sections:
- id: rlvr
  title: 可验证任务给强化学习带来什么
- id: ppo
  title: PPO 的新旧策略比率
- id: grpo
  title: GRPO 用组内比较替代显式 value
- id: bias
  title: 长度、标准化与偏差
- id: recipe
  title: 推理训练配方与系统循环
- id: review
  title: 练习与自测
---
## 1. 可验证任务给强化学习带来什么 {#rlvr}

RLHF 中的偏好奖励难免带有主观性。数学答案、可执行代码测试等场景，可利用程序验证生成结果，形成 RLVR：Reinforcement Learning with Verifiable Rewards。

验证只是将部分目标变得可计算，不保证评分器绝对正确。数学答案抽取可能被格式误导，有限代码测试可能漏掉错误。模型最终优化的是你实际写出的奖励函数，因此验证器设计也是训练方法的一部分。

**讲义定位：第 2–4、25–38 页。**下文算例和检查模板为补充。

## 2. PPO 的新旧策略比率 {#ppo}

令旧策略采样 token，当前策略的比率为：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}.
$$

PPO 的一个核心代理目标是：

$$
\mathbb E_t\left[\min\left(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right].
$$

优势 $$A_t$$ 衡量动作相对基准有多好；value 模型可用于估计它。clip 限制代理目标从过大策略变化中得到的收益，不是硬性保证每个 token 的概率或 KL 都被限制在固定范围。

补充算例：A=1、$$\epsilon=0.2$$、$$\rho=1.5$$ 时，该项取 min(1.5,1.2)=1.2；A 为负时要按同一 min 公式处理，不能只照抄正优势直觉。

**讲义定位：第 5–16 页。**

## 3. GRPO 用组内比较替代显式 value {#grpo}

对同一提示采样 G 个回答，用奖励构造组内标准化优势：

$$
A_i=\frac{r_i-\bar r}{\operatorname{std}(r_1,\ldots,r_G)+\delta}.
$$

再将回答优势赋给相关 token，用策略比率、裁剪和可能的 KL 项更新。这样可以省去独立 value 模型，但仍需保存采样概率、管理策略版本并正确处理 mask。

补充算例：奖励 `[0,0,1,1]`，若用总体标准差 0.5，忽略极小稳定项，则优势为 `[-1,-1,1,1]`。若一组全对或全错，中心化奖励为零，单靠这部分通常没有区分信号。实现中的标准差约定与零方差处理必须写清。

**讲义定位：第 17–24 页。**

## 4. 长度、标准化与偏差 {#bias}

对每条序列按自身长度平均、对全批所有 token 平均，或按固定分母归一化，会给不同长度的回答不同权重。组内标准差也会改变各问题的有效权重，不能把标准化只当成不影响目标的数值小技巧。

更长的推理文本可能提高解题率，也可能只是增加成本或放大长度偏差。需要一起观察正确率、长度分布、每题采样量与计算预算；不能把训练中出现更长思考直接当成普遍因果机制。

补充检查：把一批回答的 loss 手算到 token 级，核对 padding、截断与 EOS；对相同正确率但不同长度的两组样本，比较实际梯度权重。

**讲义定位：第 22–24、28–30、43 页。**

## 5. 推理训练配方与系统循环 {#recipe}

公开案例常组合冷启动 SFT、推理 RL、再次整理数据、通用对齐和蒸馏。不同模型采用不同数据与阶段，不能将其中任意一个步骤视为全部能力来源。

蒸馏用更强模型的轨迹或回答训练较小模型，和较小模型自己通过在线 RL 学习不同。Agent RL 则还要运行环境、工具与多步反馈，奖励延迟和基础设施更加重要。

训练系统通常循环执行：采样 → 验证 → 计算优势 → 更新 → 同步用于采样的权重。生成阶段与训练阶段的硬件利用不同，策略过旧会使重要性比率和优化更难控制。

**讲义定位：第 25–61 页，包含 R1、Kimi、Qwen 和 agentic RL 案例。**

## 6. 练习与自测 {#review}

补充练习：为一道数学题生成 4 个候选，人工核验奖励、优势和长度归一化；再构造一个能骗过简单答案抽取器的错误回答，用来改进验证器。

<details><summary>GRPO 为什么不需要单独 value 模型？</summary><p>常用版本由同一问题的一组回答奖励构造相对优势。但这改变了估计方式，并不说明没有方差、偏差或奖励设计问题。</p></details>
<details><summary>一组回答全错时为什么可能学不到区分？</summary><p>若奖励完全相同，减去组均值后都为零。需要更合适的任务分布、采样或额外信号，而不是任意伪造差异。</p></details>
<details><summary>测试通过率提高能否证明代码真正正确？</summary><p>只能说明通过了所用测试。还需隐藏测试、覆盖分析和防止利用验证器漏洞，并明确评估范围。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf) · [官方视频](https://www.youtube.com/watch?v=dIFAi87Ws4E)。整理核对日期：2026-11-20。
