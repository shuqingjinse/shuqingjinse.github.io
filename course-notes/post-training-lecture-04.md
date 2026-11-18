---
layout: course-note
title: "第 4 讲 · DPO 原理：从偏好对到直接优化"
description: "由带 KL 正则的奖励目标推到 DPO，理解参考模型、偏好概率、beta 与离线训练的边界。"
permalink: /course-notes/post-training-lecture-04/
course_id: post-training
lecture: 4
original_lessons: "Basics of DPO"
source_note: "课程主线是用好坏回答进行偏好优化；公式推导、数值例子和边界说明依据 DPO 原始论文补充。"
sections:
  - {id: pairs, title: "为什么要成对比较"}
  - {id: derive, title: "从奖励目标到 DPO 损失"}
  - {id: interpret, title: "怎样读懂这个公式"}
  - {id: beta, title: "参考模型和 beta"}
  - {id: limits, title: "离线偏好的能力边界"}
  - {id: review, title: "本讲自测"}
---

## 1. 为什么要成对比较 {#pairs}

对于开放式任务，写一份完美回答很难，但比较两份现有回答往往更容易。比如两份解释都正确，一份简洁、切题，另一份冗长、回避重点；标注者可以表达更偏好前者，而不必重写全部内容。

**直接偏好优化（Direct Preference Optimization，DPO）**使用三元组：

$$
(x,y_w,y_l),
$$

其中 $$x$$ 为相同上下文，$$y_w$$ 是 chosen / preferred，$$y_l$$ 是 rejected / less preferred。下标表示这对回答里的相对偏好，不表示 chosen 在所有方面都完美、rejected 在所有情况下都错误。

偏好必须有标准：准确性、遵循指令、帮助性、特定格式，或某种明确的表达习惯。若每条数据的标准不一致，模型会收到冲突信号。

SFT 主要提升示范回答的似然；DPO 显式利用两份回答之间的关系。它省去了经典 RLHF 流程里「先拟合独立奖励模型，再用在线 RL 优化策略」这两个阶段的直接实现，但仍需要高质量偏好标签。

## 2. 从奖励目标到 DPO 损失 {#derive}

以下为补充推导，符号与核心关系参照 [DPO 原始论文](https://arxiv.org/html/2305.18290v2)。

### 第一步：既追求奖励，也约束偏离

设 $$\pi_{\mathrm{ref}}$$ 为固定参考策略，考虑：

$$
\max_\pi\;
\mathbb E_{x,y\sim\pi}[r(x,y)]
-\beta\,\mathbb E_x
D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

第一项想提高奖励，第二项惩罚离参考分布太远。对固定 $$x$$，在相应假设与支持集条件下，最优策略具有：

$$
\pi^*(y\mid x)
=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
$$

整理可得：

$$
r(x,y)=
\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

### 第二步：用奖励差解释偏好

Bradley–Terry 偏好模型把「更偏好 $$y_w$$」的概率写成：

$$
P(y_w\succ y_l\mid x)
=\sigma\!\left(r(x,y_w)-r(x,y_l)\right),
$$

$$\sigma(z)=1/(1+e^{-z})$$ 是 sigmoid。把上式的奖励表达代入，同一个问题的 $$Z(x)$$ 项相消。

### 第三步：直接用策略参数化奖励差

用待训练策略 $$\pi_\theta$$ 替代最优策略的参数化，得到标准 DPO 损失：

$$
\mathcal L_{\mathrm{DPO}}
=-\mathbb E_{\mathcal D}\log\sigma
\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$

训练时只需计算给定回答的 log probability 并反向传播；标准离线 DPO 的更新不需要每一步都生成新的回答，也不需要单独拟合一个显式奖励网络。

## 3. 怎样读懂这个公式 {#interpret}

令：

$$
s_\theta(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x).
$$

那么损失就是 $$-\log\sigma(\beta(s_w-s_l))$$：希望 chosen 相对于参考模型的提升，超过 rejected 相对于参考模型的提升。

| 比较量 | 含义 |
| --- | --- |
| $$\log\pi_\theta(y\mid x)$$ | 当前模型对这整段回答的对数概率 |
| $$\log\pi_{\mathrm{ref}}(y\mid x)$$ | 参考模型的对应对数概率 |
| $$s_w-s_l$$ | 对 chosen 的相对偏好改变量 |
| $$\sigma(\beta(s_w-s_l))$$ | 在该参数化下解释偏好标签的概率 |

### 一个检查方向的数值例子

设初始 policy 与 reference 完全相同，则两份回答的 $$s$$ 都为 0，单对损失是 $$-\log0.5\approx0.693$$。

更新后若 chosen 的 log probability 相对提高 0.6，rejected 相对降低 0.4，则差为 1.0。取 $$\beta=0.1$$，偏好概率为 $$\sigma(0.1)\approx0.525$$，损失约为 0.644，方向符合目标。

反过来，若提升了 rejected 的相对概率，损失就可能增加。这个例子只检查数学方向，不代表真实实验中的概率变化。

### 「提高好答案、降低坏答案」是直觉概括

DPO 优化的是相对差。即使 chosen 的绝对概率下降，只要 rejected 下降更多，偏好间隔仍可能变大。因此不能把它理解成「每次更新后，所有 chosen 的概率一定上升」。

此外，整段回答的 log probability 是逐 token 对数概率之和。长度与数据分布会影响比较；标准 DPO 的序列概率不能随手改成平均 token 概率而仍声称是完全相同的目标。

## 4. 参考模型和 beta {#beta}

**参考模型**通常是本次偏好优化开始时的一个冻结快照，常来自已经做过 SFT 的指令模型。标准离线 DPO 中，policy 持续更新，reference 保持不动；如果二者一起更新，目标就变了。

$$\beta$$ 有两层理解：

- 在最初的正则化奖励目标里，它控制偏离参考策略的代价。
- 在 DPO 损失里，它缩放 log-ratio 差，影响偏好概率和梯度。

在理想化的奖励目标中，更大的正则系数倾向于让最优策略更靠近 reference。**实际有限数据训练中，beta 与学习率、步数、偏好噪声共同作用**，不能靠一句「越大越保守」预测每次实验结果。

DPO 不必把两套完整权重同时放在显存里：可以预计算固定 reference 的 log probabilities，也有适配器相关实现。但这些是实现优化，不意味着 reference 在数学上不存在。

## 5. 离线偏好的能力边界 {#limits}

标准 DPO 对固定数据训练。模型如果产生一种数据里没出现过的新型错误，离线偏好对不会自动为它提供反馈。数据越偏离当前模型会生成的分布，越可能出现训练指标改善而实际体验不佳。

| 问题 | 表现 | 可考虑的改进 |
| --- | --- | --- |
| 标签噪声 | 同类回答偏好相互矛盾 | 统一标准、复核争议样本 |
| 长度偏差 | 总是偏好长回答 | 控制回答长度和标注标准 |
| 只改表面特征 | 自称对了，内容质量变差 | 加入能力保留评估 |
| 数据覆盖不足 | 训练问法有效，改写无效 | 收集多样化真实问法 |
| 分布逐渐漂移 | 固定偏好对不再代表当前错误 | 迭代采样、重新标注或采用在线方法 |

课程用身份自称变化做实验，适合隔离一个简单行为，但模型说出某个名字并不证明训练来源、所有权或事实身份发生改变。

## 6. 本讲自测 {#review}

<details><summary>DPO 是否完全不需要奖励概念？</summary><p>不是。它把策略与参考策略的对数概率比用作隐式奖励的参数化，省去单独训练和调用显式奖励模型的步骤。</p></details>

<details><summary>为什么 chosen 与 rejected 必须对应相同上下文？</summary><p>偏好比较表达的是同一个条件下谁更好。若系统指令或问题不同，差异可能来自任务条件，无法解释为同题回答的偏好。</p></details>

<details><summary>DPO 比 SFT 多了哪些核心计算？</summary><p>要计算 chosen 和 rejected 两条回答在 policy 与固定 reference 下的 log probability，再用两者的相对差构造损失。具体显存和吞吐取决于是否预计算 reference 等实现。</p></details>

实践接口参考：[TRL 0.19.1 DPOTrainer](https://huggingface.co/docs/trl/v0.19.1/dpo_trainer)。
