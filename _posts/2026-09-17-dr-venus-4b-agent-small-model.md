---
layout: post
title: "DR-Venus-4B 调研：怎么训练一个 Agent 小模型"
description: "从 SFT 数据清洗、IGPO 长轨迹强化学习到在线推理，拆解 DR-Venus-4B 的训练流程。"
categories: [AI]
tags: [Agent, 强化学习, GRPO, IGPO, 模型训练]
---

# 一、背景

## 1.1 repo的目标
会调用搜索工具的语言模型 -> 能进行长链路调查研究的 Agent 

## 1.2 出品方
inclusionAI（IAI）是蚂蚁集团旗下的AGI研究组织，定位为开源、普惠的通用人工智能平台

# 二、核心流程

## 2.1 第一步： SFT 监督微调

将 REDSearcher 的约 1 万条开放数据轨迹清洗成统一格式，规范 search / visit 调用、删除重复工具调用、过滤错误答案，并对长轨迹进行重采样，最终形成约 1.87 万条 SFT 样本。

SFT/data_clean/prepare_trajectories.py 

流程如下：
```
 读取输入文件
    -> 统一问题、答案和 system prompt
    -> 统一消息格式、工具调用格式
    -> 删除不允许的工具调用及其响应
    -> 删除重复的 search / visit 轨迹
    -> 校验对话结构
    -> 按轨迹长度重采样
    -> 输出 parquet / jsonl / stats.json
```


### 校验对话结构
```
  system
  user
  assistant(tool call)
  user(tool response)
  assistant(tool call)
  user(tool response)
  ...
  assistant(final answer)
```

### 重采样

| turn | 复制倍数 |
| --- | --- | 
| <=50 | 1 |  
| 51-100 | 2 |  
| >100 | 5 |  

重采样的目的，是增加长轨迹在训练集中的比例，让模型更重视多步搜索、网页阅读和交叉验证能力，而不是只学习短问答。


## 2.2 第二步：RL 长轨迹强化学习

从 SFT 模型继续训练，使用基于 GRPO 的 IGPO 方法。除了最终答案是否正确，还会根据每次工具调用后模型对正确答案的信念提升量，提供逐轮的信息增益奖励。

这解决了长链路 Agent 中的几个典型问题：
  - 只看最终奖励，导致中间步骤没有反馈；
  - 多条轨迹最终得分相同，优势函数接近失效；
  - 模型不知道哪一次搜索或网页阅读真正推进了答案。

主要实现见 RL/verl/trainer/ppo/info_gain_advantage.py 和 DRAgentLoop (RL/scrl/llm_agent/dr_agent_loop.py)


### 2.2.1 DRAgentLoop 生成长轨迹
让模型真正跑出一条“搜索 → 阅读 → 再搜索 → 回答”的长轨迹。

1、始化 Agent 运行环境
```
最大交互轮数
总上下文上限
每一轮最多生成8192token
工具超时
系统提示词
tokenizer池和线程锁
rollout trace保存目录
可选的重复生成、格式错误和重复工具调用重试
可选的rollout-time IG计算
```

2、准备一条 rollout
turn_boundaries 记录每轮生成开始前的状态。作用是 后面算 V_t时，知道每个C_t对应序列的 哪一段。

```
  {
      "step": step,
      "token_len": len(current_ids),
      "msg_count": len(messages),
      "tool_name": None
  }
```

后续 IG 计算就依靠这些边界判断：

“模型在看到第几轮工具结果之后，对正确答案更有把握了多少？”



3、每轮重新构造完整上下文
从结构化 messages 重建当前完整上下文，确保 system/user/assistant 标记一致


4、三种强制结束条件

正常情况下模型可以持续调用工具，但以下情况会强制要求它回答：

  - 上下文接近 max_len；
  - 异步 rollout 达到全局 completion cutoff；
  - 已经到达 max_turns。

代码会把最后一条用户消息替换为类似：

```
You've reached the maximum number of tool calls.
Provide your final answer in <answer></answer> tags now.
```

因此“200 轮”是上限，不代表每条轨迹一定运行 200 轮。


5、模型生成和解析

模型生成后，代码会清理终止 token，并尝试修复部分标签：

缺少 </think>  -> 自动补齐
只有 </think>  -> 补充 <think>
出现 tool_response -> 截断后面的内容

随后 _parse_single_response() 将输出分为三类：

_Flag.END    # 得到 <answer>
_Flag.CALL   # 得到 <tool_call>
_Flag.ERROR  # 格式错误或输出不完整

有效工具调用支持：

```
  {"name": "search", "arguments": {"query": ["..."]}}
  {"name": "visit", "arguments": {"url": ["..."], "goal": "..."}}
```


6、三个分支的处理

（1）最终回答
将 assistant 输出加入消息列表，结束循环。

（2）格式错误
把错误输出保留在轨迹中，然后追加：
```
<tool_response>
Your response was malformed.
Please call a tool or provide a final answer.
</tool_response>
```
这让模型有机会在下一轮自行修正。

（3）工具调用

通过 _call_tool() 在线执行 search 或 visit，并把结果放回上下文：
```
assistant: <think>...</think><tool_call>...</tool_call>
user:      <tool_response>...</tool_response>
```
同步工具函数被包装进线程池和 asyncio.wait_for()，因此不同样本可以异步等待网络请求，不会让一个慢网页阻塞整个 rollout batch。


7、可选的健壮性重试

  当 ROLLOUT_ROBUST=1 时，一轮生成可以因为以下原因重试：

  - 输出尾部出现明显重复；
  - <think> / <tool_call> 格式无法解析；
  - 重复调用完全相同的工具参数。

  不过 RL/train_igpo.sh:69 默认关闭该功能，以保持论文训练过程可复现。

8、输出训练轨迹

循环结束后，代码重新用 chat template 构造：

  prompt_ids   = 初始问题
  response_ids = 后续所有 assistant/tool 交互

  这里的 response 不是单次回答，而是整条多轮轨迹。

  最终返回：

```
  AgentLoopOutput(
      prompt_ids=prompt_ids,
      response_ids=response_ids,
      response_mask=response_mask,
      num_turns=num_turns,
      extra_fields={
          "turn_boundaries": turn_boundaries,
          "messages": messages,
          ...
      },
  )
```


### 2.2.2 info_gain_advantage.py 计算信息增益

源码路径： RL/verl/trainer/ppo/info_gain_advantage.py

![截图](/assets/images/dr-venus-igpo.png)

**IGPO 的轮级信息增益定义为模型在相邻两轮中，对 Ground Truth 答案的长度归一化对数概率的变化量、记做IG_t，t为轮次turn；用信息论的角度看，等价于熵的减少量，如果某一轮交互后，模型对正确答案的置信度上升，信息增益为正，说明这一轮减少了不确定性，即“这一轮让我变得多确定”。**

把最终答案奖励和每轮信息增益转换成每个 token 的 PPO advantage。

用每轮真值的平均对数概率作为价值的度量：
$$
V_t = \frac{1}{N} \sum_{i=1}^{N}\log P(y_i \mid C_t, y_{<i}) 
$$

每轮信息增益为：
$$IG_t = V_t - V_{t-1}$$

 含义非常直接：

  - IG > 0：这轮搜索或阅读让模型更接近正确答案；
  - IG < 0：这轮信息让模型更困惑；
  - IG ≈ 0：这轮工具调用基本没有提供有效信息。

例如：
```
  初始问题             V0 = -4.0
  搜索之后             V1 = -3.2   IG1 = +0.8
  阅读网页之后         V2 = -1.5   IG2 = +1.7
  无关搜索之后         V3 = -1.8   IG3 = -0.3
```

#### 1、训练实际路径

DRAgentLoop 内置了 rollout-time IG：直接让 vLLM 在 rollout 过程中计算 Ground Truth 的 prompt logprobs。

默认配置是：USE_ROLLOUT_IG=false

论文复现路径实际上是：

```
  1. DRAgentLoop 先完成整条轨迹；
  2. 保存 turn_boundaries；
  3. RL/verl/trainer/ppo/ray_trainer.py:1629 在 rollout 后调用 FSDP Actor；
  4. 使用 KV cache 批量计算各边界的 Ground Truth log probability；
  5. 得到 info_gain_rewards。
```

vLLM 和 FSDP Actor 的 logprob 可能存在数值差异。默认路径更慢一点，但训练信号与实际更新的 Actor 更一致。

更详细点说是：

（1）在 rollout 后调用 FSDP Actor：Rollout 阶段由推理引擎（vLLM / SGLang）完成，生成完整轨迹后，训练侧调用 FSDP 包裹的 Actor 模型，进入"重计算 / 打分"阶段。 这么做的根本原因是：rollout 用的权重和训练用的权重不是同一份，必须对齐。具体如下：

- Rollout 时用的是推理引擎（vLLM / SGLang 为了速度，可能用 FP16 / BF16、不同的 kernel、不同的 attention 实现；甚至可能不保留 log prob），权重可能是另一份副本
- 算 log probability 要用当前训练权重，保证 on-policy（训练需要的是"当前策略"的概率）
- FSDP Actor 就是当前正在训练的模型，用它来算概率才准确，训练是异步的:
```
训练 step k 更新 θ_k
   → 把 θ_k 同步给推理引擎
   → 推理引擎用 θ_k 生成轨迹
   → 但训练侧可能已经走到 θ_{k+1}
```

（2）使用 KV cache 批量计算各边界的 Ground Truth log probability

对每个边界 C_t ，用 Actor 模型算 Ground Truth 的 log probability，即 V_t

```
把 C_t + Ground Truth 拼成一条序列；
前向计算，得到每个 Ground Truth token 的条件概率
取平均 log 概率
```
​
KV cache 的作用：
```
不同 C_t  共享大量前缀
用 KV cache 可以复用前面已算过的 key/value，避免每个 C_t  都从头算一遍
批量计算：把所有边界拼成一个 batch，一次前向算完，提升 GPU 利用率
```
（3）得到 info_gain_rewards
有了所有 V_t ，就能算信息增益 IG_t，后面会
```
和最终答案奖励（LLM Judge / F1）结合，构造 token-level rewards；

在同题 8 条轨迹内做 GRPO 归一化；

按轮倒序折扣累积；

广播到每个 token，用 PPO 更新 Actor
```

#### 2、IG_TOOL_FILTER="visit" 的含义，不做区分

默认只在“上一轮执行了 visit”的边界计算 IG。
```
  C0: 初始问题                    计算基线 V0
  search -> response
  C1: 看过搜索摘要                跳过
  visit -> response
  C2: 阅读网页后                  计算 V2，得到 V2 - V0
```
 需要注意：这里如果跳过了搜索边界，那么 V2 - V0 实际包含了搜索和 visit 之间的累计变化，只是奖励落在 visit 对应的位置上，并不能严格分离两者的贡献。

#### 3、如何构造 token-level reward

最终会形成一个稀疏奖励序列：

```
  第 1 轮结束 token: IG1

  第 2 轮结束 token: IG2

  第 3 轮结束 token: IG3

  最终答案末 token:  Outcome Reward

  其他 token:         0
```


最终答案奖励来自：

  - 默认的 LLM Judge；
  - 或配置成 F1；
  - 格式错误时可被 -1 格式惩罚覆盖。

中间轮格式错误也会用 -1 替换该轮 IG。

映射逻辑主要在 info_gain.compute_score() (RL/verl/utils/reward_score/info_gain.py:244)

映射逻辑：

（1）把轮次级别的标量 IG 值，分配到该轮最后一个 token 上（一般每轮最后一个 token 是 <|im_end|> 这个特殊控制 token），作为该 token 的 reward

（源码：通过环境变量 IGPO_IG_DIMINISH_RATE 控制，公式为 ig / (1 + i * rate)，越往后的轮次 IG 衰减越多，鼓励早期高效获取信息）；

（2）其余 token 的 reward 为 0

（源码中：替换为 1e-10，防止完全无梯度）。

#### 4、info_gain_advantage.py：从 reward 到 advantage

（1）通过mark区分 区分最后一个有效token（f1_mask）、其他非零奖励(ig_mask）
（2）按同一个问题进行GRPO分组
同一个问题采样 agent_grpo.n 条轨迹，共享相同uid、index参数

代码按UID计算 组内均值和标准差

更新依据：同一道题的 8 种搜索策略里，哪条轨迹、哪几个步骤相对更好？

（3）四种归一化模式

归一化的原因是：Outcome Reward 最终奖励 和 每轮信息增益 IG Reward 量级不一样，混在一起归一化会让某一方被淹没。

默认是 scaled_separate：

  - joint：最终奖励和 IG 混在一起计算均值、方差；
  - separate：最终奖励和 IG 分开归一化；
  - scaled_separate：分开归一化，再把 IG 幅度缩放到接近最终奖励；
  - raw_ig：最终奖励正常做 GRPO，IG 只除标准差、不减均值，保留 IG 的正负方向。

scaled_separate 会计算：

$$scale = \frac{\operatorname{mean}|\hat R_{outcome}|}  {\operatorname{mean}|\hat R_{IG}|+\epsilon}$$

然后：
$$\hat R_{IG} \leftarrow scale \cdot \hat R_{IG}$$

并把 scale 最大限制为 10，避免 IG 被异常放大。

```
先各自归一化
再算一个 scale，让 IG 的平均绝对值 ≈ Outcome 的平均绝对值
这样 IG 不会因为归一化被过度放大，也不会因为原始量级小被忽略
scale 上限设为 10，防止 IG 平均绝对值接近 0 时 scale 爆炸，避免 IG 被异常放大、主导训练
```
（4）按轮倒序折扣累积
假设三轮奖励为：
```
  r1 = 第一轮信息增益
  r2 = 第二轮信息增益
  R  = 最终答案奖励
```
默认 gamma=0.95，则：
```
A_3 = R
A_2 = r_2 + 0.95A_3
A_1 = r_1 + 0.95A_2
```
这意味着第一轮搜索不仅承担自己的信息增益，还会得到后续研究成功带来的折扣信用。

实现位于 _compute_turn_level_advantage() (RL/verl/trainer/ppo/info_gain_advantage.py:17)


（5）将轮级 advantage 广播到 token

奖励只落在每轮末尾，但 PPO 需要每个生成 token 的 advantage。


所以代码把一轮的 advantage 广播给该轮所有 token：

  第一轮所有模型 token -> A1
  第二轮所有模型 token -> A2
  最终回答所有 token   -> A3

工具返回本身虽然位于 response 序列中，但 Actor 更新时另有 loss_mask；默认 MASK_TOOL_RESPONSE=true，因此工具返回 token 不参与策略梯度，模型自己生成的思考、工具调用和答案 token 才会被训练。

```
1、PPO 更新发生在 advantage 全部算好、广播完之后。
先算 advantage，再用 advantage 更新，不是边算边更新。

2、PPO 更新是根据 advantage 更新

3、PPO 更新是对minibatch 内所有 token 一起算 loss，一次反向传播
```



#### 回顾"从 reward 到 advantage"的完整链路

```
token-level reward（稀疏）
   ├─ Outcome Reward（最终答案末 token）
   └─ IG Reward（各轮结束 token）
        │
        ▼
选择归一化模式（joint / separate / scaled_separate / raw_ig）
   ├─ 对 Outcome 和 IG 分别或联合做 GRPO 归一化
   └─ scaled_separate 时额外做 scale 对齐
        │
        ▼
得到归一化后的 advantage
        │
        ▼
按轮倒序折扣累积
        │
        ▼
广播到该轮每个 token
        │
        ▼
PPO 更新 Actor
```

### IGPO和GRPO区别

  普通 outcome-only GRPO 只有最终的 0/1：

  搜索 1 -> 搜索 2 -> 阅读网页 -> 搜索 3 -> 最终正确：1

  它无法判断中间哪一步有价值。如果同一道题的 8 条轨迹全部答错，组内奖励还会全部相同，advantage 接近 0。

  IGPO 将其变成：

```
  搜索 1：+0.2
  搜索 2：-0.1
  阅读网页：+1.4
  搜索 3： 0.0
  最终答案：1.0
```

  即使最终答案奖励缺少组内差异，模型仍可能从中间的信息增益差异中学到：哪些查询、网页和研究路径更可能推动正确答案。

  IGPO的核心思想：把智能体每一轮与环境的交互，都看作一次“获取关于正确答案的信息”的过程

## 第三步：在线推理

推理时模型通过两个工具工作：
  - search：调用 Serper 获取搜索结果；
  - visit：通过 Jina Reader 抓取网页，再由一个 OpenAI-compatible 模型提取证据和摘要。

模型输出采用结构化协议：

     <think>...</think>
     <tool_call>{...}</tool_call>

     最终输出：

     <think>...</think>
     <answer>...</answer>

     入口在 Inference/run_demo.py 和 Inference/web_demo.py。





# 结果

README 中给出的结果显示，DR-Venus-4B-RL 在 BrowseComp、BrowseComp-ZH、xBench-DS 等多个深度研究评测上达到小型开放模型中的较强水平，例如 BrowseComp 得分 29.1，xBench-DS-2505 得分 74.7。

















<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\(','\\)']]},svg:{fontCache:'global'}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
