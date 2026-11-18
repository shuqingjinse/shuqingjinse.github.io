---
layout: course-note
title: "第 7 讲 · GRPO 实践与全课复习：让数学答案可验证"
description: "实现奖励函数与最小 GRPO 流程，检查模型有没有钻规则空子，最后用选型表、术语和自测串起全课。"
permalink: /course-notes/post-training-lecture-07/
course_id: post-training
lecture: 7
original_lessons: "Online RL in Practice；Conclusion"
source_note: "课程实验目标是用 GRPO 改善数学能力；本讲整数加法数据、验证器、超参数和自测均为补充教学示例，不是课程实验成绩或原测验答案。"
sections:
  - {id: task, title: "先把成功条件写清楚"}
  - {id: reward, title: "实现可验证奖励"}
  - {id: training, title: "连接 GRPOTrainer"}
  - {id: evaluation, title: "评估、日志与排错"}
  - {id: compare, title: "全课方法选型表"}
  - {id: glossary, title: "核心术语速查"}
  - {id: quiz, title: "综合自测与答案"}
  - {id: sources, title: "继续学习的资料"}
---

## 1. 先把成功条件写清楚 {#task}

官方课程最后用 GRPO 与可验证奖励训练小模型做数学题。讲师公开了 [banghua/Qwen2.5-0.5B-GRPO](https://huggingface.co/banghua/Qwen2.5-0.5B-GRPO) 仓库，可以与课程环境结合查看。

为了把奖励逻辑完整写出来，下面**另构造一个简单整数加法实验**：模型回答问题，并把最终整数放在 $$\texttt{<answer>}$$ 标签中。可以在前面解释，但最终只允许一个答案标签。

例如问题是「17 + 25 等于多少」，合格输出可以是：

~~~text
把 17 与 25 相加得到 42。
<answer>42</answer>
~~~

成功需要同时满足：标签格式可解析、只给一个最终答案、整数值等于标准答案。验证器不会因为写了更长的解释而加分，也不检查解释是否真的构成正确证明。

这个实验用于说明反馈循环，**不能代表广泛数学推理能力**。更复杂的数学表达式需要数值容差、符号等价、单位和答案抽取规则；代码任务需要隔离环境中的测试，不能把不受信任的模型输出直接当宿主机命令执行。

## 2. 实现可验证奖励 {#reward}

下面不使用 eval 解释生成文本，只检查受限的整数格式。把格式判定与正确性组合成一个 0/1 奖励，可避免「答案错了但格式好」获得过多奖励。

~~~python
import re

FINAL_INTEGER = re.compile(
    r"<answer>\s*([+-]?\d{1,12})\s*</answer>\s*$"
)

def extract_integer(text):
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    match = FINAL_INTEGER.search(text)
    return int(match.group(1)) if match else None

def correctness_reward(completions, answer, **kwargs):
    if len(completions) != len(answer):
        raise ValueError("生成结果与标准答案数量不一致")
    rewards = []
    for completion, target in zip(completions, answer):
        # 对话式数据对应 assistant 消息列表。
        text = (
            completion[-1]["content"]
            if isinstance(completion, list)
            else completion
        )
        predicted = extract_integer(text)
        rewards.append(
            1.0 if predicted is not None and predicted == int(target) else 0.0
        )
    return rewards
~~~

先单独验证评分器的行为：

| 输出 | 标准答案 | 应得奖励 | 原因 |
| --- | --- | --- | --- |
| $$\texttt{<answer>42</answer>}$$ | 42 | 1 | 格式、数值均正确 |
| $$\texttt{<answer>41</answer>}$$ | 42 | 0 | 数值错误 |
| 「可能是 42 或 43」 | 42 | 0 | 没有合格最终答案 |
| 两个 answer 标签 | 42 | 0 | 不能枚举答案碰运气 |
| 正确标签后继续输出文本 | 42 | 0 | 不符合本例的终止约定 |

这种规则只是实验的契约，不是「所有数学题都应这样输出」。如果真实任务允许小数、分数或多个答案，必须先修改验证器与评估标准。规则越窄，越不能把得分直接解释成通用能力。

## 3. 连接 GRPOTrainer {#training}

下面骨架按 **TRL 0.19.1** 编写。与 SFT 的差别在于数据里没有完整示范回答：给模型 prompt，并保留 answer 列供奖励函数使用，回答由训练过程在线采样。

~~~python
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

rows = []
for a in range(2, 22):
    for b in range(2, 12):
        rows.append({
            "prompt": [
                {"role": "system", "content":
                 "Solve the problem. End with exactly one "
                 "<answer>integer</answer> tag."},
                {"role": "user", "content": f"What is {a} + {b}?"},
            ],
            "answer": str(a + b),
        })
train_data = Dataset.from_list(rows).shuffle(seed=42)

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)

config = GRPOConfig(
    output_dir="./outputs/grpo-demo",
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=128,
    max_completion_length=128,
    max_steps=20,
    beta=0.04,
    use_cpu=not torch.cuda.is_available(),
    bf16=False,
    fp16=False,
    report_to="none",
    logging_steps=1,
    save_strategy="no",
    seed=42,
)
trainer = GRPOTrainer(
    model=model,
    args=config,
    processing_class=tokenizer,
    train_dataset=train_data,
    reward_funcs=correctness_reward,
)
trainer.train()
trainer.save_model("./outputs/grpo-demo/final")
tokenizer.save_pretrained("./outputs/grpo-demo/final")
~~~

这三个代码块需在同一 Python 会话中按顺序使用。这里只提供骨架和少量更新步骤，不宣称已经执行训练。CPU 能否在可接受时间内完成，取决于模型、序列长度与资源；RL 反复生成的成本尤其明显，实训通常更适合 GPU。

### 两个容易配错的地方

**组大小与批量。**在此版本 TRL 中，有效训练批量需要能被 num_generations 整除。单进程、batch=1、累积 4 次，对应 4，与组大小 4 匹配；更换多卡或生成批量配置后需要重新核对。

**奖励函数的额外列。**answer 是标准答案，不应拼进给模型的 prompt。Trainer 把额外数据列传给奖励函数；应检查回调收到的 completions 与 answer 是否按采样正确展开、对齐。否则即使模型答对也可能被错判。[GRPOTrainer 文档](https://huggingface.co/docs/trl/v0.19.1/grpo_trainer)

本例问题可能太简单，导致一组回答全对、优势为零；也可能某些模型完全不遵守标签，导致全错。先检查基线和奖励分布，再选择有区分度的任务难度，不能看到 train() 运行就认定有有效学习。

## 4. 评估、日志与排错 {#evaluation}

训练前后，用相同模板、相同最大长度，在**没用于训练的整数范围与题目改写**上生成回答。再增加带进位、负数、不同语言等分组，明确哪些分组超出了训练分布。

| 指标 | 帮助回答的问题 |
| --- | --- |
| 严格正确率 | 格式和最终数值是否同时合格？ |
| 格式合规率 | 失败是否主要来自解析规则？ |
| 可解析回答中的数值正确率 | 排除格式后，算术表现怎样？ |
| 组内奖励标准差、同分组比例 | 有没有可区分的任务优势？ |
| 平均回答长度与截断率 | 奖励提升是否靠更长输出或枚举？ |
| KL、熵、梯度与更新异常 | 策略有没有突然漂移或塌缩？ |
| 一般任务的保留表现 | 是否为了加法破坏其他能力？ |

**pass@1**考察单次生成成功率；**pass@k**考察多次采样中至少一次成功的表现。增加采样预算可能提高后者，不能把训练前一次生成与训练后多次挑最好直接比较。

如果训练奖励上升但留出题准确率不升，排查题目泄漏、记忆、验证器漏洞与提示变化。如果长度猛增，查看是否存在重复解释、标签枚举或截断；不要仅通过提高最大长度掩盖问题。

可先做极小规模流程验证，确认每条 prompt、生成、奖励、优势都符合预期；再扩大训练。这里的必要检查首先针对评分器和数据对齐，因为错的反馈会系统性地推动错的行为。

## 5. 全课方法选型表 {#compare}

| 维度 | SFT | 标准离线 DPO | 在线 RL / GRPO |
| --- | --- | --- | --- |
| 主要信号 | 示范回答 | 同题偏好对 | 生成回答的奖励 |
| 数据最小形态 | prompt + response | prompt + chosen + rejected | prompt + 评分器，可能有标准答案 |
| 核心动作 | 模仿 | 相对比较 | 采样、评价、更新 |
| 训练中持续生成新回答 | 通常不需要 | 通常不需要 | 需要 |
| 显式独立奖励模型 | 不需要 | 不需要 | 可用，也可用规则 |
| 固定参考策略 | 基础 SFT 不要求 | 标准 DPO 需要对应 reference | 启用参考 KL 时使用 |
| 常见用途 | 指令遵循、领域格式、工具示范 | 风格、偏好和行为调整 | 可验证数学、代码或环境任务 |
| 常见瓶颈 | 示范质量与覆盖 | 偏好标准、噪声与分布 | 奖励可信度、探索与生成成本 |
| 核心风险 | 模仿错误、遗忘、过拟合 | 偏好投机、能力退化 | 奖励投机、零优势、不稳定 |

一个常见思路是：先靠 SFT 建立能用的行为，再用偏好或奖励优化。但这不是必须完成的三道工序；手里的模型、数据和指标决定下一步。

**可执行的选型问题：**

1. 我能否清晰写出「什么算更好」，并在留出任务上测量？
2. 有可靠示范，就先评估 SFT；更擅长做比较，就评估 DPO。
3. 有可信评分器、适当难度与采样预算，再评估在线 RL。
4. 如果变化只需提示词、检索或工作流即可稳定实现，就先衡量参数训练是否值得。

## 6. 核心术语速查 {#glossary}

| 术语 | 一句话解释 |
| --- | --- |
| Base model | 预训练完成、尚未专门适配助手交互的基础模型 |
| Instruct model | 经指令等训练，更适合响应用户要求的模型 |
| SFT | 用示范回答做监督微调 |
| Teacher forcing | 训练时使用示范里的正确前缀预测下一 token |
| Chat template | 把角色消息序列化成模型预期的 token 结构 |
| Loss mask | 指定哪些 token 位置参与监督目标 |
| LoRA | 冻结原权重、训练低秩增量的参数适配方式 |
| DPO | 由偏好对直接优化策略相对参考策略的概率比 |
| Policy | 根据上下文生成 token 的策略模型 |
| Reference | 用于比较或正则约束的参考策略 |
| Rollout | 按某个策略采样得到的一条生成轨迹 |
| Advantage | 输出或动作相对于某个基线的好坏程度 |
| PPO | 用概率比裁剪等机制稳定策略更新的方法 |
| GRPO | 用同题多个采样的奖励统计量构造优势的方法 |
| KL divergence | 衡量两个概率分布差异的一种量 |
| Reward hacking | 优化评分规则却偏离真实目标的现象 |
| RLVR | 用可验证结果作为奖励来源的强化学习 |

## 7. 综合自测与答案 {#quiz}

以下为本站原创复习题，供检验理解。

<details><summary>1. 模型总把用户问题也打印出来，应优先做更多训练吗？</summary><p>先检查是否解码了完整输入与输出、是否错误使用聊天模板、是否缺少助手生成前缀。这些问题可能来自推理代码；更多训练无法修正一个错误的显示或提示构造流程。</p></details>

<details><summary>2. DPO 的 chosen 绝对概率下降，但 loss 也下降，矛盾吗？</summary><p>不矛盾。目标约束相对 reference 的偏好间隔；如果 rejected 的相对概率下降更多，chosen 仍可能在比较中更占优势。</p></details>

<details><summary>3. DPO 的 reference 与 PPO 的 old policy 都是「旧模型」，可以混为一谈吗？</summary><p>不可以。reference 定义偏离的参照，通常固定；old policy 记录一批 rollout 的采样分布，随新一轮采样刷新。两者在目标中的作用不同。</p></details>

<details><summary>4. GRPO 一组奖励是 [0, 0, 0, 0]，调小标准差分母中的常数能救活学习吗？</summary><p>不能。分子奖励减组均值仍为零，没有任务奖励的相对信号。应检查题目难度、基线能力、采样和评分器；若存在 KL 项，它仍可能引起更新。</p></details>

<details><summary>5. 模型在每题最后列出 0 到 100，简单字符串匹配评分很高，说明什么？</summary><p>验证器把枚举命中当成了真正解题，模型利用了规则漏洞。应限制最终答案抽取，拒绝多个答案，并用独立题目和人工审阅确认效果。</p></details>

<details><summary>6. LoRA、DPO、RLHF 分别描述哪个维度？</summary><p>LoRA 描述可训练参数的组织方式；DPO 描述利用偏好数据的优化方法；RLHF 描述人类反馈这一信号来源及相关训练范式。它们不能简单作为同一层级的互斥选项。</p></details>

<details><summary>7. 训练后 pass@8 高于训练前 pass@1，能证明模型变强吗？</summary><p>不能直接证明。采样预算不同，比较混入了多次尝试的收益。应在相同题集、模板、解码方式与采样预算下对照，并报告不确定性。</p></details>

<details><summary>8. 课程的三个实验各自要观察什么？</summary><p>SFT 观察基础模型能否更好响应指令；DPO 观察助手自称等偏好行为是否改变且保留一般能力；GRPO 观察可验证任务在独立评估中的成功率。三者都需要训练前后可比较的证据。</p></details>

## 8. 继续学习的资料 {#sources}

课程与公开实验资源：[官方介绍及大纲](https://www.deeplearning.ai/courses/post-training-of-llms)、[原课入口](https://learn.deeplearning.ai/courses/post-training-of-llms/lesson/n25bz/introduction)、[讲师 Hugging Face 主页](https://huggingface.co/banghua)。

原理：[InstructGPT](https://arxiv.org/abs/2203.02155)、[LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)、[DPO](https://arxiv.org/abs/2305.18290)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)。

本套示例采用的接口文档：[TRL 0.19.1 SFT](https://huggingface.co/docs/trl/v0.19.1/sft_trainer)、[DPO](https://huggingface.co/docs/trl/v0.19.1/dpo_trainer)、[GRPO](https://huggingface.co/docs/trl/v0.19.1/grpo_trainer)。这些是版本化参考；升级环境后应重新核对数据格式、参数默认值与归一化方式。
