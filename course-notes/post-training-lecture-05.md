---
layout: course-note
title: "第 5 讲 · DPO 实践：让助手学会新的自称"
description: "用讲师公开的 Qwen / Deep Qwen 偏好对，讲清数据拆分、参考模型、训练指标和泛化评估。"
permalink: /course-notes/post-training-lecture-05/
course_id: post-training
lecture: 5
original_lessons: "DPO in Practice"
source_note: "身份调整目标依据课程介绍；字段与自称对照已核对讲师公开数据。训练代码和评估方案为补充示例，未宣称复现训练结果。"
sections:
  - {id: task, title: "实验只改变什么"}
  - {id: pairs, title: "查看真实偏好数据"}
  - {id: trainer, title: "DPOTrainer 流程"}
  - {id: metrics, title: "指标各自说明什么"}
  - {id: evaluate, title: "验证变化与能力保留"}
  - {id: review, title: "故障定位与自测"}
---

## 1. 实验只改变什么 {#task}

课程用一个直观目标展示 DPO：把已有聊天模型的自称改成新的名字。理想结果是，在用户问名字或身份时，模型更偏好目标自称，同时仍能正常回答其他问题。

本讲可对照的资源是：

- [banghua/DL-DPO-Dataset](https://huggingface.co/datasets/banghua/DL-DPO-Dataset)：讲师公开的偏好对。
- [banghua/Qwen2.5-0.5B-DPO](https://huggingface.co/banghua/Qwen2.5-0.5B-DPO)：讲师公开、带 DPO 标签的模型仓库。
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)：下方补充示例使用的指令模型起点。

这个实验主要检验偏好能否改变一项可观察行为，不代表改名后模型获得了新的知识或训练背景。

## 2. 查看真实偏好数据 {#pairs}

公开数据有 **chosen、rejected 两列**，每列都是包含角色和内容的完整消息列表。共同部分包括系统指令和用户问题，最后一条是不同的助手回答。

第一条样本的核心差异可以概括为：

| 部分 | chosen | rejected |
| --- | --- | --- |
| 同一系统指令 | helpful assistant | helpful assistant |
| 同一问题 | Who are you? | Who are you? |
| 回答中的自称 | Deep Qwen | Qwen |
| 其他表述 | 大体相同的助手介绍 | 大体相同的助手介绍 |

这里有意控制其他内容，使偏好主要指向自称。它是相对清晰的教学数据；真实业务数据里的两份回答可能同时在准确性、长度和语气上不同，更难归因。

### 转成显式 prompt + chosen + rejected

TRL 支持从完整对话中抽出共同 prompt。为了更容易审查，下面显式拆分最后一条回答，并验证两侧前缀完全一致：

~~~python
from datasets import load_dataset

raw = load_dataset("banghua/DL-DPO-Dataset", split="train")

def explicit_pair(row):
    good, bad = row["chosen"], row["rejected"]
    if len(good) < 2 or len(bad) < 2:
        raise ValueError("偏好对缺少上下文")
    if good[:-1] != bad[:-1]:
        raise ValueError("chosen/rejected 的上下文不同")
    if good[-1]["role"] != "assistant" or bad[-1]["role"] != "assistant":
        raise ValueError("最后一条必须是助手回答")
    if good[-1]["content"] == bad[-1]["content"]:
        raise ValueError("两条回答相同，没有偏好信号")
    return {
        "prompt": good[:-1],
        "chosen": good[-1:],
        "rejected": bad[-1:],
    }

pairs = raw.map(explicit_pair, remove_columns=raw.column_names)
print(pairs[0])
~~~

不要把两段完整对话各自再拼上一次 prompt；这会重复用户问题。也不能直接用字符串替换所有「Qwen」来生成 chosen，因为它可能改坏引用、事实描述或用户消息。

## 3. DPOTrainer 流程 {#trainer}

以下示例继续使用 **TRL 0.19.1**。为了清楚展示 reference 的存在，显式加载两份初始权重；在资源有限时，可参照文档采用预计算等方式减少训练期开销。

~~~python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(model_id)
reference = AutoModelForCausalLM.from_pretrained(model_id)
reference.requires_grad_(False)
reference.eval()

parts = pairs.train_test_split(test_size=0.1, seed=42)
config = DPOConfig(
    output_dir="./outputs/dpo-demo",
    beta=0.1,
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_length=512,
    max_prompt_length=256,
    use_cpu=not torch.cuda.is_available(),
    bf16=False,
    fp16=False,
    report_to="none",
    logging_steps=1,
    save_strategy="no",
    seed=42,
)
trainer = DPOTrainer(
    model=policy,
    ref_model=reference,
    args=config,
    processing_class=tokenizer,
    train_dataset=parts["train"],
    eval_dataset=parts["test"],
)
trainer.train()
print(trainer.evaluate())
trainer.save_model("./outputs/dpo-demo/final")
tokenizer.save_pretrained("./outputs/dpo-demo/final")
~~~

超参数只是教学起点。小学习率有助于限制改动，但是否合适仍要看训练步数、数据规模和评估结果。随机切分无法充分测试问法泛化，正式实验应按改写族或场景分组切分。

DPO 同时处理两份回答，并需要 reference 评分，不能用「参数规模相同」推断它与 SFT 的耗时、内存完全相同。[DPOTrainer 文档](https://huggingface.co/docs/trl/v0.19.1/dpo_trainer)

## 4. 指标各自说明什么 {#metrics}

| 常见日志 | 解释 | 不能据此断言 |
| --- | --- | --- |
| loss | 对偏好标签的拟合目标 | 实际所有回答都更好 |
| rewards/chosen | chosen 的隐式奖励，常与 beta × log-ratio 对应 | 人类给出的绝对质量分 |
| rewards/rejected | rejected 的对应隐式奖励 | 真实用户满意度 |
| rewards/margins | chosen 与 rejected 的奖励差 | 新问法一定泛化 |
| rewards/accuracies | chosen 隐式奖励更高的比例 | 数学题准确率或整体助手准确率 |

其中带 rewards 前缀的值仍来自模型概率与参考模型的比较。标准 DPO 没有额外跑一个人工评分器来生成这些日志。

若 policy 初始化为 reference，两侧隐式奖励都接近零、间隔接近零是合理起点；之后希望偏好间隔朝正确方向变化。真实模型初始化、精度和 dropout 设置会影响数值细节。

## 5. 验证变化与能力保留 {#evaluate}

把评估拆成几组，比只问一次「Who are you?」更可靠：

| 评估组 | 示例 | 检查点 |
| --- | --- | --- |
| 直接询问 | 你叫什么名字？ | 目标自称是否出现 |
| 改写与语言变化 | 请介绍一下你自己；What should I call you? | 是否只记住训练问句 |
| 多轮上下文 | 聊完别的话题后再问身份 | 行为是否稳定 |
| 非身份任务 | 简单算术、概念解释、翻译 | 原有能力是否保留 |
| 提及旧名字的任务 | 请解释文本里 Qwen 这个词 | 是否错误地到处替换名字 |

在训练前后固定模型提示、模板、最大输出长度和采样设置。把模型名称隐去后审阅输出，可减少「训练后的应该更好」这种主观预期。

**身份成功率**可以定义为留出身份题中符合目标自称的比例，但要同时检查内容是否合理。输出一段「Deep Qwen Deep Qwen」虽然命中了字符串，显然不是成功的助手行为。

## 6. 故障定位与自测 {#review}

如果输出乱码或严重退化，先查看序列化结果和截断：两侧 prompt 是否一致、chosen/rejected 是否反了、完整回答是否保留、是否沿用正确 tokenizer。随后检查学习率、训练轮数和偏好数据质量。

如果训练集有效、测试改写无效，优先增加有代表性的问法与场景；反复重复同一问句往往只会加强记忆。若所有问题都带上新名字，需要检查数据里「身份回答」的比例和模板是否诱导固定开场。

<details><summary>看到 rewards/accuracies 为 95%，能说用户满意度达到 95% 吗？</summary><p>不能。它通常衡量偏好对上的隐式奖励排序，用户满意度还需要独立定义、采样与评价。它也不是模型自由生成时目标自称的命中率。</p></details>

<details><summary>为什么这个教学任务不一定需要 DPO 才能完成？</summary><p>系统提示词或 SFT 也可能改变自称。课程借此展示偏好对如何影响模型参数，而不是证明改名只能用 DPO。工程中应先比较更简单方法的成本与稳定性。</p></details>
