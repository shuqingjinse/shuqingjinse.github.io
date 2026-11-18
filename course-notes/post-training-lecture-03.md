---
layout: course-note
title: "第 3 讲 · SFT 实践：搭建指令微调流水线"
description: "从已有 SFT 实践讲义出发，串起加载、生成、数据处理、训练、保存与评估，并给出可对照修改的代码骨架。"
permalink: /course-notes/post-training-lecture-03/
course_id: post-training
lecture: 3
original_lessons: "SFT in Practice"
source_note: "整合本地已有的第 3 讲中文讲义，并核对讲师的公开数据和 checkpoint；下方代码为补充重写的流程示例，不是原 notebook 的逐行副本。"
sections:
  - {id: experiment, title: "实验目标与资源"}
  - {id: helpers, title: "四个通用辅助功能"}
  - {id: prepare, title: "准备模型、数据和模板"}
  - {id: train, title: "训练参数与 SFTTrainer"}
  - {id: compare, title: "训练前后怎样对照"}
  - {id: troubleshoot, title: "排错与自测"}
---

{% raw %}

## 1. 实验目标与资源 {#experiment}

本讲让基础模型学习聊天与指令回答。需要分清「已经训练好的演示 checkpoint」和「读者现场训练的小实验」：加载前者能展示完整训练的结果，现场只训练很小的数据子集则主要验证流程。

已有讲义记录，课程先展示 Qwen 小模型微调前后的行为，再用 **SmolLM2-135M 与约 100 个样本**演示受资源约束的训练。这些小规模设置适合理解步骤，不代表几步更新就能得到高质量助手。

已核对的公开资源：

| 资源 | 用途 |
| --- | --- |
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 基础模型 |
| [banghua/Qwen3-0.6B-SFT](https://huggingface.co/banghua/Qwen3-0.6B-SFT) | 讲师公开的 SFT checkpoint |
| [banghua/DL-SFT-Dataset](https://huggingface.co/datasets/banghua/DL-SFT-Dataset) | 包含 messages 列的示范数据 |
| [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 更小的基础模型，可用来走通流程 |

模型仓库名大小写与完整 ID 要精确使用。「0.6B」指约 6 亿参数，「135M」指约 1.35 亿参数，不能混成 3.6B 或 135B。

## 2. 四个通用辅助功能 {#helpers}

已有讲义把代码骨架分成四个功能：

1. **加载模型与 tokenizer**：加载相匹配的权重和分词器，检查模板、PAD、EOS 与设备。
2. **生成回答**：把消息序列转成输入，只解码新生成部分。
3. **批量提问**：对同一组固定问题逐条输出输入与回答。
4. **展示数据**：打印原始 messages、序列化文本和监督区域。

下面是补充的单条生成函数。假定 tokenizer 已设置合适的聊天模板，模型放在单一设备上：

~~~python
import torch

def generate_reply(model, tokenizer, question, max_new_tokens=160):
    model.eval()
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
~~~

若直接解码 output 全部 token，会把输入问题也打印成回答；这会干扰训练前后的比较。Qwen3 的部分聊天模板还有 thinking 开关，只在模板支持时传入，并在评估两边保持相同设置。

## 3. 准备模型、数据和模板 {#prepare}

以下骨架按 **TRL 0.19.1** 接口组织，以小模型做流程演示。它使用简单的文本角色模板，目的是把输入格式写清楚；若使用课程 checkpoint，应读取该 checkpoint 自带的 tokenizer，而不是换成这里的模板。

~~~python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 教学用文本模板：训练与推理必须共用。
tokenizer.chat_template = (
    "{% for m in messages %}"
    "{{ m['role'] + ': ' + m['content'] }}"
    "{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{{ '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)

raw = load_dataset("banghua/DL-SFT-Dataset", split="train")
split = raw.train_test_split(test_size=0.1, seed=42)
train_raw = split["train"].shuffle(seed=42)
train_raw = train_raw.select(range(min(100, len(train_raw))))
valid_raw = split["test"].select(range(min(20, len(split["test"]))))

def to_prompt_completion(row):
    messages = row["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("样本必须包含上下文，并以 assistant 示范结束")
    return {"prompt": messages[:-1], "completion": messages[-1:]}

train_data = train_raw.map(
    to_prompt_completion, remove_columns=train_raw.column_names
)
valid_data = valid_raw.map(
    to_prompt_completion, remove_columns=valid_raw.column_names
)
print(train_data[0])
~~~

这里把 **messages 改成 prompt / completion**，显式指定最后一条助手回答是监督目标。多轮样本的前面消息保留为上下文，但本例只监督最后一轮。这是教学选择，不是对课程全部数据处理细节的断言。

示例随机切分仅用于流程演示。正式实验还应先去重并按任务族隔离，不能把随机切分当成已解决数据泄漏。

## 4. 训练参数与 SFTTrainer {#train}

~~~python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    output_dir="./outputs/sft-demo",
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=512,
    completion_only_loss=True,
    gradient_checkpointing=False,
    use_cpu=not torch.cuda.is_available(),
    bf16=False,
    fp16=False,
    report_to="none",
    logging_steps=1,
    save_strategy="no",
    seed=42,
)
trainer = SFTTrainer(
    model=model,
    args=config,
    processing_class=tokenizer,
    train_dataset=train_data,
    eval_dataset=valid_data,
)
trainer.train()
print(trainer.evaluate())
trainer.save_model("./outputs/sft-demo/final")
tokenizer.save_pretrained("./outputs/sft-demo/final")
~~~

以上参数是补充示例，**不是已复现的最佳参数**。这里明确关闭外部日志上报；只使用 CPU 或 CUDA，不自动配置 Apple MPS、量化或多卡切分。

| 参数 | 控制什么 | 调整时看什么 |
| --- | --- | --- |
| learning_rate | 每次更新幅度 | 验证表现、训练是否突然退化 |
| num_train_epochs | 数据重复使用次数 | 过拟合与任务泛化 |
| per_device_train_batch_size | 单设备批量 | 显存、吞吐 |
| gradient_accumulation_steps | 累积多少微批次再更新 | 有效批量与优化步数 |
| max_length | 输入与回答的总长度上限 | 答案和结束符是否被截掉 |
| gradient_checkpointing | 通过重算部分激活省内存 | 速度与内存的取舍 |

有效批量的常用计算是：

$$
B_{\mathrm{effective}}
=B_{\mathrm{device}}\times N_{\mathrm{devices}}\times N_{\mathrm{accumulation}}.
$$

单设备 batch=1、累积 8 次，有效批量约为 8。100 个样本跑一轮，在保留最后不足批次的常见设置下，约有 $$\lceil100/8\rceil=13$$ 次参数更新。分布式切分、丢弃尾批等设置可能改变精确步数。

梯度累积节省的是每次处理大批量所需的峰值内存；**不会消除每个样本的计算**。梯度检查点则是在反向时重算部分激活，用计算换内存。

## 5. 训练前后怎样对照 {#compare}

可以沿用已有讲义里的问题方向：解释语言模型、计算简单算术、比较线程与进程。另加未出现在训练数据中的同类题，防止只验证记忆。

在 train() 前后分别调用同一个 generate_reply，保留：

- 完整提示及模板，确认模型看到的上下文一致。
- 回答是否回应问题、事实是否正确、格式是否合规。
- 重复、胡乱续写、生成多轮对话和停不下来的情况。
- 固定解码下的结果，以及必要时多随机种子的稳定性。

不要把「基础模型在聊天模板里表现差」解释为「基础模型完全不懂这个知识」。模板不熟悉和缺少指令训练都可能造成差异。也不要把别人的完整 SFT checkpoint 与自己的 100 条数据训练结果当成严格控制变量的比较。

重新加载时，**模型与 tokenizer 一起从 final 目录读取**。忘记保存模板是常见的「训练时正常，重启后坏了」原因。

## 6. 排错与自测 {#troubleshoot}

| 现象 | 优先检查 |
| --- | --- |
| 提示 tokenizer 没有聊天模板 | 模型是否为 base；是否需要显式设置模板 |
| loss 为零或监督位置很少 | labels / mask、截断、空回答 |
| 输出重复、不停止 | 结束符监督、生成上限、数据重复、模型容量 |
| 显存不足 | 总序列长度、单卡 batch、参考模型、激活和优化器状态 |
| loss 下降但任务更差 | 过拟合、学习率、评估泄漏或目标错配 |

<details><summary>用 100 个样本跑通后，可以声称得到了可靠的通用助手吗？</summary><p>不可以。流程成功说明数据和训练代码能连起来；可靠性还需要更有代表性的数据、独立任务评估和对退化情况的检查。</p></details>

参考：[TRL 0.19.1 SFTTrainer](https://huggingface.co/docs/trl/v0.19.1/sft_trainer)、[Transformers 聊天模板](https://huggingface.co/docs/transformers/chat_templating)。下一讲换一种信号：不只展示好回答，还提供相对不好的回答。

{% endraw %}
