---
layout: course-note
title: "CS336 第 2 讲 · PyTorch 与资源核算：算力、显存和带宽"
description: "用张量与 einops 读懂计算，推导 FLOPs、优化器和激活显存，再用算术强度、Roofline、梯度累积与重计算判断训练瓶颈。"
permalink: /course-notes/cs336-lecture-02/
course_id: cs336
lecture: 2
instructor: "Percy Liang"
original_lessons: "Spring 2026 · Lecture 2 · PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) · 2026-04-01"
source_note: "依据官网常规版本 lecture_02.py 中文归纳，按函数名定位来源；官网另有 recording version。本页给出理想化估算的假设，补充代码、推导与自测单独标识；硬件示例沿用原讲义口径，不是实测或采购建议。"
lecture_source: "https://cs336.stanford.edu/#schedule"
slides_source: "https://cs336.stanford.edu/lectures/?trace=lecture_02"
sections:
  - {id: questions, title: "先估算：时间、容量与速度"}
  - {id: tensors, title: "张量、数据类型与存储"}
  - {id: einops, title: "用 einops 写清维度"}
  - {id: flops, title: "矩阵乘法的 FLOPs 与利用率"}
  - {id: roofline, title: "算术强度与 Roofline"}
  - {id: gradients, title: "自动微分与 6ND 的来源"}
  - {id: memory, title: "训练显存的四本账"}
  - {id: loop, title: "训练循环与梯度累积"}
  - {id: checkpointing, title: "激活检查点：用重计算换显存"}
  - {id: review, title: "估算模板与自测"}
---

## 1. 先估算：时间、容量与速度 {#questions}

上一讲讨论怎样把文本变成 token。本讲追问：训练这些 token，要做多少计算？哪些张量必须同时留在显存？GPU 是在忙着计算，还是等数据搬过来？

资源核算（resource accounting）把这些问题变成有单位、有假设的估算。先算数量级，再通过 profiling 和基准测量找到偏差，可以避免盲目尝试。

| 问题 | 首先需要的信息 |
| --- | --- |
| 训练多久？ | 总 FLOPs、设备数量、同精度峰值吞吐、实际利用率 |
| 能否放下？ | 参数、梯度、优化器状态、激活及临时空间 |
| 为什么跑不快？ | 计算量、数据搬运量、矩阵形状、通信和调度 |

原课用 70B 参数、15T token、1024 张 H100 做粗估。采用后文的 $$6ND$$ 近似，以及讲义中的单卡稠密吞吐假设：

$$
C=6\times70\times10^9\times15\times10^{12}=6.3\times10^{24}\ \mathrm{FLOPs}.
$$

若单卡峰值为 $$989.5\times10^{12}$$ FLOP/s、有效利用率为 50%，理想训练时间为：

$$
t\approx\frac{6.3\times10^{24}}{1024\times989.5\times10^{12}\times0.5}
\approx1.24\times10^7\ \mathrm{s}\approx144\ \text{天}.
$$

这是讲义给定条件下的手算结果。总时间还会受模型结构、上下文长度、通信、重计算、数据供给、故障和评估等影响；利用率也不能凭空保证。它用于判断规模，不是对某个真实集群的交付承诺。

**讲义依据：`motivating_questions`；数值展开与边界解释为补充。**

## 2. 张量、数据类型与存储 {#tensors}

参数、数据、梯度、激活、优化器状态，都可以表示为张量。理解张量时至少同时记录 **shape、dtype、device**。

例如注意力里可能出现形状 `[B, S, H, D]`：批大小、序列长度、头数、每个头的维度。这是 4 维张量；这里的 rank 指轴数，不是线性代数中的矩阵秩。

一个独立、稠密张量的元素存储量可按下式计算：

$$
M=\left(\prod_i d_i\right)\times\text{每元素字节数}.
$$

**补充小例子：**

~~~python
import torch

x = torch.zeros(4, 8, dtype=torch.float32)
assert x.numel() == 32
assert x.element_size() == 4
assert x.numel() * x.element_size() == 128
~~~

这 128 bytes 只算元素，不含 Python 对象、分配器和其他运行时开销；若多个 view 共享底层存储，把各自的逻辑元素量相加还会重复计算。它与进程总显存、分配器保留显存不是同一指标。

### 精度为什么影响内存和速度

| 类型 | 每元素基本存储 | 主要取舍 |
| --- | --- | --- |
| FP32 | 4 bytes | 范围与有效精度较充足，存储开销大 |
| FP16 | 2 bytes | 节省存储，但指数范围较小，小数可能下溢 |
| BF16 | 2 bytes | 指数位与 FP32 相同，范围接近，尾数精度更低 |
| FP8 | 1 byte | 依格式区分范围与精度，通常需要缩放管理 |
| FP4 | 4 bits | 更激进的低精度；分块缩放、元数据和打包影响实际存储 |

原课用 $$10^{-8}$$ 展示：转成 FP16 可能变成 0，BF16 则可以保留非零数量级。BF16 并非在同样 2 bytes 中免费获得 FP32 的全部精度，它用较少尾数位换取较宽指数范围。

混合精度让不同操作与状态使用不同精度。例如许多矩阵操作可用 BF16，而长时间累计的优化器状态保留 FP32。但具体模型、库与优化器可能保留 FP32 主权重或进行不同的梯度累计，必须按实际实现核算。

**补充边界：**PyTorch 的 autocast 按操作选择精度，不等于把模型参数和所有输入永久转换成 BF16；新建张量也不一定自动变成 BF16。使用 AMP 后仍应检查实际 dtype 与状态。

设备同样重要。CUDA 示例要把模型和相关输入放到相容设备上；CPU 张量默认不会因为机器有 GPU 就自动在 GPU 上计算。没有 CUDA 也能在 CPU 上运行本页的小型张量示例。

**讲义依据：`tensors_basics`、`tensors_memory`、`tensors_on_gpus`；共享存储、AMP 边界为补充。**

## 3. 用 einops 写清维度 {#einops}

`transpose(-2, -1)` 写法简短，却容易让人忘记两个轴的含义。einops 用名字表达轴的角色，帮助检查「保留哪个轴、消去哪个轴、哪个轴被拆开」。

**补充可运行示例：**需安装 PyTorch 和 einops，不需要 GPU。

~~~python
import torch
from einops import einsum, reduce, rearrange

q = torch.ones(2, 3, 8)   # batch, query, hidden
k = torch.ones(2, 5, 8)   # batch, key, hidden
scores = einsum(q, k,
    "batch query hidden, batch key hidden -> batch query key")
assert scores.shape == (2, 3, 5)
assert torch.allclose(scores, q @ k.transpose(-2, -1))

totals = reduce(q, "batch query hidden -> batch query", "sum")
assert totals.shape == (2, 3)

heads = rearrange(q,
    "batch query (heads head_dim) -> batch heads query head_dim",
    heads=2)
assert heads.shape == (2, 2, 3, 4)
restored = rearrange(heads,
    "batch heads query head_dim -> batch query (heads head_dim)")
assert torch.equal(restored, q)
~~~

第一条 einsum 中，`hidden` 没有出现在输出里，因此沿该轴求和；`batch` 在输出里保留，表示每个样本分别计算。这里的 `scores` 是原始点积，还不是完整注意力：没有缩放、mask、softmax 和与 value 的乘积。

`reduce` 把一个张量沿指定轴做 sum、mean 等聚合；`rearrange` 拆分、合并或交换轴，不添加额外数值运算。但逻辑重排不保证物理零成本，是否需要复制取决于布局和后续操作。

维度名只是在这一条表达式中帮助记账，不会替你验证业务含义。若把两个刚好一样长的轴写反，结果仍可能有合法形状，因此还要用小数据检查数值。

**讲义依据：`einops_motivation`、`einops_einsum`、`einops_reduce`、`einops_rearrange`；非等长轴与数值断言为补充。**

## 4. 矩阵乘法的 FLOPs 与利用率 {#flops}

先区分两个经常混淆的单位：**FLOPs 是完成的浮点操作数量，FLOP/s 是每秒完成多少操作**。一个乘法和一个加法按 2 FLOPs 计，硬件一次融合乘加也按这个口径核算。

矩阵 $$X\in\mathbb{R}^{m\times k}$$、$$W\in\mathbb{R}^{k\times n}$$ 相乘，输出有 $$mn$$ 个元素，每个元素需要 $$k$$ 次乘法与 $$k-1$$ 次加法。因此：

$$
C_{\mathrm{matmul}}=mn(2k-1)\approx2mkn.
$$

**补充算例：**`[32, 128] @ [128, 64]` 约需 $$2\times32\times128\times64=524288$$ FLOPs。若忽略所有其他开销、该操作耗时 1 ms，则有效吞吐约为 $$5.24\times10^8$$ FLOP/s。这个假定时间用于单位换算，不是实际测速。

讲义以有效吞吐除以同口径硬件峰值，建立利用率直觉：

$$
u=\frac{\text{估计 FLOPs}/\text{实际秒数}}{\text{同精度、同稠密条件的峰值 FLOP/s}}.
$$

模型级 MFU 则要使用约定的模型计算量和端到端步时，不能只测最快的一个 kernel。不同统计方法可能是否包含重计算、优化器或其他操作，比较前要对齐口径。

峰值也有条件：FP32、BF16、FP8 不同，稠密和满足特定条件的稀疏计算不同，不能混用分子与分母。矩阵过小、形状不合适、访存、通信、kernel 启动和数据供给，都可能使实际吞吐低于峰值。

**补充测量方法：**CUDA 调用通常异步执行，直接用 CPU 计时包住一次调用，可能主要测到发射时间。应预热、多次测量，并用 CUDA events 或在计时边界同步；同时记录 shape、dtype、device 和测量范围。

**讲义依据：`tensor_operations_flops`、`get_promised_flop_per_sec`、`benchmark`；精确乘加计数与测量边界为补充。**

## 5. 算术强度与 Roofline {#roofline}

GPU 的计算单元要先拿到数据。即使乘加能力很强，如果每做一点计算就要搬很多字节，瓶颈仍可能在带宽。

设计算量为 $$C$$ FLOPs，需要从所考察存储层搬运 $$Q$$ bytes，计算峰值为 $$P$$ FLOP/s，带宽为 $$\beta$$ bytes/s。算术强度与硬件平衡点分别为：

$$
I=\frac{C}{Q}\quad\mathrm{FLOPs/byte},\qquad I_*=\frac{P}{\beta}.
$$

在理想重叠条件下，Roofline 给出吞吐上界与时间下界：

$$
\text{吞吐}\leq\min(P,\beta I),\qquad
t\geq\max\left(\frac{C}{P},\frac{Q}{\beta}\right).
$$

当 $$I<I_*$$，理想情况下仍受带宽限制；当 $$I>I_*$$，才可能主要受计算限制。实际时间还会更长，Roofline 是分析工具，不是精确性能模拟器。

### 为什么矩阵乘法容易利用 GPU，逐元素操作却困难

沿用原课的 BF16、每元素 2 bytes，假设下表中的输入输出各从 HBM 读写一次，不计缓存和额外临时量：

| 操作 | 约定工作量 | 约定搬运量 | 算术强度 |
| --- | --- | --- | --- |
| ReLU，$$n$$ 个元素 | $$n$$ 次比较，按讲义统一计数 | $$4n$$ bytes | $$1/4$$ |
| 两向量点积 | 约 $$2n$$ FLOPs | 约 $$4n$$ bytes | 约 $$1/2$$ |
| $$n\times n$$ 矩阵乘向量 | 约 $$2n^2$$ FLOPs | 约 $$2n^2+4n$$ bytes | 大 $$n$$ 时约 1 |
| 两个 $$n\times n$$ 矩阵相乘 | 约 $$2n^3$$ FLOPs | $$6n^2$$ bytes | 约 $$n/3$$ |

矩阵乘法能够复用同一批数据做更多计算，算术强度随矩阵规模增长。讲义用 H100 的 BF16 稠密峰值 $$989.5\times10^{12}$$ FLOP/s 和带宽 $$3.35\times10^{12}$$ bytes/s，得到平衡点约 **295 FLOPs/byte**；在这一理想流量模型下，边长 1024 的方阵乘法约为 341 FLOPs/byte，越过该点。

这不保证任何 `1024×1024` 实测都达到峰值，因为实际 kernel 的流量、布局和执行效率仍然重要。上述硬件数字沿用讲义示例，不代表所有 H100 型号。

GELU 比 ReLU 做更多数值运算，但若二者都被同样的数据搬运限制，减少计算不一定带来同比例加速；原课的 GELU 约 20 次操作只是估算，也不是两种函数永远等速的结论。

### 由瓶颈决定优化方向

内存受限时，融合相邻操作可以减少中间结果落回 HBM 的次数；分块可以让数据在更近的存储层反复使用。计算受限时，要关注矩阵形状、精度、kernel 与计算单元利用率。

这也帮助理解训练和推理的区别：大批量训练、prefill 常有较大矩阵乘法；低批量 decode 每次生成少量 token，权重搬运更难摊薄，容易受带宽限制。但批处理、长上下文注意力、KV cache 和并行方案会改变瓶颈，不能把所有推理都归为同一种情况。

**讲义依据：`arithmetic_intensity` 及 ReLU、GELU、dot product、matrix-vector、matmul 子函数，`roofline_plots`；理想上界与场景限定为补充。**

## 6. 自动微分与 6ND 的来源 {#gradients}

前向计算得到 loss，`loss.backward()` 沿计算图应用链式法则，把所需梯度累积到叶子参数的 `.grad`。它负责求导，不会自动更新参数。

**补充标量损失例子：**

~~~python
import torch

x = torch.tensor([1., 2., 3.])
w = torch.tensor([1., 1., 1.], requires_grad=True)
loss = 0.5 * (x @ w - 5).square()
loss.backward()
assert torch.allclose(w.grad, torch.tensor([1., 2., 3.]))
~~~

此时预测为 6、误差为 1，所以梯度是误差乘输入。梯度值取决于整个计算图，不能只看参数本身。

现在看一层线性变换 $$Y=XW$$，令上游梯度为 $$G=\partial L/\partial Y$$：

$$
\frac{\partial L}{\partial X}=GW^\top,\qquad
\frac{\partial L}{\partial W}=X^\top G.
$$

若 $$X$$ 为 $$[B,d_{\mathrm{in}}]$$、$$W$$ 为 $$[d_{\mathrm{in}},d_{\mathrm{out}}]$$，前向约 $$2Bd_{\mathrm{in}}d_{\mathrm{out}}$$ FLOPs；反向的两个矩阵乘法各有相近计算量。因此对这样的层，反向约为前向的 2 倍，总共约 3 倍。

把主要参数都按每个输入位置使用一次的稠密线性层相加，就得到：

$$
C_{\mathrm{forward}}\approx2ND,\qquad
C_{\mathrm{backward}}\approx4ND,\qquad
C_{\mathrm{train}}\approx6ND.
$$

这里 **$$N$$ 是参数规模，$$D$$ 是整个训练过程中处理的 token 数**，不是上一节每个张量的隐藏维度；对单步，可以用本步 token 数替代 $$D$$。重复训练的 token 也要按处理次数计入。

这个近似有范围：长上下文注意力、MoE 的激活参数与总参数区别、嵌入 / 输出层、重计算、优化器和其他操作，都可能需要单独核算。第一层若不需要输入梯度，其反向也会略少。应先说清估算目标，再用 $$6ND$$ 判断数量级。

**讲义依据：`gradients_basics`、`gradients_flops`、`deep_network`；符号统一与适用边界为补充。**

## 7. 训练显存的四本账 {#memory}

只看模型文件大小，无法判断训练能否放下。训练中的主要内存可以分为：

| 项目 | 保存什么 | 简化估算 |
| --- | --- | --- |
| 参数 | 当前模型权重 | BF16 时 $$2N$$ bytes |
| 梯度 | 对参数的导数 | BF16 时 $$2N$$ bytes |
| 优化器状态 | 更新规则的历史统计量 | 取决于优化器和状态精度 |
| 激活 | 反向需要的中间值 | 取决于层数、批量、长度与结构 |

讲义通过 AdaGrad 解释优化器为何需要状态：它累计历史梯度平方，用于调整各参数的有效步长。对累计量 $$s_t$$，可写为：

$$
s_t=s_{t-1}+g_t^2,\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t+\epsilon}}.
$$

平方、除法均按元素进行；这里按讲义把 $$\epsilon$$ 写在根号内，具体库实现可能采用不同约定。FP32 累计量需要约 $$4N$$ bytes；Adam 则通常有一阶、二阶两个状态，合计约 $$8N$$ bytes。

由此，在 **BF16 参数 + BF16 梯度 + 两份 FP32 Adam 状态，且不保留额外主权重** 的特定假设下，静态主要状态约：

$$
M_{\mathrm{states}}\approx(2+2+4+4)N=12N\ \mathrm{bytes}.
$$

**补充算例：**1B 参数的上述状态约为 12 GB（十进制），还没加激活和临时空间。如果另保留 FP32 主权重，再增加约 4 GB；如果梯度改用 FP32，账也会改变。因此「每参数固定多少字节」必须附带训练配置。

原课还用 8 张 80 GB 显卡的总容量除以 12，得到约 53.3B 参数的理论上限。**补充条件：这要求把相关状态合理分片，并忽略激活、临时缓冲和运行开销。普通数据并行会复制模型状态，不能直接把 8 张卡当成一块 640 GB 显存。**

在讲义的简化深层网络中，若批量为 $$B$$、每层宽度为 $$d$$、层数为 $$L$$，仅按每层一份 BF16 激活估计为 $$2BdL$$ bytes。Transformer 还需要考虑序列轴、注意力中间量等；融合实现可能改变实际保存的内容。

**讲义依据：`optimizer`、`AdaGrad`、`motivating_questions`；混合精度与多卡分片条件为补充。**

## 8. 训练循环与梯度累积 {#loop}

一次普通训练更新的顺序是获取批次、前向算 loss、反向算梯度、优化器更新、清梯度。下面是**补充的结构示意**，其中 `model`、`optimizer`、`batches` 和 `loss_fn` 由具体任务提供：

~~~python
for x, y in batches:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
~~~

PyTorch 的 `.backward()` 默认累积参数梯度；不清梯度就会把不同批次叠加起来。普通训练要避免意外累积，梯度累积则有意利用这一语义。

### 用小批次得到较大的有效批次

假设想让一次更新使用 64 个样本，但显存一次只能容纳 8 个，可以分成 8 个 micro-batch。每次只保留当前 micro-batch 的前向图，反向后释放图，但保留参数梯度，最后更新一次。

**补充伪代码：假设每组正好 K 个等大小 micro-batch，loss_fn 返回每个 micro-batch 的平均 loss。**

~~~python
for micro_batches in update_groups:
    optimizer.zero_grad(set_to_none=True)
    K = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / K
        loss.backward()
    optimizer.step()
~~~

除以 $$K$$ 是为了保持与大批次平均梯度相同的尺度。若有效 token 数不同，应按总有效 token 数加权；不能只平均各 micro-batch 的平均 loss。最后一个不足整组的批次也需要按实际大小处理。

**补充限定：**等价性假设损失可按样本或 token 分解，且参数在整组累积期间保持不变。BatchNorm 等批次相关运算、随机性和浮点求和顺序可能使结果不同；梯度裁剪通常应作用于整组累积后的梯度。多卡还要考虑何时同步梯度。

它主要降低峰值激活内存，不会让模型参数或优化器状态消失；micro-batch 太小也可能降低 GPU 利用率。更大的有效 batch 也不自动意味着模型更好，仍需调参与评估。

**讲义依据：`train_loop`、`gradient_accumulation`；一致的批量算例、归一化和边界为补充。**

## 9. 激活检查点：用重计算换显存 {#checkpointing}

反向需要前向的中间结果。如果全保存，深层网络的激活内存会随层数增长。激活检查点（activation / gradient checkpointing，也叫 rematerialization）只保存部分节点，反向时重新计算缺少的中间值。

~~~text
全部保留：输入 → h1 → h2 → h3 → h4 → h5 → h6
检查点：  输入 ─────────→ h3 ─────────→ h6
反向需要 h4、h5 时，从 h3 再算出这一段
~~~

它不修改训练目标，也不是通过少算梯度节省内存，而是把「长期保存」换成「需要时再算」。因此通常增加计算量；若省下的内存允许更合理的 batch 或更少的切分，整体吞吐变化还要实测。

对层大小相近的简单链式网络，假设每段含 $$k$$ 层，需要保存约 $$L/k$$ 个段边界，反向重算当前段时还需要约 $$k$$ 份临时激活。激活量级可写为：

$$
M_{\mathrm{act}}\propto\frac{L}{k}+k.
$$

在 $$k\approx\sqrt L$$ 时，这个简化模型达到 $$O(\sqrt L)$$ 量级，而原先全存是 $$O(L)$$。它只讨论简化链式网络的激活，并不意味着完整 Transformer 的总显存都会变为平方根增长。

**补充实践边界：**重计算需与原前向在逻辑上相容，随机数与状态变化要正确管理；具体 checkpoint API 的参数和限制以所用 PyTorch 版本为准。这里解释原理，不把教学示意直接当成完整训练代码。

| 手段 | 主要改变什么 | 代价或边界 |
| --- | --- | --- |
| 降低精度 | 单元素存储与计算方式 | 数值稳定性、硬件与 kernel 支持 |
| 梯度累积 | 同时存在的 micro-batch 激活 | 更多串行执行，小批量效率与同步 |
| 激活检查点 | 前向中间值的保留量 | 反向时重新计算 |
| 状态分片 | 每卡持有的状态份额 | 通信、调度与实现复杂度 |

前三项在本讲展开，状态分片属于后续系统学习内容。它们针对不同账目，通常需要组合使用。

**讲义依据：`activation_checkpointing`、`DeepNetworkCheckpointed`；分段推导与实践边界为补充。**

## 10. 估算模板与自测 {#review}

**补充练习：**给一个假设训练任务写一页资源表。先记录 token 数、参数量、batch、长度和 dtype，再分别计算 FLOPs、状态内存、激活内存，最后说明多卡复制 / 分片策略和测量范围。不要把训练状态大小当成推理内存，也不要把估算峰值写成实测吞吐。

| 数量 | 快速估算 | 先检查的假设 |
| --- | --- | --- |
| 张量元素量 | 各维长度乘积 | 是否为稠密独立存储 |
| 张量存储 | 元素量 × 每元素 bytes | dtype、共享 view、额外元数据 |
| 矩阵乘法 | $$2mkn$$ FLOPs | 乘加按 2 次操作 |
| 稠密模型训练 | 约 $$6ND$$ FLOPs | 参数如何被使用、上下文、重计算 |
| Adam 主要状态 | 本页配置为 $$12N$$ bytes | 梯度 / 状态精度、主权重 |
| 算术强度 | FLOPs / 搬运 bytes | 从哪层存储搬、缓存与融合 |
| 时间下界 | 计算时间与搬运时间的较大者 | 需要另计通信、调度等开销 |

<details>
<summary>1. [16, 32, 64] 的 BF16 张量，元素存储量是多少？</summary>
<p>16 × 32 × 64 × 2 = 65536 bytes，即 64 KiB；不含对象、分配器或其他额外开销。</p>
</details>

<details>
<summary>2. [8, 64] @ [64, 32] 约有多少 FLOPs？</summary>
<p>按常用近似为 2 × 8 × 64 × 32 = 32768 FLOPs。若逐项精确计数，乘法加法合计为 8 × 32 × (2 × 64 − 1) = 32512 次。</p>
</details>

<details>
<summary>3. 为什么把算力翻倍，内存受限操作未必明显加速？</summary>
<p>若数据搬运时间占主导而带宽不变，计算单元仍在等数据。应先减少搬运或提高数据复用，而不是只看峰值 FLOP/s。</p>
</details>

<details>
<summary>4. 2B 参数模型，按本页 Adam 假设需要多少主要状态内存？</summary>
<p>12 × 2 × 10⁹ = 24 GB（十进制）。这没有计算激活、临时空间和额外 FP32 主权重，不能据此保证一张 24 GB 显卡能训练它。</p>
</details>

<details>
<summary>5. 8 次 backward 后才 step，一定等于一个 8 倍大 batch 吗？</summary>
<p>需要正确归一化，累积期间不更新参数，并考虑有效 token 数、批次相关运算与随机性。没有除以 8 时，等大小 micro-batch 的平均 loss 梯度会放大 8 倍。</p>
</details>

<details>
<summary>6. 梯度累积与激活检查点各省哪部分内存？</summary>
<p>梯度累积减少一次前向反向同时处理的数据量；激活检查点减少长时间保留的中间结果。两者主要针对激活，都不会直接消除参数和优化器状态。</p>
</details>

原始材料：[Lecture 2 常规可运行讲义](https://cs336.stanford.edu/lectures/?trace=lecture_02)、[官网另附的 recording version](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)、[2026 版课程目录](https://cs336.stanford.edu/#schedule)。继续阅读第 3 讲，了解现代 Transformer 的架构与超参数取舍。
