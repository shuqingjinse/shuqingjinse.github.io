---
layout: course-note
title: CS336 第 6 讲 · Kernel 与 Triton：从正确算子到高效实现
description: 学习基准测试、性能分析、Triton 的分块思维，以及融合 GELU、softmax、行归约和矩阵乘法。
permalink: /course-notes/cs336-lecture-06/
course_id: cs336
lecture: 6
instructor: Percy Liang
original_lessons: Spring 2026 · Lecture 6 · Kernels, Triton · 2026-04-15
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_06
sections:
- id: measure
  title: 优化前先建立证据
- id: fusion
  title: 为什么一行 PyTorch 可能启动多个 kernel
- id: triton
  title: Triton 用一块数据描述工作
- id: reduction
  title: softmax 与跨块归约
- id: matmul
  title: 分块 matmul 与融合输出
- id: review
  title: 练习与自测
---
## 1. 优化前先建立证据 {#measure}

benchmark 回答一次完整操作花多久；profiler 回答时间花在哪里。二者都需要，不能看到某个热点就假定修改后端到端一定更快。

GPU 调用通常异步，CPU 发射完任务不等于设备执行完。稳态测试应先预热，排除编译和首次分配，再用 CUDA events 或在计时边界同步。还要固定形状、精度、布局和硬件，重复测量并报告波动。

补充的最小记录表：实现名、输入 shape / stride、dtype、容差、预热次数、重复次数、耗时、峰值显存。先验证正确性，再比较速度；分别记录编译耗时和稳态耗时。

**讲义定位：`benchmarking`、`benchmark`、`profiling`、`profile`。**

## 2. 为什么一行 PyTorch 可能启动多个 kernel {#fusion}

Python 表达式里的加法、乘法、tanh 等操作，可能分别读取和写入大张量。GELU 的组合公式如果逐项执行，会产生临时量；内置实现或编译器可以融合成更少的 kernel。

真正需要算的是数据流：输入被读取几次，哪些中间量写回 HBM，是否存在额外启动与同步。不能用 Python 行数判断设备执行次数，也不能把某次编译器成功融合推广到所有形状和运算。

**讲义定位：`naive_vs_builtin_vs_compiled_gelu`、`naive_gelu`、`builtin_gelu`。**

## 3. Triton 用一块数据描述工作 {#triton}

Triton 常用 program instance 处理一块数据，再由编译器映射到底层执行。`program_id` 定位块，`arange` 建立块内偏移，mask 保护边界。以下为补充伪代码，展示向量 kernel 的骨架，不是完整可运行程序：

~~~python
pid = program_id(0)
offsets = pid * BLOCK + arange(0, BLOCK)
valid = offsets < n
x = load(x_ptr + offsets, mask=valid, other=0)
y = f(x)
store(y_ptr + offsets, y, mask=valid)
~~~

若 $$n$$ 不是 BLOCK 的倍数，最后一块会包含越界位置；load 和 store 都必须正确处理。带 stride 的二维数据还要使用真实行列步长，不能默认所有输入连续。查看生成 PTX 能帮助理解全局读写和线程映射，但代码更短不等于寄存器占用更少。

**讲义定位：`triton_introduction`、`triton_gelu_example`、`triton_gelu_kernel`。**

## 4. softmax 与跨块归约 {#reduction}

对一行 $$x$$，数值稳定的 softmax 为：

$$
m=\max_i x_i,\qquad p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

若整行适合一个 program，可以在片上完成最大值、指数、求和与归一化，减少中间流量。最大值归约的无效位置应使用负无穷，求和归约通常使用零；不能不加区分地为所有 padding 填零。

当行太长，需要分块累积。普通行求和可累加各 tile 的和；softmax 还需像第 5 讲那样维护最大值与缩放后的和。程序边界、同步范围与中间状态都是算法的一部分。

**讲义定位：`triton_softmax_example`、`naive_softmax`、`triton_softmax_kernel`、`triton_row_sum_example`。**

## 5. 分块 matmul 与融合输出 {#matmul}

矩阵乘法 program 通常负责一个输出 tile，循环读取 A 的行块和 B 的列块，把乘积累积到寄存器中的输出块。最终可以在写回前融合 ReLU 等逐元素操作。

| 参数 | 增大可能带来的好处 | 可能的代价 |
| --- | --- | --- |
| 输出 tile | 提高数据复用 | 更多累加器、较低并发 |
| 归约 tile | 减少循环调度 | 片上资源压力 |
| program 数量 | 提供更多并行工作 | 单块工作变小、复用降低 |
| 融合范围 | 少启动、少临时量 | 寄存器压力与编译限制 |

应覆盖矩形、非对齐维度与不同 stride，不能只对一个方阵宣布实现优于库。性能优化也受 Amdahl 定律约束：若目标算子只占 10% 时间，即使它无限快，总加速也最多约 1.11 倍。

**讲义定位：`triton_matmul_relu_example`、`matmul_relu_kernel`；Amdahl 算例为补充。**

## 6. 练习与自测 {#review}

补充练习：先做 1024、1025 两种长度的向量算子正确性检查，再测多个尺寸。输出中记录绝对误差、相对误差以及是否存在 NaN / Inf。

<details><summary>为什么要把 1025 也作为检查尺寸？</summary><p>它能暴露最后一块的边界问题。只测整除 BLOCK 的长度，可能漏掉越界读取和写入。</p></details>
<details><summary>为什么不能直接用 CPU 时间包围一次 CUDA 调用？</summary><p>它可能主要测到异步任务发射；需用设备事件或显式同步，并把编译与预热单独处理。</p></details>
<details><summary>融合越多越好吗？</summary><p>不一定。融合可能增加寄存器与片上内存压力，限制并发。需要在真实工作负载上测端到端收益。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_06) · [官方视频](https://www.youtube.com/watch?v=xnDHaNUvHBg)。整理核对日期：2026-11-20。
