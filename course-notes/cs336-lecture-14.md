---
layout: course-note
title: CS336 第 14 讲 · 数据处理：过滤、去重、混合与合成
description: 掌握正文抽取、质量分类、Jaccard、MinHash 与 LSH，并用可验证实验选择混合比例和合成数据。
permalink: /course-notes/cs336-lecture-14/
course_id: cs336
lecture: 14
instructor: Percy Liang
original_lessons: 'Spring 2026 · Lecture 14 · Data: filtering, deduplication, mixing, synthetic data · 2026-05-13'
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_14
sections:
- id: transform
  title: 处理链条的顺序会改变结果
- id: filter
  title: 过滤是在定义什么叫好数据
- id: dedup
  title: 精确去重与近重复
- id: lsh
  title: MinHash 与 LSH 怎样避免两两比较
- id: mix
  title: 混合比例与合成数据
- id: review
  title: 练习与自测
---
## 1. 处理链条的顺序会改变结果 {#transform}

HTML、PDF、代码目录先转成适合训练的文本，再过滤、去重、混合。抽取阶段若把菜单当正文或打乱公式，后面的质量模型和去重都会在错误输入上工作。

转换不是无损操作：表格、布局、图片说明与代码缩进可能承载重要信息。应为主要来源分别抽样检查，并保存处理版本，便于发现指标变化究竟来自抽取还是筛选。

**讲义定位：`transformation`。**下文算例和实验要求为补充。

## 2. 过滤是在定义什么叫好数据 {#filter}

用目标样本 T 和原始池 R 学一个分数，再从 R 选出更像 T 的内容。语言识别、质量、领域和有害内容分类都可纳入此框架，但它们的目标不同。

生成模型可用目标分布概率打分，分类器可估计属于目标类的概率。分类器高分通常只表示接近训练标签，并非事实正确性的证书。阈值越严格，保留率、覆盖面和误删成本都可能变化。

补充实践：为每个阈值记录保留 token、人工精度、稀有领域覆盖和小模型验证结果。不要只优化某一个综合分数；过滤器自身的训练数据与模型来源也要记录。

**讲义定位：`filtering`。**

## 3. 精确去重与近重复 {#dedup}

精确去重可对规范化文本计算哈希，再验证候选。规范化规则决定哪些差异被忽略；整篇、段落和句子级别删除的语义后果也不同。从文档中间删掉若干段，可能破坏上下文。

近重复常把文档转为 token n-gram 集合，使用 Jaccard 相似度：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

补充算例：$$A=\{a,b,c\}$$、$$B=\{b,c,d\}$$，相似度为 2/4=0.5。集合表示忽略出现次数，也只保留所选 n-gram 粒度的信息，因此选择表示和阈值都属于算法设计。

**讲义定位：`deduplication`、`hash_functions`、`exact_deduplication`、`jaccard_minhash`。**

## 4. MinHash 与 LSH 怎样避免两两比较 {#lsh}

理想的随机排列 MinHash 满足：两个集合的最小哈希一致的概率等于它们的 Jaccard。多个哈希构成签名，匹配比例估计相似度；单次碰撞并不能断言超过阈值。

LSH 将签名分成 b 个 band，每个 band 有 r 项；同一 band 内全部匹配时生成候选。在独立近似下，相似度 s 的候选概率为：

$$
P(\text{候选}\mid s)=1-(1-s^r)^b.
$$

它是一条概率曲线，不是绝对阈值。候选还应按选定标准核验；需要关注漏检、误匹配与聚类后的保留规则。MinHash 在有限样本下也有统计波动，不能把某次小实验误差当成必然上界。

**讲义定位：`jaccard_minhash`、`locality_sensitive_hashing`。**

## 5. 混合比例与合成数据 {#mix}

按文档数均匀、按 token 比例、手动提高高质量来源权重，得到的是不同分布。若来源 s 的有效唯一 token 数为 $$U_s$$，采样概率为 $$p_s$$，总处理 D token，则预期使用量为 $$Dp_s$$，近似重复轮数为 $$Dp_s/U_s$$。

提高小来源的比例可能很快进入多轮重复。应在小规模上做配比实验，再检验扩大模型后是否仍有效。合成数据可补充任务、格式与难度，但需要验证器、去污染与多样性控制；老师模型输出并不自动正确。

补充实验：保持总 token 和模型不变，比较 3 种混合比例，并在通用、代码与目标领域分别评估。对合成样本保存生成条件与验收结果，保留无法验证和被拒绝的比例。

**讲义定位：`data_mixing`、`post_training_data`。**

## 6. 练习与自测 {#review}

<details><summary>LSH 找到的候选都应直接删除吗？</summary><p>不应。候选只提高了相似可能性，需要进一步核验和制定保留规则，避免误删并破坏文档完整性。</p></details>
<details><summary>一个 1B token 来源在 20B token 训练中占 20%，意味着什么？</summary><p>预期处理该来源 4B token，粗略相当于 4 轮。不能当成看到了 4B 独立新 token。</p></details>
<details><summary>为什么合成数据要独立验收？</summary><p>生成器会犯错，也可能复制评测题或固定风格。独立检查能减少错误放大、污染与分布收缩。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_14) · [官方视频](https://www.youtube.com/watch?v=5sxHosTLPF8)。整理核对日期：2026-11-20。
