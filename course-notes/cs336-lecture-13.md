---
layout: course-note
title: CS336 第 13 讲 · 预训练数据：从原始来源到可用语料
description: 梳理网页、百科、代码、论文与公开语料，区分实时服务、数据快照和处理后数据，并维护来源记录。
permalink: /course-notes/cs336-lecture-13/
course_id: cs336
lecture: 13
instructor: Percy Liang
original_lessons: 'Spring 2026 · Lecture 13 · Data: sources, datasets · 2026-05-11'
source_note: 依据官网当前公开课件归纳，按页码或函数名定位来源；公式展开、算例、实验模板与自测为补充解释，不是逐字课堂实录。
lecture_source: https://cs336.stanford.edu/#schedule
slides_source: https://cs336.stanford.edu/lectures/?trace=lecture_13
sections:
- id: pipeline
  title: 数据不是从互联网上直接落进模型
- id: sources
  title: 几种来源各有什么价值和偏差
- id: datasets
  title: 数据集名字背后是一套处理选择
- id: provenance
  title: 来源信息为什么要跟着文档走
- id: audit
  title: 如何做一个小型数据盘点
- id: review
  title: 练习与自测
---
## 1. 数据不是从互联网上直接落进模型 {#pipeline}

一个真实数据链条通常是「在线服务 → 抓取或导出快照 → 提取与清理 → 可采样文档 → token」。每个步骤都改变分布：爬虫能访问什么、什么时候下载、是否支持动态页面、抽取器保留哪些内容，都会影响模型最后学到什么。

预训练提供大规模一般文本；中期训练通常提高特定能力或数据质量；后训练更直接塑造交互和决策。边界并不绝对，但应区分「原始来源多大」「清理后多大」「实际训练处理多少 token」。

**讲义定位：`motivation`、`raw_sources`。**下文数据卡和练习为补充。

## 2. 几种来源各有什么价值和偏差 {#sources}

| 来源 | 可能的价值 | 常见处理难点 |
| --- | --- | --- |
| Common Crawl 网页 | 覆盖广、多样 | 垃圾、模板、重复、抓取偏差 |
| Wikipedia | 结构化知识与相对统一风格 | 版本、语言覆盖与编辑分布 |
| GitHub 等代码 | 程序结构、API 与开发文本 | fork、自动生成文件、许可证与密钥 |
| arXiv 等论文 | 科学写作、数学与专业内容 | PDF / TeX 提取、公式、重复版本 |
| 图书与问答 | 长文、解释、领域讨论 | 获取范围、授权、格式与质量差异 |

原始网页快照不是已经清洗好的自然语言。代码仓库也不是每个文件都适合训练：依赖、压缩文件、大量重复的 vendored 代码可能主导 token 数。

**讲义定位：`common_crawl`、`wikipedia`、`github`、`arxiv`。**

## 3. 数据集名字背后是一套处理选择 {#datasets}

早期模型常组合百科、图书、网页等来源；之后的公开项目分别尝试规则过滤网页、多来源混合、质量分类器和更透明的数据发布。

| 讲义中的实例 | 适合带着什么问题读 |
| --- | --- |
| WebText、C4、CCNet | 网页如何被发现和筛选？ |
| The Pile、Dolma | 多来源怎么组合与记录？ |
| RefinedWeb、DCLM | 大规模网页清理和质量选择如何比较？ |
| The Stack | 代码去重与来源如何处理？ |
| Common Pile | 明确来源与许可的语料怎样构建？ |

这些不是可直接互换的「同一种网页数据」。应查看每个版本的抽取、语言识别、去重和采样规则。历史论文中的数量和配比也不能作为现时下载版本的保证。

**讲义定位：`bert` 至 `common_pile` 的数据案例。**

## 4. 来源信息为什么要跟着文档走 {#provenance}

数据工程需要把来源、快照日期、语言、处理版本、保留理由和删除标记与文档关联。公开可访问不自动说明适合任意使用；不同来源的授权、服务规则和隐私情况应依据原始资料核对，本页不作法律结论。

补充的数据记录示意：

~~~json
{"doc_id":"sample-001","source":"example-corpus","snapshot":"2026-04","language":"zh","pipeline":"extract-v2","license_record":"source-card","content_hash":"..."}
~~~

这能帮助定位质量问题、更新语料、重建实验和按来源分析模型表现。训练只留下 token、丢掉所有来源，后续很难解释哪一批数据带来了退化。

**讲义定位：`raw_sources`、`copyright`；记录格式为补充。**

## 5. 如何做一个小型数据盘点 {#audit}

先定义目标能力，再抽样查看原文，不要只看总 token 数。按来源统计长度、语言、重复率、解析失败和异常内容；对少数重要领域保留独立验证集。

补充方案：从三类来源各取 100 篇，人工检查每类 20 篇，记录模板占比、乱码、缺失标题、公式破坏和代码完整性。发现大量导航文字时，先修抽取；发现同一内容大量复制时，再处理去重。

数量、质量和多样性有取舍。强过滤可能提高某个验证指标，也可能删掉少数语言、口语或稀有领域。应通过分领域评估判断，不能把质量分类分数当成普适真理。

## 6. 练习与自测 {#review}

<details><summary>为什么训练 token 数可能大于唯一语料 token 数？</summary><p>因为数据可能重复采样或训练多轮。计算成本统计处理次数，数据覆盖则需另记唯一内容。</p></details>
<details><summary>两个都来自 Common Crawl 的数据集为何可能表现不同？</summary><p>快照、正文抽取、语言过滤、质量筛选、去重和采样都可能不同，原始来源相同不等于训练分布相同。</p></details>
<details><summary>为什么代码数据要按仓库或近重复组划分评估？</summary><p>fork 和复制代码可能跨训练与验证集泄漏，使测试表现被重复内容抬高。划分应考虑真实关联。</p></details>

原始材料：[官方讲次目录](https://cs336.stanford.edu/#schedule) · [原始课件或录像](https://cs336.stanford.edu/lectures/?trace=lecture_13) · [官方视频](https://www.youtube.com/watch?v=-qm0ln33G24)。整理核对日期：2026-11-20。
