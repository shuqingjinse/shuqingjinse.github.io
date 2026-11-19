# Straight 的个人知识库

这是基于 GitHub Pages + Jekyll 的个人知识博客。

## 发布

首次使用时，在 GitHub 创建公开仓库 `shuqingjinse.github.io`，然后执行：

```bash
git push -u origin main
```

随后在仓库的 `Settings -> Pages` 中选择 `Deploy from a branch`、`main`、`/(root)`。

## 写文章

在 `_posts/` 新建 `YYYY-MM-DD-title.md`，文件头示例：

```yaml
---
layout: post
title: "文章标题"
categories: [AI]
tags: [实践]
---
```

文章支持标准 Markdown、代码块、表格和 MathJax 公式。提交后 GitHub Pages 会自动构建。

## 课程笔记

`/course-notes/` 按课程展示入口，每门课有独立的讲义目录与课程官网链接。课程信息及讲义索引统一维护在 `_data/courses.yml`，由 `_layouts/course.html` 渲染课程页。

- CS2680：`/course-notes/cs2680/`，现有 6 篇笔记，讲义原地址保持可用。
- CS146S：`/course-notes/cs146s/`，The Modern Software Developer，现有 2 篇笔记。
- CS193T：`/course-notes/cs193t/`，Thinking with AI，对应官方归档中公开的 8 讲，一讲一篇；各篇附课件链接、页码、应用练习和自测。第 4 讲说明嘉宾课公开材料的范围。
- Post-training of LLMs：`/course-notes/post-training/`，7 讲完整中文笔记，涵盖后训练全景、SFT、DPO、PPO / GRPO 的原理与实践，附来源、自测及全课选型表。

后训练笔记与 CS193T 共用 `_layouts/course-note.html`，通过 `course_id` 和 `lecture` 关联目录，自动生成上一讲 / 下一讲；`sections` 维护页内目录。课程数据的 `instructor`、`note_label` 提供讲师与页眉，单篇可覆盖 `instructor`；`lecture_source`、`slides_source` 提供来源链接。Markdown 公式使用 Kramdown 的双美元分隔符（行内与独立公式均支持），代码中出现 Jinja / Liquid 模板时需要用 Liquid `raw` 块保护。新增笔记后应检查 Jekyll 构建、公式显示、页内锚点与手机排版。

新增课程时，先在 `_data/courses.yml` 添加课程信息（`id`、`code`、`school`、`title`、`english_title`、`description`、`url`、`source_url`、`source_domain`、`lessons`），再创建课程页面，例如：

```yaml
---
layout: course
title: "CS146S · 现代软件开发者"
permalink: /course-notes/cs146s/
course_id: cs146s
---
```

新增讲义时，在课程的 `lessons` 中登记 `number`、`title`、`url`、`description`，并添加对应正文页面；笔记数量会自动更新。尚无讲义的课程使用 `lessons: []`。

## 小游戏栏目

首页「探索内容」与主导航提供「小游戏」入口，栏目地址为 `/games/`。
《烽火破城》位于 `/games/fire-siege/`，使用独立页面显示完整战场，支持键盘、触屏、返回栏目和浏览器全屏。当前为俯视二维召唤版，WASD 移动主将，1—4 召唤投弹组、投石组、骑兵队与火枪组；97 名初始守军及最多三批增援，破门后斩将夺旗。
游戏脚本、样式和合成音效全部内嵌，不需要后端或额外服务；最高分保存在当前浏览器。

更新游戏时，先在游戏项目运行 `npm run build`，再从本仓库执行：

```sh
node scripts/sync-fire-siege.mjs '/path/to/烽火破城.html'
```

同步脚本只给构建产物添加网站返回链接和全屏按钮。游戏预览图为 `assets/images/games/fire-siege.webp`；栏目介绍与按键说明在 `games/index.html`。

### 手机同款

手机专用入口为 `/games/fire-siege/mobile/`。栏目提供电脑版与手机版两个按钮，游戏开场和操作说明也可切换版本。手机版采用纵向战场：我军大营在下方，城门和帅旗在上方，摇杆向上推进。使用双指摇杆与蓄力、大按钮召兵、纵向小地图和触屏军令，适配安全区域；同一浏览器中的最高分与电脑版共用。

更新手机版构建后同步：

```sh
node scripts/sync-fire-siege.mjs '/path/to/烽火破城手机版.html' --mobile
```

手机版从 `../../` 返回小游戏栏目，从 `../` 切换电脑版，均为相对链接，可在站点子路径下工作。

### 首页与栏目维护

- 首页“最近更新”自动合并博客文章与 `_data/updates.yml` 中的课程、游戏动态，按日期排列。新增讲义或作品后，在该文件登记标题、链接、分类和完整发布时间。
- 文章的 `description` 用作列表摘要；一两句话说明文章解决的问题即可。首页推荐阅读取带有 `featured: true` 的文章，目前为 PPO 图解与交互实验室。
- 栏目颜色延续为课程蓝、总结绿、面试金、游戏紫。手机导航将关于和 RSS 收进“更多”，手机首页使用紧凑双列栏目，并优先展示更新。
- 游戏栏目提供电脑／手机实机预览切换，手机默认显示竖版预览并优先提供手机版入口。截图资源为 `assets/images/games/fire-siege-mobile.webp`。


### 手机版三关战役与兵力平衡

手机版入口 `/games/fire-siege/mobile/` 已升级为元帅战役：五兵种含佛朗机炮，两名将军拥有冲锋／恢复鼓舞技能。三关依次为玄武关、虎牢关、云内州，胜利解锁并保存在当前浏览器。守军开局 141／181／221 人，另有分批增援；护城河、桥梁、瓮城和内外双门实际影响行军与胜负。详细规则见游戏内帮助。

游戏源项目位于相邻 `古代战争` 目录，新增 `campaign-config.ts`、`campaign.ts` 与 `tests/campaign.test.ts`。发布前运行规则测试、手机完整三关触控验收和独立构建，再通过同步脚本仅更新手机版入口。

手机版 6.1：攻方总兵员 180／210／240、同时在场上限 80／90／100；炮组按 3 名兵员计算，阵亡不返还。将军每关最多返营一次并消耗兵员。守军三关配齐投弹兵、佛朗机炮、骑兵，增援有限且提前预警；敌炮有可躲避的瞄准线和落点圈。火枪克制骑兵，骑兵克制炮组。26 项规则测试、15 场平衡对照和三关触控验收覆盖兵员、路径、火力与关卡解锁。
