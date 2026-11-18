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
- CS146S：`/course-notes/cs146s/`，The Modern Software Developer，暂无笔记时显示「笔记待更新」。

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

手机专用入口为 `/games/fire-siege/mobile/`。栏目提供电脑版与手机版两个按钮，游戏开场和操作说明也可切换版本。手机用双指摇杆与蓄力、大按钮召兵、小地图和触屏军令，横竖屏适配安全区域；同一浏览器中的最高分与电脑版共用。

更新手机版构建后同步：

```sh
node scripts/sync-fire-siege.mjs '/path/to/烽火破城手机版.html' --mobile
```

手机版从 `../../` 返回小游戏栏目，从 `../` 切换电脑版，均为相对链接，可在站点子路径下工作。
