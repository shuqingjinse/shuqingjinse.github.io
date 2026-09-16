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
