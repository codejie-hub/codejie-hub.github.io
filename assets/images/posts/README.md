# 文章原始图片与预览

当前主分类来自 `_data/domains.yml`。本目录保留既有封面、论文原图及手写存档；新增专属封面在 `assets/images/covers/`，可编辑教学 SVG 在 `assets/images/figures/`，演示实际截图在 `assets/images/demos/`。

视觉伺服原图的轻量预览位于 `visual-servoing/previews/`；原图按需打开，保留原始文件。正文统一使用 `_includes/figure.html`，不要将正文图固定高度裁切。完整清单见 `reports/blog-organization.md`。

以下为原有封面目录与 front matter 用法，路径仍然有效。

## 目录结构

```
posts/
├── robotics/        # 机器人相关
├── llm/            # 大模型相关
├── 3d-vision/      # 3D视觉相关
└── general/        # 通用图片
```

## 图片规范

- **尺寸建议**: 1200x630px (适合社交媒体分享)
- **格式**: JPG/PNG
- **命名**: 使用文章日期+主题，如 `2026-04-02-pi0.jpg`
- **大小**: 建议 < 500KB

## 使用方法

在文章 frontmatter 中添加：

```yaml
image:
  path: /assets/images/posts/robotics/2026-04-02-pi0.jpg
  alt: π₀ VLA 模型架构图
```

或简写：

```yaml
image: /assets/images/posts/robotics/2026-04-02-pi0.jpg
```
