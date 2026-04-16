# 文章封面图片目录

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
