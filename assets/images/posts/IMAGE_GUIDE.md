# 📸 文章封面图使用指南

## 已配置的文章

以下文章已添加封面图配置，需要你添加对应的图片文件：

### 🤖 机器人抓取
- `robotics/pi0-cover.jpg` - π₀ VLA 模型文章封面

### 👁️ 3D视觉  
- `3d-vision/3dgs-cover.jpg` - 3D Gaussian Splatting 文章封面

### 🧠 大模型应用
- `llm/transformer-cover.jpg` - Transformer 文章封面

## 🎨 获取封面图的方法

### 1. 从论文中提取
- 使用论文的架构图、示意图
- 截取关键的可视化结果
- 建议使用 Figma/Photoshop 添加标题和美化

### 2. 使用 AI 生成
- **Midjourney**: `/imagine robotics grasping, technical illustration, clean background`
- **DALL-E 3**: "A technical illustration of transformer architecture"
- **Stable Diffusion**: 使用 technical diagram 风格

### 3. 免费图库资源
- **Unsplash**: https://unsplash.com (高质量免费图片)
- **Pexels**: https://pexels.com (科技类图片)
- **Pixabay**: https://pixabay.com

### 4. 自己设计
使用 Canva 模板：
1. 访问 https://canva.com
2. 搜索 "Blog Banner" 或 "Social Media Post"
3. 尺寸设置为 1200x630px
4. 添加渐变背景 + 文章标题 + 图标

## 📐 图片规范

```yaml
尺寸: 1200x630px (推荐) 或 1920x1080px
格式: JPG (照片) 或 PNG (图表)
大小: < 500KB (使用 TinyPNG 压缩)
命名: 小写字母+连字符，如 transformer-cover.jpg
```

## 💡 设计建议

### 配色方案
- **机器人**: 蓝色系 (#2563EB, #3B82F6)
- **3D视觉**: 紫色系 (#7C3AED, #A855F7)  
- **大模型**: 绿色系 (#059669, #10B981)
- **论文笔记**: 橙色系 (#EA580C, #F97316)

### 构图建议
1. **简洁为主**: 避免过于复杂的背景
2. **突出主题**: 使用图标或关键词
3. **保持一致**: 同系列文章使用相似风格
4. **可读性**: 确保标题清晰可读

## 🔧 图片压缩工具

- **在线**: https://tinypng.com
- **命令行**: `npm install -g imagemin-cli`
- **批量处理**: 使用 Photoshop 批处理功能

## 示例代码

在文章 frontmatter 中添加：

```yaml
# 简单模式
image: /assets/images/posts/robotics/pi0-cover.jpg

# 完整模式（带 alt 文本，利于 SEO）
image:
  path: /assets/images/posts/robotics/pi0-cover.jpg
  alt: π₀ VLA 模型架构示意图
  
# 带 lqip（低质量图片占位符，可选）
image:
  path: /assets/images/posts/robotics/pi0-cover.jpg
  alt: π₀ VLA 模型架构示意图
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSAwAAAARBxAR/Q9ERP8DAABWUDggGAAAABQBAJ0BKhAACAAFADQlpAADcAD++/1QAA==
```

## 📝 待添加封面的其他文章

你可以继续为以下文章添加封面：
- 2025-12-14-roboLLM.md (Robot+LLM 综述)
- 2026-04-01-VGGT.md (VGGT 论文)
- 2026-03-26-priorDA.md (深度估计)
- 2026-04-01-flowmatching.md (流匹配)
- 其他文章...

按照相同的方式在 frontmatter 中添加 `image` 字段即可。
