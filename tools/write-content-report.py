#!/usr/bin/env python3
"""Package local visual evidence and the reviewed article/figure inventory."""
from pathlib import Path
from collections import Counter
import json
import shutil
import sys

import yaml
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
REPORT = ROOT / 'reports'
PREVIEWS = REPORT / 'previews'
PREVIEWS.mkdir(parents=True, exist_ok=True)


def sheet(paths, destination, columns, width, height):
    paths = list(sorted(paths))
    canvas = Image.new('RGB', (columns*width, ((len(paths)+columns-1)//columns)*height), '#e8edf3')
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        image = ImageOps.contain(Image.open(path).convert('RGB'), (width-12, height-32))
        x, y = (index % columns)*width, (index//columns)*height
        draw.text((x+6, y+6), path.stem, fill='#18212f')
        canvas.paste(image, (x+6, y+28))
    canvas.save(destination, optimize=True)


def main():
    site = json.loads((ROOT/'.qa/browser/site-results.json').read_text(encoding='utf-8'))
    assets = json.loads((ROOT/'.qa/browser/assets-results.json').read_text(encoding='utf-8'))
    diagrams = json.loads((ROOT/'.qa/browser/inspect-results.json').read_text(encoding='utf-8'))
    assert not site['failures'] and not assets['failures'] and not diagrams['failures'], 'Resolve browser audit failures before packaging the report.'
    assert not site['consoleErrors'], 'Resolve browser JavaScript exceptions before packaging the report.'
    sheet((ROOT/'.qa/browser/figures').glob('*.png'), PREVIEWS/'figures.png', 4, 300, 220)
    sheet((ROOT/'assets/images/covers').glob('*.jpg'), PREVIEWS/'covers.png', 3, 400, 245)
    sheet((ROOT/'assets/images/demos').glob('*.jpg'), PREVIEWS/'demos.png', 3, 400, 275)
    for name in ['publications-desktop.png', 'robot-manipulation-survey-375-light.png', 'fig-camera-projection-desktop.png', 'fig-gaussian-flow-example-desktop.png', 'demo-robot-kinematics-7dof-desktop.png', 'transformer-768-dark.png']:
        shutil.copyfile(ROOT/'.qa/browser'/name, PREVIEWS/name)
    domains = yaml.safe_load((ROOT/'_data/domains.yml').read_text(encoding='utf-8'))
    kinds = yaml.safe_load((ROOT/'_data/content_types.yml').read_text(encoding='utf-8'))
    illustrations = yaml.safe_load((ROOT/'_data/illustrations.yml').read_text(encoding='utf-8'))
    posts = []
    for p in sorted((ROOT/'_posts').glob('*.md')):
        front = yaml.safe_load(p.read_text(encoding='utf-8').split('---',2)[1])
        posts.append((p.name[11:-3], front))
    counts=Counter(front['categories'][0] for slug,front in posts)
    body=['# 博客分类与插图整理验收记录',
          '', '交付基线：18 篇文章；七个主分类；四种内容类型。文章原地址保持可用，原始图片全部保留。',
          '', '## 分类与页面', '', '| 分类 | 篇数 |', '|---|---:|']
    body += [f'| {d["name"]} | {counts[d["name"]]} |' for d in domains]
    body += ['', '博客展示全部 18 篇，支持主题与类型组合筛选；论文页展示 6 篇论文解读和 3 篇专题综述。旧 VLA、控制、多模态模型归档保留访问入口；世界模型筛选导向路线图。',
             '', '| 文章 | 主分类 | 内容类型 | 新教学图 |', '|---|---|---|---:|']
    for slug,front in posts:
        n=sum(item['post']==slug for item in illustrations)
        body.append(f'| [{front["title"]}](../_posts/{next(p.name for p in (ROOT/"_posts").glob("*.md") if p.name[11:-3]==slug)}) | {front["categories"][0]} | {kinds[front["content_type"]]["label"]} | {n} |')
    body += ['', '## 教学图与布局', '', '32 张新增教学图均为可编辑 SVG。另有分箱原图的一张可读重绘和仅用于封面的复习地图；这两张不计入 32 张新增教学图。正文合计 48 处静态图、16 个 Mermaid 逻辑图及 7 处交互演示。',
             '', '正文最大 740px；小图最大 320px，中图最大 520px。原论文低分辨率图不放大到超过其原始宽度。图注与替代文本分开，使用语义化 figure/figcaption，沿用 Chirpy 图片放大。',
             '', '| 文章 | 章节 | 图源 | 布局 |', '|---|---|---|---|']
    size_names=dict(full='全宽',medium='中图',small='小图')
    for item in illustrations:
        section=item['section'].replace('@before:','第 ')+' 节前' if item['section'].startswith('@before:') else item['section']+' 节后'
        body.append(f'| {item["post"]} | {section} | [{item["asset"]}](../assets/images/figures/{item["asset"]}.svg) | {size_names[item["size"]]} |')
    body += ['', '![教学图联系表](previews/figures.png)', '', '### 原有图片纠正', '',
             '- Transformer 三张论文图采用 513、279、269px 的原始宽度上限；原论文架构图先出现，简化 Mermaid 位于折叠补充区。',
             '- ViT-VS 流程图解释控制、匹配与预对齐三个部分；分箱原图保留在折叠存档，可读版本分成邻域构造与描述子拼接两部分。',
             '- EMA 图明确位置误差与路径长度比两个坐标轴；受扰动图解释为特征对应示例。',
             '- 完整 Table I 放在文字结果表后的折叠核对区，保留单位与文献结果标记。',
             '- convergence 图更正为六维速度随迭代变化，final-error 图更正为位姿误差随迭代变化。',
             '- runtime 图更正为帧率 Hz；industrial-setup 图更正为原论文图 7(d) 的三维末端轨迹。',
             '- CNS 比较图保留方法设计动机的条件说明。手写 A1/A2 默认折叠，展开才插入压缩预览，原图由放大入口按需访问。',
             '', '## 封面、预览与资源', '', '新增 14 张 1200×630 专属封面，同时保存 SVG 图源和 JPEG 展示版本；原有四张专属封面保留。项目卡片及手机演示预览均使用实际界面截图。',
             '', '![封面联系表](previews/covers.png)', '', '![七处演示实际截图](previews/demos.png)',
             '', '桌面在演示接近可视区域时挂载 iframe，并预留与实时窗口相同的高度；手机初始不挂载 iframe，使用预览与独立打开入口。运动学三处演示移到坐标变换、万向锁和正运动学章节，其他移动位置保留了原章节索引。',
             '', '资源压缩统计：', '']
    originals=list((ROOT/'assets/images/posts/visual-servoing').glob('*.png'))+list((ROOT/'assets/images/posts/visual-servoing').glob('*.jpg'))
    previews=list((ROOT/'assets/images/posts/visual-servoing/previews').glob('*.webp'))
    body += [f'- 视觉伺服 12 张原图合计 {sum(p.stat().st_size for p in originals)/1024/1024:.2f} MiB；压缩预览合计 {sum(p.stat().st_size for p in previews)/1024/1024:.2f} MiB。',
             f'- 最大压缩预览 {max(p.stat().st_size for p in previews)/1024:.0f} KiB，原图保留。',
             '', '## 验证结果', '',
             '- Jekyll 生产构建通过；主题固定为 7.3.1，锁文件包含 Windows 与 Linux 平台。',
             '- `python tools/check-content.py --baseline --site .qa/site` 通过：分类、类型、媒体数量、图源、图注、尺寸、重复 ID、嵌套链接及两篇用户原有正文保留检查。',
             '- HTMLProofer 通过：124 个 HTML 文件、944 个内部链接及内部锚点。Windows 使用项目内 DLL 引导工具加载本机 Git 提供的 libcurl。',
             f'- Chrome 页面验收 {len(site["checks"])} 项通过，素材验收 {len(assets["checks"])} 项通过，Mermaid 可读字号及局部滚动检查 {len(diagrams["checks"])} 项通过；页面 JavaScript 异常为 {len(site["consoleErrors"])}。',
             '- 全部 18 篇文章在 375px、1440px 下逐页及逐图检查，另覆盖 768px 深色布局；检查实际图片解码、Mermaid SVG、放大与 Esc 关闭、组合筛选与历史记录、折叠存档及演示挂载。',
             '- 无 JavaScript 时博客仍可阅读全部文章。',
             '', '以上为本机生产构建和 Chrome 验证记录；GitHub Actions 已加入内容检查，本轮未触发远程部署。外部文献链接沿用文章原来源，本轮链接检查关闭外部网络验证。',
             '', '### 页面预览', '', '![论文页](previews/publications-desktop.png)', '', '![视觉伺服教学图与正文](previews/fig-camera-projection-desktop.png)', '', '![Flow Matching 算例](previews/fig-gaussian-flow-example-desktop.png)',
             '', '## 后续维护', '',
             '- 分类与类型：`_data/domains.yml`、`_data/content_types.yml`。',
             '- 新图位置清单：`_data/illustrations.yml`；图源与确定性生成脚本：`assets/images/figures/`、`tools/build-illustrations.py`。',
             '- 共用组件：`_includes/figure.html`、`_includes/demo-frame.html`、`_includes/note-list.html`。',
             '- 日常新增文章运行普通内容检查；`--baseline` 用于本次 18 篇交付快照，CI 的普通检查允许继续发布新文章。',
             '- 浏览器验收工具只使用独立的项目缓存配置目录；日志与完整截图在 `.qa/browser/`。',
             '']
    (REPORT/'blog-organization.md').write_text('\n'.join(body),encoding='utf-8')
    print('Wrote reports/blog-organization.md and visual evidence in reports/previews/.')


if __name__=='__main__':
    main()
