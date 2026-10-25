#!/usr/bin/env python3
"""Apply the reviewed article metadata/media manifest without rewriting prose.

This migration is idempotent. It updates only named metadata, media blocks and
the explicitly reviewed quiz structure. Run build-illustrations.py first.
"""
from pathlib import Path
import json
import re
import shutil
import sys

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')

# basename, topic, type, cover title, subtitle, first diagram used on the cover
POSTS = [
    ('2025-09-11-llm', '大模型基础', 'tutorial', 'LoRA / QLoRA', '低秩更新 · 参数效率 · 存储与计算', 'lora-branches'),
    ('2025-11-26-transformer', '大模型基础', 'tutorial', None, '', 'attention-masks'),
    ('2025-12-03-llms', '大模型基础', 'tutorial', 'LLM 模块笔记', 'Tokenizer · RoPE · SwiGLU', 'token-pipeline'),
    ('2025-12-08-quiz', '大模型基础', 'qa', '大模型复习清单', '结构 · 注意力 · 训练 · 推理', 'quiz-map'),
    ('2025-12-14-roboLLM', '机器人学习与 VLA', 'survey', 'Robot + LLM', '从视觉语言理解到机器人执行', 'robot-llm-loop'),
    ('2026-03-26-priorDA', '三维视觉', 'paper', 'Prior Depth Anything', '相对结构 + 度量先验', 'prior-depth-pipeline'),
    ('2026-04-01-3DGSS', '三维视觉', 'tutorial', None, '', 'gaussian-parameters'),
    ('2026-04-01-flowmatching', '生成模型', 'tutorial', 'Flow Matching', '条件插值 · 边际速度 · 概率输运', 'conditional-marginal'),
    ('2026-04-01-VGGT', '三维视觉', 'paper', 'VGGT', '多视图输入 · 前馈几何预测', 'vggt-heads'),
    ('2026-04-02-pi0', '机器人学习与 VLA', 'paper', None, '', 'pi0-architecture'),
    ('2026-04-17-robot-kinematics', '运动学与控制', 'tutorial', None, '', 'coordinate-chain'),
    ('2026-09-18-pid-mpc-control', '运动学与控制', 'tutorial', 'PID 与 MPC', '反馈纠偏 · 滚动优化 · 显式约束', 'pid-responses'),
    ('2026-09-19-diffusion-policy', '机器人学习与 VLA', 'paper', 'Diffusion Policy', '多峰动作 · 条件去噪 · 滚动执行', 'action-modes'),
    ('2026-09-19-rl-base', '强化学习', 'tutorial', '强化学习基础', 'Bellman · MC / TD · 策略优化', 'bellman-backup'),
    ('2026-09-19-visual-servoing-survey', '视觉伺服', 'survey', '视觉伺服', 'IBVS · ViT-VS · CNS', 'camera-projection'),
    ('2026-09-21-hilserl', '强化学习', 'paper', 'HIL-SERL', '人工纠偏 · 回放学习 · 真机控制', 'hil-actor-critic'),
    ('2026-09-21-pi0.5', '机器人学习与 VLA', 'paper', 'π0.5', '异构数据 · FAST · 连续动作', 'pi05-data'),
    ('2026-09-23-robot-manipulation-survey', '机器人学习与 VLA', 'survey', '机器人操作路线', '七条路线 · 数据 · 泛化 · 闭环', 'research-routes'),
]

# Figure placements refer to existing section numbers, so prose edits are preserved.
FIGURES = {
    'llm': [('2.', 'lora-branches', 'full', '原权重 W 保持冻结，输入同时经过低秩分支 A、B；两路相加得到输出，矩阵形状沿用正文约定。'), ('7.', 'qlora-precision', 'full', 'QLoRA 将量化权重存储、计算精度与适配器更新分开；图为数据流示意，不代表全部张量都以 4-bit 计算。')],
    'transformer': [('9.', 'attention-masks', 'full', '三种掩码的允许位置示意。行是 Query，列是 Key；组合掩码同时排除未来位置和填充 Key。')],
    'llms': [('2.', 'token-pipeline', 'full', '文本、Token、整数 ID 与 Embedding 是不同层次；图中的切分与 ID 仅作教学示例，不对应某个真实 tokenizer。'), ('5.', 'rope-rotation', 'medium', 'RoPE 对成对分量施加位置相关旋转，保留每对向量的范数；维度如何配对必须与实现保持一致。'), ('6.', 'swiglu-gate', 'full', 'SwiGLU 的 gate 与 up 分支先逐元素相乘，再经 down 投影回到模型宽度。')],
    'roboLLM': [('1.', 'robot-llm-loop', 'full', '感知、语义接地、高层规划、动作生成与真实反馈构成研究问题链；各方法可能只覆盖其中一部分。')],
    'priorDA': [('2.', 'prior-depth-pipeline', 'full', '相对深度提供稠密几何结构，度量先验提供尺度约束；粗对齐、重加权和条件细化将两者结合。')],
    '3DGSS': [('2.', 'gaussian-parameters', 'medium', '二维截面示意：高斯中心决定位置，尺度与旋转共同决定协方差；透明度参与后续合成。'), ('3.', 'gaussian-projection', 'full', '三维高斯投影为屏幕空间椭圆，并沿深度顺序进行透明度合成；图为表示与渲染流程示意。')],
    'flowmatching': [('3.', 'conditional-marginal', 'full', '不同条件路径可以在相同时间、相同位置提供不同速度标签；平方损失回归到它们的条件平均，而非记住单条配对路径。'), ('6.', 'gaussian-flow-example', 'full', '正文独立高斯算例的解析计算：方差先缩小再增大，真实 ODE 轨迹随时间变速；一步 Euler 将所有起点送到均值 2。')],
    'VGGT': [('2.', 'vggt-heads', 'full', '多视图输入经过共享表征与跨视图信息交互，连接相机、深度、点图与跟踪等输出；示意图省略具体网络层。')],
    'pi0': [('4.', 'pi0-architecture', 'full', '视觉语言骨干提供条件表征，动作 expert 在观测与流时间条件下预测动作速度场；此图为正文方法整理。'), ('5.', 'pi0-training-inference', 'full', '训练时采样加噪动作与速度标签；推理时从新噪声积分生成动作块，只执行一段，再用新观测重新规划。')],
    'robot-kinematics': [('2.6', 'coordinate-chain', 'full', '上标表示目标坐标系：将相机中的目标经手眼关系与当前末端位姿变换到基座系；乘法次序与正文约定一致。'), ('4.5', 'quaternion-slerp', 'medium', '单位四元数的 q 与 −q 表示同一旋转；SLERP 前选择符号以沿较短球面弧插值。图为低维球面截面示意。'), ('6.3', 'jacobian-singularity', 'full', '单位关节速度球经位置雅可比映射为末端速度椭球；接近奇异位形时，某些方向的可达瞬时速度收缩。')],
    'pid-mpc-control': [('4.', 'pid-responses', 'full', '统一一阶对象上的教学仿真：G(s)=1/(s+1)，单位阶跃；P: Kp=2，PI: Kp=2、Ki=1，PID 另取 Kd=0.2 并对测量求导。未加入饱和与噪声，不代表普遍整定结论。'), ('7.', 'mpc-horizon', 'full', '每次根据当前状态优化预测窗口内的控制，只执行第一个控制量；下一次收到新测量后重新求解。')],
    'diffusion-policy': [('6.', 'action-modes', 'full', '定性示意：左右两种可行绕行策略形成多峰分布，直接回归的均值可能落在障碍上；曲线不表示论文实验统计。'), ('8.', 'action-windows', 'full', '观测窗口 To、预测动作窗口 Tp 与实际执行窗口 Ta 分工不同；执行一段后获取新观测，闭环发生在再次生成动作块时。')],
    'rl-base': [('2.', 'bellman-backup', 'full', '给定策略下，对动作与后继状态按概率加权，得到即时奖励加折扣后续价值；最优方程才对动作取最大值。'), ('6.', 'mc-td-timeline', 'full', 'MC 等待整段回报，TD(0) 用一步奖励与后继价值自举；二者使用的目标与更新时间不同。')],
    'visual-servoing-survey': [('2.', 'camera-projection', 'medium', '静止三维点在针孔模型下投影到归一化像平面；相机沿 +X 平移时，相机系的 X 与归一化像坐标 x 减小。图中 X 轴画为竖直方向。'), ('4.', 'servo-velocity-chain', 'full', '特征误差先映射为相机速度，再按坐标系与参考点约定转换到机器人执行量；速度变换是否需要杠杆臂项取决于参考点。')],
    'hilserl': [('6.', 'hil-actor-critic', 'full', 'Critic 用终止掩码、奖励和目标网络构造 Bellman 目标；Actor 在熵正则目标下利用当前 Critic 的价值与动作梯度。示意图省略双 Critic 等实现细节。'), ('8.', 'hil-asynchronous', 'full', 'Actor 持续与机器人交互，Learner 从示范和在线回放采样更新；参数同步与经验传输并非每个控制时刻都阻塞执行。')],
    'pi0.5': [('3.', 'pi05-data', 'full', '六类数据提供不同监督，图文样本没有动作标签；HL 可来自机器人数据再标注，六类并非互斥小时数。VI 用于聚焦后的高层学习。'), ('5.', 'fast-encoding', 'full', 'FAST 将动作块按时间维做 DCT、量化并编码为离散序列；推理连续动作时仍使用动作 expert，不能把两种输出路径混为一谈。'), ('7.', 'pi05-attention-mask', 'full', '行是 Query，列是 Key：前缀不看动作，离散输出保持因果，连续动作不看真实 FAST 标签；这是防止答案泄漏的可见性结构。')],
    'robot-manipulation-survey': [('@before:1.', 'research-routes', 'full', '七条路线按核心研究问题组织，名称与站内路线图共享数据；箭头表示阅读关联，不表示方法之间的性能排名。')],
}

MERMAID_CAPTIONS = {
    'transformer': ['Encoder–Decoder 的简化计算路径；与上方论文架构图对应阅读。'],
    'flowmatching': ['从端点耦合、条件速度回归到网络 ODE 的理论主线。'],
    'robot-kinematics': ['运动学在感知、轨迹与闭环执行中的位置。'],
    'pid-mpc-control': ['参考值、误差、控制器与对象组成负反馈闭环。', 'MPC 用模型预测与带约束优化生成当前执行量，再根据新测量重新求解。'],
    'diffusion-policy': ['训练流程：观测条件固定后，采样噪声等级并回归噪声。', '运行流程：内部多步去噪与外部滚动观测是两个时间层级。'],
    'rl-base': ['从 MDP、Bellman 方程到价值学习与策略优化的阅读路线。', '正文使用的两步决策例子，终止后的后续价值为零。'],
    'visual-servoing-survey': ['视觉前端、控制器与机器人执行共同形成视觉反馈闭环。', 'ViT-VS 与 CNS 在视觉对应及控制映射中的分工；两者不是必须串联的模块。'],
    'hilserl': ['HIL-SERL 系统：执行、人工接管、经验回放与参数学习相互连接。', '在线接管经验进入两类回放池，训练 batch 从两池分别采样。'],
    'pi0.5': ['同一个 VLA 分别完成高层语义决策与低层连续动作生成。', '两阶段训练及训练后运行路径，数据选择随阶段改变。', 'Knowledge Insulation 关联工作的梯度隔离示意；不应当作 π0.5 原论文的同义名称。'],
}

DEMO_MOVES = {
    '3DGSS': [('3d-gaussian-ellipsoid', '2.')],
    'flowmatching': [('flow-matching-ot', '3.1')],
    'VGGT': [('semantic-gaussian-grasp', '5.')],
    'robot-kinematics': [('robot-kinematics-transform', '2.6'), ('robot-kinematics-gimbal', '3.4'), ('robot-kinematics-7dof', '5.3')],
}


def headings(text):
    result, fence, offset = [], None, 0
    for line in text.splitlines(keepends=True):
        f = re.match(r'^\s*(`{3,}|~{3,})', line)
        if f:
            fence = None if fence else f[1][0]
        elif not fence:
            h = re.match(r'^(#{2,6})\s+(.+)', line)
            if h:
                result.append((offset, len(h[1]), h[2].strip()))
        offset += len(line)
    return result


def place(text, section, block):
    if block in text:
        return text
    marker = re.search(r'\bid="(fig-[^"]+)"', block)
    if marker and f'id="{marker[1]}"' in text:
        pattern = r'{% include figure\.html\s+(?:(?!%}).)*\bid="' + re.escape(marker[1]) + r'"(?:(?!%}).)*%}'
        return re.sub(pattern, lambda m: block, text, count=1, flags=re.S)
    before = section.startswith('@before:')
    section = section.removeprefix('@before:')
    hs = headings(text)
    found = next((h for h in hs if h[2].startswith(section + ' ')), None)
    if found is None:
        raise ValueError(f'Missing section {section}')
    start, level, _ = found
    end = start if before else next((h[0] for h in hs if h[0] > start and h[1] <= level), len(text))
    return text[:end].rstrip() + '\n\n' + block.strip() + '\n\n' + text[end:].lstrip()


def figure(slug, size, caption, **kwargs):
    path = ROOT / 'assets/images/figures' / f'{slug}.svg'
    svg = path.read_text(encoding='utf-8')
    width, height = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups()
    params = dict(src=f'/assets/images/figures/{slug}.svg', id=f'fig-{slug}', alt=caption.split('；')[0], caption=caption, size=size, width=width, height=height, source='根据正文整理的教学示意')
    params.update(kwargs)
    return '{% include figure.html ' + ' '.join(f'{k}="{v}"' for k, v in params.items()) + ' %}'


def wrap_mermaid(text, slug):
    if 'data-mermaid-figure' in text:
        return text
    captions = iter(MERMAID_CAPTIONS.get(slug, []))
    def wrap(m):
        caption = next(captions)
        out = '<figure class="article-figure mermaid-figure" data-mermaid-figure markdown="1">\n\n' + m[0] + '\n\n<figcaption><span class="figure-number" aria-hidden="true"></span>' + caption + '</figcaption>\n</figure>'
        if slug == 'transformer':
            out = '<details class="media-archive" markdown="1">\n<summary>展开：Encoder–Decoder 的简化流程</summary>\n\n' + out + '\n\n</details>'
        return out
    return re.sub(r'(?m)^(```|~~~)mermaid[^\n]*\n.*?^\1\s*$', wrap, text, flags=re.S)


def group_hil_system(text):
    if 'subgraph execution' in text:
        return text
    match = re.search(r'(?m)^```mermaid\n(flowchart TD\n.*?)^```', text, re.S)
    if not match:
        return text
    graph = match[1]
    nodes = dict(re.findall(r'(\w+)\["([^"]+)"\]', graph))
    groups = [('execution', '执行闭环', ['O','P','S','C','N']), ('human', '人工监督', ['H','I']), ('replay', '经验回放', ['R','D','B']), ('learning', '参数学习', ['L'])]
    lines = ['flowchart TD']
    for key, label, ids in groups:
        lines.append(f'    subgraph {key}["{label}"]')
        for node in ids:
            lines.append(f'      {node}["{nodes[node]}"]')
        lines.append('    end')
    lines.extend(re.sub(r'(\w+)\["[^"]+"\]', r'\1', line) for line in graph.splitlines()[1:] if line.strip())
    return text[:match.start(1)] + '\n'.join(lines) + '\n' + text[match.end(1):]


def sync_figure_dimensions(text):
    def update(match):
        block = match[0]
        src = re.search(r'\bsrc="([^"]+)"', block)
        if src and src[1].endswith('.svg'):
            path = ROOT / src[1].lstrip('/')
            width, height = re.search(r'viewBox="0 0 (\d+) (\d+)"', path.read_text(encoding='utf-8')).groups()
            block = re.sub(r'\bwidth="\d+"', f'width="{width}"', block)
            block = re.sub(r'\bheight="\d+"', f'height="{height}"', block)
        return block
    return re.sub(r'{% include figure\.html\s+.*?%}', update, text, flags=re.S)


def transformer_supplement(text):
    match = re.search(r'<details class="media-archive" markdown="1">\n<summary>展开：Encoder–Decoder 的简化流程</summary>.*?</details>', text, re.S)
    if not match:
        return text
    block = match[0]
    existing_image = re.search(r'{% include figure\.html\s+(?:(?!%}).)*transform_architecture\.png(?:(?!%}).)*%}', text, re.S)
    if existing_image and existing_image.end() < match.start() and not text[existing_image.end():match.start()].strip():
        return text
    text = text[:match.start()] + text[match.end():]
    image = re.search(r'{% include figure\.html\s+(?:(?!%}).)*transform_architecture\.png(?:(?!%}).)*%}', text, re.S)
    assert image
    return text[:image.end()] + '\n\n' + block + text[image.end():]


def replace_existing_images(text, slug):
    if slug not in ['transformer', 'visual-servoing-survey']:
        return text
    vit = re.search(r'(?m)^\[vit-paper\]:\s*(\S+)', text)
    cns = re.search(r'(?m)^\[cns-paper\]:\s*(\S+)', text)
    vit_url = vit[1] if vit else ''
    cns_url = cns[1] if cns else ''
    specs = {
        'transform_architecture.png': ('full', '原始 Transformer 的 Encoder–Decoder 架构：左侧编码源序列，右侧通过因果自注意力和交叉注意力逐步生成目标序列。', 'https://arxiv.org/abs/1706.03762', 513),
        'dot-product-attention.png': ('small', '缩放点积注意力：Q 与 K 计算分数，缩放并施加掩码后经 softmax，再对 V 加权求和。', 'https://arxiv.org/abs/1706.03762', 279),
        'MHA.png': ('small', '多头注意力：各头在投影空间计算注意力，拼接后通过输出投影；图中的各头不是人工指定的语义角色。', 'https://arxiv.org/abs/1706.03762', 269),
        'vit-vs-pipeline.png': ('full', 'ViT-VS 原论文图 2：左侧为视觉控制闭环，右上为对应匹配，右下为初始旋转补偿。', vit_url, None),
        'feature-binning.png': ('full', '原笔记分箱示意图存档。ViT-L/14 为图中示例，正文典型实验配置为 ViT-S/14；比较时须先对齐配置。', '', None),
        'vit-vs-ema.png': ('full', 'EMA 新值权重 α 的对比：左轴为位置误差（mm），右轴为路径长度比；两条曲线应结合阅读。', vit_url, None),
        'vit-vs-perturbations.png': ('full', 'ViT-VS 原论文图 3：期望图像与受扰动当前图像之间的特征对应；图像扰动概率不等于遮挡面积比例。', vit_url, None),
        'vit-vs-results-table.png': ('full', 'ViT-VS 完整 Table I：终点位置误差为 mm，APE 平移为 cm；带星号的文献结果需核对实验设置。', vit_url, None),
        'vit-vs-convergence.png': ('full', '六维速度随迭代变化的收敛曲线：左轴为线速度（cm/s），右轴为角速度（deg/s）；该图不是不同方法的成功率比较。', vit_url, None),
        'vit-vs-final-error.png': ('full', '位置与旋转误差随迭代变化：左轴为位置误差（cm），右轴为旋转误差（deg）；该图不是跨试验的终点误差分布。', vit_url, None),
        'vit-vs-runtime.png': ('full', '模型、输入分辨率与分箱层级对应的帧率（Hz）；吞吐率不等于图像采集到命令执行的端到端时延。', vit_url, None),
        'vit-vs-industrial-setup.png': ('medium', 'ViT-VS 原论文图 7(d)：不同初始位置的工具中心点三维轨迹对齐到目标；当前素材未包含设备场景图 7(a–c)。', vit_url, None),
        'cns-comparison.png': ('full', '经典对应求解、图像端到端控制与 CNS 对应编码控制的结构比较；底部优劣概括是论文的设计动机，结果仍取决于具体设置。', cns_url, None),
        'ibvs-handwritten-1.jpg': ('full', '相机坐标下的点运动与归一化投影求导手写存档；统一符号及可复制公式以正文第 2 节为准。', '', None),
        'ibvs-handwritten-2.jpg': ('full', '整理六自由度图像交互矩阵的手写存档；阅读时与正文第 2 节的矩阵列顺序对应。', '', None),
    }
    lines = text.splitlines()
    out, skip = [], False
    for i, line in enumerate(lines):
        if skip:
            skip = False
            continue
        name = next((n for n in specs if n in line and ('![' in line or '<img ' in line)), None)
        if not name:
            out.append(line)
            continue
        size, caption, url, natural = specs[name]
        original = '/assets/images/llm/' + name if slug == 'transformer' else '/assets/images/posts/visual-servoing/' + name
        deferred = name.startswith('ibvs-handwritten')
        src = original if slug == 'transformer' else '/assets/images/posts/visual-servoing/previews/' + Path(name).stem + '.webp'
        image = Image.open(ROOT / src.lstrip('/'))
        params = dict(src=src, original=original, alt=caption.split('；')[0], caption=caption, size=size, width=image.width, height=image.height)
        if url:
            params.update(source_url=url, source='原论文')
        else:
            params['source'] = '原笔记存档'
        if natural:
            params['max_width'] = natural
        if deferred:
            params.update(deferred='true', archive='A1' if '-1.' in name else 'A2')
        block = '{% include figure.html ' + ' '.join(f'{k}="{v}"' for k, v in params.items()) + ' %}'
        if name == 'feature-binning.png':
            block = figure('feature-binning-readable', 'full', '分箱的两步：先构造中心细邻域与外围粗区域，再按固定顺序拼接；外围粗区域内部先平均池化。') + '\n\n<details class="media-archive">\n<summary>展开原笔记分箱图与配置说明</summary>\n' + block + '\n</details>'
        if name == 'vit-vs-results-table.png':
            block = '<details class="media-archive">\n<summary>展开论文完整实验表，核对单位与方法设置</summary>\n' + block + '\n</details>'
        out.append(block)
        if i + 1 < len(lines) and lines[i + 1].startswith('_'):
            skip = True
    return '\n'.join(out).rstrip() + '\n'


def move_demos(text, slug):
    for demo, target in DEMO_MOVES.get(slug, []):
        pattern = r'{% include demo-frame\.html\s+(?:(?!%}).)*' + re.escape(demo + '.html') + r'(?:(?!%}).)*%}'
        match = re.search(pattern, text, re.S)
        if not match:
            raise ValueError(f'Missing demo {demo}')
        block = match[0]
        # Keep the original section as a useful index, including existing anchors.
        text = text[:match.start()] + f'[跳转到本篇演示：{demo}](#demo-{demo})' + text[match.end():]
        if slug == 'VGGT':
            block = '> **应用扩展：**下方为语义高斯与抓取意图的教学原型，非 VGGT 的原始推理流程，也不作为 VGGT 实验结果。\n\n' + block
        text = place(text, target, block)
    return text


def quiz_structure(text):
    if '## 1. 模型结构与归一化' in text:
        return text
    a, fm, body = text.split('---', 2)
    groups = [
        ('1. 模型结构与归一化', ['Transformer 与 LLaMA 的归一化位置有什么不同？', 'RMSNorm、BatchNorm、LayerNorm 的区别是什么？', '如何手写 RMSNorm？', 'ReLU 与 SwiGLU 有何区别？', 'Encoder–Decoder、Encoder-only 与 Decoder-only 分别适合什么任务？', 'BERT 的预训练任务与 Embedding 如何设计？']),
        ('2. 注意力与位置编码', ['解释 Scaled Dot-Product Attention 的计算步骤。', '为什么点积要除以维度的平方根？', 'Self-Attention 与 Cross-Attention 有何区别？', '如何手写 Cross-Attention？', '多头注意力为什么要进行维度拆分？', 'MHA、MQA、GQA 有何区别？', '位置编码解决什么问题？', 'Attention 与传统 Seq2Seq、全连接层分别有什么区别？']),
        ('3. 训练与数据', ['如何估计模型训练时间？', '预训练数据怎样清洗与处理？', '训练过程中应监控什么？', 'Transformer 在哪里共享权重，为什么？']),
        ('4. 推理与系统设计', ['文本从输入到输出经过哪些步骤？', '常见 Decoding 策略有哪些？', '端到端大模型与多个小模型的组合如何取舍？', '如何设计并验证 Prompt？', '大模型结构近年有哪些变化？']),
        ('5. 分布式训练', ['DP 与 DDP 的执行、同步和资源使用有何不同？']),
    ]
    body = '\n\n这是复习问题清单，尚未完成的问题标记为“待补充”。建议先阅读 [Transformer](/posts/transformer/)、[大模型模块笔记](/posts/llms/) 和 [LoRA / QLoRA](/posts/llm/)。\n'
    for title, questions in groups:
        body += '\n## ' + title + '\n\n'
        body += '\n'.join('- **待补充**：' + q for q in questions) + '\n'
        if title.startswith('1.'):
            body += '\n**已有笔记：**原始 Transformer 使用 Post-LN，LLaMA 使用 Pre-Norm，并采用 RMSNorm。比较时应同时区分放置位置与归一化计算。\n'
    return '---' + fm + '---' + body


def configure_pages():
    for layout, prefix, label, intro in [
        ('blog', 'blog', '全部文章', '按主题与内容类型浏览机器人学习、感知、控制和模型基础笔记。'),
        ('publications', 'paper', '全部论文与综述', '论文解读与专题综述：关注问题设定、方法逻辑、实验依据与阅读思考。'),
    ]:
        pool = "{% assign note_posts = site.posts | where_exp: 'item', 'item.hidden != true' %}"
        if layout == 'publications':
            pool += "\n{% assign note_posts = note_posts | where_exp: 'item', 'item.content_type == \"paper\" or item.content_type == \"survey\"' %}"
        extra = '<nav class="blog-resources" aria-label="更多浏览方式"><span>更多浏览方式</span><a href="{{ \'/categories/\' | relative_url }}">分类索引</a><a href="{{ \'/tags/\' | relative_url }}">标签</a><a href="{{ \'/archives/\' | relative_url }}">时间归档</a><a href="{{ \'/feed.xml\' | relative_url }}">RSS</a></nav>' if layout == 'blog' else ''
        text = f'''---
layout: default
refactor: true
---
{pool}
<article class="section-page {layout}-page">
  <header class="section-page__header"><p class="section-page__eyebrow">{'Writing' if layout == 'blog' else 'Papers'}</p><h1>{{{{ page.title }}}}</h1><p>{intro}</p></header>
  {{% include note-list.html posts=note_posts prefix="{prefix}" label="{label}" %}}
  {extra}
</article>
'''
        (ROOT / f'_layouts/{layout}.html').write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n', encoding='utf-8')
    # Override only the upstream post template; retain its TOC, sharing and footer.
    theme = Path('D:/Ruby33-x64/lib/ruby/gems/3.3.0/gems/jekyll-theme-chirpy-7.3.1/_layouts/post.html')
    post_layout = ROOT / '_layouts/post.html'
    if not post_layout.exists():
        if not theme.exists():
            raise RuntimeError('Copy the pinned Chirpy 7.3.1 _layouts/post.html before running this migration.')
        text = theme.read_text(encoding='utf-8')
        text = text.replace('<h1 data-toc-skip>{{ page.title }}</h1>', '<h1 data-toc-skip>{{ page.title }}</h1>\n    {% include post-classification.html %}')
        # A cover caption is presentation text, not the image alt text.
        text = text.replace('<div class="mt-3 mb-3">', '<figure class="post-cover mt-3 mb-3">').replace('{%- if page.image.alt -%}\n            <figcaption class="text-center pt-2 pb-2">{{ page.image.alt }}</figcaption>\n          {%- endif -%}\n        </div>', '</figure>')
        post_layout.write_text(text, encoding='utf-8')
    head_path = ROOT / '_includes/head.html'
    if not head_path.exists():
        head = (theme.parent.parent / '_includes/head.html').read_text(encoding='utf-8')
        head = head.replace('width=device-width, user-scalable=no initial-scale=1, shrink-to-fit=no, viewport-fit=cover', 'width=device-width, initial-scale=1, viewport-fit=cover')
        needle = '<script src="{{ \'/assets/js/dist/theme.min.js\' | relative_url }}"></script>'
        assert needle in head
        head = head.replace(needle, needle + '\n  <script src="{{ \'/assets/js/theme-compat.js\' | relative_url }}"></script>')
        head_path.write_text(head, encoding='utf-8')
    for relative in ['_layouts/home.html', '_layouts/research.html']:
        p = ROOT / relative
        text = p.read_text(encoding='utf-8-sig')
        text = text.replace('多模态模型与世界模型', '大模型基础与三维感知')
        text = text.replace("'/publications/' | relative_url }}#paper-domain-control", "'/blog/' | relative_url }}#blog-domain-control")
        text = text.replace("'/publications/' | relative_url }}#paper-domain-multimodal-models", "'/blog/' | relative_url }}#blog-domain-multimodal-models")
        text = text.replace("'/publications/' | relative_url }}#paper-domain-world-models", "'/roadmap/' | relative_url }}#planning")
        text = text.replace('多模态笔记</a>', '大模型基础</a>')
        if relative.endswith('research.html') and '#paper-domain-3d-vision' not in text:
            text = text.replace('>世界模型笔记</a>', '>规划与世界模型路线</a>\n          <a href="{{ \'/publications/\' | relative_url }}#paper-domain-3d-vision">三维视觉笔记</a>')
        p.write_text(text, encoding='utf-8')
    p = ROOT / '_layouts/projects.html'
    text = p.read_text(encoding='utf-8-sig')
    for old, name in [('posts/robotics/kinematics-cover.jpg', 'robot-kinematics-7dof'), ('posts/3d-vision/3dgs-cover.jpg', 'semantic-gaussian-grasp'), ('posts/placeholders/paper-note.svg', 'flow-matching-ot')]:
        text = text.replace('/assets/images/' + old, '/assets/images/demos/' + name + '.jpg')
    text = text.replace('Flow Matching 最优传输轨迹演示', 'Flow Matching 随机配对条件插值演示')
    text = text.replace('以二维轨迹原型理解流匹配与向量场，为 VLA 中的连续动作生成和机器人轨迹建模提供方法验证。', '拖动时间观察随机配对的条件插值与速度标签；此原型不表示 OT 解，也不表示训练后模型的生成轨迹。')
    text = text.replace("'/posts/3DGSS/' | relative_url", "'/posts/VGGT/' | relative_url")
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'README.md'
    text = p.read_text(encoding='utf-8-sig').replace('本项目使用 Jekyll 和 Chirpy 7.x。', '本项目使用 Jekyll 和固定版本 Chirpy 7.3.1，提交 Gemfile.lock 以保持本地与 CI 一致。')
    if 'tools/check-content.py' not in text:
        text += '\n## 内容与插图维护\n\n主分类来自 `_data/domains.yml`，每篇只设置一个 `categories`，并填写 `content_type: tutorial / paper / survey / qa`。博客展示全集，论文页展示 paper 与 survey。\n\n正文图片统一使用 `_includes/figure.html`，交互演示使用 `_includes/demo-frame.html`。SVG 为可编辑图源；不要用封面代替方法图。\n\n```powershell\npython tools/build-illustrations.py\npython tools/check-content.py\nbundle exec jekyll build\nbundle exec htmlproofer _site --disable-external\n```\n\n图文与分类清单见 `reports/blog-organization.md`。`tools/organize-content.py` 为本次幂等迁移脚本；新增文章通常只需编辑元数据与正文，无需重跑迁移。\n'
    p.write_text(text, encoding='utf-8')


def main():
    manifest = []
    for basename, topic, kind, cover, subtitle, motif in POSTS:
        p = ROOT / '_posts' / f'{basename}.md'
        text = p.read_text(encoding='utf-8-sig')
        slug = basename[11:]
        if 'content_type:' not in text.split('---', 2)[1]:
            text = re.sub(r'(?m)^categories:.*$', f'categories: [{topic}]\ncontent_type: {kind}', text, count=1)
        text = text.replace('depth complete', 'depth-completion')
        text = re.sub(r'(?m)^(tags:.*)(?:paper-notes, |paper-survey, )', r'\1', text)
        if cover:
            text = re.sub(r'(?m)^(  path:) /assets/images/posts/placeholders/.*$', r'\1 /assets/images/covers/' + slug + '-cover.jpg', text)
        if slug == 'quiz':
            text = quiz_structure(text)
        if 'data-mermaid-figure' not in text:
            text = wrap_mermaid(text, slug)
        if slug == 'hilserl':
            text = group_hil_system(text)
        text = replace_existing_images(text, slug)
        if slug == 'transformer':
            text = transformer_supplement(text)
        for section, name, size, caption in FIGURES.get(slug, []):
            text = place(text, section, figure(name, size, caption))
            manifest.append(dict(post=slug, section=section, asset=name, size=size, caption=caption))
        # Each moved include is only migrated once; original section anchors stay valid.
        if '跳转到本篇演示：' not in text:
            text = move_demos(text, slug)
        # Correct prose adjacent to the reviewed figure labels, retaining qualifications.
        if slug == 'visual-servoing-survey':
            text = text.replace('硬件、输入大小、backbone 与分箱层级必须一起说明。', '硬件、输入大小、backbone 与分箱层级必须一起说明；帧率曲线不能代替端到端时延测量。')
        text = sync_figure_dimensions(text)
        p.write_text(text, encoding='utf-8')
    configure_pages()
    (ROOT / '_data/illustrations.yml').write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(f'Organized {len(POSTS)} posts and {len(manifest)} new teaching figures.')


if __name__ == '__main__':
    main()
