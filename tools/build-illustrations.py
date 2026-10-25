#!/usr/bin/env python3
"""Build editable SVG teaching figures, cover sources and compressed previews.

Standard library + Pillow + PyYAML. No generative assets or external data.
All numerical curves below state their analytical model / teaching assumptions.
"""
from pathlib import Path
from html import escape
import importlib.util
import math
import re
import sys

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
OUT = ROOT / 'assets/images/figures'
BLUE, TEAL, GOLD, PINK, INK = '#4264ad', '#198477', '#b57836', '#b15e7b', '#253247'
COLORS = [BLUE, TEAL, GOLD, PINK, '#8063a9', '#6d8845', '#4b819c']


class SVG:
    def __init__(self, title, subtitle='', w=1000, h=600):
        self.title, self.subtitle, self.w, self.h = title, subtitle, w, h
        self.items = []
        self.text(40, 52, title, 30, weight=700)
        if subtitle:
            self.text(40, 88, subtitle, 19, color='#637084')

    def add(self, value):
        self.items.append(value)

    def text(self, x, y, value, size=24, color=INK, anchor='start', weight=400):
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')

    def multiline(self, x, y, lines, size=24, color=INK, anchor='middle', gap=34):
        for i, line in enumerate(lines):
            self.text(x, y + gap * i, line, size, color, anchor)

    def rect(self, x, y, w, h, fill='#f3f6fa', stroke='#d7dfe9', radius=12, dash=False):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"' + (' stroke-dasharray="8 6"' if dash else '') + '/>')

    def box(self, x, y, w, h, lines, color=BLUE):
        self.rect(x, y, w, h, '#f8fafc', color)
        if isinstance(lines, str):
            lines = lines.split('\n')
        self.multiline(x + w / 2, y + h / 2 - (len(lines) - 1) * 16 + 8, lines, 24, color)

    def line(self, x1, y1, x2, y2, color='#7c899a', width=3, dash=False, arrow=False):
        self.add(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"' + (' stroke-dasharray="8 6"' if dash else '') + (' marker-end="url(#arrow)"' if arrow else '') + '/>')

    def arrow(self, x1, y1, x2, y2, color=BLUE):
        self.line(x1, y1, x2, y2, color, arrow=True)

    def path(self, points, color=BLUE, width=4, dash=False, fill='none'):
        d = ' '.join(('M' if i == 0 else 'L') + f'{x:.2f},{y:.2f}' for i, (x, y) in enumerate(points))
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}"' + (' stroke-dasharray="8 6"' if dash else '') + '/>')

    def circle(self, x, y, r, fill=BLUE, stroke='none'):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')

    def ellipse(self, x, y, rx, ry, angle=0, color=BLUE):
        self.add(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" transform="rotate({angle} {x} {y})" fill="{color}" fill-opacity=".10" stroke="{color}" stroke-width="3"/>')

    def axes(self, x, y, w, h, xlabel='', ylabel=''):
        self.arrow(x, y, x + w, y, '#8b97a7')
        self.arrow(x, y, x, y - h, '#8b97a7')
        self.text(x + w - 4, y + 33, xlabel, 20, anchor='end')
        self.text(x + 3, y - h - 12, ylabel, 20)

    def footer(self, text):
        self.line(40, self.h - 56, self.w - 40, self.h - 56, '#e2e7ef', 1)
        self.text(40, self.h - 24, text, 18, '#69778a')

    def render(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="title desc">
<title id="title">{escape(self.title)}</title><desc id="desc">{escape(self.subtitle or self.title)}。根据文章整理的教学示意。</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="context-stroke"/></marker></defs>
<style>text{{font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',Arial,sans-serif}}</style>
<rect width="100%" height="100%" fill="white"/>
''' + '\n'.join(self.items) + '\n</svg>\n'

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'{name}.svg').write_text(self.render(), encoding='utf-8')


def matrix(s, x, y, n, allowed, cell=36, label='', group=0):
    for r in range(n):
        for c in range(n):
            s.rect(x + c * cell, y + r * cell, cell - 3, cell - 3, BLUE if allowed(r, c) else '#edf0f5', 'none', 3)
    if label:
        s.text(x + n * cell / 2, y + n * cell + 36, label, 22, anchor='middle')


def chain(s, labels, y=205, colors=None):
    n = len(labels)
    gap = 35
    w = (920 - (n - 1) * gap) / n
    for i, label in enumerate(labels):
        x = 40 + i * (w + gap)
        s.box(x, y, w, 105, label, (colors or COLORS)[i % len(colors or COLORS)])
        if i:
            s.arrow(x - gap + 5, y + 52, x - 8, y + 52)


def build():
    s = SVG('LoRA：冻结主路，学习低秩更新', '列向量约定：x∈Rⁿ，W∈Rᵐˣⁿ，A∈Rʳˣⁿ，B∈Rᵐˣʳ')
    s.box(45, 255, 130, 85, '输入 x')
    s.box(300, 140, 310, 100, ['W · x', '冻结 W'], BLUE)
    s.box(270, 355, 160, 100, ['A · x', 'n → r'], TEAL)
    s.box(500, 355, 160, 100, ['B · Ax', 'r → m'], TEAL)
    s.circle(780, 295, 34, '#fff', GOLD); s.text(780, 305, '+', 32, GOLD, 'middle')
    for a in [(175, 280, 300, 190), (175, 315, 270, 405), (430, 405, 495, 405), (610, 190, 745, 280), (660, 405, 755, 325), (815, 295, 945, 295)]: s.arrow(*a)
    s.text(935, 260, '输出 y', 24, anchor='end')
    s.text(500, 515, 'y = Wx + (α/r) BAx     ·     参数量：mn → r(n+m)', 27, anchor='middle')
    s.footer('低秩分支学习的是权重更新量；合并推理时 W′ = W + (α/r)BA。'); s.save('lora-branches')

    s = SVG('QLoRA：存储、计算、更新分开理解', '量化基础权重保持冻结，适配器参与梯度更新')
    chain(s, [['4-bit 存储', '量化基础权重'], ['反量化', '进入计算格式'], ['矩阵乘法', '如 BF16 计算']], 170)
    s.box(330, 355, 340, 100, ['LoRA 适配器（输入 x）', '浮点参数与梯度'], TEAL)
    s.circle(865,405,30,'#fff',GOLD);s.text(865,415,'+',30,GOLD,'middle')
    s.arrow(820,280,855,370,BLUE);s.arrow(670,405,830,405,TEAL)
    s.text(500, 510, '低比特存储 ≠ 所有运算都使用 4-bit', 28, anchor='middle')
    s.footer('具体计算格式、优化器状态与量化方案由实现配置决定。'); s.save('qlora-precision')

    s = SVG('Attention Mask：行是 Query，列是 Key', '蓝色允许注意；浅灰色屏蔽。示例最后两列为 Padding Key。', h=480)
    matrix(s, 65, 145, 6, lambda r, c: c < 4, label='Padding')
    matrix(s, 390, 145, 6, lambda r, c: c <= r, label='Causal')
    matrix(s, 715, 145, 6, lambda r, c: c <= r and c < 4, label='组合：取交集')
    s.footer('图仅展示 Key 可见性；实际实现还应正确处理 Padding Query 与全屏蔽行。'); s.save('attention-masks')

    s = SVG('从文本到 Embedding', '切分粒度、Token ID、向量维度是三个不同问题')
    chain(s, [['文本', '机器人学习'], ['Token', '机器人 / 学习'], ['整数 ID', '例如 42 / 87']], 160)
    s.box(335, 350, 330, 105, ['Embedding 查表', '[T, d_model]'], TEAL)
    s.arrow(810, 265, 665, 380)
    for i, c in enumerate([BLUE, TEAL, GOLD, PINK, BLUE, TEAL]): s.rect(100 + i * 24, 360, 19, 80, c, 'none', 2)
    s.text(110, 485, '每个 Token 对应向量', 22)
    s.footer('图中切分与 ID 是解释格式的例子，不是某个真实模型的 tokenizer 输出。'); s.save('token-pipeline')

    s = SVG('RoPE：每对分量做位置相关旋转', '二维平面示意；旋转前后向量范数相同', w=780,h=620)
    s.axes(240,340,205,190,'分量 1','分量 2')
    s.circle(240,340,150,'none','#dbe3ed')
    s.arrow(240,340,385,301,BLUE);s.text(400,302,'q',26,BLUE)
    s.arrow(240,340,303,204,TEAL);s.text(315,185,'R(mθ)q',26,TEAL)
    s.multiline(615,255,['成对旋转，范数不变','Q/K 的相对相位','随相对位置变化'],22,gap=42)
    s.multiline(615,435,['相邻配对 / 半区配对','由实现约定决定'],21,gap=36)
    s.footer('图中圆为单位圆缩放示意；高维向量由多对旋转平面组成。'); s.save('rope-rotation')

    s = SVG('SwiGLU：两个分支，一次门控', '中间宽度 d_ff 与模型宽度 d_model 分开标记')
    s.box(40, 245, 125, 100, ['输入 x', 'd_model'])
    s.box(280, 145, 230, 95, ['W_gate', 'SiLU'], TEAL)
    s.box(280, 365, 230, 95, ['W_up', '线性'], BLUE)
    s.circle(650, 295, 35, '#fff', GOLD); s.text(650, 307, '⊙', 32, GOLD, 'middle')
    s.box(795, 245, 165, 100, ['W_down', '→ d_model'], PINK)
    for a in [(165, 280, 275, 190), (165, 315, 275, 410), (510, 190, 625, 270), (510, 410, 625, 325), (685, 295, 790, 295)]: s.arrow(*a)
    s.text(500, 510, 'FFN(x) = [SiLU(xW_gate) ⊙ (xW_up)] W_down', 25, anchor='middle')
    s.footer('⊙ 表示逐元素乘法；图中省略偏置及具体框架的矩阵存储转置约定。'); s.save('swiglu-gate')

    s = SVG('Robot + LLM：把语义连接到执行', '方法家族围绕同一条问题链展开')
    labels = [['视觉感知', '几何 / 物体'], ['语义接地', '可供性 / 部件'], ['高层规划', '子任务 / 监控']]
    chain(s, labels, 155)
    s.box(330, 350, 320, 110, ['动作生成与机器人执行', 'VLA / 生成式策略'], TEAL)
    s.arrow(820, 260, 650, 380); s.arrow(330, 405, 80, 300)
    s.text(140, 365, '新观测 / 执行反馈', 22, TEAL)
    s.footer('模型名称及方法分支见正文；总览图不表示每个方法都覆盖完整闭环。'); s.save('robot-llm-loop')

    s = SVG('PriorDA：结构与尺度在不同阶段汇合', '单目相对结构 + 任意可用的度量先验')
    s.box(40, 140, 250, 90, ['单目图像', '稠密相对深度'], BLUE)
    s.box(40, 340, 250, 90, ['度量先验', '可能稀疏 / 不完整'], TEAL)
    s.box(390, 235, 230, 110, ['粗度量对齐', 'kNN / 标定 / 重加权'], GOLD)
    s.box(710, 235, 250, 110, ['条件细化', '输出稠密度量深度'], PINK)
    s.arrow(290, 185, 390, 260); s.arrow(290, 385, 390, 320); s.arrow(620, 290, 705, 290)
    s.text(500, 485, '几何条件保留结构，度量条件约束尺度', 27, anchor='middle')
    s.footer('条件分支与主干的具体连接方式见正文第 4–5 节。'); s.save('prior-depth-pipeline')

    s = SVG('一个高斯：位置、形状与方向', '协方差的二维截面示意，不代表完整三维投影', w=780)
    s.axes(140, 430, 470, 270, 'x', 'y')
    s.ellipse(370, 300, 170, 65, -28, BLUE)
    s.circle(370, 300, 7, TEAL); s.text(385, 330, '中心 μ', 24, TEAL)
    s.arrow(370, 300, 515, 222, BLUE); s.arrow(370, 300, 333, 241, GOLD)
    s.text(460, 185, '尺度 + 旋转 → Σ', 24)
    s.text(390, 495, '透明度 α 与颜色在渲染阶段参与合成', 22, anchor='middle')
    s.footer('三维协方差 Σ = R diag(s²) Rᵀ；椭圆展示其中一个平面截面。'); s.save('gaussian-parameters')

    s = SVG('Splatting：从三维高斯到二维像素', '投影足迹 → 深度排序 → 前向透明度合成')
    s.ellipse(170, 245, 95, 52, -25, BLUE); s.ellipse(170, 300, 65, 38, 20, TEAL)
    s.text(170, 405, '三维高斯', 24, anchor='middle')
    s.arrow(290, 285, 380, 285)
    s.rect(400, 160, 205, 225, '#fafbfd')
    s.ellipse(500, 245, 72, 37, -10, BLUE); s.ellipse(505, 275, 50, 25, 20, TEAL)
    s.text(500, 435, '屏幕空间足迹', 24, anchor='middle')
    s.arrow(625, 285, 710, 285)
    s.box(730, 190, 225, 170, ['按深度合成', 'Σ Tᵢ αᵢ cᵢ', 'Tᵢ = ∏ⱼ＜ᵢ(1−αⱼ)'], GOLD)
    s.footer('图为渲染链路示意；实际屏幕协方差还涉及相机变换与投影局部线性化。'); s.save('gaussian-projection')

    s = SVG('条件速度标签与边际速度不是一回事', '相同 t、相同 X_t，可以对应不同端点配对')
    s.box(65, 150, 260, 100, ['配对路径 A', '速度标签 U₁'], BLUE)
    s.box(65, 355, 260, 100, ['配对路径 B', '速度标签 U₂'], PINK)
    s.circle(485, 300, 38, '#fff', TEAL); s.text(485, 309, 'x', 30, TEAL, 'middle')
    s.arrow(325, 200, 450, 280, BLUE); s.arrow(325, 405, 450, 325, PINK)
    s.box(650, 240, 310, 125, ['平方误差回归', 'u_t(x) = E[U_t | X_t=x]'], TEAL)
    s.arrow(525, 300, 640, 300)
    s.text(500, 510, '模型学习条件平均速度；生成时再沿该速度场积分', 27, anchor='middle')
    s.footer('示意节点表示同一时间切片；不将单条训练插值当成生成轨迹。'); s.save('conditional-marginal')

    s = SVG('高斯算例：直线训练不保证匀速生成', 'X₀~N(0,1)，X₁~N(2,1)，独立配对；解析解 Z_t=2t+√a_t z₀', h=650)
    # Two distinct analytical panels, with the axes explicitly identified.
    s.axes(75, 425, 375, 260, 't：0 → 1', '均值 / 方差')
    for fn, color in [(lambda t: 2*t, BLUE), (lambda t: (1-t)**2+t*t, TEAL)]:
        s.path([(75+340*i/100, 425-110*fn(i/100)) for i in range(101)], color)
    s.text(85, 470, '蓝：均值 2t', 22, BLUE); s.text(85, 505, '绿：方差 a_t', 22, TEAL)
    s.axes(570, 425, 350, 260, 't：0 → 1', 'Z_t 的位置')
    for z, color in [(-1, PINK), (0, TEAL), (1, BLUE)]:
        s.path([(570+315*i/100, 355-62*(2*i/100+math.sqrt((1-i/100)**2+(i/100)**2)*z)) for i in range(101)], color)
        s.path([(570, 355-62*z), (885, 355-124)], color, 2, True)
    s.text(575, 470, '实线：真实 ODE', 22); s.text(575, 505, '虚线：一步 Euler → 2', 22)
    s.footer('解析计算；右图示例起点 z₀∈{−1,0,1}，不是模型实验结果。'); s.save('gaussian-flow-example')

    s = SVG('VGGT：多视图到统一几何预测', '此图概括输入输出，省略具体 token 与网络层实现')
    for i, c in enumerate([BLUE, TEAL, GOLD]):
        s.rect(55+i*23, 160+i*23, 170, 110, '#f3f6fa', c)
    s.text(155, 370, '多视图图像', 24, anchor='middle')
    s.box(340, 230, 285, 125, ['共享表征', '跨视图信息交互'], BLUE)
    s.arrow(260, 270, 335, 285)
    for i, label in enumerate(['相机参数', '深度', '三维点图', '点跟踪']):
        y=130+i*90; s.box(765, y, 195, 65, label, COLORS[i]); s.arrow(625, 290, 760, y+32)
    s.footer('下游语义高斯抓取演示属于应用扩展，不是 VGGT 原始网络的一部分。'); s.save('vggt-heads')

    s = SVG('π₀：视觉语言表征与连续动作生成', '观测与指令提供条件，Action Expert 预测动作速度场')
    s.box(40, 145, 250, 105, ['图像 + 语言指令', '视觉语言骨干'], BLUE)
    s.box(40, 355, 250, 95, ['机器人状态', '本体条件'], TEAL)
    s.box(400, 235, 260, 125, ['Action Expert', 'vθ(观测, 动作, t)'], PINK)
    s.box(750, 235, 210, 125, ['积分采样', '连续动作块'], GOLD)
    s.arrow(290, 195, 400, 265); s.arrow(290, 400, 400, 330); s.arrow(660, 295, 745, 295)
    s.text(530, 160, '噪声动作 + 流时间 t', 24, anchor='middle'); s.arrow(530, 178, 530, 230)
    s.footer('图为方法结构概括；张量维度与具体本体配置以正文和官方实现为准。'); s.save('pi0-architecture')

    s = SVG('π₀：训练与执行使用不同的数据路径', '训练不需要每次积分；推理从新噪声逐步生成', h=680)
    s.text(40, 145, 'A · 训练', 25, BLUE, weight=700)
    chain(s, [['示范动作 + 噪声', '采样时间 t'], ['构造加噪动作', '计算速度标签'], ['条件速度回归', '最小化预测误差']], 175)
    s.text(40, 360, 'B · 运行', 25, TEAL, weight=700)
    chain(s, [['新观测 + 新噪声', '固定当前条件'], ['多步速度积分', '生成动作块'], ['执行短片段', '获取新观测']], 395)
    s.line(920, 515, 920, 565, TEAL); s.line(920, 565, 90, 565, TEAL); s.arrow(90, 565, 90, 510, TEAL)
    s.footer('采样内部循环与真实机器人反馈循环分开；图省略数据归一化等工程步骤。'); s.save('pi0-training-inference')

    s = SVG('把相机里的目标放到机器人基座系', 'T 的上标表示目标系，下标表示源系；图以末端安装相机为例')
    chain(s, [['目标坐标', '相机系 C'], ['手眼关系', '末端系 E'], ['当前末端位姿', '基座系 B']], 185)
    s.text(500, 405, 'ᴮT_target = ᴮT_E · ᴱT_C · ᶜT_target', 32, anchor='middle')
    s.text(500, 475, '从右向左作用；每一步的相邻坐标系必须匹配', 25, anchor='middle')
    s.footer('外置相机的变换链不同；正文统一的是坐标系约定，而非固定安装方式。'); s.save('coordinate-chain')

    s = SVG('SLERP：单位四元数与较短球面弧', '低维球面截面，不把四元数当成三维位置', w=780)
    s.circle(290, 310, 170, 'none', '#d5dee9')
    for angle, label, color in [(-140, 'q₀', BLUE), (-50, 'q₁', TEAL), (130, '−q₁', PINK)]:
        a=math.radians(angle); x=290+170*math.cos(a); y=310+170*math.sin(a)
        s.circle(x,y,8,color); s.text(x+15,y+5,label,26,color)
    s.path([(290+170*math.cos(math.radians(a)),310+170*math.sin(math.radians(a))) for a in range(-140,-49)],TEAL,6)
    s.multiline(605,240,['q 与 −q：同一旋转','若点积为负，先变号','再沿较短球面弧插值'],23)
    s.footer('SLERP 中使用的夹角位于四维单位球面；图只呈现其二维截面直觉。'); s.save('quaternion-slerp')

    s = SVG('雅可比与奇异性：球如何变成椭球', '位置雅可比 J_v 把关节速度映射到末端线速度')
    s.circle(170,290,105,'#f4f7fb',BLUE); s.text(170,445,'单位关节速度球',23,anchor='middle')
    s.arrow(295,290,400,290); s.text(350,255,'J_v',26,anchor='middle')
    s.ellipse(535,290,115,72,-25,TEAL); s.text(535,445,'良好位形',23,anchor='middle')
    s.ellipse(830,290,115,9,-25,PINK); s.text(830,445,'接近奇异位形',23,anchor='middle')
    s.text(680,505,'小奇异值对应速度能力收缩的方向',25,anchor='middle')
    s.footer('二维截面示意；位置雅可比与包含旋转的完整任务雅可比应分别解释。'); s.save('jacobian-singularity')

    s = SVG('P、PI、PID：统一对象上的教学阶跃响应', 'G(s)=1/(s+1)；Kp=2，PI/PID 的 Ki=1，PID 的 Kd=0.2（对测量求导）')
    s.axes(80,445,820,290,'时间 t / s','输出 y')
    s.line(80,195,900,195,'#a5afbc',2,True); s.text(905,200,'1',20)
    curves=[]
    for kind,color in [('P',BLUE),('PI',TEAL),('PID',PINK)]:
        value,integ,points=0.0,0.0,[];dt=.002
        for i in range(5001):
            t=i*dt;err=1-value
            ki=0 if kind=='P' else 1; kd=.2 if kind=='PID' else 0
            deriv=(-value+2*err+ki*integ)/(1+kd)
            if i%10==0:points.append((80+80*t,445-250*value))
            value+=dt*deriv;integ+=dt*err
        s.path(points,color,4);curves.append((kind,color))
    for i,(kind,color) in enumerate(curves):s.line(180+i*240,500,220+i*240,500,color,4);s.text(235+i*240,508,kind,23,color)
    s.footer('数值积分 dt=0.002s；无饱和、无噪声。参数用于教学比较，不是普遍最优整定。'); s.save('pid-responses')

    s = SVG('MPC：优化一个窗口，只执行第一步', '每次收到新测量后，把优化问题向前滚动')
    s.axes(85,430,820,280,'离散时刻','预测状态 / 约束')
    for y in [195,355]:s.line(85,y,890,y,GOLD,2,True)
    s.text(840,180,'约束边界',22,GOLD)
    s.path([(100,380),(230,315),(360,270),(490,242),(620,225),(750,217),(880,214)],BLUE,4)
    for i in range(7):s.circle(100+i*130,[380,315,270,242,225,217,214][i],6,BLUE)
    s.rect(100,455,115,45,TEAL,'none',4)
    for i in range(1,6):s.rect(100+i*130,455,115,45,'#e7eef8',BLUE,4)
    s.text(155,528,'执行 u(k)',22,TEAL,anchor='middle');s.text(630,528,'其余计划在下一时刻重新优化',23,anchor='middle')
    s.footer('轨迹为定性示意；实际可行性与稳定性还取决于模型、约束及终端设计。'); s.save('mpc-horizon')

    s = SVG('多峰动作：均值可能不是可行策略', '定性绕行例子，不表示论文实验分布')
    s.circle(230,300,60,'#e7e9ef','#8b97a7')
    s.circle(230,470,10,INK);s.circle(230,135,10,TEAL)
    s.path([(230,470),(120,380),(125,220),(230,135)],BLUE,5)
    s.path([(230,470),(345,380),(340,220),(230,135)],PINK,5)
    s.line(230,460,230,150,GOLD,3,True)
    s.text(230,525,'左 / 右绕行与均值路径',23,anchor='middle')
    s.axes(550,430,380,245,'动作方向','密度（示意）')
    s.path([(550+i*3.5,430-170*(math.exp(-((i-25)/11)**2)+math.exp(-((i-78)/11)**2))) for i in range(101)],TEAL,4)
    s.line(730,430,730,240,GOLD,3,True);s.text(730,215,'均值',23,GOLD,anchor='middle')
    s.footer('多峰建模提供表达不同动作模式的能力，不自动保证碰撞约束满足。'); s.save('action-modes')

    s = SVG('观测、预测、执行：三个窗口', 'To：观测帧数；Tp：预测动作数；Ta：实际执行片段')
    s.line(80,230,925,230,'#8b97a7',3,arrow=True)
    for i in range(3):s.box(70+i*100,145,80,60,f'o₋{2-i}' if i<2 else 'o₀',BLUE)
    for i in range(6):s.box(385+i*90,280,75,65,f'a{i}',TEAL if i<2 else BLUE)
    s.text(200,115,'历史观测 To',24,BLUE,anchor='middle')
    s.text(650,395,'预测动作窗口 Tp',25,BLUE,anchor='middle')
    s.line(385,365,550,365,TEAL,7);s.text(465,445,'执行 Ta',24,TEAL,anchor='middle')
    s.arrow(550,465,780,465,TEAL);s.text(785,475,'新观测 → 重采样',22,TEAL)
    s.footer('窗口长度仅为说明相对关系的示例；动作时间单位由训练数据与控制接口决定。'); s.save('action-windows')

    s = SVG('Bellman Backup：奖励与后续价值分开', '给定策略下，对动作和状态转移求期望')
    s.circle(125,290,55,'#edf2fa',BLUE);s.text(125,300,'s',30,BLUE,'middle')
    s.box(340,150,230,90,['动作 a₁','概率 π(a₁|s)'],TEAL)
    s.box(340,355,230,90,['动作 a₂','概率 π(a₂|s)'],PINK)
    s.arrow(180,265,335,195);s.arrow(180,320,335,400)
    for y,c in [(195,TEAL),(400,PINK)]:
        s.box(735,y-45,225,90,['r + γV(s′)','按 P(s′|s,a) 加权'],c);s.arrow(570,y,730,y,c)
    s.text(500,515,'Vπ(s) = Eπ[r + γVπ(s′) | s]',29,anchor='middle')
    s.footer('策略评价求期望；Bellman 最优方程对动作取最大值，不混用二者。');s.save('bellman-backup')

    s = SVG('MC 与 TD(0)：等到什么时候更新？', '两条路径使用相同经历，但回归目标不同')
    for row,(label,color) in enumerate([('MC',BLUE),('TD(0)',TEAL)]):
        y=175+row*210;s.text(45,y+35,label,27,color,weight=700)
        for i,node in enumerate(['sₜ','sₜ₊₁','…','终止']):
            x=220+i*200;s.box(x,y,130,70,node,color)
            if i:s.arrow(x-60,y+35,x-8,y+35,color)
        s.text(300,y+120,'等待完整回报 Gₜ' if row==0 else '一步后：rₜ + γV(sₜ₊₁)',25,color)
    s.footer('终止状态的后续价值为零；截断与真实终止的处理需分别检查。');s.save('mc-td-timeline')

    s = SVG('针孔投影：相机沿 +X 平移', 'X 在图中竖直向上，Z 为向右的光轴；归一化像坐标 x=X/Z',w=780,h=620)
    s.line(110,300,690,300,BLUE,2,True);s.line(110,230,690,230,TEAL,2,True)
    s.circle(110,300,7,BLUE);s.circle(110,230,7,TEAL)
    s.arrow(110,300,110,238,PINK);s.text(40,260,'+vₓ',24,PINK)
    s.line(300,125,300,420,'#9aa9b9',3)
    s.circle(620,150,8,GOLD);s.text(575,125,'静止点 P',24,GOLD)
    s.line(110,300,620,150,BLUE,3);s.line(110,230,620,150,TEAL,3)
    s.circle(300,244,6,BLUE);s.circle(300,200,6,TEAL)
    s.line(325,244,325,300,BLUE,4);s.text(350,285,'原像高',22,BLUE)
    s.line(325,200,325,230,TEAL,4);s.text(350,212,'新像高',22,TEAL)
    s.text(110,355,'相机中心',22);s.text(645,335,'+Z 光轴',22)
    s.text(390,475,'x_new = (X − ΔX) / Z < X / Z',27,anchor='middle')
    s.text(390,520,'像点相对各自光轴的坐标减小',24,TEAL,anchor='middle')
    s.footer('理想平移截面，Z 不变；颜色区分移动前后的相机中心、光轴与投影射线。');s.save('camera-projection')

    s = SVG('从像素误差到机器人执行量', '速度表达坐标系与速度参考点必须同时说明')
    chain(s,[['图像误差 e','交互矩阵 L'],['相机速度 Vc','阻尼伪逆等控制律'],['机器人执行量','速度换系 + 运动学']],185)
    s.box(230,390,540,100,['相机 / 末端 / 基座：变换链一致','同一参考点换基 vs 改变速度参考点'],TEAL)
    s.footer('改变速度参考点时通常包含角速度 × 杠杆臂项；仅换基与换参考点不同。');s.save('servo-velocity-chain')

    s = SVG('HIL-SERL：价值学习与策略更新', '训练数据来自示范与在线交互；图省略双 Critic 等实现细节')
    s.box(40,145,265,100,['回放转移','(s,a,r,s′,done)'],BLUE)
    s.box(390,145,260,100,['Critic 目标','r + γ(1−done)V_target'],TEAL)
    s.box(735,145,225,100,['Critic 回归','Q(s,a) → 目标'],TEAL)
    s.arrow(305,195,385,195);s.arrow(650,195,730,195)
    s.box(390,365,350,110,['Actor 更新','熵正则目标：α logπ − Q'],PINK)
    s.arrow(845,250,740,395,TEAL);s.arrow(305,245,390,390,BLUE)
    s.text(145,410,'人的纠偏',24,GOLD);s.text(145,448,'改变经验分布',22,GOLD)
    s.footer('成功分类器提供奖励；人工接管片段参与回放学习，并非单独的在线模仿梯度。');s.save('hil-actor-critic')

    s = SVG('异步 Actor–Learner：执行不等于训练', '环境交互、经验传输和参数更新具有不同节奏')
    s.box(45,160,265,120,['Actor + 机器人','观测 → 策略 / 人工接管','执行 → 新经验'],BLUE)
    s.box(380,160,250,120,['双回放池','示范 + 在线交互','采样训练 batch'],GOLD)
    s.box(700,160,255,120,['Learner','更新 Critic / Actor','与 Actor 同步参数'],TEAL)
    s.arrow(310,220,375,220);s.arrow(630,220,695,220)
    s.line(825,285,825,410,TEAL);s.line(825,410,175,410,TEAL);s.arrow(175,410,175,290,TEAL)
    s.text(500,455,'参数同步：不要求每个控制周期都阻塞等待训练',25,TEAL,anchor='middle')
    s.footer('具体同步频率、队列延迟与控制频率由实现及设备预算决定。');s.save('hil-asynchronous')

    s = SVG('π0.5：六类数据，几种不同监督', '数据标签并非互斥物理小时数；HL 可能来自机器人数据再标注',h=680)
    for i,(label,desc) in enumerate([('MM','移动机器人 / 家庭'),('ME','固定机器人 / 家庭'),('CE','跨本体 / 实验室'),('HL','子任务与框标注'),('WD','图文 / VQA / 定位'),('VI','执行中高层指导')]):
        x=40+(i%3)*320;y=135+(i//3)*145
        s.box(x,y,280,105,[label,desc],COLORS[i])
    s.box(150,475,700,100,['共享视觉语言表征 → 对应任务的监督目标','离散序列交叉熵 / 连续动作 Flow Matching'],TEAL)
    s.footer('预训练与后训练选择不同数据组合；互联网图文样本不凭空拥有机器人动作标签。');s.save('pi05-data')

    s = SVG('FAST：动作块到离散序列的编码', '对时间维进行变换；动作序列的表示与动作执行区分开')
    chain(s,[['连续动作块','时间 × 动作维'],['DCT + 量化','压缩时间结构'],['序列编码','离散 Token']],175)
    s.box(205,380,590,100,['解码：离散 Token → 系数 → 逆变换','π0.5 的连续运行路径仍由 Action Expert 采样'],TEAL)
    s.footer('图为编码链概括；量化、词表及压缩规则以 FAST 实现为准。');s.save('fast-encoding')

    s = SVG('π0.5：可见性阻止跨动作标签泄漏', '蓝色可见；浅灰屏蔽；每个分组用 3 个示意 Token 表示',h=650)
    def allowed(r,c):
        a,b=r//3,c//3
        return b==0 or (a==1 and b==1 and c<=r) or (a==2 and b==2)
    matrix(s,130,160,9,allowed,36)
    for i,label in enumerate(['P 前缀','D 离散动作','F 连续动作']):
        s.text(184+i*108,140,label,19,anchor='middle');s.text(113,220+i*108,label,19,anchor='end')
    s.multiline(745,240,['行：Query，列：Key','前缀 P 不读取动作','D 只因果读取 D','F 不读取真实 D','F 块内双向可见'],23,gap=45)
    s.footer('离散目标使用移位输入；图中的自注意力对角线不等于读取当前位置待预测答案。');s.save('pi05-attention-mask')

    routes=yaml.safe_load((ROOT/'_data/roadmap.yml').read_text(encoding='utf-8'))['routes']
    s = SVG('机器人操作：七条互相关联的研究路线', '路线名称与站内 ROADMAP 共享数据；按研究问题选择阅读入口',h=850)
    s.box(320,125,360,85,['感知 · 决策 · 真实执行'],BLUE)
    s.line(500,215,500,668,'#a9b4c2',2)
    for i,route in enumerate(routes):
        x=45+(i%2)*485;y=265+(i//2)*120
        s.box(x,y,425,85,[f'{i+1:02d}  '+route['label']],route['color'])
        s.line(500,y+42,470 if i%2==0 else 530,y+42,route['color'],2)
    s.footer('各路线的代表论文、代码与首个实验建议见正文和 ROADMAP；此图不作性能排名。');s.save('research-routes')

    s = SVG('特征分箱：先聚合，再按顺序拼接', '中心保留细节；外围粗区域先池化为摘要，再参与描述子构造',h=790)
    s.text(45,145,'A · 邻域构造',26,BLUE,weight=700)
    for r in range(3):
        for c in range(3):
            s.rect(100+c*62,190+r*62,56,56,'#e9f2fa',BLUE,4)
            s.text(128+c*62,225+r*62,'X' if (r,c)==(1,1) else 'f',24,BLUE,'middle')
    for r in range(3):
        for c in range(3):
            if (r,c)==(1,1):continue
            s.rect(520+c*90,160+r*90,70,70,'#e8f3ef',TEAL,5)
            s.text(555+c*90,203+r*90,'Avg',19,TEAL,'middle')
    s.text(190,425,'中心细邻域：按序保留',23,BLUE,anchor='middle')
    s.text(645,465,'外围粗区域：每区先平均池化',23,TEAL,anchor='middle')
    s.text(45,530,'B · 构造描述子',26,BLUE,weight=700)
    s.box(65,570,870,100,['中心 + 每层 8 个邻域摘要 → 固定顺序拼接','Dβ = (1 + 8β)D；β=0 时仅保留中心'],TEAL)
    s.footer('第 1 层包含中心周围 8 个细摘要；更外层追加 8 个粗区域摘要，不能重复计入内部向量。');s.save('feature-binning-readable')

    # This motif is used only by the quiz cover, not counted as a teaching figure.
    s = SVG('大模型复习地图','先定位知识模块，再用问题检查理解')
    chain(s,[['模型结构','Norm / FFN'],['注意力','Mask / RoPE'],['训练与推理','数据 / 解码']],175)
    s.box(300,370,400,100,['分布式与系统设计','监控 / 时间预算 / 工程取舍'],TEAL)
    s.footer('文章为未完成的复习清单；尚未填写的答案明确标记待补充。');s.save('quiz-map')


def cover_sources():
    spec=importlib.util.spec_from_file_location('organization',ROOT/'tools/organize-content.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    domains=yaml.safe_load((ROOT/'_data/domains.yml').read_text(encoding='utf-8'))
    palette={d['name']:d['color'] for d in domains}
    out=ROOT/'assets/images/covers';out.mkdir(parents=True,exist_ok=True)
    for basename,topic,kind,title,subtitle,motif in module.POSTS:
        if not title:continue
        color=palette[topic];s=SVG('',w=1200,h=630);s.items=[]
        s.rect(0,0,1200,630,'#f7f9fc','none',0)
        s.rect(48,65,8,500,color,'none',4)
        s.text(86,115,'CODEJIE  /  RESEARCH NOTES',20,color,weight=700)
        s.text(86,185,topic,25,'#64748b')
        size=52 if len(title)>16 else 61
        # Keep all title text in the left 630 px; long English names wrap.
        title_lines=title.split(' Depth ') if title=='Prior Depth Anything' else [title]
        if len(title_lines)>1:title_lines=[title_lines[0],'Depth '+title_lines[1]]
        for i,line in enumerate(title_lines):s.text(86,285+i*70,line,size,weight=700)
        parts=subtitle.split(' · ')
        for i in range(0,len(parts),2):s.text(88,425+(i//2)*38,' · '.join(parts[i:i+2]),24,'#637084')
        raw=(OUT/f'{motif}.svg').read_text(encoding='utf-8')
        w,h=map(int,re.search(r'viewBox="0 0 (\d+) (\d+)"',raw).groups())
        body=raw[raw.index('<rect width="100%"'):raw.rindex('</svg>')]
        body=body.replace('<rect width="100%" height="100%"',f'<rect width="{w}" height="{h}"',1)
        # Crop only the decorative motif on the cover; the article uses the full figure.
        scale=min(430/w,300/h)
        s.rect(705,145,455,360,'#fff','#e0e6ee',14)
        s.add(f'<g transform="translate(710 175) scale({scale})">{body}</g>')
        s.text(88,565,'方法理解  /  公式推导  /  工程笔记',19,color)
        (out/f'{basename[11:]}-cover.svg').write_text(s.render(),encoding='utf-8')
    print('Built 14 editable cover SVG sources.')


def previews():
    root=ROOT/'assets/images/posts/visual-servoing'
    target=root/'previews';target.mkdir(parents=True,exist_ok=True)
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in ['.png','.jpg']:continue
        image=Image.open(p).convert('RGB')
        image.thumbnail((1280,1280))
        quality=90
        dest=target/(p.stem+'.webp')
        image.save(dest,'WEBP',quality=quality,method=6)
        while dest.stat().st_size>390*1024 and quality>65:
            quality-=5;image.save(dest,'WEBP',quality=quality,method=6)
    print('Built 12 compressed previews; original files retained.')


if __name__=='__main__':
    build();cover_sources();previews()
    print('Built 32 teaching figures, a feature-binning redraw and a cover-only quiz motif.')
