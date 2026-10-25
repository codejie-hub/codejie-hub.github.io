#!/usr/bin/env python3
"""Editable, deterministic diagrams for the five embodied-learning tutorials.

Run: python tools/build-embodied-figures.py
Render the five JPEG covers with browser-audit.mjs series-assets afterwards.
Existing images are never moved or removed.
"""
from pathlib import Path
import importlib.util
import math

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('drawing', ROOT / 'tools/build-illustrations.py')
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)
SVG = drawing.SVG
BLUE, TEAL, GOLD, PINK, INK = drawing.BLUE, drawing.TEAL, drawing.GOLD, drawing.PINK, drawing.INK
PURPLE = '#8064a8'


def figure(title, subtitle, height=500):
    return SVG(title, subtitle, 960, height)


def build():
    s = figure('模仿学习：训练分布与执行分布', '正常示范提供主路径；偏离状态上的标签提供恢复方向。')
    for left, title in [(40, '只有专家示范'), (510, '访问偏离状态，再查询专家')]:
        s.rect(left, 125, 410, 285)
        s.text(left + 205, 160, title, 23, anchor='middle', weight=700)
        s.line(left+40, 355, left+370, 355, '#c5cfdc', arrow=True)
        s.text(left+35, 390, '任务进度', 19, color='#637084')
        s.line(left+40, 265, left+365, 265, BLUE, 4)
        for x in range(left+60, left+350, 48): s.circle(x, 265, 5, BLUE)
        s.path([(left+40,265),(left+140,267),(left+210,300),(left+280,330),(left+360,338)], PINK, 4, dash=True)
        if left == 510:
            for x,y in [(210,300),(280,330),(360,338)]:
                s.circle(left+x,y,6,PINK);s.arrow(left+x,y-8,left+x,274,TEAL)
        else: s.text(left+195, 220, '未被数据约束的偏离', 20, PINK, 'middle')
    s.footer('蓝：专家路径  ·  粉：学习者偏离  ·  绿：专家恢复标签  |  概念示意');s.save('embodied-distribution')

    s = figure('动作块：预测长度与反馈间隔分开看', 'H = 4，K = 2；方块是未来动作索引，不代表推理内部步骤。', 460)
    s.arrow(210,140,890,140,'#7c899a');s.text(880,115,'真实时间 →',20,anchor='end')
    s.text(40,210,'观测与推理',23,weight=700)
    s.circle(230,200,8,TEAL);s.text(230,240,'oₜ',23,TEAL,'middle')
    s.rect(260,178,170,45,'#eef2fa',BLUE);s.text(345,207,'推理 / 通信',21,BLUE,'middle')
    s.text(40,300,'本轮预测',23,weight=700)
    for i in range(4):
        x=455+i*105
        s.rect(x,268,94,58,TEAL if i<2 else '#f1f4f8','none',5)
        s.text(x+47,304,f'a{i}',25,'white' if i<2 else INK,'middle')
    s.text(552,365,'执行前两步',23,TEAL,'middle')
    s.text(765,365,'后两步可被重规划替换',21,'#637084','middle')
    s.arrow(660,250,660,185,PINK);s.text(680,205,'新的反馈时刻',20,PINK)
    s.footer('H：预测范围  ·  K：执行前缀  ·  延迟 L：图像采样至命令生效的时间');s.save('embodied-action-timing')

    s = figure('抓取坐标链：从像素到法兰', 'Tᴮ꜀ 将相机系坐标变换到基座系；所有位姿的方向约定保持一致。')
    labels=[['像素 + 深度','(u, v, z)'],['相机点云','pᶜ = z K⁻¹ ũ'],['抓取候选','Tᶜɢ + 开度'],['基座目标','Tᴮɢ = Tᴮ꜀ Tᶜɢ']]
    for i,l in enumerate(labels):
        x=35+235*i;s.box(x,145,200,105,l,PURPLE if i<3 else BLUE)
        if i:s.arrow(x-30,197,x-8,197,PURPLE)
    s.box(310,310,360,80,['法兰目标：Tᴮꜰ = Tᴮɢ (Tꜰɢ)⁻¹'],TEAL)
    s.line(840,255,840,350,BLUE);s.arrow(840,350,680,350,BLUE)
    s.multiline(145,325,['TCP 与法兰','可能不是同一点'],21,color='#637084',gap=32)
    s.footer('先检查单位、曝光时间与外参；再检查夹爪轴序和工具中心偏置。');s.save('embodied-grasp-chain')

    s = figure('夹持受力：摩擦锥与空间力矩', '左：理想平面受力。右：两个点接触无法抵抗绕连线的扭矩。')
    s.rect(40,120,430,290);s.rect(490,120,430,290)
    s.rect(185,195,140,125,'#eef0fa',PURPLE,15)
    s.rect(145,210,25,95,'#d8dfeb','none',4);s.rect(340,210,25,95,'#d8dfeb','none',4)
    s.add(f'<path d="M170 255 L235 220 L235 290 Z" fill="{TEAL}" opacity=".18"/>')
    s.add(f'<path d="M340 255 L275 220 L275 290 Z" fill="{TEAL}" opacity=".18"/>')
    s.arrow(95,255,175,255,BLUE);s.text(95,232,'N',22,BLUE)
    s.arrow(415,255,335,255,BLUE);s.text(410,232,'N',22,BLUE)
    s.arrow(255,310,255,370,PINK);s.text(278,360,'mg',23,PINK)
    s.text(255,160,'|fₜ| ≤ μN',25,TEAL,'middle')
    s.ellipse(705,245,105,70,0,PURPLE)
    s.circle(600,245,8,TEAL);s.circle(810,245,8,TEAL)
    s.line(545,245,865,245,BLUE,3,dash=True)
    s.add(f'<path d="M720 187 C765 162 775 321 720 300" fill="none" stroke="{PINK}" stroke-width="4" marker-end="url(#arrow)"/>')
    s.text(705,365,'绕接触连线的扭矩：缺少约束',21,PINK,'middle')
    s.footer('点接触与软指面接触模型不同；几何可行还应结合载荷、力矩与执行器上限。');s.save('embodied-grasp-contact')

    s = figure('阻抗：允许外力引起可控的运动偏移', '一维教学模型 · 固定参考 · 机器人质量 M · 环境对机器人的力 Fext',480)
    s.rect(95,160,15,190,'#dce4ef','none',1);s.text(102,125,'参考 xᵈ',24,anchor='middle')
    pts=[(110,205),(140,205)]+[(150+i*22,190 if i%2==0 else 220) for i in range(9)]+[(350,205),(395,205)]
    s.path(pts,BLUE,3);s.text(245,170,'虚拟刚度 K',23,BLUE,'middle')
    s.line(110,290,215,290,TEAL);s.line(215,270,215,310,TEAL)
    s.path([(215,270),(285,270),(285,310),(215,310)],TEAL,3)
    s.line(245,280,245,300,TEAL);s.line(245,290,395,290,TEAL)
    s.text(255,350,'虚拟阻尼 B',23,TEAL,'middle')
    s.rect(400,185,120,130,'#edf2f9',BLUE,12);s.text(460,260,'M',36,BLUE,'middle')
    s.text(460,350,'末端 x',23,anchor='middle')
    s.rect(740,155,120,185,'#f0ebe6',GOLD,4)
    for i in range(7): s.line(745+i*16,160,740+i*16,330,'#d8c6ad',1)
    s.text(800,125,'环境',24,anchor='middle')
    s.arrow(720,250,545,250,PINK);s.text(625,220,'Fext',26,PINK,'middle')
    s.arrow(570,365,850,365,'#7c899a');s.text(850,397,'+x',22,anchor='end')
    s.footer('目标关系：M ë + B ė + K e = Fext；e = x − xᵈ。控制律还取决于真实动力学。');s.save('embodied-impedance')

    s = figure('阻抗与导纳：看清控制接口', '两种路径都需要处理反馈、采样和真实机器人动力学。')
    for y,title,color,labels in [(150,'阻抗实现',BLUE,[['运动参考','误差 e'],['动力学 / 反馈','计算力或力矩'],['机器人','产生运动']]),(315,'导纳实现',TEAL,[['测量外力','力误差'],['虚拟动态','运动参考'],['位置内环','真实机器人']])]:
        s.text(40,y+45,title,25,color,weight=700)
        for i,l in enumerate(labels):
            x=210+240*i;s.box(x,y,205,90,l,color)
            if i:s.arrow(x-30,y+45,x-8,y+45,color)
    s.footer('导纳外环输出的是参考，实际运动还由位置内环决定；参数单位与符号需一致。');s.save('embodied-control-interfaces')

    s = figure('潜状态：真实观测更新，先验预测未来', '确定性记忆 h 与随机潜变量 z 联合表示历史；此图省略解码器细节。',520)
    s.text(40,145,'观测阶段',24,TEAL,weight=700)
    s.box(210,110,185,80,['历史 + 动作','更新 hₜ'],BLUE)
    s.box(470,110,185,80,['后验 q','zₜ | hₜ, oₜ'],TEAL)
    s.box(730,110,185,80,['表示 (hₜ,zₜ)','评估 / 行动'],BLUE)
    s.arrow(400,150,460,150);s.arrow(660,150,720,150)
    s.box(475,218,175,55,'真实观测 oₜ',TEAL);s.arrow(562,214,562,197,TEAL)
    s.line(40,295,920,295,'#d7dfe9',1)
    s.text(40,355,'想象阶段',24,PURPLE,weight=700)
    for i,l in enumerate([['当前潜状态','+ 候选动作'],['转移先验 p','采样下个 z'],['预测后果','奖励 / 继续']]):
        x=210+260*i;s.box(x,320,185,90,l,PURPLE)
        if i:s.arrow(x-65,365,x-10,365,PURPLE)
    s.text(565,445,'未来观测尚未发生，不能用未来图像纠正模型',22,PINK,'middle')
    s.footer('想象能力需用多步开环和任务相关误差评估，不能只看后验重建质量。');s.save('embodied-latent-model')

    s = figure('滚动规划：每次只兑现短前缀', '在模型中比较候选；在真实世界执行后，用新观测修正状态。',480)
    labels=[['观测更新','当前状态'],['候选序列','模型评分'],['选择方案','执行第一步'],['新观测','重新规划']]
    for i,l in enumerate(labels):
        x=35+235*i;s.box(x,135,200,95,l,[BLUE,PURPLE,TEAL,BLUE][i])
        if i:s.arrow(x-30,182,x-8,182)
    s.axes(100,355,770,80,'未来步骤','')
    s.path([(120,335),(260,300),(400,290),(540,270),(680,285),(820,265)],PURPLE,3,dash=True)
    s.path([(120,335),(260,315),(400,265),(540,270),(680,250),(820,260)],PINK,3,dash=True)
    s.line(120,335,260,315,TEAL,6);s.circle(260,315,7,TEAL)
    s.text(420,335,'真实执行至此后重新估计',20,TEAL)
    s.footer('虚线：尚未执行的候选未来  ·  绿线：短前缀执行  |  概念示意，非实验轨迹');s.save('embodied-planning-loop')

    s = figure('离线自举：没有样本约束的高 Q 值', '教学 MDP · 即时奖励 r = 0 · 折扣 γ = 0.9',480)
    s.box(45,190,175,100,['状态 s₀','数据动作 a₀'],BLUE)
    s.box(330,190,150,100,['状态 s₁','需要后续价值'],BLUE)
    s.arrow(225,240,320,240,BLUE)
    s.box(635,120,275,100,['已见动作 A','Q = 1'],TEAL)
    s.box(635,290,275,100,['未见动作 B','网络猜测 Q = 10'],PINK)
    s.arrow(485,225,625,170,TEAL);s.arrow(485,260,625,340,PINK)
    s.text(220,365,'max 自举目标 = 9',25,PINK,'middle',700)
    s.text(220,400,'数据支持的后续值：0.9',21,TEAL,'middle')
    s.footer('“未见”并不表示真实回报为零；问题是训练数据无法校正这个虚高估计。');s.save('embodied-offline-support')

    s = figure('Expectile 不等于相同百分位的 Quantile', '教学分布：P(X=0)=P(X=2)=0.5；τ=0.7 时最优不对称平方拟合值为 1.4。',440)
    s.arrow(110,250,855,250,'#7c899a')
    for x,label,col,y in [(150,'X = 0',BLUE,300),(450,'均值 1.0',TEAL,205),(570,'Expectile 1.4',PURPLE,350),(750,'70% 分位数 2.0',PINK,180)]:
        s.circle(x,250,8,col);s.line(x,250,x,y+(-28 if y>250 else 15),col,2)
        s.text(x,y,label,23,col,'middle',700)
    s.text(750,300,'X = 2',23,BLUE,'middle')
    s.footer('一阶条件：(1−τ)v − τ(2−v) = 0，所以 v = 2τ；此例取 v = 1.4。');s.save('embodied-expectile')


def covers():
    entries=[
        ('imitation-learning','模仿学习','数据、动作与闭环','机器人学习与 VLA',BLUE),
        ('grasp-geometry','三维抓取几何','从深度到可执行位姿','三维视觉',PURPLE),
        ('contact-control','接触与阻抗控制','运动、力与柔顺响应','运动学与控制',GOLD),
        ('world-models','世界模型与规划','估计状态，想象未来','机器人学习与 VLA',BLUE),
        ('offline-rl','离线强化学习','数据支持与策略改进','强化学习','#6f8844'),
    ]
    out=ROOT/'assets/images/covers';out.mkdir(parents=True,exist_ok=True)
    for i,(slug,title,subtitle,topic,color) in enumerate(entries,1):
        s=SVG(title,subtitle,1200,630);s.items=[]
        s.rect(0,0,1200,630,'#f5f7fb','none',0)
        s.rect(0,0,22,630,color,'none',0)
        s.text(80,95,'CODEJIE  /  EMBODIED INTELLIGENCE',22,color,weight=700)
        s.text(80,174,f'具身学习系列  {i:02}  /  {topic}',25,'#637084')
        s.text(76,290,title,58,INK,weight=700)
        s.text(80,360,subtitle,32,'#637084')
        s.line(80,425,620,425,'#d7dfe9',2)
        s.text(80,480,'原理推导  ·  数值算例  ·  学习实验',25,color)
        s.text(80,566,'LEARN  /  REASON  /  BUILD',19,'#748196')
        s.rect(735,160,395,340,'white','#dce3ee',24)
        s.text(1090,215,f'{i:02}',32,color,'end',700)
        if i==1:
            for y,c in [(300,color),(370,PINK)]:
                pts=[(780,335),(845,335),(910,y),(975,y),(1060,y)]
                s.path(pts,c,5,dash=y==370)
            for x in [910,975,1060]:s.arrow(x,365,x,310,TEAL)
            s.text(930,450,'DEMONSTRATE → RECOVER',18,color,'middle')
        elif i==2:
            for j in range(55):
                a=j*2.4;r=20+math.sqrt(j)*7
                s.circle(925+r*math.cos(a),340+r*.7*math.sin(a),3,PURPLE)
            s.path([(840,285),(825,285),(825,400),(855,400)],color,9)
            s.path([(1010,285),(1025,285),(1025,400),(995,400)],color,9)
            s.arrow(930,335,995,310,TEAL);s.arrow(930,335,930,260,PINK)
            s.text(930,450,'PIXEL → POINT → POSE',20,color,'middle')
        elif i==3:
            s.path([(780,330),(800,330)]+[(810+j*15,315 if j%2==0 else 345) for j in range(8)]+[(930,330)],color,4)
            s.rect(930,285,70,90,'#f6f0e8',color,12)
            s.arrow(1080,330,1010,330,PINK)
            s.text(930,450,'M ë + B ė + K e = F',23,color,'middle')
        elif i==4:
            s.circle(800,335,15,color)
            for end,c in [(260,TEAL),(335,PURPLE),(405,PINK)]:
                s.path([(820,335),(880,335),(960,end),(1070,end)],c,4,dash=True)
                s.circle(1070,end,9,c)
            s.text(930,455,'OBSERVE → IMAGINE → ACT',18,color,'middle')
        else:
            for j,(height,c) in enumerate([(65,color),(120,color),(165,PINK)]):
                s.rect(795+j*90,410-height,55,height,c,'none',6)
            s.line(785,410,1085,410,'#b6c2d2',2)
            s.text(930,455,'SUPPORT → VALUE → POLICY',18,color,'middle')
        (out/f'embodied-{slug}-cover.svg').write_text(s.render(),encoding='utf-8')


if __name__=='__main__':
    build();covers()
    print('Built 10 teaching SVGs and 5 cover SVG sources for the embodied series.')
