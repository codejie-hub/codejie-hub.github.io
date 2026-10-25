---
title: "【具身学习·03】接触与阻抗控制：从弹簧阻尼模型到机器人柔顺操作"
date: 2026-10-24 00:03:00 +0800
categories: [运动学与控制]
content_type: tutorial
tags: [embodied-ai, impedance-control, admittance-control, force-control, contact, passivity]
description: 从一维接触动力学推导阻抗和导纳，计算刚度与环境误差如何决定接触力，解释被动性、采样延迟、混合力位控制及学习策略与底层控制器的接口，并提供可运行的接触仿真。
image:
  path: /assets/images/covers/embodied-contact-control-cover.jpg
  alt: 机器人通过虚拟弹簧和阻尼与环境接触的示意
math: true
---

> **位置控制回答“到哪里去”，接触控制还必须回答“碰到东西以后，允许怎样偏离目标”。** 阻抗控制的核心是设计运动与外力之间的动态关系。

{% include embodied-series.html %}

拿起杯子之前，末端应准确靠近；手指接触杯身后，继续追踪一个穿过杯子的刚性位置目标就不合理了。插接、擦拭、拧盖和装配都需要处理这样的阶段变化。本篇从一维模型推导，不要求先掌握完整刚体动力学；建议先读 [PID 与 MPC]({% post_url 2026-09-18-pid-mpc-control %}) 以及 [机器人运动学]({% post_url 2026-04-17-robot-kinematics %})。

文中的参数和仿真用于理解公式。真实系统还受到驱动器、传感器、机构柔性与接口带宽的限制，具体参数需要结合相应系统辨识。

## 1. 为什么刚性位置跟踪在接触中会产生矛盾

把机器人末端简化为质量 $$m$$，沿 $$x$$ 轴运动。设重力已补偿，控制器施加力 $$u$$，环境施加在机器人上的力为 $$F_{\mathrm{ext}}$$：

$$
m\ddot x=u+F_{\mathrm{ext}}.
$$

若环境是一面位于 $$x=0$$ 的墙，正方向指向墙内部，用单边弹簧近似：

$$
F_{\mathrm{ext}}=-K_e\max(x,0).
$$

目标 $$x_d>0$$ 表示控制器希望末端进入墙内。理想刚性墙不允许这个位置，理想无限刚度位置控制又不允许位置误差；两个要求不能同时满足。真实系统通过弹性形变、力矩饱和、跟踪误差甚至振动来“解决”矛盾。

这里的 $$F_{\mathrm{ext}}$$ 是墙对机器人施加的力，机器人对墙的力符号相反。推导时先明确作用对象，可以避免后续力反馈符号接反。

## 2. 阻抗：把误差动力学设计成质量、弹簧与阻尼

### 2.1 目标关系与控制律不是同一个层次

令 $$e=x-x_d$$。常见目标关系为：

$$
\boxed{M_d\ddot e+B_d\dot e+K_d e=F_{\mathrm{ext}}.}
$$

$$M_d,B_d,K_d$$ 分别代表希望末端表现出的等效质量、阻尼和刚度。它不是对所有机器人的直接力矩命令；实现这个关系，需要结合实际动力学和控制接口。

先看 $$M_d=m$$ 的简化情况。选择：

$$
u=m\ddot x_d-B_d(\dot x-\dot x_d)-K_d(x-x_d),
$$

代入真实动力学，就得到目标误差关系。若改变目标惯量 $$M_d\ne m$$，理想模型下可写：

$$
u=m\ddot x_d+\frac m{M_d}(-B_d\dot e-K_d e)
+\left(\frac m{M_d}-1\right)F_{\mathrm{ext}}.
$$

这说明惯量整形通常引入额外的模型与力反馈要求，不能简单把弹簧刚度改小就称为完整的惯量整形。实际机器人还要处理耦合和带宽限制。阻抗控制的原始思想见 [Hogan 的论文][hogan]，机器人控制接口可参考 [MIT 操作控制讲义][force]。

{% include figure.html id="fig-embodied-impedance" src="/assets/images/figures/embodied-impedance.svg" alt="末端质量通过虚拟弹簧阻尼连接目标位置，接触环境后受到反向力" caption="一维阻抗模型：目标、末端与环境具有不同位置。环境力作用在末端上；目标刚度决定末端偏离参考位置时控制器如何响应。" size="full" width="960" height="480" source="本文绘制" %}

### 2.2 三个参数分别控制什么

无外力、参考固定时：

$$
M_d\ddot e+B_d\dot e+K_d e=0.
$$

自然频率和阻尼比分别为：

$$
\omega_n=\sqrt{K_d/M_d},\qquad
\zeta=\frac{B_d}{2\sqrt{M_dK_d}}.
$$

增大刚度会减小同一静态外力引起的偏移，但也提高动态响应频率。只提高刚度、不调整阻尼和采样，可能让系统更容易振荡。$$\zeta=1$$ 是这个理想单自由度自由空间模型的临界阻尼条件，不是接触所有材料时都临界阻尼。

### 2.3 “阻抗”的频域含义

固定参考、零初始条件下：

$$
\frac{F_{\mathrm{ext}}(s)}{E(s)}=M_ds^2+B_ds+K_d.
$$

机械阻抗通常定义为力与速度的比值，因此：

$$
Z(s)=\frac{F_{\mathrm{ext}}(s)}{sE(s)}
=M_ds+B_d+\frac{K_d}s.
$$

把“力/位移”和“力/速度”都简称阻抗时容易混淆单位；阅读论文时应先查看作者的定义。

## 3. 接触力由机器人刚度和环境刚度共同决定

固定目标 $$x_d>0$$，接触平衡点满足：

$$
K_d(x_d-x)=K_e x.
$$

从而：

$$
x^*=\frac{K_d}{K_d+K_e}x_d,\qquad
F_c^*=\frac{K_dK_e}{K_d+K_e}x_d,
$$

其中 $$F_c^*=-F_{\mathrm{ext}}$$ 为正的压入力大小。系数正是两根串联弹簧的等效刚度。

取 $$K_e=2000\,\mathrm{N/m}$$、$$x_d=0.01\,\mathrm m$$：

| 机器人刚度 | 稳态压入量 | 稳态接触力 |
| --- | --- | --- |
| $$K_d=200\,\mathrm{N/m}$$ | 0.909 mm | 1.818 N |
| $$K_d=800\,\mathrm{N/m}$$ | 2.857 mm | 5.714 N |

目标位置相同，力却不同；环境刚度变了，力也会改变。因此单纯设置一个“墙后位置”并不是精确力控制。

在硬环境极限 $$K_e\to\infty$$ 下，$$F_c^*\to K_dx_d$$。若实际墙面位置估计误差为 $$\delta x$$，接触力误差近似为 $$K_d\delta x$$。这也是几何标定误差与控制刚度之间的直接联系：同样 2 mm 的误差，在不同刚度下会产生不同结果。

## 4. 导纳：从测得的力生成运动参考

如果机器人开放的是高性能位置接口，而不是直接力矩接口，可以在外层计算一个柔顺位置参考。设名义轨迹为 $$x_0$$，偏移 $$\delta x$$ 满足：

$$
M_a\delta\ddot x+B_a\delta\dot x+K_a\delta x
=F_{\mathrm{ext}}-F_{\mathrm{ref}},\qquad
x_{\mathrm{cmd}}=x_0+\delta x.
$$

这里 $$F_{\mathrm{ref}}$$ 也使用“环境对机器人”的符号。外层把力误差变成运动，内层再跟踪 $$x_{\mathrm{cmd}}$$。如果内层跟踪不够快，实际末端不会严格遵循外层虚拟模型。

在速度/力定义下，导纳是阻抗的倒数；工程上说“导纳控制”，通常指这种**力输入、运动参考输出**的控制结构。不能据此断言某个控制器一定更稳定或更适合某种任务，效果还取决于硬件与环境。

{% include figure.html id="fig-embodied-control-interfaces" src="/assets/images/figures/embodied-control-interfaces.svg" alt="阻抗控制的运动误差到力矩路径与导纳控制的测力到运动参考路径对照" caption="两类实现接口。阻抗目标关系需由底层动力学实现；导纳外环生成参考后，还要经过位置伺服。箭头表示计算依赖，不是两者的性能排名。" size="full" width="960" height="500" source="本文绘制" %}

### 4.1 有弹簧项就不一定消除稳态力误差

平衡时 $$K_a\delta x=F_{\mathrm{ext}}-F_{\mathrm{ref}}$$。只要偏移非零，力误差就可以非零。若需要精确跟踪接触力，要进一步设计外环和积分机制，不能仅靠把导纳方程写出来。

去掉刚度项可以允许持续位移，但也可能因传感器偏置产生缓慢漂移。需要结合任务阶段定义接触、自由空间、回撤和参考重置的逻辑。

## 5. 从一维推广到机器人：坐标和功率必须一致

机器人动力学常写为：

$$
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=\tau+J(q)^\top F_{\mathrm{ext}}.
$$

在一致的速度/wrench 排列与表达系下，有功率恒等式：

$$
\tau^\top\dot q=F^\top J\dot q=F^\top v.
$$

这解释了为什么末端力通过 $$J^\top$$ 映射到关节力矩。但写下 $$\tau=J^\top(-Ke-D\dot e)+g$$，并不自动实现任意设定的末端惯量；真实惯量、耦合项和零空间运动仍然存在。

六维误差的前三维与后三维单位不同。位置误差可用米，姿态误差可用局部旋转向量；旋转刚度的单位则是 $$\mathrm{N\,m/rad}$$。姿态误差、角速度、外部力矩必须在兼容坐标系中表达，旋转向量在大角度和接近 $$\pi$$ 的区域还需要注意分支问题。

在接触面坐标系中设置各向异性刚度往往更有意义：沿表面切向需要较好的轨迹跟踪，法向允许偏移以调节接触力。换到基座系后，刚度矩阵也应进行相应坐标变换，不能只旋转目标位置。

## 6. 混合力位控制为何按方向分配目标

设已知接触面法向为单位向量 $$n$$，平移空间投影为：

$$
P_n=nn^\top,\qquad P_t=I-P_n.
$$

擦拭桌面时，可以在切向控制轨迹，在法向控制接触力。一个教学形式为：

$$
F_{\mathrm{cmd}}
=P_tF_{\mathrm{motion}}+P_nF_{\mathrm{force}}.
$$

这并不是在同一个刚性受约束方向同时精确指定任意位置和任意力。法向位置已受到接触几何约束，力目标是在该约束下分配作用；切向轨迹则描述允许运动的方向。

上式仅用于解释方向分工。真实六维混合控制还应依据接触约束雅可比和运动/力的对偶关系构造投影。若法向估计错了，所谓“切向移动”可能实际包含持续压入，这会同时影响轨迹和力。

## 7. 被动性、参考变化与离散实现

### 7.1 一个可以亲自推导的储能函数

对固定参考、常量正定参数，定义：

$$
V=\frac12M_d\dot e^2+\frac12K_de^2.
$$

沿目标动力学求导：

$$
\dot V=\dot e(M_d\ddot e+K_de)
=F_{\mathrm{ext}}\dot e-B_d\dot e^2.
$$

因此输入端口提供的能量一部分储存，一部分被阻尼耗散。这是理想连续模型的被动性解释。它没有包含采样、时延、力估计误差、执行器饱和，也没有证明任意机器人实现都被动。

若刚度随时间改变，会出现额外项 $$\tfrac12\dot K_de^2$$；若参考运动，参考本身也可向系统输入能量。学习策略动态调节刚度时，不能只检查每一时刻 $$K_d>0$$ 就认为整个过程具有同样的被动性保证。

### 7.2 正参数也可能产生数值不稳定

写成状态空间 $$\dot z=Az$$，显式 Euler 离散化为：

$$
z_{k+1}=(I+\Delta t A)z_k.
$$

连续模型稳定需要 $$A$$ 的特征值实部为负；离散 Euler 还要求 $$I+\Delta t A$$ 的谱半径小于 1。提高刚度会改变特征值，因此原来合适的步长可能不再合适。

接触时等效刚度通常包含 $$K_e$$。仅按自由空间响应选阻尼和步长，未必适合硬接触。滤波降低力噪声但引入相位延迟，延迟又会影响闭环，因此“滤得更平滑”不等于“控制得更稳定”。

## 8. 可运行的一维接触仿真

下面是教学数值实验：实际质量 1 kg、环境单边弹簧 2000 N/m、目标在墙内 1 cm。使用半隐式 Euler，分别比较两种机器人刚度；阻尼按接触区线性模型的临界值设置。没有噪声、延迟、摩擦或驱动器限幅。

```python
import math

def simulate(k_robot, dt=1e-4, duration=3.0):
    mass, k_wall, target = 1.0, 2000.0, 0.01
    damping = 2.0 * math.sqrt(mass * (k_robot + k_wall))
    x, velocity = -0.02, 0.0
    max_force = 0.0
    for _ in range(round(duration / dt)):
        external = -k_wall * max(x, 0.0)
        control = k_robot * (target - x) - damping * velocity
        acceleration = (control + external) / mass
        velocity += dt * acceleration
        x += dt * velocity
        max_force = max(max_force, k_wall * max(x, 0.0))
    analytic = k_robot * k_wall * target / (k_robot + k_wall)
    return k_wall * max(x, 0.0), analytic, max_force

for stiffness in (200.0, 800.0):
    final, equilibrium, peak = simulate(stiffness)
    print(stiffness, round(final, 3), round(equilibrium, 3), round(peak, 3))
    assert abs(final - equilibrium) < 0.01
```

首先比较数值终值与第 3 节解析解；再把步长减半，检查结论是否稳定。进一步练习可固定刚度、增加一个采样周期的测量延迟，观察峰值和振荡如何变化。不能将这里的接触峰值解释为某款机器人的实验性能。

## 9. 学习策略应该控制哪一层

一个 VLA 或模仿学习策略可以输出末端参考、动作块、夹爪命令，甚至刚度参数。底层控制器则把这些参考转化为高速反馈。两者的职责需要写进数据协议：

| 上层输出 | 底层需要知道 | 常见混淆 |
| --- | --- | --- |
| 位姿目标 | 坐标系、插值时间、刚度 | 把目标位姿当成真实轨迹 |
| 位置增量 | 参考时刻、表达系、每步周期 | 不同频率下仍使用同一增量 |
| 接触力目标 | 作用对象、坐标系、接触阶段 | 力的符号或零点错误 |
| 刚度参数 | 单位、范围、变化率、耦合 | 正值就被当成稳定保证 |

训练数据由低刚度控制器产生，部署却换成高刚度控制器，即使策略输出完全相同，物体也可能受到完全不同的力。这是具身策略的一个重要特性：**动作含义由模型输出和执行接口共同决定。**

## 10. 复习题与参考答案

**问题一：减小刚度是否一定使机器人更容易完成插接？** 不一定。柔顺能吸收部分几何误差，也可能导致对准能力下降，需结合任务方向、阻尼和接触模型。

**问题二：为什么导纳控制中的测力偏置会导致位置偏移？** 外环把偏置当成真实外力；有刚度时产生静态偏移，无刚度时可能持续漂移。

**问题三：阻抗目标方程成立，是否等于真实系统稳定？** 不等于。目标关系和实际实现之间还隔着模型、采样、延迟与执行器动力学。

**问题四：同一动作数据在不同底层刚度下还能直接混合训练吗？** 应显式记录控制模式或统一动作语义，否则相同标签可能对应不同物理结果。

下一篇讨论 [世界模型与规划]({% post_url 2026-10-24-embodied-world-models %})：机器人如何预测这些动作对环境造成的后果，并在执行前比较不同选择。

## 参考资料

- [Hogan. Impedance Control: An Approach to Manipulation. ACC 1984.][hogan] — 阻抗控制的原始思想；本篇的简化算例另行构造。
- [Tedrake. Robotic Manipulation — Manipulator Control.][force] — 机器人动力学、间接力控制与控制接口。
- [Lynch and Park. Modern Robotics，官方教材与课程入口.][mr] — 动力学、wrench、控制与接触约束。

[hogan]: https://newmanlab.mit.edu/wp-content/uploads/2017/06/1984-impedance-control-an-approach-to-manipulation.pdf
[force]: https://manipulation.csail.mit.edu/force.html
[mr]: https://modernrobotics.northwestern.edu/
