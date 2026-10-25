---
title: "【具身学习·04】世界模型与规划：从部分可观测状态到潜空间中的行动选择"
date: 2026-10-24 00:04:00 +0800
categories: [机器人学习与 VLA]
content_type: tutorial
tags: [embodied-ai, world-model, pomdp, model-based-rl, latent-dynamics, mpc, planning]
description: 从 POMDP 的信念更新出发，建立动作条件世界模型，推导潜状态训练目标与误差累积，解释 RSSM、CEM-MPC、模型不确定性和想象式策略学习，并用可运行的一维规划实验验证闭环思想。
image:
  path: /assets/images/covers/embodied-world-models-cover.jpg
  alt: 真实观测更新潜状态，模型想象多种未来并选择动作的闭环
math: true
---

> **控制用世界模型的价值，在于比较“如果执行不同动作，会发生什么”。** 预测画面足够逼真只是一个可能的训练目标，不能单独说明模型适合规划。

{% include embodied-series.html %}

机器人要把杯子推到托盘里。直接策略输出一个动作；世界模型则允许它先比较轻推、重推、换个接触点等候选的后果。本篇从这一区别出发，建立状态估计、模型学习和行动选择的连接。先修内容是 [强化学习基础]({% post_url 2026-09-19-rl-base %}) 中的 MDP，以及 [MPC]({% post_url 2026-09-18-pid-mpc-control %}) 的滚动优化思想。

这是基础教程，不按“最新模型排行榜”组织。PlaNet、PETS、Dreamer 作为不同方法设计的例子；公式采用统一教学记号，不逐行复制任何一个实现。

## 1. 策略、价值函数和世界模型分别回答什么

| 对象 | 典型形式 | 回答的问题 |
| --- | --- | --- |
| 策略 | $$\pi(a\mid h,g)$$ | 给定历史和任务，做什么动作？ |
| 价值 | $$Q(h,a)$$ | 做这个动作，之后按某策略行动，预计累计收益怎样？ |
| 动力学模型 | $$p(s'\mid s,a)$$ | 做这个动作，下一状态是什么？ |
| 观测模型 | $$p(o\mid s)$$ | 这个状态会产生什么传感器观测？ |
| 奖励/终止模型 | $$p(r,c\mid s,a)$$ | 后果是否有用，是否继续？ |

一个动作生成模型学习的是 $$p(a\mid o,g)$$；一个前向世界模型学习的是动作到后果的关系。两者可以同时使用扩散或 Transformer 架构，但学习目标不同。把“生成未来动作”叫作“生成未来世界”会混淆问题。

## 2. 为什么首先需要状态估计

### 2.1 单帧观测通常不够

杯子在同一位置可能正在向左滑、向右滑，或静止。图像相同，下一状态不同。如果模型只用当前图像预测未来，它必须把未观测速度造成的差异当成随机性。

设真实状态为 $$s_t$$，观测历史为 $$h_t^{\mathrm{obs}}=(o_{0:t},a_{0:t-1})$$。POMDP 的充分信息可以用信念分布表示：

$$
b_t(s)=P(s_t=s\mid h_t^{\mathrm{obs}}).
$$

收到新观测时，贝叶斯更新为：

$$
b_{t+1}(s')\propto
p(o_{t+1}\mid s')\int p(s'\mid s,a_t)b_t(s)\,ds.
$$

积分是“执行动作后的预测”，乘上观测似然是“新证据纠正预测”。这个两阶段结构非常重要：**有真实观测时应更新状态；想象未来时没有未来观测可用。**

### 2.2 潜状态是信念的近似，而非自动得到真实物理状态

高维图像下无法显式维护所有物理状态的分布，可以学习压缩表示。理想表示既保留任务相关信息，又能在动作作用下预测后果。

仅靠重建当前画面，模型可能很关注背景纹理，却忽略接触是否建立。只用奖励监督，又可能丢失未来换任务需要的几何信息。表示是否有用，必须与下游问题共同评估。

## 3. 一个潜状态模型怎样区分记忆、先验与后验

用 $$h_t$$ 表示确定性记忆，$$z_t$$ 表示随机潜变量。一种 RSSM 风格的结构是：

$$
h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}),
$$

$$
\text{先验：}\quad p_\theta(z_t\mid h_t),
\qquad
\text{后验：}\quad q_\phi(z_t\mid h_t,o_t).
$$

再用 $$(h_t,z_t)$$ 预测观测，用 $$(h_t,z_t,a_t)$$ 预测动作之后的奖励和继续标记。这里规定 $$r_t$$ 是执行 $$a_t$$ 后获得的奖励，与一些论文将奖励附着到到达状态的下标习惯不同。

{% include figure.html id="fig-embodied-latent-model" src="/assets/images/figures/embodied-latent-model.svg" alt="有真实观测时用后验修正潜变量，无未来观测时用动作条件先验递推潜状态" caption="RSSM 风格的教学结构。训练和在线观测阶段可以使用后验；想象阶段只能通过先验和动作推进。符号按本文约定。" size="full" width="960" height="520" source="本文绘制" %}

确定性记忆负责跨时间传播信息，随机潜变量表示仍未确定的部分。随机性可以用于表达遮挡、不可见物理属性或多种可能后果；它不是“模型一定知道自己不知道什么”的保证。

[PlaNet][planet] 使用包含确定性与随机成分的潜动态模型，并在潜空间做在线规划。这里抽取的是结构思想，不包含其完整网络、训练技巧和多步目标。

## 4. 从似然到训练目标：模型到底在优化什么

对一段轨迹，假设潜变量后验按时间因果分解，观测、奖励、继续标记 $$c_t$$ 在给定潜状态与动作时按选定结构条件独立。一个教学损失为：

$$
\mathcal L=\sum_t\mathbb E_q\Big[
-\log p_\theta(o_t\mid h_t,z_t)
-\log p_\theta(r_t\mid h_t,z_t,a_t)
-\log p_\theta(c_t\mid h_t,z_t,a_t)
\Big]
+\beta\sum_t\mathbb E_q\Big[
D_{\mathrm{KL}}\big(q_\phi(z_t\mid h_t,o_t)\Vert p_\theta(z_t\mid h_t)\big)
\Big].
$$

在相应生成模型、因果后验和 $$\beta=1$$ 下，这是负变分下界的形式；改权重、截断 KL 或加入额外正则后，应称为训练目标的修改，不能笼统说都严格等于原始似然。

各项有不同作用：观测项让表示保留可解释信息，奖励/继续项面向任务，KL 项让依靠历史得到的先验与看到当前观测后的后验相容。若先验很差、后验靠真实图像不断纠正，重建可能很好，开环多步想象仍会失真。

### 4.1 为什么要专门评估多步预测

训练时每一步都喂入真实观测，下一步的潜状态就总能被纠正。部署规划却需要连续使用模型自己的预测。两种条件不同，因此应从某个真实历史起点出发，屏蔽未来观测，测试 1、5、10、20 步的任务相关误差。

可以比较物体位置、接触阶段、奖励排序或到达概率，而不只比较像素误差。对控制而言，“两段候选轨迹谁更好”有时比逐像素重建更直接。

## 5. 从模型到规划：CEM 与滚动时域

### 5.1 优化的是未来动作序列

给定当前状态估计 $$b_t$$，考虑：

$$
A^*=\arg\max_{a_{t:t+H-1}}
\mathbb E\left[
\sum_{k=0}^{H-1}\gamma^k r_{t+k}
+\gamma^HV(s_{t+H})\right].
$$

期望同时覆盖当前状态不确定性和未来动态随机性。动作边界、碰撞约束、速度限制可以进入优化问题；把违规惩罚加进奖励是软约束，并不保证每条候选都满足硬约束。

### 5.2 CEM 为什么能用于不可微的模型

交叉熵方法在动作序列空间维护一个采样分布。每轮采样若干序列，用模型展开后评分，取最优的一部分“精英”，再更新采样分布的均值和方差。它不要求通过动态模型反向传播。

随后只执行第一个动作或一个短前缀，获取新观测，再优化。这就是模型学习与 MPC 的连接。CEM 是近似随机优化，不保证找到全局最优；维度、规划长度、样本数、初始分布都会影响结果。

{% include figure.html id="fig-embodied-planning-loop" src="/assets/images/figures/embodied-planning-loop.svg" alt="观测更新状态，采样候选动作并在模型中评分，仅执行第一步后重新观测的规划闭环" caption="滚动规划将模型预测与真实反馈交替使用。虚线表示未执行的预测未来；下一轮可以重新选择，不要求沿旧计划走到底。" size="full" width="960" height="480" source="本文绘制" %}

### 5.3 可运行的一维 CEM-MPC 实验

令 $$x_{t+1}=x_t+0.1a_t$$，$$a_t\in[-1,1]$$，目标 $$x=1$$。代价为每步距离平方与动作惩罚之和。此处模型完全已知，先用它隔离“规划器是否正确”这一问题。

```python
import numpy as np

rng = np.random.default_rng(7)

def cost(x, sequences):
    predicted = x + 0.1 * np.cumsum(sequences, axis=1)
    return np.sum((predicted - 1.0) ** 2 + 0.01 * sequences ** 2, axis=1)

def plan(x, horizon=15, population=256, elites=32, iterations=4):
    mean, std = np.zeros(horizon), np.ones(horizon)
    for _ in range(iterations):
        candidates = np.clip(rng.normal(mean, std, (population, horizon)), -1, 1)
        values = cost(x, candidates)
        selected = candidates[np.argsort(values)[:elites]]
        mean = selected.mean(axis=0)
        std = np.maximum(selected.std(axis=0), 0.05)
    return mean

x = 0.0
positions = [x]
for _ in range(20):
    sequence = plan(x)
    x += 0.1 * sequence[0]  # 只执行第一步，再基于新状态重规划。
    positions.append(x)
print("final position:", round(x, 3))
assert abs(x - 1.0) < 0.15
```

这里用裁剪高斯、对角方差和固定方差下限，是简化教学实现。实验二可将真实系数改为 0.08、规划模型仍用 0.1，比较一次性执行整段计划与滚动规划的终点误差；实验三加入观测延迟，检查闭环纠正能力的限制。

## 6. 模型误差为什么会被规划放大

### 6.1 递推误差与规划长度

假设在考虑的区域内，真实动态 $$f$$ 对状态是 $$L$$-Lipschitz，单步模型误差不超过 $$\epsilon$$。比较相同动作序列下的真实轨迹与预测轨迹，误差满足：

$$
e_{k+1}\leq Le_k+\epsilon.
$$

若起点准确，则：

$$
e_H\leq\epsilon\sum_{i=0}^{H-1}L^i.
$$

$$L=1$$ 时是线性累积；$$L>1$$ 时界可能快速增大。接触切换和碰撞附近未必满足统一的小 Lipschitz 常数，所以公式只是解释误差传播的局部模型。

若两个候选的预测回报差小于模型评价误差量级，规划器对它们的排序就不可靠。提高搜索预算可能找到更好的真实方案，也可能更充分地利用模型漏洞。

### 6.2 模型利用错误的例子

训练数据只有小力推杯子，模型却把动作幅度与位移关系外推为始终线性。优化器发现极大动作能瞬间到达目标，于是选择数据未覆盖的区域。模型在训练集上一阶误差很小，并不能阻止这个选择。

缩短规划时域、限制动作范围、增加有针对性的交互数据和检查不确定性，都可能缓解这一问题；具体效果需要实验，不存在只改一个参数就普遍解决的办法。

## 7. 不确定性：随机结果与知识不足

若多个模型对同一输入给出不同均值，可以粗略表示模型间分歧；单个概率模型内部的方差则可表示条件结果的随机性。一个方差分解写为：

$$
\operatorname{Var}(Y\mid x)
=\mathbb E_m[\operatorname{Var}(Y\mid x,m)]
+\operatorname{Var}_m(\mathbb E[Y\mid x,m]).
$$

第一项是模型内部的不确定性，第二项是模型间预测均值的差异。把第二项解释为认知不确定性，是一种近似：多个网络可能共享同样的数据缺口和偏差，全部自信地犯错。

[PETS][pets] 将概率动态模型集成与轨迹采样结合，为理解“模型分歧如何进入规划”提供代表性例子。教学上可比较平均回报与一个风险惩罚目标：

$$
J_{\mathrm{risk}}(A)=\mathbb E[R(A)]-\lambda\sqrt{\operatorname{Var}[R(A)]}.
$$

后者只是可选的风险偏好，不是 PETS 所有配置的统一公式，也不是安全保证。过强惩罚可能让机器人永远避开需要接触、但有少量随机性的必要动作。

## 8. 另一条路线：在想象中训练策略

每次在线搜索都消耗时间。另一种方式是在模型内生成潜状态轨迹，用预测奖励和价值学习 actor，执行时直接由 actor 给出动作。[Dreamer][dreamer] 是这一思路的代表工作。

理解两者可比较：

| 路线 | 主要计算发生在哪里 | 优点与代价 |
| --- | --- | --- |
| 在线潜空间规划 | 每次选择动作时 | 可针对当前情况重新搜索；执行计算较重 |
| 想象训练策略 | 训练阶段 | 执行快速；策略可能把模型偏差固化下来 |

一个通用的有限想象轨迹价值目标可写为：

$$
G_t^\lambda=r_t+\gamma c_t\left[(1-\lambda)V(z_{t+1})+\lambda G_{t+1}^\lambda\right],
$$

末端以价值函数自举；这里省略确定性记忆仅为排版。$$\lambda$$ 在短步自举和长步模型累计之间权衡。不同 Dreamer 版本的潜变量、梯度处理、归一化与训练技巧并不相同，不能把这一个公式当作完整算法。

## 9. 视频预测怎样才与机器人行动发生联系

一个只根据过去画面预测未来的视频模型，可能主要学到行为数据中的相关性。要用于控制，至少要进一步检查：

1. 动作条件是否对应真实可执行的量、坐标和时间间隔？
2. 对同一个起点输入不同动作，预测后果是否合理地区分？
3. 训练数据是否覆盖候选动作，还是依靠没有证据的外推？
4. 可从预测中提取什么任务评价，是否对接触和失败敏感？
5. 推理时间是否允许在需要的反馈周期内完成选择？

这是一份控制用途的判断框架，不要求世界模型一定输出像素。点云、物体状态、潜变量或接触模式，都可以成为合适的预测对象。模型是否足够好，应由闭环决策质量与任务相关误差共同回答。

## 10. 复习题与参考答案

**问题一：为什么后验重建准确不能证明模型会规划？** 后验使用当前真实观测纠错，想象却要连续使用先验；两者面对的误差积累条件不同。

**问题二：为什么增加 CEM 样本数有时会让真实表现变差？** 搜索可能更充分地找到模型外推错误，而不是更好的真实动作。

**问题三：模型集成意见一致是否等于可信？** 不等于，成员可能共享训练分布和结构偏差；需用实际留出区域和闭环试验校验。

**问题四：世界模型是否必须生成视频？** 不需要。若潜状态足以预测任务相关的后果并支持决策，就可以不在每次规划中重建像素。

最后一篇讨论 [离线强化学习]({% post_url 2026-10-24-embodied-offline-rl %})：当只能使用已有数据时，怎样限制价值估计和策略改进的外推。

## 参考资料

- [Hafner et al. Learning Latent Dynamics for Planning from Pixels. ICML 2019.][planet] — PlaNet、潜状态与在线规划。
- [Chua et al. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018.][pets] — PETS、模型集成与轨迹采样。
- [Hafner et al. Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020.][dreamer] — 在潜空间想象中学习行为。

[planet]: https://proceedings.mlr.press/v97/hafner19a.html
[pets]: https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
[dreamer]: https://openreview.net/pdf?id=S1lOTC4tDS
