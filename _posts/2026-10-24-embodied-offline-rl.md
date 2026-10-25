---
title: "【具身学习·05】离线强化学习：从分布外价值误差到 CQL 与 IQL"
date: 2026-10-24 00:05:00 +0800
categories: [强化学习]
content_type: tutorial
tags: [embodied-ai, offline-rl, cql, iql, value-learning, robot-learning, policy-improvement]
description: 解释固定机器人数据上的价值外推与自举误差，推导 CQL 正则梯度、IQL 的 expectile 与优势加权策略提取，用手算和可运行代码比较均值、分位数和 expectile，并梳理奖励、终止与离线评估的限制。
image:
  path: /assets/images/covers/embodied-offline-rl-cover.jpg
  alt: 在已有机器人数据的支持范围内估计价值并改进策略
math: true
---

> **离线强化学习需要改进决策，却不能随时向环境询问“这个新动作到底好不好”。** 它的核心矛盾是策略改进与数据支持之间的取舍。

{% include embodied-series.html %}

如果已经有许多抓取轨迹，其中既有成功，也有滑落和失败，能否比单纯模仿做得更好？这正是离线强化学习值得研究的场景之一。本篇默认读者了解 [Bellman 方程与 TD 学习]({% post_url 2026-09-19-rl-base %})，并把 CQL 与 IQL 当作理解机制的两个例子。

本文所有小型 MDP、Q 数值和代码输出均为教学构造，不是机器人基准成绩。算法背景可参考 [Levine 等人的离线强化学习教程][tutorial]。

## 1. 离线、离策略和模仿学习有什么区别

数据集记为：

$$
\mathcal D=\{(s_i,a_i,r_i,s'_i,d_i)\}_{i=1}^{N},
$$

其中 $$d_i=1$$ 表示真正的终止。视觉机器人中，$$s$$ 可以是经过历史编码的状态表示；若单帧不满足马尔可夫假设，使用同样的公式也不会自动消除部分可观测性。

| 设定 | 是否可继续采集 | 优化依据 |
| --- | --- | --- |
| 行为克隆 | 训练本身不要求 | 贴近示范动作 |
| 在线离策略 RL | 可以 | 回放旧经验，同时用新交互修正价值 |
| 离线 RL | 固定数据阶段不可以 | 从已有奖励和转移中学习较好决策 |

“离策略”表示更新所用数据可以来自不同于当前策略的行为；“离线”表示训练阶段没有新的环境数据可纠正错误。把在线算法的采集循环关掉，并不保证它就成为可靠的离线方法。

## 2. 一个没有数据约束的 Q 值怎样污染整个训练

考虑状态 $$s_1$$ 有两个动作。数据里只记录了 $$a_{\mathrm{seen}}$$，它的真实回报为 1；从未出现的 $$a_{\mathrm{new}}$$ 的真实回报未知。网络可能给出：

$$
Q(s_1,a_{\mathrm{seen}})=1,\qquad Q(s_1,a_{\mathrm{new}})=10.
$$

前一个值可以被样本校正，后一个值没有直接监督。若数据包含 $$s_0\to s_1$$、即时奖励为零，普通最优 Bellman 目标在 $$\gamma=0.9$$ 时变成：

$$
y(s_0,a_0)=0+0.9\max_{a'}Q(s_1,a')=9.
$$

但数据支持的后续行动只提供 0.9 的价值。虚高估计已经通过自举传播到一个**确实出现在数据中的状态—动作对**。因此，只检查训练动作上的 TD 误差不够：目标本身可能依赖未受约束的动作。

{% include figure.html id="fig-embodied-offline-support" src="/assets/images/figures/embodied-offline-support.svg" alt="未见动作的虚高 Q 值通过最大化进入上一状态 Bellman 目标，与数据支持路径对照" caption="两步教学 MDP：未见动作被网络估为 10，便可能把上一步目标推到 9。这个数值只说明外推与自举机制，不代表任何真实任务的回报。" size="full" width="960" height="480" source="本文绘制" %}

双 Q、目标网络和较小学习率可以缓解某些优化或过估计问题，但不能凭空产生缺失动作的真实信息。两个 Q 网络也可能在同一区域一起高估。

## 3. 离线改进的三种限制方式

一类方法限制策略偏离行为数据，例如在 actor 目标中加入行为克隆项。以确定性策略的教学形式表示：

$$
\mathcal L_\pi=\mathbb E_{(s,a)\sim\mathcal D}
[-\lambda Q(s,\pi(s))+\|\pi(s)-a\|^2].
$$

第一项鼓励高价值，第二项限制偏移。[TD3+BC][td3bc] 是这一简洁思路的代表，具体实现还涉及归一化和权重选择。平方距离近不意味着位于数据支持内：两个模式之间的平均动作可能恰好没有出现过。

另一类方法直接约束 Q 值，例如 CQL；第三类方法在价值学习时避免显式查询未见动作，例如 IQL。三者不是互斥的哲学，实际系统可以组合模型、行为约束、价值正则和在线微调。

## 4. CQL：对数据外的高价值保持保守

先看离散动作版的 CQL(H) 形式。令固定目标网络给出的 Bellman 目标为 $$y$$，则：

$$
\mathcal L_{\mathrm{CQL}}
=\frac12\mathbb E_{\mathcal D}[(Q(s,a)-y)^2]
+\alpha\mathbb E_{s\sim\mathcal D}\left[
\log\sum_a\exp Q(s,a)
-\mathbb E_{a\sim\hat\pi_\beta(\cdot\mid s)}Q(s,a)
\right].
$$

这里 $$\hat\pi_\beta$$ 表示数据中的经验行为分布，log-sum-exp 温度取 1。正则项一边压低总体高值，一边抵消对数据动作的压低，形成相对保守的偏好。[CQL 原论文][cql]

### 4.1 对两个动作手算梯度

对某个动作 $$a_j$$，正则梯度是：

$$
\frac{\partial\mathcal R}{\partial Q(s,a_j)}
=\operatorname{softmax}(Q(s,\cdot))_j-\hat\pi_\beta(a_j\mid s).
$$

如果数据只包含动作 A，而 $$Q_A=1,Q_B=4$$，则 B 的 softmax 概率约为 0.953。梯度下降会强烈压低 B；A 的正则梯度约为 -0.953，倾向于抬高 A，Bellman 项再把 A 锚定到数据支持的回报附近。

这个算例说明它不是“把所有 Q 都统一减去一个常数”。正则的作用与动作之间的相对估计、数据频率和 $$\alpha$$ 有关。

### 4.2 连续动作与理论边界

连续动作下要处理对动作空间的积分，通常通过采样估计，涉及采样分布及相应密度修正。将离散动作的 `logsumexp` 直接套在随便采出的若干动作上，通常不能声称与原始连续目标完全相同。

CQL 关于保守价值的理论依赖相应假设，主要讨论策略期望意义下的界；不能据此宣称任意神经网络训练得到的每个 $$Q(s,a)$$ 都是严格下界。保守性过强也可能低估有价值但稀少的行为，需要与数据质量和任务一起分析。

## 5. IQL：先在数据动作中估计“偏好的价值”

IQL 的价值学习使用数据中的动作，不在 Bellman 目标中显式执行对未见动作的最大化。用目标 Q 构造 V 的上侧 expectile，再用 V 自举 Q，最后通过优势加权的行为克隆提取策略。[IQL 原论文第 4 节][iql]

### 5.1 Expectile 是不对称平方误差的最优点

对残差 $$u=x-v$$，定义：

$$
\rho_\tau(u)=|\tau-\mathbf1(u<0)|u^2.
$$

当 $$\tau=0.5$$，最优 $$v$$ 是均值；当 $$\tau>0.5$$，低估较大的 $$x$$ 受到更强惩罚，最优值会向上移动。

考虑等概率取 $$X=0$$ 或 $$X=2$$ 的教学分布。在 $$0\leq v\leq2$$ 内：

$$
L(v)=\tfrac12(1-\tau)v^2+\tfrac12\tau(2-v)^2.
$$

求导得：

$$
(1-\tau)v-\tau(2-v)=0
\quad\Rightarrow\quad v=2\tau.
$$

所以 $$\tau=0.7$$ 对应 1.4。它既不是最大值 2，也不是该分布的 70% 分位数 2。Expectile 使用误差的大小，quantile 使用不对称绝对误差，两者不能混称。

{% include figure.html id="fig-embodied-expectile" src="/assets/images/figures/embodied-expectile.svg" alt="等概率零和二的数值轴上，均值一、七成 expectile 一点四和七成分位数二的对照" caption="等概率 X∈{0,2} 的手算例子。τ=0.7 时 expectile 为 1.4，70% 分位数为 2；改变 τ 并不等于选择相同百分位。" size="full" width="960" height="440" source="本文公式计算" %}

### 5.2 三个目标及其数据依赖

设 $$\bar Q$$ 为目标 Q；常用双 Q 实现会取两者较小值。IQL 的核心目标可统一写为：

$$
\mathcal L_V=\mathbb E_{(s,a)\sim\mathcal D}
[\rho_\tau(\operatorname{sg}[\bar Q(s,a)]-V(s))],
$$

$$
\mathcal L_Q=\mathbb E_{\mathcal D}
\left[(Q(s,a)-\operatorname{sg}[r+\gamma(1-d)V(s')])^2\right],
$$

$$
\mathcal L_\pi=-\mathbb E_{\mathcal D}
\left[\operatorname{sg}[w(s,a)]\log\pi(a\mid s)\right],\qquad
w=\min\{w_{\max},\exp(\beta(\bar Q-V))\}.
$$

$$\operatorname{sg}$$ 表示停止梯度；本文 $$\beta$$ 是逆温度，越大越偏好高优势动作。权重裁剪是实践中的稳定化措施。V 目标偏好同一状态下价值较高的数据动作，Q 的普通平方损失则对随机转移做期望拟合。

为什么不直接对 $$r+\gamma V(s')$$ 做上侧 expectile？因为高目标可能只是随机转移碰巧走运。对数据动作的预期价值做偏好，与偏好环境随机性中最幸运的样本，是不同操作。

### 5.3 不查询未见动作，不等于部署永远不会出分布

IQL 的训练目标没有消除函数逼近、状态覆盖和 actor 泛化的问题。连续策略在新图像上仍可能输出未记录过的动作；加权回归也可能在多峰数据间产生不理想的平均。

因此更准确的说法是：IQL 在价值更新中改变了如何利用数据支持，而不是提供了一个硬约束的机器人执行屏障。官方代码可用于核对目标网络、权重与实现细节。[IQL 官方实现][iqlcode]

## 6. 可运行的小实验：expectile 与优势加权

下面用二分求解一阶条件，验证手算例子，再计算两个动作的加权行为分布。仅依赖 NumPy。

```python
import numpy as np

def expectile(values, tau, steps=100):
    values = np.asarray(values, dtype=float)
    low, high = values.min(), values.max()
    for _ in range(steps):
        v = (low + high) / 2.0
        residual = values - v
        weights = np.where(residual >= 0, tau, 1.0 - tau)
        # 一阶条件：sum(weights * (values - v)) = 0。
        if np.sum(weights * residual) > 0:
            low = v
        else:
            high = v
    return (low + high) / 2.0

q = np.array([0.0, 2.0])
v = expectile(q, tau=0.7)
assert np.isclose(v, 1.4)
behavior = np.array([0.5, 0.5])
log_weight = q - v  # beta = 1；先减最大值以稳定指数计算。
weight = np.exp(log_weight - log_weight.max())
policy = behavior * weight
policy /= policy.sum()
print("expectile:", v, "weighted policy:", policy)
# expectile: 1.4；加权概率约为 [0.1192, 0.8808]。
```

这里对离散动作概率做精确归一化，未使用权重裁剪。在单一状态内，公共项 $$-V(s)$$ 会被归一化抵消；真实训练中 V 还用于 Bellman 更新，跨状态权重、函数共享和裁剪也会影响优化，所以不能从这个例子推断 V 没有作用。

再把行为概率改成 `[0.99, 0.01]`，会发现同样优势下，稀有高价值动作的最终概率仍受数据频率影响。这可以帮助理解为什么“偏好高价值”与“直接取最大值”不是一回事。

## 7. 轨迹拼接为什么可能改进，什么时候不可能

假设数据有一条轨迹从起点到中间状态后失败，另一条从相同中间状态到目标成功。动态规划可以将后半段的高价值传播到前半段，从而组合出数据里没有完整出现过的成功路线。

前提是这个“相同中间状态”真的包含决策所需信息。如果两张图像看起来相同，但一条已经夹牢物体、另一条没有接触，那么拼接就可能不成立。状态表示、接触记忆和动力学一致性共同决定是否能复用后续经验。

离线方法也不能从没有覆盖的关键转移中创造证据。所有轨迹都没能接触到把手时，不能仅凭价值优化保证学会开门。额外先验、模型或新交互可以提供帮助，但此时要明确新增了什么信息。

## 8. 机器人离线数据的四个关键细节

### 8.1 奖励定义改变“更好”的含义

只有成功/失败标签时，如何把奖励放到时间轴上必须写清楚。终点奖励与每步距离奖励具有不同偏好；后者可能鼓励停在目标附近，却没有真正完成夹持。失败数据有价值，但要有一致的奖励和状态解释。

### 8.2 终止与时间截断

真正终止应令后续价值为零。采集时间上限导致的截断，不一定代表环境终止；若仍有合法末状态且任务定义为继续过程，通常保留自举。如果任务本身是固定时限，应把剩余时间纳入状态，并按任务结束处理。不能简单把所有文件末尾都标为同一种 `done`。

### 8.3 动作和控制模式

同一个末端增量在不同控制频率、阻抗刚度下造成不同转移。数据混用多个执行接口时，要统一语义或把模式作为条件，参照 [接触控制]({% post_url 2026-10-24-embodied-contact-control %}) 与 [模仿学习]({% post_url 2026-10-24-embodied-imitation-learning %}) 两篇。

### 8.4 奖励缩放影响优化

奖励乘以常数会改变 Q 和优势的尺度，进而影响 CQL 的相对正则强度以及 IQL 的指数权重。即使理想最优策略在正比例缩放下不变，有限样本训练的超参数也未必可以直接照搬。

## 9. 没有在线交互，怎样评估策略

离线评估本身也是一个估计问题。固定策略的拟合 Q 评估（FQE）会以策略动作构造 Bellman 目标；如果这些动作缺少数据支持，它仍受外推误差影响。重要性采样需要行为概率与支持覆盖，长轨迹的权重乘积还可能具有很高方差。

因此学习实验至少应同时报告数据划分、奖励定义、离线指标、随机种子、实际闭环测试条件。仿真评估可以用于模型选择，但它对现实的解释范围仍由仿真误差决定。若训练期间反复用真实测试任务选超参数，应披露这部分交互预算，不能把整个过程都称为纯离线。

一个合理的实验矩阵是：相同数据、相同视觉编码器、相同动作接口下比较 BC、行为正则方法和离线价值学习，再分别改变成功/失败比例及状态覆盖。先回答“价值学习是否帮助当前数据”，再研究更大的模型。

## 10. 复习题与参考答案

**问题一：训练动作上的 Q 拟合很好，为什么仍可能失败？** 目标可能包含分布外动作的虚高估计，策略也可能访问数据未覆盖的状态。

**问题二：0 和 2 等概率，0.7-expectile 是多少？** 1.4；70% 分位数是 2。二者优化的损失不同。

**问题三：增大 IQL 的逆温度是否总能提高策略？** 不一定。它会放大优势估计误差，令少数样本主导训练，还可能降低行为覆盖。

**问题四：离线 RL 是否一定超过 BC？** 不一定。奖励噪声、覆盖不足、状态混淆和优化误差都可能使价值学习得不偿失。

在具备离线基线后，可继续阅读 [HIL-SERL]({% post_url 2026-09-21-hilserl %})，理解引入在线交互与人工纠偏后，数据分布和优化问题又发生了哪些变化。

## 参考资料

- [Levine et al. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. 2020.][tutorial] — 问题设定与评估背景。
- [Kumar et al. Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS 2020.][cql] — 保守价值正则。
- [Kostrikov, Nair, Levine. Offline Reinforcement Learning with Implicit Q-Learning. ICLR 2022；预印本 2021.][iql] — Expectile、价值更新与策略提取。
- [IQL 官方代码.][iqlcode] — 可用于核对实现；本文算例独立编写。
- [Fujimoto and Gu. A Minimalist Approach to Offline Reinforcement Learning. NeurIPS 2021.][td3bc] — TD3+BC 基线。

[tutorial]: https://arxiv.org/abs/2005.01643
[cql]: https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf
[iql]: https://arxiv.org/abs/2110.06169
[iqlcode]: https://github.com/ikostrikov/implicit_q_learning
[td3bc]: https://arxiv.org/abs/2106.06860
