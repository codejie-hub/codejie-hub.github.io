---
title: "【机器人学习】Diffusion Policy：从条件去噪到闭环动作生成的原理与推导"
date: 2026-09-19 00:00:00 +0800
categories: [机器人学习]
tags: [diffusion-policy, imitation-learning, diffusion, robotics, visuomotor-policy, action-chunking]
description: 从多模态行为克隆出发，推导 Diffusion Policy 的前向加噪、反向后验、噪声预测目标与条件 score；结合手算例子、官方代码和 PyTorch 实现，解释视觉条件、动作序列、滚动执行、DDIM 加速及其与 MPC、Flow Matching 的关系。
image:
  path: /assets/images/posts/placeholders/robotics.svg
  alt: Diffusion Policy 条件扩散与机器人闭环动作生成学习笔记
math: true
mermaid: true
---

> **Diffusion Policy 用条件扩散模型学习“给定当前观测，哪些动作序列是合理的”，再从这个分布中生成一段动作，只执行其中一部分，获取新观测后重新生成。**
>
> 理解它需要同时看清两个过程：**模型在计算机里把噪声变成动作序列**，以及**机器人在真实时间里执行动作并接收反馈**。去噪步数不是机器人移动的步数。

这篇笔记围绕一个问题展开：假设机器人要绕过障碍物去抓取物体，示范中既有“从左侧绕行”，也有“从右侧绕行”，怎样学到一个既能保留两种行为、又能连续稳定执行的策略？

Diffusion Policy 的答案包含三个相互配合的设计：

1. 用**条件生成模型**表达多种可能的动作，而不把它们平均成一条动作。
2. 联合生成**动作序列**，学习多个未来时刻之间的相关性。
3. 使用**滚动时域执行**，在行为一致性与环境反馈之间取舍。

本文以 Chi 等人的 [Diffusion Policy 论文][dp]、[官方项目页][project]及[官方代码][repo]为依据。原工作发表于 RSS 2023，随后有 IJRR 2024 扩展版；下文重点解释方法本身，不将论文某个任务上的经验结论当成普遍保证。

## 阅读路线与符号约定

建议先读第 1—2 节建立整体认识，再沿第 3—6 节完成扩散推导；第 7—10 节解释机器人实现，第 11 节对应代码，第 12—13 节用于串联已有知识和复习。

| 符号 | 本文约定 |
| --- | --- |
| $$t$$ | 机器人真实控制时间的离散索引 |
| $$h$$ | 一次预测动作块中的位置索引 |
| $$k=0,\ldots,K$$ | 扩散噪声等级；$$k=0$$ 为干净动作，$$k=K$$ 接近纯噪声 |
| $$o_t,a_t$$ | 在时刻 $$t$$ 获得的观测，以及随后执行的动作 |
| $$T_o,T_p,T_a$$ | 观测窗口长度、预测动作窗口长度、每次实际执行的动作数 |
| $$O_t$$ | 观测历史 $$(o_{t-T_o+1},\ldots,o_t)$$ |
| $$A_t^0$$ | 干净动作块；上标 $$0$$ 表示未加扩散噪声 |
| $$A_t^k$$ | 同一个动作块在噪声等级 $$k$$ 下的状态 |
| $$d_a$$ | 单步动作维度，动作块形状为 $$T_p\times d_a$$ |
| $$\beta_k,\alpha_k,\bar\alpha_k$$ | 加噪方差、单步信号系数与累积信号系数 |
| $$\epsilon_\theta(A^k,k,c)$$ | 预测所加高斯噪声的网络 |
| $$c=f_\phi(O_t)$$ | 观测编码器提取的条件特征 |

推导时把动作块展平成 $$D=T_p d_a$$ 维向量；实现时保留 `[batch, horizon, action_dim]` 形状。未特别说明时，省略真实时间下标 $$t$$。

**扩散方向统一为：干净动作 $$A^0$$ → 加噪动作 $$A^K$$；生成则从 $$K$$ 走回 $$0$$。** 这与本博客 [Flow Matching 笔记]({% post_url 2026-04-01-flowmatching %})采用的“时间从噪声走向数据”方向相反。

## 1. 学习目标：从单个动作回归到动作分布

### 1.1 它首先是一个模仿学习问题

数据集由示范轨迹组成：

$$
\mathcal D
=
\left\{
\tau^{(n)}
=
(o_0,a_0,o_1,a_1,\ldots)
\right\}_{n=1}^{N}.
$$

通过滑动窗口，可以构造许多训练对：

$$
(O_t,A_t^0).
$$

希望学到的策略是：

$$
\pi_\theta(A_t^0\mid O_t)
\approx
p_{\mathrm{demo}}(A_t^0\mid O_t).
$$

这里的目标分布来自示范。原始 Diffusion Policy 的核心训练不需要奖励函数、价值函数或与环境交互的策略梯度；它属于行为克隆的一种生成式实现。

这个区别直接影响方法的能力边界：示范中的多种行为可以被学习，但“能生成动作”本身并不意味着模型会超越示范、主动寻找最优策略或理解任务代价。

### 1.2 为什么普通 MSE 会平均掉多种行为

若使用确定性回归：

$$
\min_f\;
\mathbb E\left[
\|f(O)-A\|^2
\right],
$$

固定观测 $$O=o$$，对输出 $$f(o)$$ 求导：

$$
\frac{\partial}{\partial f(o)}
\mathbb E[\|f(o)-A\|^2\mid O=o]
=
2\left(f(o)-\mathbb E[A\mid O=o]\right).
$$

因此最优预测是：

$$
\boxed{f^\star(o)=\mathbb E[A\mid O=o].}
$$

假设左右两条可行轨迹分别为 $$A_L,A_R$$，概率相同，则：

$$
f^\star(o)=\frac{A_L+A_R}{2}.
$$

平均轨迹可能恰好穿过障碍物。平方回归没有算错，它只是优化了“离所有示范尽量近”，而这个目标不一定对应一个可执行行为。

并不是所有行为克隆都会遇到同一种平均问题：高斯混合模型、自回归策略、离散动作模型等也可以表达多模态。Diffusion Policy 的选择，是通过逐步去噪生成高维连续动作块。[Diffusion Policy 第 1—2 节][dp]

### 1.3 为什么需要联合生成动作序列

考虑一个更简单的两步例子：

$$
A_L=(-1,-1),\qquad A_R=(+1,+1).
$$

如果分别为两步独立采样左右模式，除了正确的两种序列，还会得到：

$$
(-1,+1),\qquad (+1,-1).
$$

这两种序列可能对应动作突然反向。问题出在独立假设：

$$
p(a_t,a_{t+1}\mid O_t)
\neq
p(a_t\mid O_t)\,p(a_{t+1}\mid O_t).
$$

Diffusion Policy 直接建模整块联合分布：

$$
p(A_t\mid O_t),
\qquad
A_t\in\mathbb R^{T_p\times d_a}.
$$

虽然加入的高斯噪声通常在各坐标上独立，去噪网络会跨动作时间位置处理整个序列，从数据中学习相关性。**独立噪声不等于独立的生成动作。**

## 2. 先把完整流程放在一起

### 2.1 训练：人为破坏已知的示范，再预测破坏量

训练时，干净动作块 $$A^0$$ 已知。采样噪声等级 $$k$$ 和高斯噪声 $$\epsilon$$，直接构造：

$$
A^k
=
\sqrt{\bar\alpha_k}A^0
+
\sqrt{1-\bar\alpha_k}\epsilon.
$$

网络看到的是 $$A^k,k,O$$，监督标签是已知的 $$\epsilon$$。

~~~mermaid
flowchart LR
    O["观测窗口 O"] --> E["观测编码器"]
    E --> C["条件特征 c"]
    A["示范动作块 A⁰"] --> Q["随机选 k 并直接加噪"]
    N["高斯噪声 ε"] --> Q
    Q --> X["加噪动作块 Aᵏ"]
    X --> M["噪声预测网络"]
    C --> M
    K["噪声等级 k"] --> M
    M --> L["预测噪声与 ε 的 MSE"]
~~~

一次训练更新通常只采样一个噪声等级，不需要把完整的 $$K$$ 步反向生成过程展开后再反向传播。

### 2.2 执行：内部生成一个动作块，外部执行一段

推理时没有干净动作。先固定当前观测条件，从随机动作噪声出发：

$$
A_t^K\sim\mathcal N(0,I),
$$

然后逐步生成：

$$
A_t^K\rightarrow A_t^{K-1}\rightarrow\cdots\rightarrow A_t^0.
$$

提取其中接下来可执行的 $$T_a$$ 个动作，交给机器人控制器，再读取新观测重新生成。

~~~mermaid
flowchart TD
    O["读取最近 To 帧观测"] --> E["编码一次，得到条件 c"]
    E --> D["从高斯噪声初始化动作块"]
    D --> S["固定 c，完成内部多步去噪"]
    S --> A["还原动作单位并取执行片段"]
    A --> R["机器人执行 Ta 个动作"]
    R --> O
~~~

这里有三个完全不同的轴：

| 轴 | 含义 | 一个例子 |
| --- | --- | --- |
| 真实时间 $$t$$ | 机器人已经走到第几个控制周期 | 当前获得第 100 帧观测 |
| 动作位置 $$h$$ | 这次动作块中的第几个位置 | 第 3 个未来动作 |
| 扩散等级 $$k$$ | 当前候选动作块还有多少噪声 | 第 60 个噪声等级 |

在机器人尚未执行动作时，模型就可能已经完成了几十次去噪网络调用。反过来，一次去噪更新通常会同时更新整个动作块，不是只更新一个未来动作。

## 3. 前向扩散：为什么任意噪声等级都能直接采样

### 3.1 单步加噪过程

定义：

$$
q(A^k\mid A^{k-1})
=
\mathcal N
\left(
A^k;
\sqrt{\alpha_k}A^{k-1},
\beta_k I
\right),
$$

其中：

$$
\alpha_k=1-\beta_k,
\qquad 0<\beta_k<1.
$$

等价地：

$$
A^k
=
\sqrt{\alpha_k}A^{k-1}
+
\sqrt{\beta_k}\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,I).
$$

信号被略微缩小，同时加入高斯噪声。这里是在**动作表示空间**中加噪，不是让真实机器人执行随机扰动。

### 3.2 展开两步，看到累积方差

继续展开：

$$
A^k
=
\sqrt{\alpha_k\alpha_{k-1}}A^{k-2}
+
\sqrt{\alpha_k\beta_{k-1}}\epsilon_{k-1}
+
\sqrt{\beta_k}\epsilon_k.
$$

独立高斯项相加仍是高斯，方差为：

$$
\alpha_k\beta_{k-1}+\beta_k
=
\alpha_k(1-\alpha_{k-1})+(1-\alpha_k)
=
1-\alpha_k\alpha_{k-1}.
$$

递推到干净动作，令：

$$
\bar\alpha_k=\prod_{j=1}^{k}\alpha_j,
\qquad
\bar\alpha_0=1,
$$

得到：

$$
\boxed{
q(A^k\mid A^0)
=
\mathcal N
\left(
\sqrt{\bar\alpha_k}A^0,
(1-\bar\alpha_k)I
\right).
}
$$

于是可以直接重参数化采样：

$$
\boxed{
A^k
=
\sqrt{\bar\alpha_k}A^0
+
\sqrt{1-\bar\alpha_k}\epsilon,
\quad
\epsilon\sim\mathcal N(0,I).
}
$$

这个 $$\epsilon$$ 是等效的累积噪声，不要求保存每一次单步噪声 $$\epsilon_1,\ldots,\epsilon_k$$。这是 DDPM 训练高效的关键。[DDPM 第 2 节][ddpm]

### 3.3 为什么最后可以从标准高斯开始生成

当 $$\bar\alpha_K$$ 足够小：

$$
q(A^K\mid A^0)\approx\mathcal N(0,I),
$$

端点几乎不再依赖示范动作。因此生成过程可以用简单高斯作为起点。

但这个近似需要检查具体噪声日程。**把训练步数设成 100，并不自动保证终点接近纯噪声。** 日程决定了：

$$
\operatorname{SNR}_k
=
\frac{\bar\alpha_k}{1-\bar\alpha_k}.
$$

这也是为什么不能只修改步数、却不检查最终的 $$\bar\alpha_K$$ 和采样器设定。

## 4. 反向去噪：从高斯后验推导网络更新

### 4.1 真正困难的分布是什么

希望学习：

$$
p_\theta(A^{k-1}\mid A^k,O).
$$

给定当前加噪动作和观测，应该往哪个更干净的动作移动？由于干净动作 $$A^0$$ 未知，这个反向条件分布通常不能直接计算。

但是训练时知道 $$A^0$$，所以可以先推导：

$$
q(A^{k-1}\mid A^k,A^0).
$$

前向加噪不依赖观测，因此给定 $$A^0$$ 后，再加入 $$O$$ 不会改变这个已知加噪过程的后验。

### 4.2 用 Bayes 公式合并两个高斯

记 $$x=A^{k-1}$$。忽略与 $$x$$ 无关的常数：

$$
q(x\mid A^k,A^0)
\propto
q(A^k\mid x)\,q(x\mid A^0).
$$

取负对数：

$$
-\log q(x\mid A^k,A^0)
=
\frac{\|A^k-\sqrt{\alpha_k}x\|^2}{2\beta_k}
+
\frac{\|x-\sqrt{\bar\alpha_{k-1}}A^0\|^2}
{2(1-\bar\alpha_{k-1})}
+
C.
$$

对 $$k\geq2$$，把它整理成关于 $$x$$ 的二次函数。后验精度为：

$$
\frac{1}{\widetilde\beta_k}
=
\frac{\alpha_k}{\beta_k}
+
\frac{1}{1-\bar\alpha_{k-1}}
=
\frac{1-\bar\alpha_k}
{\beta_k(1-\bar\alpha_{k-1})}.
$$

因此：

$$
\boxed{
\widetilde\beta_k
=
\frac{1-\bar\alpha_{k-1}}
{1-\bar\alpha_k}\beta_k.
}
$$

一次项给出后验均值：

$$
\boxed{
\widetilde\mu_k(A^k,A^0)
=
\frac{\sqrt{\bar\alpha_{k-1}}\beta_k}
{1-\bar\alpha_k}A^0
+
\frac{\sqrt{\alpha_k}(1-\bar\alpha_{k-1})}
{1-\bar\alpha_k}A^k.
}
$$

于是：

$$
q(A^{k-1}\mid A^k,A^0)
=
\mathcal N(\widetilde\mu_k,\widetilde\beta_kI).
$$

在 $$k=1$$ 时，给定 $$A^0$$ 的后验退化为该干净动作，$$\widetilde\beta_1=0$$；它需要作为边界情况处理。[DDPM 第 2—3 节][ddpm]

### 4.3 为什么预测噪声就能得到反向均值

前向公式可以重排为：

$$
A^0
=
\frac{A^k-\sqrt{1-\bar\alpha_k}\epsilon}
{\sqrt{\bar\alpha_k}}.
$$

将其代入后验均值，并合并系数：

$$
\boxed{
\widetilde\mu_k
=
\frac{1}{\sqrt{\alpha_k}}
\left(
A^k
-
\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon
\right).
}
$$

真实噪声在生成时未知，就用网络预测替代：

$$
\mu_\theta(A^k,k,c)
=
\frac{1}{\sqrt{\alpha_k}}
\left(
A^k
-
\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\epsilon_\theta(A^k,k,c)
\right).
$$

再设：

$$
p_\theta(A^{k-1}\mid A^k,c)
=
\mathcal N(\mu_\theta,\sigma_k^2I),
$$

便得到反向采样：

$$
\boxed{
A^{k-1}
=
\frac{1}{\sqrt{\alpha_k}}
\left(
A^k
-
\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\epsilon_\theta(A^k,k,c)
\right)
+
\sigma_k z.
}
$$

其中 $$z\sim\mathcal N(0,I)$$。一种常用选择是 $$\sigma_k^2=\widetilde\beta_k$$，最后一步不再添加随机噪声。

注意：推导中的 $$q(A^{k-1}\mid A^k,A^0)$$ 是已知干净端点时的高斯后验；对未知端点混合后的真实反向分布一般并不严格是一个高斯。有限步模型使用参数化反向转移来逼近它，不能把两者完全等同。

## 5. 训练目标：噪声 MSE、ELBO 与条件期望

### 5.1 条件扩散模型的似然

模型的完整反向链为：

$$
p_\theta(A^{0:K}\mid O)
=
p(A^K)
\prod_{k=1}^{K}
p_\theta(A^{k-1}\mid A^k,O).
$$

直接积分掉全部中间变量来计算 $$p_\theta(A^0\mid O)$$ 很困难。DDPM 使用前向过程构造变分界：

$$
\begin{aligned}
-\log p_\theta(A^0\mid O)
\leq{}&
\operatorname{KL}
\bigl(q(A^K\mid A^0)\,\|\,p(A^K)\bigr)\\
&+
\sum_{k=2}^{K}
\mathbb E_q\left[
\operatorname{KL}
\bigl(
q(A^{k-1}\mid A^k,A^0)
\,\|\,
p_\theta(A^{k-1}\mid A^k,O)
\bigr)
\right]\\
&-
\mathbb E_q[\log p_\theta(A^0\mid A^1,O)].
\end{aligned}
$$

三部分分别是终点先验匹配、中间转移匹配与最后一步重建。负对数似然的这个上界，等价于对数似然的 ELBO 取负。

严格计算似然时，最后的重建项需要单独定义合适的解码分布。前面给定干净端点的后验在最后一步退化，以及实践中采样器最后不再注入噪声，都不意味着可以直接把零方差高斯当成普通密度代入这个对数似然。

### 5.2 高斯 KL 如何变成噪声回归

若反向方差 $$\sigma_k^2$$ 固定，对于 $$k\geq2$$，与均值参数有关的 KL 项为：

$$
\frac{1}{2\sigma_k^2}
\|\widetilde\mu_k-\mu_\theta\|^2.
$$

代入两者的噪声参数化：

$$
\widetilde\mu_k-\mu_\theta
=
\frac{\beta_k}
{\sqrt{\alpha_k}\sqrt{1-\bar\alpha_k}}
(\epsilon_\theta-\epsilon).
$$

所以得到：

$$
\boxed{
\frac{\beta_k^2}
{2\sigma_k^2\alpha_k(1-\bar\alpha_k)}
\|\epsilon-\epsilon_\theta(A^k,k,c)\|^2.
}
$$

这解释了噪声预测目标从何而来。但实际常用的简化目标会移除这组随时间变化的权重：

$$
\boxed{
\mathcal L_{\mathrm{simple}}(\theta,\phi)
=
\mathbb E_{
(A^0,O)\sim\mathcal D,\,
k,\,
\epsilon
}
\left[
\|\epsilon-\epsilon_\theta(A^k,k,f_\phi(O))\|^2
\right].
}
$$

因此，**未加权噪声 MSE 与完整负 ELBO 不是数值上相同的目标**。它是一种常用的重加权训练选择；不能把“具有变分推导依据”简化成“两者完全相等”。[DDPM 第 3.4 节][ddpm]

### 5.3 网络不是在恢复每一次随机噪声

平方回归的最优解是：

$$
\boxed{
\epsilon^\star(A^k,k,O)
=
\mathbb E[\epsilon\mid A^k,k,O].
}
$$

同一个加噪状态可以由多个干净动作和不同噪声产生，网络无法唯一恢复原来那一次随机采样，只能预测后验均值。

这也意味着即使模型达到最优，噪声 MSE 一般也不为零。类似 [Flow Matching 中的条件方差分解]({% post_url 2026-04-01-flowmatching %})：

$$
\begin{aligned}
\mathbb E[\|\epsilon-\epsilon_\theta\|^2]
={}&
\mathbb E[\|\epsilon-\epsilon^\star\|^2]\\
&+
\mathbb E[\|\epsilon^\star-\epsilon_\theta\|^2].
\end{aligned}
$$

第一项是当前输入下无法消除的标签不确定性，第二项才是预测函数相对于最优条件均值的误差。

因此，不能要求噪声 loss 必须趋近于零，也不能只凭训练 loss 小就判断机器人任务成功率高。

## 6. 为什么去噪可以保留多种动作模式

### 6.1 从噪声预测推导条件 score

令加噪动作的条件分布为：

$$
q_k(A^k\mid O)
=
\int
q(A^k\mid A^0)\,
p_{\mathrm{demo}}(A^0\mid O)
\,\mathrm dA^0.
$$

我们关心其 score：

$$
s_k(A^k,O)
=
\nabla_{A^k}\log q_k(A^k\mid O).
$$

单个高斯条件分布的 score 是：

$$
\nabla_{A^k}\log q(A^k\mid A^0)
=
-\frac{A^k-\sqrt{\bar\alpha_k}A^0}
{1-\bar\alpha_k}
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_k}}.
$$

对混合密度求导，并使用 Bayes 权重：

$$
\begin{aligned}
s_k(A^k,O)
&=
\mathbb E\left[
\nabla_{A^k}\log q(A^k\mid A^0)
\mid A^k,O
\right]\\
&=
-\frac{
\mathbb E[\epsilon\mid A^k,O]
}{
\sqrt{1-\bar\alpha_k}
}.
\end{aligned}
$$

于是：

$$
\boxed{
s_\theta(A^k,k,O)
=
-\frac{\epsilon_\theta(A^k,k,O)}
{\sqrt{1-\bar\alpha_k}}.
}
$$

Diffusion Policy 学到的是动作分布的**对数密度梯度**。Score 本身已经是梯度，不需要再称它为“score 的梯度”。

### 6.2 一个能手算的双峰动作例子

固定一个观测，只考虑一维动作：

$$
p_{\mathrm{demo}}(a^0\mid O)
=
\frac12\mathcal N(-m,\sigma_0^2)
+
\frac12\mathcal N(+m,\sigma_0^2).
$$

两峰可以理解为左右两种操作。直接回归动作的最优输出是 $$0$$，但当两峰相距较远时，$$0$$ 恰好处于低概率区域。

在噪声等级 $$k$$，分布仍是双高斯混合。定义：

$$
b_k=\sqrt{\bar\alpha_k}m,
\qquad
v_k=\bar\alpha_k\sigma_0^2+(1-\bar\alpha_k).
$$

则：

$$
q_k(x\mid O)
=
\frac12\mathcal N(x;-b_k,v_k)
+
\frac12\mathcal N(x;+b_k,v_k).
$$

两个模式的后验权重之比是：

$$
\frac{w_+(x)}{w_-(x)}
=
\exp\left(\frac{2b_kx}{v_k}\right).
$$

因此：

$$
w_+(x)-w_-(x)
=
\tanh\left(\frac{b_kx}{v_k}\right).
$$

把两个高斯的 score 按后验权重平均：

$$
\boxed{
s_k(x,O)
=
\frac{
b_k\tanh(b_kx/v_k)-x
}{v_k}.
}
$$

这个公式说明，方向依赖当前带噪动作 $$x$$。它不是把所有输入都指向同一个平均动作。

- 噪声很大时，两峰混合得较充分，分布可能只有一个明显峰。
- 噪声变小时，两个模式逐渐分离，当前位置会影响更可能靠近哪一峰。
- 在完全对称的 $$x=0$$ 处，score 确实为零；采样不会因为这个单点就都停在中心，因为初始噪声是连续随机变量，且随机采样器还会加入扰动。

这里的 score 图景有助于理解模式形成，但实际 DDPM 更新还包含信号缩放与随机项，不是对一个固定函数做普通梯度上升。

### 6.3 同样使用 MSE，为什么没有重新平均掉行为

直接行为回归预测：

$$
\mathbb E[A^0\mid O].
$$

噪声回归预测：

$$
\mathbb E[\epsilon\mid A^k,k,O].
$$

后者多了一个不断变化的输入 $$A^k$$，并通过一串反向转移构造完整分布。不同初始噪声会走出不同的生成结果，因而不需要把最终输出压缩成单个条件均值。

但是，“能够表达多模态”不保证“所有模式都学得一样好”。示范不平衡、网络容量、噪声日程和采样误差仍可能让某些行为模式缺失。

### 6.4 Score 与机器人任务代价不是同一个东西

若写成能量形式：

$$
q_k(A\mid O)\propto\exp[-E_k(A,O)],
$$

则：

$$
s_k(A,O)=-\nabla_A E_k(A,O).
$$

这个能量表达的是**带噪示范动作的统计分布**。它不天然等于碰撞代价、抓取成功率、机器人动力学代价或强化学习的负 Q 值。

“沿高密度方向去噪”意味着向示范分布中的动作靠近，不能直接解释为“在线求解最优控制”。[Diffusion Policy 第 4.4—4.5 节][dp]

## 7. 网络如何同时理解观测、时间和动作

### 7.1 视觉条件只需在一轮生成开始时编码

一次策略调用首先构造：

$$
c_t=f_\phi(O_t).
$$

$$O_t$$ 可以包含多帧图像、多路相机和机器人本体状态。随后整轮去噪重复使用同一个 $$c_t$$：

$$
\epsilon_\theta(A_t^k,k,c_t),
\qquad k=K,\ldots,1.
$$

这样就不必在每个扩散等级重新运行视觉编码器。当前相机图像是条件，而不是需要一起从噪声生成的目标。

在概念上，其单次推理成本约为：

$$
C_{\mathrm{call}}
\approx
C_{\mathrm{vision}}
+
N_{\mathrm{eval}}C_{\mathrm{denoiser}}.
$$

这里 $$N_{\mathrm{eval}}$$ 是去噪网络调用次数。官方策略实现先计算观测特征，再进入采样循环，可以直接核对这个执行顺序。[官方图像策略实现][image-policy]

有限帧历史也不自动成为完整状态。如果图像遮挡、接触力或历史模式无法从窗口中辨别，策略仍面临部分可观测性。

### 7.2 1D U-Net：沿动作时间轴做卷积

输入动作块通常为：

$$
A^k\in\mathbb R^{B\times T_p\times d_a}.
$$

时序 U-Net 将动作通道与序列轴适当转置后，沿动作位置轴卷积、下采样与上采样，并使用跳跃连接保留多尺度信息。

这里的“一维”指**卷积沿动作时间序列展开**，不是说动作只有一个自由度。图像通常已经由独立视觉编码器处理。

噪声等级 $$k$$ 经时间嵌入，和观测特征一起作为条件注入网络。例如 FiLM：

$$
h'
=
\gamma_\theta(k,c)\odot h
+
b_\theta(k,c).
$$

缩放和偏置通常按通道生成，在序列位置上广播。它让同一个带噪动作块在不同观测或噪声等级下，产生不同的噪声预测。[官方 ConditionalUnet1D][unet]

U-Net 的输出形状仍是 $$B\times T_p\times d_a$$，表示每个动作位置、每个动作维度的预测噪声。

### 7.3 Transformer：动作位置与扩散等级需要不同编码

Transformer 版本把不同动作位置转换成 token，并引入：

- 区分动作序列位置 $$h$$ 的位置编码；
- 区分噪声等级 $$k$$ 的扩散时间编码；
- 通过条件 token 或交叉注意力传入的观测特征。

原论文的时序 Transformer 使用动作位置上的因果注意力设计。但这**不意味着生成时必须像语言模型一样，一个动作 token 接着一个动作 token 地采样**：一次去噪网络前向仍可以输出整段动作的噪声预测，然后统一更新所有位置。[论文第 3.1 节][dp]、[官方 Transformer 实现][transformer]

论文报告了 U-Net 与 Transformer 在不同动作变化特征下的差异；架构选择仍需任务验证，不能把某个骨干网络理解为扩散策略成立的必要条件。

## 8. 动作序列与滚动执行：闭环到底发生在哪里

### 8.1 三个 horizon 分别控制什么

$$T_o$$ 决定策略能看多少历史；$$T_p$$ 决定一次联合生成的动作窗口有多长；$$T_a$$ 决定在获取新反馈前，承诺执行多少动作。

为方便理解，若动作块恰好从当前时刻开始，可以写：

$$
A_t^0=(a_t,a_{t+1},\ldots,a_{t+T_p-1}).
$$

这时只执行前 $$T_a$$ 个动作，再在 $$t+T_a$$ 重新规划。

但具体代码的数据窗口可能从最早观测时刻开始对齐，此时“执行前几个动作”就不再准确。

### 8.2 为什么官方代码从 `n_obs_steps - 1` 开始截取

在本文核对的官方图像策略中：

~~~python
start = n_obs_steps - 1
end = start + n_action_steps
action = action_pred[:, start:end]
~~~

原因是动作预测窗口与观测窗口的起点对齐。设：

$$
T_o=2,\qquad T_p=16,\qquad T_a=8.
$$

在当前时刻 $$t$$，对应关系为：

| 内容 | 真实时间对应 |
| --- | --- |
| 输入观测 | $$o_{t-1},o_t$$ |
| 预测动作槽位 0 | $$a_{t-1}$$ |
| 预测动作槽位 1 | $$a_t$$ |
| 预测动作槽位 8 | $$a_{t+7}$$ |
| 预测动作槽位 15 | $$a_{t+14}$$ |
| 实际取出的 `action_pred[:, 1:9]` | $$a_t,\ldots,a_{t+7}$$ |

也就是说，槽位 0 对应已经过去的位置，不能再作为当前动作执行。这个配置下应满足：

$$
T_a\leq T_p-(T_o-1).
$$

这里的 $$2,16,8$$ 来自官方 U-Net hybrid 配置示例，并非 Diffusion Policy 的定义要求；若自己的数据从当前时刻开始排列，截取起点也应相应变化。[官方配置][config]、[官方截取实现][hybrid-policy]

### 8.3 开环片段与闭环系统并不矛盾

一段动作执行期间，策略层没有重新利用新图像生成计划；在这个意义上，它暂时开环。每执行完一个片段又读取新观测，因此整体构成闭环。

但低层伺服控制器可能一直在利用关节位置、速度或力反馈。**策略层按片段更新，不代表机器人电机层也没有反馈。**

$$T_a$$ 体现了取舍：

- 较小：更快利用反馈，但重规划更频繁、计算压力更大，也可能在行为模式之间切换。
- 较大：更能维持一段动作的连贯性，但对环境变化反应更迟。
- 联合预测鼓励时间一致性，却不等于对速度、加速度或碰撞显式施加了约束。

论文采用滚动执行来平衡这两类需求；不能简单认为执行越少或预测越长就一定越好。[Diffusion Policy 第 2.3、4.3 节][dp]

### 8.4 推理延迟会改变“下一个动作”的含义

假设控制周期为 $$\Delta t$$，从观测采集到动作可用的总延迟为 $$\ell$$。如果推理期间旧动作继续执行，新动作可用时，原计划中的某些时刻已经过去。

粗略对应的跳过步数是：

$$
L=\left\lceil\frac{\ell}{\Delta t}\right\rceil.
$$

在观测起点对齐的窗口下，新的执行起点可近似为：

$$
h_{\mathrm{start}}=T_o-1+L.
$$

这只是均匀时间网格下的解释；实际实现应根据观测时间戳、动作目标时间戳和执行队列选择仍在未来的动作，而不是机械地套一个整数偏移。

例如控制周期为 $$50\,\mathrm{ms}$$、总延迟为 $$120\,\mathrm{ms}$$，则约有 3 个动作槽位需要重新考虑。这个数字是说明时间对齐的算例，不是论文硬件的实测速度。

若采用异步流水线，希望计算能隐藏在当前动作片段的执行期间，通常需要：

$$
\ell\lesssim T_a\Delta t.
$$

这只是基本时间预算，不是稳定性定理。

### 8.5 与 MPC 的共同点和区别

| 方面 | Diffusion Policy | 典型 MPC |
| --- | --- | --- |
| 序列来源 | 根据示范学到的条件动作分布采样 | 基于动力学与代价函数求解优化问题 |
| 是否显式预测未来状态 | 基本动作扩散策略不要求 | 通常依赖状态预测 |
| 执行方式 | 执行一段，再读取观测重新生成 | 执行前一部分，再更新状态重新优化 |
| 约束与保证 | 取决于额外设计，不能由去噪目标自动得到 | 可在优化中显式写入约束，但保证仍有条件 |
| 优化目标 | 拟合示范动作分布 | 最小化给定控制代价 |

两者共享滚动时域的执行思想，内部求解的问题却不同。可以结合 [PID 与 MPC 笔记]({% post_url 2026-09-18-pid-mpc-control %})理解策略生成、轨迹规划与底层跟踪各自承担的职责。

## 9. DDIM 加速：不能把完整反向链简单跳着执行

### 9.1 先估计干净动作

在噪声等级 $$k$$，由预测噪声估计：

$$
\widehat A^0
=
\frac{
A^k-\sqrt{1-\bar\alpha_k}\epsilon_\theta(A^k,k,c)
}{
\sqrt{\bar\alpha_k}
}.
$$

若希望从 $$k$$ 跳到更低等级 $$j<k$$，需要使用与该跨度一致的更新式。

### 9.2 任意两个噪声等级之间的 DDIM 更新

令：

$$
\sigma_{k\rightarrow j}
=
\eta
\sqrt{\frac{1-\bar\alpha_j}{1-\bar\alpha_k}}
\sqrt{1-\frac{\bar\alpha_k}{\bar\alpha_j}}.
$$

DDIM 家族的更新为：

$$
\boxed{
A^j
=
\sqrt{\bar\alpha_j}\widehat A^0
+
\sqrt{1-\bar\alpha_j-\sigma_{k\rightarrow j}^2}
\,\epsilon_\theta(A^k,k,c)
+
\sigma_{k\rightarrow j}z.
}
$$

当 $$\eta=0$$ 时，没有新的随机噪声注入；给定初始 $$A^K$$ 与条件，生成过程是确定性的。但初始高斯噪声仍可改变，所以仍然可以生成多个动作模式。[DDIM 论文][ddim]

对相邻等级 $$j=k-1$$，$$\eta=1$$ 时该方差恢复 $$\widetilde\beta_k$$，在对应未裁剪参数化下可联系到 DDPM 更新。

### 9.3 训练等级数与推理调用次数分开看

训练可能使用 $$K$$ 个噪声等级，推理只选其中 $$S$$ 个，且 $$S<K$$。这并不要求一定重新训练，但需要匹配的采样器、时间网格与边界设置。

例如，“训练 100 个等级、使用 DDIM 进行 10 次网络调用”与“把 DDPM 循环随意砍到前 10 次”不是同一个操作。后者可能还没有去到干净动作端点。

少步推理应比较：

- 在同一数据、同一初始噪声条件下，不同步数的动作分布与轨迹质量；
- 完整策略调用延迟，而不只统计网络前向耗时；
- 闭环任务成功率，而不只比较离线重建 MSE。

去噪步数越少，积分或离散化近似通常越粗；不能从“模型是 diffusion”直接推出某个固定步数足够。

## 10. 数据与动作表示决定学到什么

### 10.1 位置、增量、速度与力矩不能混用

单步动作 $$a_t$$ 可以是：

- 关节目标位置；
- 末端执行器的绝对位姿或相对位姿；
- 关节或末端速度；
- 力矩、夹爪命令，或多种量的组合。

预测相同数字，在不同动作定义下会导致完全不同的执行结果。

例如速度误差会通过积分影响位置：

$$
e_p(N)
\approx
\Delta t\sum_{i=0}^{N-1}e_v(i).
$$

因此长动作片段中的速度偏差可能累积。位置目标通常由底层控制器跟踪，但也有工作空间、跟踪误差和动态限制。论文报告的位置控制优势是实验结果，不是所有机器人上的普遍结论。[Diffusion Policy 第 4.2 节][dp]

### 10.2 归一化不仅影响训练，也影响生成空间

若各维分别代表米、弧度和夹爪开度，直接对它们加入同尺度噪声，会引入不合适的相对尺度。

对普通标量动作维度，一种常见变换是：

$$
\widetilde a_j
=
2\frac{a_j-a_j^{\min}}
{a_j^{\max}-a_j^{\min}}
-1.
$$

逆变换为：

$$
a_j
=
\frac{\widetilde a_j+1}{2}
(a_j^{\max}-a_j^{\min})
+
a_j^{\min}.
$$

近乎常量的维度需要单独处理，避免除以极小范围；旋转表示也不应未经检查就照搬普通标量缩放。

归一化应与采样器一致。例如某些 scheduler 会将**估计的干净动作**裁剪到 $$[-1,1]$$；如果训练使用标准化后大量超出该区间的数据，裁剪就会改变可生成范围。[论文附录 A.1][dp]、[官方 scheduler 配置][config]

因此，采样器中的 clipping 不只是“防止数值出错”的无害开关，它会修改实际生成过程。前面未裁剪的解析式与启用裁剪的实现，需要区分。

### 10.3 窗口不能跨 episode，观测不能偷看未来

每个窗口应该来自同一次示范轨迹，并保持统一时序：

$$
o_t
\;\longrightarrow\;
a_t
\;\longrightarrow\;
o_{t+1}.
$$

如果把动作执行后的图像误当成执行前的条件，训练会出现未来信息泄露。离线指标可能很好，部署时却没有同样的信息。

轨迹边界还需要 padding 策略。官方序列采样器会对不足的边界位置重复边界值；这会影响模型看到的动作分布。[官方序列采样器][sampler]

如果改用有效位置 mask，则损失分母也要与有效元素数量一致。不要一边补零一边把这些位置当成真实动作，也不要让滑动窗口跨越两次不相干的 episode。

### 10.4 闭环并不自动消除示范分布之外的错误

模型在示范观测分布上训练：

$$
O\sim d_{\mathrm{demo}}.
$$

部署时，自己的动作决定之后看到什么：

$$
O\sim d_{\pi_\theta}.
$$

两者不一定相同。小误差导致未见状态，未见状态又可能触发更大的动作误差。滚动重规划提供了利用反馈的机会，但若数据中没有对应的恢复行为，反馈本身不会凭空产生修复能力。

因此，采集多种起始状态、接触变化与恢复轨迹，以及按完整 episode 划分训练验证集，常常比只降低去噪训练误差更有意义。

## 11. 将核心推导写成 PyTorch

下面是**条件噪声预测训练 + 完整 DDPM 反向采样**的教学实现。它不包含视觉骨干和时序 U-Net，调用方提供：

- `encoder(obs) -> condition`，条件形状为 `[B, condition_dim]`；
- `denoiser(noisy_actions, step_index, condition) -> noise`，输出形状为 `[B, T_p, d_a]`。

代码中的 `step_index=i` 从 0 开始，对应公式中的 $$k=i+1$$。动作须预先归一化，采样结果也处于归一化空间；所有张量与模型应使用匹配的设备和浮点类型。

### 11.1 噪声日程、训练损失与反向采样

~~~python
import math
import torch


def cosine_betas(num_steps=100, offset=0.008):
    # 用余弦累积信号构造日程；最后限制 beta < 1
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    s = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64)
    cumulative = torch.cos((s + offset) / (1 + offset) * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(min=1e-5, max=0.999).float()


def schedule_terms(betas):
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    previous = torch.cat([torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]])
    posterior_variance = betas * (1 - previous) / (1 - alpha_bars)
    return alphas, alpha_bars, posterior_variance


def diffusion_loss(encoder, denoiser, obs, clean_actions, betas):
    # clean_actions: [B, T_p, d_a]，已经归一化
    condition = encoder(obs)
    betas = betas.to(device=clean_actions.device, dtype=clean_actions.dtype)
    _, alpha_bars, _ = schedule_terms(betas)

    batch = clean_actions.shape[0]
    index = torch.randint(len(betas), (batch,), device=clean_actions.device)
    alpha_bar = alpha_bars[index].reshape(batch, 1, 1)
    noise = torch.randn_like(clean_actions)
    noisy_actions = (
        alpha_bar.sqrt() * clean_actions
        + (1 - alpha_bar).sqrt() * noise
    )

    predicted_noise = denoiser(noisy_actions, index, condition)
    # 对 batch、动作位置与动作维度求平均
    return (predicted_noise - noise).square().mean()


@torch.no_grad()
def sample_ddpm(encoder, denoiser, obs, action_shape, betas):
    # 调用前将 encoder 和 denoiser 设为 eval 模式
    # action_shape = (T_p, d_a)
    condition = encoder(obs)  # 每轮动作生成只编码一次
    betas = betas.to(device=condition.device, dtype=condition.dtype)
    alphas, alpha_bars, posterior_variance = schedule_terms(betas)

    batch = condition.shape[0]
    actions = torch.randn(
        (batch, *action_shape),
        device=condition.device,
        dtype=condition.dtype,
    )

    # 使用全部相邻噪声等级，不可直接改成任意跳步
    for i in range(len(betas) - 1, -1, -1):
        index = torch.full(
            (batch,), i, device=condition.device, dtype=torch.long
        )
        predicted_noise = denoiser(actions, index, condition)
        mean = (
            actions
            - betas[i] / (1 - alpha_bars[i]).sqrt() * predicted_noise
        ) / alphas[i].sqrt()

        if i > 0:
            actions = (
                mean
                + posterior_variance[i].sqrt() * torch.randn_like(actions)
            )
        else:
            actions = mean  # 最后一步不再加随机噪声

    return actions
~~~

训练时使用 `diffusion_loss`，再执行常规的清梯度、反向传播与优化器更新。生成时分别调用 `encoder.eval()`、`denoiser.eval()`，然后使用 `sample_ddpm`。

这段教学代码采用后验方差、未裁剪均值和完整相邻步采样，目的是逐项对应第 3—5 节。官方实现还包含 normalizer、EMA、裁剪和不同 scheduler 等选项，因此不应把这段核心代码当作已经完成训练的机器人策略。

### 11.2 还原物理单位后，按时间对齐取动作

假设已按训练时保存的统计量完成逆归一化，得到 `physical_actions`。若采用第 8.2 节的官方窗口排列：

~~~python
start = n_obs_steps - 1
end = start + n_action_steps

if end > physical_actions.shape[1]:
    raise ValueError("执行窗口超出预测动作窗口")

actions_to_execute = physical_actions[:, start:end]
~~~

实际机器人部署还需要把这些动作解释为正确坐标系下、正确时间戳对应的控制命令。仅有一个形状正确的张量，并不能证明时序与动作语义正确。

### 11.3 对照官方实现时建议按什么顺序读

| 文件或函数 | 优先弄清的问题 |
| --- | --- |
| [配置文件][config] | horizon、观测长度、执行长度、训练噪声等级和推理次数分别是多少？ |
| [`compute_loss`][hybrid-policy] | 归一化、噪声采样、时间采样、网络标签在哪里构造？ |
| [`predict_action` / `conditional_sample`][hybrid-policy] | 视觉何时编码？采样循环如何使用 scheduler？如何截取动作？ |
| [`ConditionalUnet1D`][unet] | 时序维度如何转置？扩散时间与观测条件如何注入？ |
| [`TransformerForDiffusion`][transformer] | 动作位置编码、时间 token、条件 attention 与 mask 如何配合？ |
| [序列采样器][sampler] | episode 边界如何处理？动作与观测窗口怎样对齐？ |

阅读时把论文符号与实现索引分开记录。例如 DDPM 公式使用 $$k=1,\ldots,K$$，代码 scheduler 常用 $$0,\ldots,K-1$$；这不表示前向过程少了一步，而是数组索引习惯不同。

## 12. 与 Flow Matching、VLA 和低层控制串起来

### 12.1 Diffusion Policy 的结构不只是一条损失函数

可以把它拆成四层：

$$
\text{观测编码}
\;\rightarrow\;
\text{条件动作生成}
\;\rightarrow\;
\text{动作块截取与滚动执行}
\;\rightarrow\;
\text{底层控制器}.
$$

其中“条件动作生成”可以由不同生成模型实现；“动作块与滚动执行”也不是扩散模型独有的机制。

### 12.2 如果将动作生成换成 Flow Matching

沿用同一个条件 $$c=f_\phi(O)$$，令 $$A_{\mathrm{data}}$$ 为示范动作块，$$\epsilon$$ 为高斯噪声。用 $$s\in[0,1]$$ 表示**噪声到动作**方向的连续时间：

$$
X_s=(1-s)\epsilon+sA_{\mathrm{data}},
$$

$$
U_s=A_{\mathrm{data}}-\epsilon.
$$

训练速度场：

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E\left[
\|v_\theta(s,X_s,c)-U_s\|^2
\right].
$$

生成则求解：

$$
\frac{\mathrm dX_s}{\mathrm ds}
=
v_\theta(s,X_s,c),
\qquad X_0\sim\mathcal N(0,I).
$$

观测编码、动作尺度与滚动执行仍然需要处理，只是生成路径、预测目标与求解器改变了。

| 维度 | 本文 DDPM 形式的 Diffusion Policy | 线性路径 Flow Matching 动作策略 |
| --- | --- | --- |
| 训练状态 | $$\sqrt{\bar\alpha_k}A^0+\sqrt{1-\bar\alpha_k}\epsilon$$ | $$(1-s)\epsilon+sA_{\mathrm{data}}$$ |
| 常见监督 | 噪声 $$\epsilon$$ | 速度 $$A_{\mathrm{data}}-\epsilon$$ |
| 生成方式 | DDPM、DDIM 等匹配的采样器 | 对速度场数值积分 |
| 条件输入 | 观测编码、任务条件等 | 同样可以使用 |
| 动作块与闭环 | 需要设计 | 同样需要设计 |

两者在合适路径下可以通过 score、噪声与速度的关系建立联系，但不能只替换变量名就认为训练权重和采样公式相同。详见 [Flow Matching 原理笔记]({% post_url 2026-04-01-flowmatching %})。

### 12.3 VLA 扩展了条件与模型能力，低层控制仍有职责

在视觉—语言—动作模型中，条件还可以包含语言指令和更强的视觉语言表征；动作生成部分则可以使用扩散、Flow Matching 或其他形式。本博客的 [π₀ 笔记]({% post_url 2026-04-02-pi0 %})可以作为后续阅读。

无论条件编码多强，生成的动作都需要由实际控制系统执行。动作策略主要决定“接下来想做什么”，伺服或阻抗控制器负责跟踪、接触响应与硬件执行。模型的统计生成能力，不能替代对坐标系、单位、时延和机器人动态特性的理解。

## 13. 学习检查：能否独立回答这些问题

**1. Diffusion Policy 输出的是噪声还是动作？**

训练中的去噪网络通常输出预测噪声；整个策略通过采样器反复调用它，最后输出动作块。网络模块的输出与完整策略的输出不是同一个对象。

**2. 为什么训练时一次前向就够，推理时却要多步？**

训练有真实干净动作，可以直接合成任意等级的加噪样本。推理没有干净动作，需要沿反向生成过程逐步构造。

**3. 为什么噪声回归的 MSE 不会必然变成平均动作？**

它预测的是给定加噪动作、噪声等级和观测后的噪声条件均值，并通过整条采样链表达分布；它不是直接输出 $$\mathbb E[A^0\mid O]$$。

**4. 为什么动作块输出联合相关，但初始噪声可以逐维独立？**

相关性由整个序列上的条件去噪网络建立，不必预先写进高斯先验。

**5. 去噪时接收了 100 次新图像吗？**

通常没有。一轮生成使用固定观测特征；完成动作片段后才进入下一次策略调用。

**6. 将 $$T_a$$ 设为 1，就一定最稳定吗？**

不一定。反馈更频繁，但计算负担、模式切换与时延也可能更明显，应该结合任务闭环评估。

**7. 为什么不能只看离线 loss 选择机器人策略？**

去噪误差只评价数据分布上的局部预测；真实执行还受到状态分布变化、动作时序、积分或采样误差与底层控制的影响。

**8. 这是否等价于 MPC 或强化学习？**

基本方法学习示范的条件动作分布，并采用滚动执行。它没有因此自动获得 MPC 的动力学约束优化，也没有自动引入强化学习的奖励最大化目标。

### 建议动手完成的三个小练习

1. **核对后验公式。** 对同一组 $$A^0,\epsilon,k$$，分别使用“干净动作加权”的后验均值与“噪声参数化”的均值，验证数值一致，并检查最后一步后验方差为零。
2. **观察双峰分布。** 取 $$m=2,\sigma_0=0.15$$，比较不同 $$\bar\alpha_k$$ 下第 6 节的 score；再比较条件均值预测和多次独立采样得到的动作分布。
3. **检查动作对齐。** 构造带时间编号的假数据，例如令 $$a_t=t$$，使用 $$T_o=2,T_p=16,T_a=8$$ 切片，确认执行的是当前时刻到之后 7 步，而不是重新执行过去的动作。

读懂这些问题后，再看一个新的扩散动作策略，可以依次记录：**条件是什么、动作怎样表示、窗口怎样对齐、预测什么目标、用什么采样器、多久重新利用一次反馈。** 这样更容易分清模型创新、控制设计与实现细节。

## 参考资料

1. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*，RSS 2023；IJRR 2024 扩展版。[论文][dp]、[项目页][project]
2. Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*，NeurIPS 2020。重点阅读前向边际、反向后验与简化训练目标。[论文][ddpm]
3. Song, Meng & Ermon, *Denoising Diffusion Implicit Models*，ICLR 2021。理解确定性采样、跳步时间网格与加速更新。[论文][ddim]
4. 官方 Diffusion Policy 仓库。建议将论文和具体配置、策略类、采样器一起阅读。[代码][repo]
5. 本站关联笔记：[Flow Matching]({% post_url 2026-04-01-flowmatching %})、[PID 与 MPC]({% post_url 2026-09-18-pid-mpc-control %})、[π₀]({% post_url 2026-04-02-pi0 %})。

[dp]: https://arxiv.org/abs/2303.04137
[project]: https://diffusion-policy.cs.columbia.edu/
[ddpm]: https://arxiv.org/abs/2006.11239
[ddim]: https://arxiv.org/abs/2010.02502
[repo]: https://github.com/real-stanford/diffusion_policy
[config]: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml
[image-policy]: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py
[hybrid-policy]: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py
[unet]: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py
[transformer]: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
[sampler]: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py
