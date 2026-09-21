---
title: "【生成模型】Flow Matching：从连续性方程到条件流匹配的原理与推导"
date: 2026-04-01 16:00:00 +0800
categories: [多模态模型]
tags: [flow-matching, diffusion, generative-model, rectified-flow, optimal-transport, ode]
description: 从概率输运与连续性方程出发，推导边际速度场、条件流匹配的等价目标和高斯概率路径；用可手算例子解释直线插值与生成轨迹的区别，并梳理最优传输、Rectified Flow、扩散及训练采样实现。
image:
  path: /assets/images/posts/placeholders/paper-note.svg
  alt: Flow Matching 的概率输运、速度场与条件回归
mermaid: true
math: true
---

> **Flow Matching（流匹配，FM）把生成建模写成一个速度场学习问题：从噪声出发，每一时刻应该朝哪个方向、以多快的速度移动，才能让最终样本服从数据分布？**
>
> 它最巧妙的地方，是把难以直接计算的**整体分布速度**，转化成可以用数据样本构造的**条件速度回归**。理解这一步，才能解释为什么训练时不需要求解 ODE，推理时却能通过 ODE 生成样本。

先看最常见的三行公式。令 $$X_0$$ 是噪声，$$X_1$$ 是数据，二者独立采样：

$$
X_t=(1-t)X_0+tX_1,
\qquad
U_t=X_1-X_0,
$$

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E\left[
\left\|v_\theta(t,X_t)-U_t\right\|^2
\right].
$$

它们很容易实现，但还留下三个问题：

1. 一个噪声可以与许多数据配对，监督速度彼此冲突，网络究竟学什么？
2. 把速度预测准确，为什么就能把噪声分布变成数据分布？
3. 训练样本走直线，是否意味着模型生成时也走直线，甚至一步就能生成？

本文从这三个问题出发。需要的基础是概率期望、条件概率、微积分和最小二乘；涉及偏微分方程的地方，会同时给出粒子运动与概率守恒的解释。核心构造依据 [Flow Matching 原论文][fm]，其余推导统一使用下面的时间与符号约定。

## 阅读路线与符号约定

**全文统一采用：$$t=0$$ 是噪声，$$t=1$$ 是数据。** 许多扩散论文采用相反方向，比较公式时必须先对齐时间。

| 符号 | 含义 |
| --- | --- |
| $$p_0,\ p_{\mathrm{data}}$$ | 源分布与目标数据分布，通常 $$p_0=\mathcal N(0,I)$$ |
| $$\pi(x_0,x_1)$$ | 端点的联合分布，即耦合；边际分别是 $$p_0$$ 和 $$p_{\mathrm{data}}$$ |
| $$X_t,\ p_t$$ | 人为构造的随机插值，以及它在时刻 $$t$$ 的分布 |
| $$U_t=\dot X_t$$ | 一条训练插值路径上的速度标签 |
| $$u_t(x)$$ | 能产生目标概率路径 $$p_t$$ 的边际速度场 |
| $$v_\theta(t,x)$$ | 神经网络预测的速度场 |
| $$Z_t,\ q_t^\theta$$ | 沿网络 ODE 积分的生成样本，以及它的分布 |
| $$\alpha_t,\beta_t$$ | 数据系数、噪声系数；上方的点表示对时间求导 |
| $$s_t(x)$$ | score，即 $$\nabla_x\log p_t(x)$$，是向量而不是概率值 |

特意用 $$X_t$$ 和 $$Z_t$$ 区分**训练插值**与**生成轨迹**。理想情况下，两者在每个时刻的分布相同，但逐条轨迹和端点配对不必相同。

~~~mermaid
flowchart TD
    A["选择源分布、数据分布与端点耦合"] --> B["构造可采样的插值 X_t"]
    B --> C["对时间求导，得到速度标签 U_t"]
    C --> D["条件速度回归：训练 vθ(t, X_t)"]
    D --> E["最优回归函数：E[U_t | X_t = x]"]
    E --> F["连续性方程：保证目标分布的演化"]
    D --> G["生成：从新噪声出发求解网络 ODE"]
    F --> G
~~~

第 1—4 节给出理论主线，第 5—6 节把它落实到高斯路径和手算例子，第 7—8 节梳理 OT、Rectified Flow 与扩散，第 9—11 节讨论数值采样、实现和交互演示。首次阅读可以先读第 3、4、6 节，再回看连续性方程。

## 1. 从生成模型到连续概率输运

### 1.1 ODE 如何生成样本

从简单分布抽取初始状态，然后求解常微分方程：

$$
Z_0\sim p_0,
\qquad
\frac{\mathrm d Z_t}{\mathrm dt}
=
v_\theta(t,Z_t),
\qquad t\in[0,1].
$$

记从初始位置到时刻 $$t$$ 的流映射为 $$\phi_t^\theta$$，则：

$$
Z_t=\phi_t^\theta(Z_0),
\qquad
q_t^\theta=(\phi_t^\theta)_\#p_0.
$$

符号 $$\#$$ 表示**推前分布**：把服从 $$p_0$$ 的随机样本通过同一个映射，得到新的分布。生成建模希望：

$$
q_1^\theta\approx p_{\mathrm{data}}.
$$

ODE 在给定初始噪声后是确定性的；**生成结果的随机性来自初始噪声**。因此，确定性动力学完全可以产生丰富的样本。

当速度场对空间满足适当的局部 Lipschitz 条件，并有保证有限时间内解存在的增长控制时，ODE 才有良好的存在唯一性。后面“速度场正确就能得到正确分布”的结论，都需要相应的正则性，而不能只凭一条损失函数推出。

### 1.2 CNF 是模型形式，FM 是训练方法

这类由连续流定义的生成模型称为 **Continuous Normalizing Flow（CNF）**。沿生成轨迹，其密度满足：

$$
\frac{\mathrm d}{\mathrm dt}\log q_t^\theta(Z_t)
=
-\nabla\cdot v_\theta(t,Z_t).
$$

所以：

$$
\log q_1^\theta(Z_1)
=
\log p_0(Z_0)
-
\int_0^1\nabla\cdot v_\theta(t,Z_t)\,\mathrm dt.
$$

其中散度是速度场 Jacobian 的迹：

$$
\nabla\cdot v_\theta
=
\sum_{i=1}^{d}
\frac{\partial v_{\theta,i}}{\partial x_i}.
$$

直接用最大似然训练 CNF，通常需要在训练中进行数值积分，并计算或估计散度。FM 提供另一条训练路线：**先指定希望经过的概率路径，再回归能实现这条路径的速度场**。基本 CFM 训练不需要求解模型 ODE，也不需要计算上述散度；若之后要评估似然，这些计算仍可能需要。参见 [FM 原论文第 2—3 节及附录 C][fm]。

## 2. 连续性方程：粒子运动怎样改变分布

### 2.1 从区域内的概率守恒理解

假设粒子以速度 $$u_t(x)$$ 移动，时刻 $$t$$ 的密度是 $$p_t(x)$$。固定区域 $$A$$ 内的概率变化，等于穿过边界流出的概率通量的负值：

$$
\frac{\mathrm d}{\mathrm dt}\int_A p_t(x)\,\mathrm dx
=
-\int_{\partial A}p_t(x)u_t(x)\cdot n(x)\,\mathrm dS,
$$

其中 $$n(x)$$ 是外法向量。用散度定理：

$$
\int_A
\left[
\partial_t p_t(x)
+
\nabla\cdot\bigl(p_t(x)u_t(x)\bigr)
\right]\mathrm dx=0.
$$

因为区域可以任意选取，得到**连续性方程**：

$$
\boxed{
\partial_t p_t+\nabla\cdot(p_tu_t)=0.
}
$$

它说的是：概率质量不会凭空消失或出现，只会随着速度场转移。

### 2.2 用测试函数推导，连接到后面的条件期望

取光滑、具有紧支撑的函数 $$f$$。沿粒子轨迹求导：

$$
\frac{\mathrm d}{\mathrm dt}\mathbb E[f(Z_t)]
=
\mathbb E\left[\nabla f(Z_t)\cdot u_t(Z_t)\right].
$$

如果 $$Z_t$$ 的密度是 $$p_t$$，右侧可以写成积分，再分部积分：

$$
\begin{aligned}
\int f(x)\partial_t p_t(x)\,\mathrm dx
&=
\int \nabla f(x)\cdot u_t(x)p_t(x)\,\mathrm dx\\
&=
-\int f(x)\nabla\cdot(p_t(x)u_t(x))\,\mathrm dx.
\end{aligned}
$$

这就是连续性方程的弱形式。它的好处是：不必把每条条件路径都写成普通密度，也能讨论概率随时间的变化。

因此，FM 的理论链条是：

> 构造目标概率路径 $$p_t$$ → 找到满足连续性方程的 $$u_t$$ → 用网络逼近 $$u_t$$ → 在相应唯一性条件下，网络 ODE 产生相同的边际分布。

还要注意，**给定 $$p_t$$，速度场不一定唯一**。若另一个场 $$w_t$$ 满足 $$\nabla\cdot(p_tw_t)=0$$，那么 $$u_t+w_t$$ 也满足同一连续性方程。FM 的路径与条件构造，为我们选出了一个具体的回归目标。

## 3. 从条件路径构造边际速度场

### 3.1 最容易采样的路径：端点线性插值

先选一个耦合：

$$
(X_0,X_1)\sim\pi,
\qquad
\pi_0=p_0,\quad \pi_1=p_{\mathrm{data}}.
$$

最简单的选择是独立耦合：

$$
\pi(x_0,x_1)=p_0(x_0)p_{\mathrm{data}}(x_1).
$$

然后定义：

$$
X_t=(1-t)X_0+tX_1.
$$

固定一对端点，对时间求导：

$$
\boxed{U_t=\dot X_t=X_1-X_0.}
$$

这就是训练所需的标签。它只是**这对端点的插值速度**，还不能直接写成整个分布在位置 $$x$$ 的速度 $$u_t(x)$$。

### 3.2 为什么需要条件平均

同一个时刻、同一个位置附近，可能有许多不同端点配对产生的训练样本，它们的速度标签不一样。网络只看到 $$t,x$$，无法知道标签来自哪一对端点。

合理的确定性预测是：

$$
\boxed{
u_t(x)
=
\mathbb E[U_t\mid X_t=x]
=
\mathbb E[X_1-X_0\mid X_t=x].
}
$$

这个平均是**给定当前位置后的后验平均**，而不是对所有端点不加区别地求平均。哪些端点更可能产生当前 $$x$$，它们就占更高权重。

为了看清这一点，令 $$C$$ 表示构造条件路径的变量，例如一个数据端点。设条件密度是 $$p_t(x\mid c)$$，条件速度场是 $$u_t(x\mid c)$$，条件变量的分布为 $$r(c)$$，则：

$$
p_t(x)=\int p_t(x\mid c)r(c)\,\mathrm dc,
$$

$$
\boxed{
u_t(x)
=
\int
u_t(x\mid c)
\frac{p_t(x\mid c)r(c)}{p_t(x)}
\,\mathrm dc.
}
$$

其中分式正是后验密度 $$r_t(c\mid x)$$。该表达式在 $$p_t(x)>0$$ 的地方成立；更一般地，可用条件期望作几乎处处的定义。[FM 原论文第 3.1 节][fm]

### 3.3 这个平均速度为什么正确

对任意光滑测试函数 $$f$$，使用训练插值本身：

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}\mathbb E[f(X_t)]
&=
\mathbb E[\nabla f(X_t)\cdot U_t]\\
&=
\mathbb E\left[
\nabla f(X_t)\cdot
\mathbb E[U_t\mid X_t]
\right]\\
&=
\mathbb E[\nabla f(X_t)\cdot u_t(X_t)].
\end{aligned}
$$

第二步使用条件期望的塔式法则。与第 2 节比较可知：

$$
\partial_t p_t+\nabla\cdot(p_tu_t)=0.
$$

因此，只要交换求导与期望合法、速度可积，并且相应 ODE 与分布演化问题适定，求解：

$$
\dot Z_t=u_t(Z_t),\qquad Z_0\sim p_0
$$

就能得到：

$$
\boxed{\operatorname{Law}(Z_t)=\operatorname{Law}(X_t)=p_t.}
$$

这个结论保证的是**每个时刻的边际分布**，不是整个随机过程完全相同。训练插值需要预先知道两个端点；生成 ODE 只需要当前状态，却能在分布层面复现同一演化。

这也解释了一个看似矛盾的现象：训练插值可以在同一时刻交叉，并携带不同速度；具有唯一解的 ODE 不能在同一时刻、同一位置给出两个不同方向。它通过条件平均重新组织轨迹，而不必保留原配对。

## 4. CFM 损失为何等价于 FM 损失

### 4.1 理想目标与可计算目标

如果知道边际速度场，可以直接最小化：

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{t,X_t}
\left[
\|v_\theta(t,X_t)-u_t(X_t)\|^2
\right].
$$

但 $$u_t$$ 包含难以计算的条件期望。条件流匹配使用容易采样的标签：

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{t,X_0,X_1}
\left[
\|v_\theta(t,X_t)-U_t\|^2
\right].
$$

下面证明二者为何具有相同的期望梯度。这也是 [FM 原论文定理 2][fm] 的核心结论。

### 4.2 展开平方：消失的交叉项

暂时简写 $$v=v_\theta(t,X_t)$$、$$u=u_t(X_t)$$、$$U=U_t$$。有：

$$
v-U=(v-u)+(u-U).
$$

所以：

$$
\|v-U\|^2
=
\|v-u\|^2
+
\|u-U\|^2
+
2(v-u)^\top(u-U).
$$

在给定 $$t,X_t$$ 后，$$v-u$$ 是确定的，而：

$$
\mathbb E[u-U\mid t,X_t]=0.
$$

因此交叉项的期望为零，得到：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)
+
\mathbb E\left[
\|U_t-u_t(X_t)\|^2
\right].
}
$$

最后一项可写成条件协方差的迹的期望：

$$
\mathbb E\left[
\operatorname{tr}\operatorname{Cov}(U_t\mid t,X_t)
\right].
$$

当端点耦合、路径和采样规则固定，且不依赖 $$\theta$$ 时，它与网络参数无关。因此：

$$
\boxed{
\nabla_\theta\mathcal L_{\mathrm{CFM}}
=
\nabla_\theta\mathcal L_{\mathrm{FM}}.
}
$$

这就是平方损失中“条件期望是最优预测”的正交分解。实际训练用一个端点对给出一个随机标签；大量样本上的回归，隐式完成了条件平均。

### 4.3 这个等价性意味着什么，又不意味着什么

- **不需要显式算出 $$p_t(x)$$ 或后验。** 能采样 $$X_t$$ 并计算 $$U_t$$ 就可以训练。
- **CFM 损失不一定收敛到零。** 不同配对的速度存在条件方差，即使网络已经精确等于 $$u_t$$，仍可能有正的回归损失。
- **梯度相同指期望相同。** 单个 minibatch 的梯度仍有随机性，不同耦合也会改变标签冲突与梯度方差。
- **不能把有限训练损失直接等同于生成质量。** 网络近似误差、ODE 稳定性、数值积分误差都会影响最终分布。

这里的 **Conditional** 指“通过简单条件路径构造训练目标”。它不自动意味着模型带有文本、类别或图像条件；后者是另一个层面的条件生成。

## 5. 高斯概率路径：从一般形式推到常用公式

### 5.1 数据与高斯噪声的仿射组合

令 $$X_1\sim p_{\mathrm{data}}$$，$$\varepsilon\sim\mathcal N(0,I)$$，并且二者独立。定义：

$$
X_t=\alpha_tX_1+\beta_t\varepsilon,
$$

$$
\alpha_0=0,\quad \beta_0=1,
\qquad
\alpha_1=1,\quad \beta_1=0.
$$

于是，条件于数据点 $$X_1=x_1$$：

$$
p_t(x\mid x_1)
=
\mathcal N(x;\alpha_tx_1,\beta_t^2I).
$$

对时间求导，直接得到可采样的速度标签：

$$
\boxed{
U_t=\dot\alpha_tX_1+\dot\beta_t\varepsilon.
}
$$

在 $$\beta_t>0$$ 时，消去噪声：

$$
\varepsilon=\frac{x-\alpha_tx_1}{\beta_t},
$$

得到条件速度场：

$$
\boxed{
u_t(x\mid x_1)
=
\dot\alpha_tx_1
+
\frac{\dot\beta_t}{\beta_t}(x-\alpha_tx_1).
}
$$

这就是一般高斯路径公式的一个常用形式。更一般地，将均值写成 $$\mu_t(x_1)$$、标准差写成 $$\sigma_t(x_1)$$，便得到：

$$
u_t(x\mid x_1)
=
\dot\mu_t(x_1)
+
\frac{\dot\sigma_t(x_1)}{\sigma_t(x_1)}
\bigl(x-\mu_t(x_1)\bigr).
$$

它是所选仿射流对应的速度场；不能据此说“能产生该高斯密度路径的所有速度场都唯一”。[FM 原论文第 4 节][fm]

### 5.2 线性路径只是其中一种选择

取：

$$
\alpha_t=t,\qquad \beta_t=1-t.
$$

便有：

$$
X_t=tX_1+(1-t)\varepsilon,
\qquad
U_t=X_1-\varepsilon.
$$

条件速度还可以写成：

$$
u_t(x\mid x_1)
=
\frac{x_1-x}{1-t},
\qquad t<1.
$$

后一种形式在 $$t\to1$$ 时出现小分母，但代回训练路径：

$$
\frac{x_1-X_t}{1-t}=x_1-\varepsilon.
$$

因此，训练时直接使用端点差，可以避免通过小分母计算标签。**沿已采样路径的标签有限，并不表示终点附近的整个速度场一定光滑。**

### 5.3 保留终点噪声时，目标分布也改变

原始 FM 论文还讨论了：

$$
\alpha_t=t,
\qquad
\beta_t=1-(1-\sigma_{\min})t,
\qquad \sigma_{\min}>0.
$$

此时：

$$
X_t=tX_1+[1-(1-\sigma_{\min})t]\varepsilon,
$$

$$
U_t=X_1-(1-\sigma_{\min})\varepsilon,
$$

$$
u_t(x\mid x_1)
=
\frac{x_1-(1-\sigma_{\min})x}
{1-(1-\sigma_{\min})t}.
$$

终点是：

$$
X_{t=1}=X_1+\sigma_{\min}\varepsilon.
$$

所以最终分布是数据分布与高斯的卷积：

$$
p_{t=1}
=
p_{\mathrm{data}}*\mathcal N(0,\sigma_{\min}^2I),
$$

而不是严格等于原始数据分布。保留小噪声可以避免条件高斯在终点退化，但同时改变了要拟合的终点分布。

如果目标是经验分布中的离散点，或严格位于低维流形上，处处正则、可逆的有限时间流不能把满维高斯精确压成奇异分布。理论上需要讨论终点极限；实践中可采用平滑数据、保留噪声或提前终止积分等处理。不能同时假定“终点严格坍缩”和“全程良好可逆”。

## 6. 一个可手算的例子：直线训练不保证匀速生成

设：

$$
X_0\sim\mathcal N(0,1),
\qquad
X_1\sim\mathcal N(2,1),
\qquad X_0\perp X_1.
$$

构造线性插值：

$$
X_t=(1-t)X_0+tX_1.
$$

### 6.1 先求出中间分布

独立高斯的线性组合仍是高斯：

$$
p_t=\mathcal N(2t,a_t),
\qquad
a_t=(1-t)^2+t^2.
$$

这里 $$a_t$$ 是方差。它从 $$1$$ 降到 $$1/2$$，再回到 $$1$$。也就是说，即使两端方差一样，**独立配对也会让中间分布先收缩再展开**。

### 6.2 再求出真正的边际速度

令 $$U=X_1-X_0$$。有：

$$
\mathbb E[U]=2,
\qquad
\operatorname{Cov}(U,X_t)=2t-1.
$$

对联合高斯变量使用条件期望公式：

$$
\boxed{
u_t(x)
=
2+\frac{2t-1}{a_t}(x-2t).
}
$$

这和“每条训练直线上的 $$X_1-X_0$$”不同：它依赖当前位置和时间。

在三个特殊时刻：

$$
u_0(x)=2-x,
\qquad
u_{1/2}(x)=2,
\qquad
u_1(x)=x.
$$

初期既向右平移又收缩，中点只平移，后期向右平移并展开，恰好对应刚才算出的均值和方差变化。

### 6.3 生成 ODE 的解析解

取一个初始噪声 $$Z_0=z_0$$，则：

$$
\boxed{
Z_t=2t+\sqrt{a_t}\,z_0.
}
$$

验证只需对它求导：

$$
\frac{\mathrm dZ_t}{\mathrm dt}
=
2+\frac{2t-1}{\sqrt{a_t}}z_0
=
u_t(Z_t).
$$

若 $$Z_0\sim\mathcal N(0,1)$$，那么：

$$
Z_t\sim\mathcal N(2t,a_t)=p_t,
\qquad
Z_1=2+Z_0\sim\mathcal N(2,1).
$$

以 $$z_0=1$$ 为例：

| $$t$$ | $$a_t$$ | $$Z_t=2t+\sqrt{a_t}$$ |
| --- | --- | --- |
| $$0$$ | $$1$$ | $$1.000$$ |
| $$0.25$$ | $$0.625$$ | $$1.291$$ |
| $$0.5$$ | $$0.5$$ | $$1.707$$ |
| $$0.75$$ | $$0.625$$ | $$2.291$$ |
| $$1$$ | $$1$$ | $$3.000$$ |

时间—位置曲线不是直线，速度也不是常数。在一维中，这体现为沿同一直线变速；在更高维中，边际速度的方向还可能变化，产生空间中的弯曲轨迹。

### 6.4 为什么一步 Euler 会失败

如果从 $$t=0$$ 直接跨到 $$t=1$$：

$$
Z_1^{\mathrm{Euler}}
=
Z_0+u_0(Z_0)
=
Z_0+(2-Z_0)
=
2.
$$

所有噪声都被映射到均值 $$2$$，方差完全丢失。可见，**即使速度场预测完全正确，一步离散化也可能严重错误**。

还可以验证最优 CFM 损失为什么不为零：

$$
\begin{aligned}
\operatorname{Var}(U\mid X_t)
&=
2-\frac{(2t-1)^2}{a_t}\\
&=
\frac{1}{a_t}>0.
\end{aligned}
$$

在 $$t=1/2$$ 时，最优预测对所有位置都是 $$2$$，但标签仍有方差 $$2$$。这是条件不确定性，不是网络“还没学好”。

## 7. Flow Matching、最优传输与 Rectified Flow

### 7.1 直线插值没有决定端点如何配对

任意耦合 $$\pi$$ 都可以定义：

$$
X_t=(1-t)X_0+tX_1.
$$

**最优传输（Optimal Transport，OT）额外优化的是耦合本身。** 以平方欧氏距离为代价：

$$
\pi^\star
\in
\arg\min_{\pi\in\Pi(p_0,p_{\mathrm{data}})}
\mathbb E_\pi[\|X_1-X_0\|^2].
$$

只有使用相应的最优耦合，线性插值才具有 Wasserstein-2 位移插值的意义。点对之间走最短线段，与所有点对的总运输代价最小，是两件事。

在适当条件下，动态 OT 将同一问题写成：

$$
\inf_{p,u}
\int_0^1\int
p_t(x)\|u_t(x)\|^2
\,\mathrm dx\,\mathrm dt,
$$

约束是：

$$
\partial_t p_t+\nabla\cdot(p_tu_t)=0,
\qquad
p_{t=0}=p_0,\quad p_{t=1}=p_{\mathrm{data}}.
$$

这里同时优化中间路径和速度，最小化整个分布的动能；普通 FM 则通常先选定一条路径，再学习对应的速度。[OT-CFM 论文][otcfm]

### 7.2 用同一个高斯例子比较耦合

第 6 节的独立配对，其运输代价为：

$$
\mathbb E[(X_1-X_0)^2]=2^2+1+1=6.
$$

而把每个点向右平移 $$2$$：

$$
X_1=X_0+2
$$

同样满足两个端点分布，代价却只有：

$$
\mathbb E[(X_1-X_0)^2]=4.
$$

任意耦合都满足 Jensen 不等式：

$$
\mathbb E[(X_1-X_0)^2]
\geq
\bigl(\mathbb E[X_1-X_0]\bigr)^2
=4.
$$

因此平移配对就是这个例子的最优耦合。使用它时：

$$
X_t=X_0+2t,\qquad u_t(x)=2,
$$

生成可以一步 Euler 精确完成。

独立配对和 OT 配对的训练公式表面上相同，但前者中间方差收缩，后者始终保持方差为 $$1$$。**耦合改变中间分布与回归难度，不改变预先指定的两个端点边际。**

有趣的是，第 6 节独立配对训练得到的理想 ODE，最终也实现了平移映射，但中间过程不是匀速平移。这进一步说明：端点映射、时间路径和数值采样难度需要分别讨论；该例的最优端点映射也不能推广成“一次 FM 总能解出 OT”。

### 7.3 “条件 OT 路径”与“全局 OT 耦合”

原始 FM 论文中的条件 OT 路径，是对固定数据点 $$x_1$$，在标准高斯与以 $$x_1$$ 为中心的小方差高斯之间使用 OT 映射。对数据点混合后，**并不自动得到从源分布到整个数据分布的全局 OT**。[FM 原论文第 4.1 节][fm]

OT-CFM 则研究通过端点的 OT 耦合构造条件路径。实际大数据训练常采用 minibatch OT：在一批噪声和数据之间构建距离代价矩阵，求解或近似求解运输计划，再据此采样配对。它是总体 OT 的近似；批大小、代价度量与正则化都会影响结果。[Tong 等人的 OT-CFM 论文][otcfm]

### 7.4 Rectified Flow 与 reflow 做了什么

Rectified Flow 的基本回归目标，与本文的线性插值 CFM 形式一致：

$$
\min_\theta
\mathbb E\left[
\|v_\theta(t,(1-t)X_0+tX_1)-(X_1-X_0)\|^2
\right].
$$

它进一步关注：能否通过重新组织端点配对，让实际 ODE 轨迹更接近匀速直线？**Reflow** 的基本步骤是：

1. 用已有速度场，从噪声 $$Z_0$$ 积分得到 $$Z_1$$。
2. 保留生成出的对应关系，形成新耦合 $$\operatorname{Law}(Z_0,Z_1)$$。
3. 对这批新端点重新做直线插值，训练下一个速度场。

新模型的标签仍是端点差，改变的是配对。Reflow 涉及生成配对时的 ODE 模拟；不能把“基础 CFM 的训练回归不需要模拟”扩大成“包括 reflow 的整个流程都不需要模拟”。

理想条件下，rectification 有边际保持、凸运输代价不增加等性质；reflow 用于进一步拉直流。但它不保证求解任意指定代价下的全局 OT，实际还会累积模型与数值误差。少步能力也不能仅凭“用了 Rectified Flow”就保证。[Rectified Flow 原论文][rf]

| 名称 | 主要回答的问题 |
| --- | --- |
| CNF | 用什么模型表示连续的分布变换？ |
| FM / CFM | 怎样通过速度回归训练这个模型？ |
| OT / OT-CFM | 怎样选取运输代价更小的端点耦合？ |
| Rectified Flow / reflow | 怎样重新组织配对，使实际生成流更容易少步积分？ |

## 8. 与扩散模型的联系：概率路径、score 与速度

### 8.1 扩散也能使用确定性 ODE

为了避免混淆，暂用 $$\tau$$ 表示**数据到噪声**方向的扩散时间，密度记为 $$r_\tau$$。对状态无关、各向同性扩散系数 $$g(\tau)$$，前向 SDE 为：

$$
\mathrm dY_\tau
=
f_\tau(Y_\tau)\,\mathrm d\tau
+
g(\tau)\,\mathrm dW_\tau.
$$

它的 Fokker–Planck 方程是：

$$
\partial_\tau r_\tau
=
-\nabla\cdot(f_\tau r_\tau)
+
\frac12g(\tau)^2\Delta r_\tau.
$$

利用：

$$
\Delta r_\tau
=
\nabla\cdot(r_\tau\nabla\log r_\tau),
$$

可改写为连续性方程：

$$
\partial_\tau r_\tau
=
-\nabla\cdot
\left[
r_\tau
\left(
f_\tau-\frac12g(\tau)^2\nabla\log r_\tau
\right)
\right].
$$

因此，对应的 **probability flow ODE** 是：

$$
\boxed{
\frac{\mathrm dY_\tau}{\mathrm d\tau}
=
f_\tau(Y_\tau)
-
\frac12g(\tau)^2\nabla\log r_\tau(Y_\tau).
}
$$

在相应条件和精确 score 下，它与 SDE 具有相同的单时刻边际分布。生成时可以沿这个 ODE 反向积分；若换成本文 $$t=1-\tau$$ 的方向，整个漂移还要乘以负号。[Song 等人的 SDE 论文，第 4.3 节][sde]

因此，“扩散一定随机、FM 才确定”并不是准确区分。FM 的特点在于**直接围绕可选概率路径构造速度回归目标**；常见扩散训练也可以直接采样任意时刻的加噪状态，不需要逐步模拟完整前向链。

### 8.2 由高斯路径推导 score 与条件均值

回到本文方向：

$$
X_t=\alpha_tX_1+\beta_t\varepsilon.
$$

本节继续假设高斯噪声与数据独立。条件高斯的 score 是：

$$
\nabla_x\log p_t(x\mid x_1)
=
-\frac{x-\alpha_tx_1}{\beta_t^2}.
$$

对混合密度求导，得到边际 score：

$$
\begin{aligned}
s_t(x)
&=
\nabla_x\log p_t(x)\\
&=
\mathbb E\left[
\nabla_x\log p_t(x\mid X_1)
\mid X_t=x
\right]\\
&=
-\frac{1}{\beta_t}
\mathbb E[\varepsilon\mid X_t=x].
\end{aligned}
$$

所以：

$$
\boxed{
\mathbb E[\varepsilon\mid X_t=x]
=
-\beta_ts_t(x).
}
$$

再对 $$X_t=\alpha_tX_1+\beta_t\varepsilon$$ 取条件期望：

$$
\boxed{
\mathbb E[X_1\mid X_t=x]
=
\frac{x+\beta_t^2s_t(x)}{\alpha_t}.
}
$$

这些式子要求对应分母非零，通常在时间区间内部使用。对于依赖端点的 OT 耦合，条件分布不再是这里的独立高斯加噪形式，不能直接照搬本节 score 恒等式。

### 8.3 速度预测与噪声预测如何互相转换

因为：

$$
u_t(x)
=
\dot\alpha_t\mathbb E[X_1\mid X_t=x]
+
\dot\beta_t\mathbb E[\varepsilon\mid X_t=x],
$$

代入前面的条件均值：

$$
\boxed{
u_t(x)
=
\frac{\dot\alpha_t}{\alpha_t}x
+
\left(
\frac{\dot\alpha_t}{\alpha_t}\beta_t^2
-
\dot\beta_t\beta_t
\right)s_t(x).
}
$$

这说明，在相同的高斯概率路径下，边际速度、最优噪声预测、最优数据预测与 score 之间存在确定的代数关系。

对线性路径 $$\alpha_t=t,\beta_t=1-t$$，有：

$$
u_t(x)=\frac{x+(1-t)s_t(x)}{t},
\qquad 0<t<1.
$$

若用网络预测速度，可以定义对应的数据和噪声估计：

$$
\boxed{
\widehat x_1=x+(1-t)v_\theta(t,x),
\qquad
\widehat\varepsilon=x-tv_\theta(t,x).
}
$$

当 $$v_\theta=u_t$$ 时，它们分别是给定当前状态后的数据后验均值和噪声后验均值。它们不必等于某一条原始训练配对的真实端点。

### 8.4 参数化可转换，不代表训练权重相同

在一条线性训练样本上：

$$
U_t=\frac{X_1-X_t}{1-t}
=\frac{X_t-\varepsilon}{t}.
$$

若通过数据预测构造速度：

$$
v_\theta=\frac{\widehat x_1-X_t}{1-t},
$$

则：

$$
\|v_\theta-U_t\|^2
=
\frac{\|\widehat x_1-X_1\|^2}{(1-t)^2}.
$$

若通过噪声预测构造速度：

$$
v_\theta=\frac{X_t-\widehat\varepsilon}{t},
$$

则：

$$
\|v_\theta-U_t\|^2
=
\frac{\|\widehat\varepsilon-\varepsilon\|^2}{t^2}.
$$

所以“可以把一种预测转换成另一种”不等于“直接使用相同未加权 MSE 就有完全一样的优化过程”。时间采样、损失权重、预条件化和端点数值处理，都会改变训练效果。

另外，有些扩散论文中的 `v-prediction` 是特定系数组合的预测参数化；它不应在没有检查定义和时间方向时，直接等同于这里的 $$\mathrm dX_t/\mathrm dt$$。

## 9. 训练不积分，生成为什么还要多步

### 9.1 一次训练更新只需采样一个时刻

基本线性 CFM 的训练步骤是：

$$
t\sim\mathcal U(0,1),\quad
X_0\sim p_0,\quad
X_1\sim p_{\mathrm{data}},
$$

$$
X_t=(1-t)X_0+tX_1,
\qquad
U_t=X_1-X_0,
$$

然后进行一次网络前向与反向传播。训练并不要求先走过 $$0$$ 到 $$t$$ 的所有时刻；$$X_t$$ 是直接构造出来的。

这就是 **simulation-free training** 的具体含义：基础回归训练不需要模拟模型轨迹。它并没有取消生成阶段的数值积分。

### 9.2 Euler 与 Heun 采样

设时间网格为 $$0=t_0<\cdots<t_N=1$$，$$h_k=t_{k+1}-t_k$$。Euler 法为：

$$
Z_{k+1}
=
Z_k+h_kv_\theta(t_k,Z_k).
$$

Heun 法先预测，再平均两端斜率：

$$
\widetilde Z_{k+1}
=
Z_k+h_kv_\theta(t_k,Z_k),
$$

$$
Z_{k+1}
=
Z_k+\frac{h_k}{2}
\left[
v_\theta(t_k,Z_k)
+
v_\theta(t_{k+1},\widetilde Z_{k+1})
\right].
$$

在足够光滑的条件下，Euler 的全局误差通常是一阶，Heun 是二阶；后一种方法每步通常需要两次网络评估。比较采样成本时，需要看 **NFE（Number of Function Evaluations）**，而不只是时间步数。

### 9.3 匀速直线为何适合少步积分

沿精确 ODE 轨迹：

$$
\frac{\mathrm d^2Z_t}{\mathrm dt^2}
=
\partial_t v_\theta(t,Z_t)
+
\left[\nabla_xv_\theta(t,Z_t)\right]v_\theta(t,Z_t).
$$

Taylor 展开给出 Euler 忽略的主要项：

$$
Z_{t+h}
=
Z_t+hv_\theta(t,Z_t)
+
\frac{h^2}{2}\frac{\mathrm d^2Z_t}{\mathrm dt^2}
+
O(h^3).
$$

当实际生成轨迹上的速度保持常数，二阶导数为零，Euler 可以精确沿这条轨迹前进。若轨迹虽在几何上是一条直线、但速度随时间变化，仍会有离散化误差；若速度方向也变化，通常就更不能大步跨越。

因此，应检查的是**学到的 ODE 轨迹是否接近匀速直线**，而不只是训练时是否写下了线性插值公式。

### 9.4 时间采样、时间重参数化是两种操作

若目标是：

$$
\int_0^1
\lambda(t)\,
\mathbb E[\|v_\theta-U_t\|^2\mid t]
\,\mathrm dt,
$$

但训练按密度 $$\rho(t)>0$$ 采样时间，无偏估计需要权重：

$$
\frac{\lambda(t)}{\rho(t)}
\|v_\theta-U_t\|^2.
$$

不做补偿，就改变了各时间段在目标中的权重。无限表达能力、严格正权重下的逐点最优条件均值可以不变，但有限网络的容量分配与优化过程会改变。

另一种操作是改用新时间 $$s$$，令 $$t=\tau(s)$$。为保持同一条状态轨迹，必须同时修改速度：

$$
\frac{\mathrm dZ}{\mathrm ds}
=
\tau'(s)\,u_{\tau(s)}(Z).
$$

重新分配采样步长、改变训练时间分布、重参数化整个动力学，不是同一件事。

## 10. PyTorch 核心实现与条件生成

### 10.1 将推导对应到代码

下面实现独立高斯噪声、线性路径、均匀时间采样和 Euler 生成。它是可复用的训练与采样核心，调用方提供数据与网络；接口约定是 `model(x, t)`，其中 `t.shape == [batch]`，输出形状与 `x` 相同。

~~~python
import torch


def flow_matching_loss(model, x1):
    # x1: [batch, ...]，已完成预处理的数据或 latent
    batch = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(batch, device=x1.device, dtype=x1.dtype)

    # 为图像、序列等数据扩展时间维，避免错误广播
    t_view = t.reshape(batch, *([1] * (x1.ndim - 1)))
    xt = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0

    prediction = model(xt, t)
    # 每个样本求向量范数平方，再对 batch 取平均
    return (prediction - target_velocity).square().reshape(batch, -1).sum(-1).mean()


@torch.no_grad()
def sample_euler(model, x0, steps=50):
    if steps < 1:
        raise ValueError("steps must be positive")

    # 调用前执行 model.eval()；x0 的形状、尺度应与训练噪声一致
    x = x0.clone()
    batch = x.shape[0]
    dt = 1.0 / steps

    for k in range(steps):
        t = torch.full(
            (batch,), k * dt, device=x.device, dtype=x.dtype
        )
        x = x + dt * model(x, t)

    return x
~~~

训练循环中调用 `loss = flow_matching_loss(model, x1)`，然后执行 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`。采样时调用 `model.eval()`，从新高斯噪声开始执行 `sample_euler`。

这段代码中的 `steps=50` 只是演示默认值，不是质量保证。实际需要在相同初始噪声下比较不同步数与求解器，才能判断误差主要来自模型还是数值积分。完整的二维网络训练示例可参考 [官方 Flow Matching notebook][code]。

### 10.2 文本或观测条件放在哪里

若生成受条件 $$c$$ 控制，例如类别、文本、图像或机器人观测，网络改成：

$$
v_\theta(t,x,c).
$$

训练时采样配对数据 $$(X_1,c)$$，保持噪声与该配对独立，目标仍为：

$$
\mathbb E\left[
\|v_\theta(t,X_t,c)-(X_1-X_0)\|^2
\right].
$$

最优预测变成：

$$
u_t(x,c)
=
\mathbb E[X_1-X_0\mid X_t=x,c].
$$

有两种“条件”需要区分：$$X_1$$ 可以是构造监督路径时才知道的端点；$$c$$ 则是生成时也提供给模型的信息。**推理时不能把尚未生成的数据端点当成已知输入。**

如果模型支持条件丢弃训练，还可组合条件与无条件速度进行 classifier-free guidance，例如：

$$
v_{\mathrm{guided}}
=
v_\theta(t,x,\varnothing)
+
w\left[
v_\theta(t,x,c)-v_\theta(t,x,\varnothing)
\right].
$$

按这个约定，$$w=0$$ 是无条件速度，$$w=1$$ 是原条件速度，$$w>1$$ 是外推。Guidance 改变实际积分的速度场，因此原先针对未引导场的精确分布结论不能直接套用；过强引导也可能增加积分难度。条件生成与引导的系统介绍见 [Flow Matching Guide and Code][guide]。

### 10.3 实现时优先检查什么

| 检查项 | 需要确认的内容 |
| --- | --- |
| 时间方向 | 数据在 $$t=1$$ 还是 $$t=0$$？速度标签与积分方向是否一致？ |
| 输入输出 | 网络是否同时接收状态与时间？输出是否是与状态同形状的速度？ |
| 数据尺度 | 训练 latent 的缩放、采样噪声的尺度、解码前的逆变换是否一致？ |
| 广播 | 每个样本的时间是否正确扩展到空间或序列维度？ |
| 标签 | 线性路径使用 $$X_1-X_0$$；改调度后要重新求导，不能沿用旧标签。 |
| 损失尺度 | 按维度求和或求平均会改变数值尺度；不同维数下不能直接比较 loss。 |
| 端点与步数 | 是否存在小分母、退化路径？增大 NFE 后生成结果是否明显改善？ |
| 条件输入 | 训练和推理的条件形式是否一致？是否误把真实终点作为推理输入？ |

## 11. 交互演示：观察条件插值，不把它误当作生成轨迹

下面的动画保留二维高斯噪声到“∞”形目标点云的演示。每对端点按：

$$
x_t=(1-t)x_0+tx_1
$$

移动，箭头表示该配对的速度标签 $$x_1-x_0$$。

**这个演示没有训练神经网络，没有求解 OT 配对，也没有积分学到的边际速度场。** 它展示的是构造训练样本时使用的条件插值；动画中的高斯尺度与曲线大小为了显示效果进行了调整。

{% include demo-frame.html src="/assets/demos/flow-matching-ot.html" title="Flow Matching 条件线性插值演示" height="620px" caption="随机配对的条件插值：拖动时间观察训练状态与速度标签；不代表 OT 解或模型生成轨迹。" %}

可以按下面的顺序观察：

1. 从 $$t=0$$ 拖到 $$t=1$$，区分“每个粒子的位置”和“所有粒子组成的分布”。
2. 打开路径线与速度箭头，寻找相邻粒子速度不一致的区域，理解条件平均为何必要。
3. 对照第 6 节：假如只给模型当前位置与时间，它需要预测的是后验平均速度，而不是记住动画中某条指定的线。
4. 要评价真实模型的少步生成效果，应另外从新噪声积分网络 ODE，并比较不同 NFE 下的终点分布。

## 12. 回到最初的三个问题

**标签冲突时，网络学什么？**

平方回归的最优解是给定当前时间与位置后的条件平均：

$$
v^\star(t,x)=\mathbb E[\dot X_t\mid X_t=x].
$$

**为什么这个速度能生成正确分布？**

这个条件平均与训练插值的边际分布满足同一连续性方程。在适当正则性、精确拟合和精确积分的理想条件下，从相同初始分布出发的 ODE 会复现这条概率路径。

**为什么直线训练仍不能保证一步生成？**

训练的直线依赖已知的两个端点，生成模型只知道当前状态。条件平均会重新组织轨迹，实际速度可能随时间与位置改变；只有实际生成轨迹足够接近匀速直线，粗粒度积分才容易准确。

阅读后续 FM 论文时，可以带着五个具体问题：**选了什么端点耦合？构造了什么概率路径？回归什么速度标签？怎样加权时间？最后用什么求解器积分？** 很多看似复杂的变体，都可以从这些选择上分清差异。

## 参考资料

1. **Flow Matching 的原始构造与等价性证明**：Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023，重点读第 3—4 节及附录 A—C。[论文][fm]
2. **Rectified Flow、reflow 与直线性的分析**：Liu, Gong & Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023。正确的 arXiv 编号是 **2209.03003**。[论文][rf]
3. **一般耦合与 minibatch OT-CFM**：Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*, TMLR 2024。[论文][otcfm]
4. **扩散 SDE 与 probability flow ODE**：Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021，重点读第 4.3 节。[论文][sde]
5. **系统教程与扩展**：Lipman et al., *Flow Matching Guide and Code*, 2024。[教程][guide]
6. **实现对照**：官方 Flow Matching 文档中的二维训练与采样 notebook。[代码示例][code]

[fm]: https://arxiv.org/abs/2210.02747
[rf]: https://arxiv.org/abs/2209.03003
[otcfm]: https://arxiv.org/abs/2302.00482
[sde]: https://arxiv.org/abs/2011.13456
[guide]: https://arxiv.org/abs/2412.06264
[code]: https://facebookresearch.github.io/flow_matching/notebooks/standalone_flow_matching.html
