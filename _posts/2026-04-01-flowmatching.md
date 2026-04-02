---
title: 【生成模型】流匹配 Flow Matching：ODE、向量场与相对扩散
date: 2026-04-01 16:00:00 +0800
categories: [paper, ML]
tags: [flow-matching, diffusion, generative-model, rectified-flow, optimal-transport]
description: 流匹配用 ODE 与向量场回归描述从噪声到数据的输运；本文整理直线路径（OT）直觉、条件流匹配训练目标，以及与扩散在轨迹与调度上的差异，并附可部署 2D 轨迹演示。
mermaid: true
math: true
---

> **流匹配（Flow Matching, FM）** 在生成建模中用 **常微分方程（ODE）** 描述样本从简单源分布（常为高斯噪声）到数据分布的演化，通过神经网络 **回归速度场（向量场）** 并在推理时 **数值积分** 得到样本。工业界大量图像模型（如 **Stable Diffusion 3**、**FLUX.1**）在实现上广泛采用 **Rectified Flow** 等 FM 变体，以在 **少步采样** 与 **训练可扩展性** 上与经典扩散形成互补。

## 1. 直觉：从「去噪随机轨迹」到「确定性流」

**扩散模型**的典型叙事是：对数据逐步加噪（前向），再学习反向去噪；连续时间形式常涉及 **随机微分方程（SDE）** 与 **噪声日程（noise schedule）**，单条生成轨迹可视为在数据流形附近 **曲折随机** 的路径。

**流匹配**更强调 **确定性流**：在时刻 $t \in [0,1]$，状态 $x_t$ 沿某条 **预先构造的概率路径** 从噪声端点 $x_0$ 移向数据端点 $x_1$。学习目标是在每个 $(t, x_t)$ 处预测 **瞬时速度** $\frac{dx}{dt}$，使积分结果与目标分布一致。下文以最常用的 **两点线性插值路径**（与 **最优传输** 中「直线耦合」的直觉一致）为例。

## 2. 线性路径与目标向量场

1. **概率路径（示例）**  
   设 $x_0 \sim p_0$（噪声），$x_1 \sim p_1$（数据），定义：
   $$
   x_t = (1-t)\, x_0 + t\, x_1,\quad t \in [0,1]
   $$

2. **目标速度场**  
   对 $t$ 求导得到 **真值速度**（条件于一对 $(x_0,x_1)$）：
   $$
   u_t(x_t) = \frac{d x_t}{dt} = x_1 - x_0
   $$
   即在已知端点时，沿该直线路径的速度 **恒为** 指向 $x_1$ 的向量 $x_1-x_0$。

3. **向量场回归**  
   用网络 $v_\theta(t, x_t)$ 拟合上述速度，常用 **$L_2$ 回归**（条件流匹配 / CFM 一类目标的简化写法）：
   $$
   \mathcal{L} = \mathbb{E}_{t,\, x_0,\, x_1}\left[\left\lVert v_\theta(t, x_t) - (x_1 - x_0) \right\rVert^2\right]
   $$
   其中 $t$ 通常在 $[0,1]$ 上均匀采样，$(x_0,x_1)$ 按训练配对方式采样（独立耦合、最优传输配对等会改变边际与方差，但 **直线插值 + 回归 $x_1-x_0$** 的核心形式不变）。

推理时从 $x_0 \sim \mathcal{N}(0,I)$ 出发，用 ODE 求解器沿 $v_\theta$ 积分到 $t=1$，得到生成样本。**步数**取决于求解器与向量场光滑程度；直线路径动机上支持 **较大步长**，实践中常见 **十步量级** 相对少步高质量采样（具体仍依赖架构与蒸馏）。

## 3. 相对经典扩散的常见优势（概括）

| 维度 | 经典扩散（示意） | 流匹配 / Rectified Flow（示意） |
| :--- | :--- | :--- |
| 轨迹几何 | 随机、多步去噪，路径常较弯 | 常显式构造 **更直** 的耦合路径，利于少步积分 |
| 训练 | 需噪声日程、预测噪声或 score 等 | **CFM** 等可避免对整个 SDE 轨迹做昂贵模拟，批内独立配对即可算损失 |
| 数学对象 | SDE + 变分下界等 | **ODE + 向量场回归**，形式紧凑 |

需注意：二者边界在文献与实现中不断融合（如 EDM、flow 蒸馏、一致性模型等），上表仅为 **阅读 FM 论文时的对照锚点**，非严格互斥分类。

## 4. 交互演示：最优传输直线路径（2D）

每个粒子满足 **$x_t=(1-t)x_0+t x_1$**（$y$ 同理），$x_0$ 来自二维高斯，$x_1$ 落在「∞」形曲线。拖动 **时间进度 $t$** 可观察 **线性概率路径**（理想化 OT 直觉）。部署后若 iframe 不显示，请用 `bundle exec jekyll serve` 本地预览或打开独立链接。

<div class="flow-matching-demo-embed" style="width:100%;max-width:640px;margin:1.25rem auto;">
  <iframe
    src="{{ site.baseurl }}/assets/demos/flow-matching-ot.html"
    title="流匹配 2D 轨迹可视化"
    width="100%"
    height="620"
    style="border:0;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,.2);display:block;background:#0f1419;"
    loading="lazy"
    allowfullscreen
  ></iframe>
  <p style="text-align:center;font-size:0.85rem;opacity:.8;margin:0.5rem 0 0;">独立打开：<a href="{{ site.baseurl }}/assets/demos/flow-matching-ot.html" rel="noopener noreferrer" target="_blank">flow-matching-ot.html</a></p>
</div>

## 5. 小结

流匹配把生成问题写成 **学习速度场 + ODE 积分**：在直线插值等简单路径下，监督信号就是 **$x_1-x_0$**，训练与推理叙事都比「全程 SDE 调度」更易模块化；与 **最优传输、Rectified Flow、少步扩散** 的联系使其成为当前图像生成主干的重要选项之一。

---

**参考（入门与经典文献入口）**

- Lipman et al., *Flow Matching for Generative Modeling*（ICLR 2023 相关线）. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. [arXiv:2309.03005](https://arxiv.org/abs/2309.03005)
- Stable Diffusion 3 技术报告（DiT + flow 采样）: [stability.ai/news/stable-diffusion-3](https://stability.ai/news/stable-diffusion-3)
