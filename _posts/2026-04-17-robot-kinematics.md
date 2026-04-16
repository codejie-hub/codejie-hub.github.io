---
title: 【机器人运动学】坐标变换、四元数与7自由度机械臂详解
date: 2026-04-17 10:00:00 +0800
categories: [机器人抓取]
tags: [kinematics, robotics, quaternion, euler-angles, coordinate-transform, 7dof]
description: 深入解析机器人运动学核心概念：坐标系转换、欧拉角、四元数、万向锁问题，以7自由度机械臂为例，配合Three.js交互式3D可视化，帮助理解空间旋转与末端位姿控制。
image: /assets/images/posts/robotics/kinematics-cover.jpg
math: true
mermaid: true
---

> 机器人运动学（Robot Kinematics）是机器人学的基础，描述机械臂各关节角度与末端执行器位姿之间的数学关系。本文从坐标系变换出发，深入讲解欧拉角、四元数、万向锁等核心概念，并以 **7自由度（7-DOF）机械臂** 为例，结合 **Three.js 交互式可视化**，帮助你建立空间几何的直观理解。

## 1. 为什么需要运动学？

在机器人抓取任务中，我们通常面临两个核心问题：

1. **正运动学（Forward Kinematics, FK）**：已知各关节角度 $\theta_1, \theta_2, \ldots, \theta_n$，求末端执行器的位置和姿态
2. **逆运动学（Inverse Kinematics, IK）**：已知目标位姿（位置 + 姿态），求各关节应该转到的角度

这两个问题的核心都依赖于 **坐标系变换** 和 **旋转表示**。

---

## 2. 坐标系与齐次变换矩阵

### 2.1 坐标系的定义

在机器人系统中，我们通常定义多个坐标系：

- **世界坐标系（World Frame）** $\{W\}$：固定在环境中的参考系
- **基座坐标系（Base Frame）** $\{B\}$：固定在机器人基座上
- **关节坐标系（Joint Frame）** $\{J_i\}$：每个关节都有自己的局部坐标系
- **末端坐标系（End-Effector Frame）** $\{E\}$：固定在末端执行器上

### 2.2 齐次变换矩阵

要描述一个坐标系相对于另一个坐标系的位置和姿态，我们使用 **齐次变换矩阵（Homogeneous Transformation Matrix）**：

$$
T = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

其中：
- $R \in SO(3)$ 是 $3 \times 3$ 旋转矩阵，描述姿态
- $\mathbf{t} \in \mathbb{R}^3$ 是平移向量，描述位置
- $T \in SE(3)$ 称为特殊欧氏群

**关键性质**：
- 旋转矩阵满足 $R^T R = I$ 且 $\det(R) = 1$
- 变换矩阵的逆：$T^{-1} = \begin{bmatrix} R^T & -R^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$
- 变换的复合：$T_{AC} = T_{AB} \cdot T_{BC}$（从 C 到 A 的变换）

---

## 3. 旋转的三种表示方法

### 3.1 旋转矩阵（Rotation Matrix）

最直接但冗余的表示（9个元素，但只有3个自由度）。

**绕基本轴旋转**：

绕 $x$ 轴旋转 $\alpha$：
$$
R_x(\alpha) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{bmatrix}
$$

绕 $y$ 轴旋转 $\beta$：
$$
R_y(\beta) = \begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}
$$

绕 $z$ 轴旋转 $\gamma$：
$$
R_z(\gamma) = \begin{bmatrix}
\cos\gamma & -\sin\gamma & 0 \\
\sin\gamma & \cos\gamma & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

### 3.2 欧拉角（Euler Angles）

用三个角度 $(\alpha, \beta, \gamma)$ 描述旋转，通过三次绕基本轴的旋转复合得到。

**常见的欧拉角约定**：
- **ZYX（Roll-Pitch-Yaw）**：先绕 $z$ 轴（偏航），再绕 $y$ 轴（俯仰），最后绕 $x$ 轴（翻滚）
  $$
  R_{ZYX}(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_x(\gamma)
  $$

- **ZYZ（经典欧拉角）**：$z$-$y$-$z$ 序列

**优点**：
- 直观易懂，符合人类思维
- 只需3个参数

**缺点**：
- 存在 **万向锁（Gimbal Lock）** 问题
- 不同约定导致混乱
- 插值困难（不适合平滑运动规划）

### 3.3 四元数（Quaternion）

四元数是一种 **4维超复数**，用于表示3D旋转，避免了万向锁问题。

**定义**：
$$
q = w + xi + yj + zk = [w, x, y, z]^T
$$

其中 $i^2 = j^2 = k^2 = ijk = -1$，且满足 **单位约束** $w^2 + x^2 + y^2 + z^2 = 1$。

**几何意义**：
- $w = \cos(\theta/2)$
- $(x, y, z) = \sin(\theta/2) \cdot \mathbf{u}$

其中 $\theta$ 是旋转角度，$\mathbf{u}$ 是旋转轴的单位向量。

**四元数乘法**（旋转复合）：
$$
q_1 \otimes q_2 = \begin{bmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{bmatrix}
$$

**优点**：
- 无万向锁问题
- 插值平滑（SLERP：球面线性插值）
- 计算效率高（相比旋转矩阵）
- 数值稳定性好

**缺点**：
- 不直观，难以理解
- 需要归一化保持单位约束

---

## 4. 万向锁问题详解

### 4.1 什么是万向锁？

**万向锁（Gimbal Lock）** 是使用欧拉角表示旋转时的一个致命缺陷：当中间旋转轴达到 $\pm 90°$ 时，第一个和第三个旋转轴重合，导致 **失去一个自由度**。

以 ZYX 欧拉角为例，当 $\beta = 90°$（俯仰角为直角）时：

$$
R_{ZYX}(\alpha, 90°, \gamma) = R_z(\alpha) R_y(90°) R_x(\gamma)
$$

此时 $R_y(90°)$ 会使 $x$ 轴旋转到原来 $z$ 轴的方向，导致 $R_z(\alpha)$ 和 $R_x(\gamma)$ 作用在同一个轴上，只能控制 $\alpha + \gamma$ 的和，而无法独立控制。

### 4.2 万向锁的后果

1. **失去一个旋转自由度**：无法表示某些姿态
2. **奇异性**：雅可比矩阵退化，逆运动学求解失败
3. **运动不连续**：微小的姿态变化可能导致关节角度突变

### 4.3 解决方案

- **使用四元数**：完全避免万向锁
- **使用旋转矩阵**：冗余但稳定
- **轴角表示（Axis-Angle）**：$(\mathbf{u}, \theta)$，直观但插值复杂

---

## 5. 7自由度机械臂的运动学

### 5.1 为什么是7自由度？

人类手臂有 **7个主要自由度**：
- 肩部：3个（屈伸、内外展、内外旋）
- 肘部：1个（屈伸）
- 腕部：3个（屈伸、偏移、旋前旋后）

**6-DOF vs 7-DOF**：
- **6-DOF**：刚好满足末端6维位姿（3位置 + 3姿态），但无冗余
- **7-DOF**：**冗余自由度**，同一末端位姿有无穷多种关节配置

**7-DOF的优势**：
1. **避障能力强**：可以在保持末端位姿不变的情况下调整肘部位置
2. **奇异性规避**：远离奇异配置，提高灵活性
3. **优化性能**：可以优化能耗、关节限位等指标

### 5.2 DH参数法（Denavit-Hartenberg）

**DH参数** 是描述相邻关节坐标系关系的标准方法，每个关节用4个参数描述：

| 参数 | 符号 | 含义 |
|------|------|------|
| 连杆长度 | $a_i$ | 沿 $x_i$ 轴从 $z_{i-1}$ 到 $z_i$ 的距离 |
| 连杆扭角 | $\alpha_i$ | 绕 $x_i$ 轴从 $z_{i-1}$ 到 $z_i$ 的角度 |
| 连杆偏距 | $d_i$ | 沿 $z_{i-1}$ 轴从 $x_{i-1}$ 到 $x_i$ 的距离 |
| 关节角 | $\theta_i$ | 绕 $z_{i-1}$ 轴从 $x_{i-1}$ 到 $x_i$ 的角度 |

**相邻坐标系变换**：
$$
T_i^{i-1} = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i \cos\alpha_i & \sin\theta_i \sin\alpha_i & a_i \cos\theta_i \\
\sin\theta_i & \cos\theta_i \cos\alpha_i & -\cos\theta_i \sin\alpha_i & a_i \sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 5.3 正运动学

末端相对于基座的变换矩阵：
$$
T_E^B = T_1^0 \cdot T_2^1 \cdot T_3^2 \cdot \ldots \cdot T_7^6
$$

从中提取：
- **位置**：$\mathbf{p} = [T_{14}, T_{24}, T_{34}]^T$
- **姿态**：$R = T_{1:3, 1:3}$（左上角 $3 \times 3$ 子矩阵）

### 5.4 逆运动学

**问题**：给定 $T_E^B$，求 $\theta_1, \ldots, \theta_7$。

**挑战**：
- 7-DOF 系统有 **无穷多解**（冗余）
- 需要额外约束（如最小关节运动、避障等）

**常用方法**：
1. **解析法**：针对特定结构（如球腕）推导闭式解
2. **数值法**：
   - **雅可比伪逆法**：$\Delta \theta = J^+ \Delta x$
   - **阻尼最小二乘法**：$\Delta \theta = J^T (JJ^T + \lambda^2 I)^{-1} \Delta x$
   - **优化法**：最小化 $\|\theta - \theta_0\|$ 约束于 $f(\theta) = x_{target}$

---

## 6. 雅可比矩阵与速度运动学

### 6.1 雅可比矩阵

**雅可比矩阵** $J(\theta)$ 描述关节速度与末端速度的关系：

$$
\begin{bmatrix}
\dot{\mathbf{p}} \\
\boldsymbol{\omega}
\end{bmatrix} = J(\theta) \dot{\boldsymbol{\theta}}
$$

其中：
- $\dot{\mathbf{p}} \in \mathbb{R}^3$：末端线速度
- $\boldsymbol{\omega} \in \mathbb{R}^3$：末端角速度
- $\dot{\boldsymbol{\theta}} \in \mathbb{R}^7$：关节角速度

**雅可比矩阵的计算**：
$$
J = \begin{bmatrix}
J_v \\
J_\omega
\end{bmatrix}, \quad
J_v^i = \mathbf{z}_{i-1} \times (\mathbf{p}_E - \mathbf{p}_{i-1}), \quad
J_\omega^i = \mathbf{z}_{i-1}
$$

其中 $\mathbf{z}_{i-1}$ 是第 $i$ 个关节的旋转轴方向。

### 6.2 奇异性

当 $\det(JJ^T) = 0$ 时，机械臂处于 **奇异配置（Singularity）**：
- 某些方向无法运动
- 逆运动学无解或无穷多解
- 关节速度趋于无穷大

**7-DOF的优势**：可以通过冗余自由度 **远离奇异配置**。

---

## 7. 交互式可视化示例

我为你准备了三个 Three.js 交互式演示，帮助理解这些抽象概念：

### 7.1 坐标系变换可视化

<iframe src="/assets/demos/robot-kinematics-transform.html" width="100%" height="500px" frameborder="0"></iframe>

**操作说明**：
- 拖动滑块调整平移和旋转参数
- 观察子坐标系相对于父坐标系的变换
- 实时显示齐次变换矩阵

### 7.2 欧拉角与万向锁演示

<iframe src="/assets/demos/robot-kinematics-gimbal.html" width="100%" height="500px" frameborder="0"></iframe>

**操作说明**：
- 调整 Roll、Pitch、Yaw 三个角度
- 当 Pitch = 90° 时观察万向锁现象
- 对比四元数插值的平滑性

### 7.3 7-DOF机械臂正运动学

<iframe src="/assets/demos/robot-kinematics-7dof.html" width="100%" height="600px" frameborder="0"></iframe>

**操作说明**：
- 调整7个关节角度
- 实时显示末端位姿（位置 + 四元数）
- 显示雅可比矩阵的条件数（奇异性指标）

---

## 8. 实践建议

### 8.1 选择合适的旋转表示

| 场景 | 推荐表示 | 原因 |
|------|----------|------|
| 存储和传输 | 四元数 | 紧凑（4个数）、无奇异性 |
| 用户界面 | 欧拉角 | 直观易懂 |
| 插值和规划 | 四元数 | SLERP平滑 |
| 数学推导 | 旋转矩阵 | 线性代数友好 |
| 传感器融合 | 四元数 | 数值稳定 |

### 8.2 四元数使用注意事项

1. **归一化**：每次运算后归一化 $q \leftarrow q / \|q\|$
2. **双重覆盖**：$q$ 和 $-q$ 表示同一旋转，插值时选择短路径
3. **转换公式**：

**四元数 → 旋转矩阵**：
$$
R = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

**欧拉角（ZYX）→ 四元数**：
$$
\begin{aligned}
w &= \cos(\alpha/2)\cos(\beta/2)\cos(\gamma/2) + \sin(\alpha/2)\sin(\beta/2)\sin(\gamma/2) \\
x &= \cos(\alpha/2)\cos(\beta/2)\sin(\gamma/2) - \sin(\alpha/2)\sin(\beta/2)\cos(\gamma/2) \\
y &= \cos(\alpha/2)\sin(\beta/2)\cos(\gamma/2) + \sin(\alpha/2)\cos(\beta/2)\sin(\gamma/2) \\
z &= \sin(\alpha/2)\cos(\beta/2)\cos(\gamma/2) - \cos(\alpha/2)\sin(\beta/2)\sin(\gamma/2)
\end{aligned}
$$

### 8.3 逆运动学求解技巧

对于7-DOF机械臂，推荐使用 **零空间优化**：

$$
\dot{\boldsymbol{\theta}} = J^+ \dot{\mathbf{x}} + (I - J^+ J) \dot{\boldsymbol{\theta}}_0
$$

其中：
- $J^+$ 是雅可比伪逆
- $(I - J^+ J)$ 是零空间投影矩阵
- $\dot{\boldsymbol{\theta}}_0$ 是次要任务（如关节限位规避、奇异性规避）

---

## 9. 代码示例

### 9.1 Python：四元数插值（SLERP）

```python
import numpy as np

def quaternion_slerp(q1, q2, t):
    """
    球面线性插值
    q1, q2: 单位四元数 [w, x, y, z]
    t: 插值参数 [0, 1]
    """
    # 确保选择短路径
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # 球面插值
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)
    
    return q1 * np.cos(theta) + q3 * np.sin(theta)
```

### 9.2 C++：旋转矩阵转四元数

```cpp
#include <Eigen/Dense>

Eigen::Quaterniond rotationMatrixToQuaternion(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q;
    double trace = R.trace();
    
    if (trace > 0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        q.w() = 0.25 / s;
        q.x() = (R(2,1) - R(1,2)) * s;
        q.y() = (R(0,2) - R(2,0)) * s;
        q.z() = (R(1,0) - R(0,1)) * s;
    } else if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2));
        q.w() = (R(2,1) - R(1,2)) / s;
        q.x() = 0.25 * s;
        q.y() = (R(0,1) + R(1,0)) / s;
        q.z() = (R(0,2) + R(2,0)) / s;
    } else if (R(1,1) > R(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2));
        q.w() = (R(0,2) - R(2,0)) / s;
        q.x() = (R(0,1) + R(1,0)) / s;
        q.y() = 0.25 * s;
        q.z() = (R(1,2) + R(2,1)) / s;
    } else {
        double s = 2.0 * std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1));
        q.w() = (R(1,0) - R(0,1)) / s;
        q.x() = (R(0,2) + R(2,0)) / s;
        q.y() = (R(1,2) + R(2,1)) / s;
        q.z() = 0.25 * s;
    }
    
    return q.normalized();
}
```

---

## 10. 总结与展望

本文系统介绍了机器人运动学的核心概念：

✅ **坐标系变换**：齐次变换矩阵是描述位姿的标准工具  
✅ **旋转表示**：欧拉角直观但有万向锁，四元数无奇异性且插值平滑  
✅ **7-DOF机械臂**：冗余自由度带来灵活性，但逆运动学求解更复杂  
✅ **雅可比矩阵**：连接关节空间与任务空间，是速度控制的关键  

**进阶方向**：
- **动力学**：考虑力、力矩和惯性（拉格朗日方程、牛顿-欧拉法）
- **轨迹规划**：时间最优、能量最优、避障约束
- **力控制**：阻抗控制、混合位置/力控制
- **学习方法**：强化学习、模仿学习在运动学中的应用

希望通过本文和交互式演示，你能对机器人运动学有更深入的理解！

---

## 参考资料

1. Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control*. Pearson.
2. Siciliano, B., et al. (2009). *Robotics: Modelling, Planning and Control*. Springer.
3. Murray, R. M., et al. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
4. Shoemake, K. (1985). "Animating rotation with quaternion curves". *SIGGRAPH*.
5. Eigen Library: https://eigen.tuxfamily.org
6. Three.js Documentation: https://threejs.org/docs/

---

**互动演示源码**：本文的 Three.js 可视化代码已开源，可在 [GitHub](https://github.com/codejie-hub/robot-kinematics-demos) 查看完整实现。
