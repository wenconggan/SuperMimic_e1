# 来源与许可说明

## 代码

本项目在非官方 SuperMimic 复现基础上，整理 E1 21 自由度适配、原生 MuJoCo sim2sim、
录屏和评测流程。原始 MIT 许可证及 `Copyright (c) 2026 ETH Zurich, Chong Zhang`
保留于 [LICENSE](LICENSE)。这是面向 E1 的非官方复现与工程适配。

方法参考：*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable
World Model*，论文链接由上游复现提供：<https://arxiv.org/abs/2610.28480>。

## 仿真依赖

Isaac Lab、Newton、MuJoCo、MuJoCo-Warp、Warp 与 PyTorch 是独立的第三方项目，
使用其各自许可证。本目录不复制它们的源码或安装环境；安装过程获取指定版本。

## E1 机器人描述

`robots/e1_21dof/mjcf`、`robots/e1_21dof/urdf` 和 `robots/e1_21dof/meshes` 来自本项目适配时提供的
E1 机器人描述。源资源中未附独立许可证文件；此处保留原文件，并将其与 MIT
授权的项目代码区分。机器人资源的来源与授权沿用原始资源提供者的说明。

## 动作数据与展示媒体

本地 `datasets/e1_lafan/*.csv` 是提供给 E1 适配使用的 LaFAN1 重定向动作。LaFAN1 由
Ubisoft La Forge 发布；上游复现引用的动作资源许可为 CC BY-NC-ND 4.0。
E1 重定向版本由项目维护者提供，目前没有独立公开下载地址。上游 G1 重定向
文件具有不同的关节布局。CSV 与训练权重在本地可用，默认不由 Git 跟踪。

`docs/assets` 的动画、图片与视频来自本项目实际 E1 仿真，包含 LaFAN1 参考动作
的可视化。这些展示资源及动作数据不因代码使用 MIT 许可证而改变其原资源许可。
