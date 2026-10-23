# E1 LaFAN1 动作数据

将已经重定向至 E1 21 自由度的 LaFAN1 CSV 放在此目录。本地整理版包含 40 条
动作，CSV 被 `.gitignore` 排除，不随 Git 提交发布。

每个 CSV 无表头、30 Hz，每帧 28 个浮点数，依次为：

1. 根位置 `x, y, z`，单位米。
2. 根姿态四元数 `qx, qy, qz, qw`。
3. 21 个关节角，单位弧度，顺序与 `supermimic/robot.py` 的 `JOINT_NAMES` 一致。

读取时重采样至 50 Hz，速度通过有限差分计算。这里需要 E1 重定向版本；
G1 的 29 自由度 CSV 不能直接使用。

数据来源和许可范围见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
