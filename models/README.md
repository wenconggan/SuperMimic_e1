# 训练检查点

本地整理版提供 `ckpt_168000.pt`，用于复现 README 中的原生 MuJoCo 演示。
模型文件由 `.gitignore` 排除；GitHub 克隆后，可将自行训练或另行获得的 E1
检查点放入此目录，并通过 `--checkpoint models/文件名.pt` 选择。

原训练目录的 `logs/<run>/ckpt_*.pt` 同样可以直接用于 sim2sim。
`last.pt` 包含优化器状态，用于恢复训练；编号检查点主要用于评测与推理。

仅有检查点还需要对应的 `datasets/e1_lafan/*.csv` 参考动作。记录中的旧日志目录不会
被 sim2sim 用作资源路径，运行时资源位置由本仓库目录与命令行参数确定。
