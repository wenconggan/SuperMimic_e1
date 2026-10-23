# supermimic_e1 安装指南

以下命令均在本仓库根目录执行。训练使用独立的 `.venv-isaaclab3`；仅运行 MuJoCo 仿真和录制视频时，可以使用更小的 `.venv-sim2sim`。两者都不依赖其他机器人部署仓库。

## 1. 系统与工具

训练环境面向 Linux x86_64、Python 3.12 和 NVIDIA GPU。已验证的训练设备为 RTX 5090 32 GB，使用 NVIDIA 570 系列驱动、CUDA 12 版 PyTorch 和 Warp。4096 个环境是默认训练规模，显存占用也受回放缓冲区、批量大小和并行任务影响。

Ubuntu/Debian 安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential ffmpeg libegl1 libgl1 libglfw3
```

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 并准备 Python 3.12：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv --version
ffmpeg -version
ffprobe -version
```

训练或使用 NVIDIA EGL 渲染前，确认系统驱动可用：

```bash
nvidia-smi
```

PyTorch/Warp 的预编译 wheel 提供所需 CUDA 运行库；这里仍需要可用的 NVIDIA 驱动。MP4 编码使用系统 `ffmpeg`，视频检查使用 `ffprobe`。

## 2. 模型与动作文件

目录约定如下：

```text
robots/
└── e1_21dof/
    ├── mjcf/E1_21dof.xml
    ├── urdf/E1_21dof.urdf
    └── meshes/
datasets/
└── e1_lafan/
    └── *.csv
models/
└── ckpt_168000.pt         # 可选：已有的训练检查点
```

机器人描述随项目提供。本地整理版包含 40 个动作 CSV 和 `models/ckpt_168000.pt`；这两类文件由 `.gitignore` 排除，不会随普通 Git 提交发布。从 Git 克隆项目时，需将有权使用的 E1 动作与检查点自行放到上述位置。训练需要动作文件；已有权重的仿真也需要对应参考动作。

本项目读取 **E1 21 自由度重定向数据**。CSV 无表头，源帧率为 30 Hz，每行 28 列：根位置 3 列、`xyzw` 根四元数 4 列、按 E1 MJCF 关节顺序排列的 21 个关节角。程序自动重采样到 50 Hz。完整动作库为 40 个 CSV；数据获取与使用条件见 [README.md](README.md) 中的来源说明。

## 3. Isaac Lab 3 训练环境

项目基于上游 SuperMimic 非官方复现，保留原始 Isaac Lab 3 / Newton / MuJoCo-Warp 训练后端。使用固定 Isaac Lab 提交与其锁文件安装依赖：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
cd IsaacLab
UV_CACHE_DIR=../.cache/uv UV_PROJECT_ENVIRONMENT=../.venv-isaaclab3 \
  uv sync --frozen --inexact --extra importers --extra test --python 3.12
cd ..
```

`importers` 提供离线 MJCF → USD 转换所需的导入器，`test` 提供检查工具。这条安装路径使用 kit-less 运行方式。Isaac Lab 的源码目录保留在本仓库内，虚拟环境中的 editable 包依赖该目录。

该提交的仓库版本是 Isaac Lab **3.0.0**；单个 Python 组件包使用独立版本号，例如 `isaaclab` 包可能显示 `41.0.0`，查看仓库提交即可确认所用版本。

### CUDA 12 兼容 wheel

上游锁文件默认选择 CUDA 13 wheel。已验证的 NVIDIA 570 驱动环境使用下面的 CUDA 12 兼容版本，保持相同 PyTorch / Warp 主版本：

```bash
uv pip install --python .venv-isaaclab3/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0' 'torchvision==0.26.0' \
  --reinstall-package torch --reinstall-package torchvision

uv pip install --python .venv-isaaclab3/bin/python --no-deps \
  'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
```

此后直接调用 `.venv-isaaclab3/bin/python` 和项目启动脚本。再次执行 `uv sync` 或默认的 `uv run` 会按上游锁文件同步依赖，可能将兼容 wheel 换回 CUDA 13；重新同步后，应再次执行上述两条替换命令。

安装依据与已验证版本：

| 组件 | 版本或来源 |
| --- | --- |
| Python | 3.12，本地验证为 3.12.7 |
| Isaac Lab | `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` |
| Newton | 上游锁定 `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3` |
| PyTorch / torchvision | `2.11.0+cu128` / `0.26.0+cu128` |
| Warp | `1.17.0+cu12` |
| MuJoCo / MuJoCo-Warp | `3.15.0`，由上游锁文件安装 |
| NumPy / SciPy | 本地验证为 `2.5.1` / `1.17.0` |
| Pillow | 本地验证为 `12.3.0` |

## 4. 检查并开始训练

先确认 PyTorch 和 Warp 都能使用 GPU：

```bash
.venv-isaaclab3/bin/python - <<'PY'
from importlib.metadata import version
import torch
import warp as wp

wp.init()
for package in ("torch", "warp-lang", "mujoco", "mujoco-warp", "isaaclab"):
    print(package, version(package))
assert torch.cuda.is_available(), "PyTorch CUDA 不可用"
assert wp.is_cuda_available(), "Warp CUDA 不可用"
print("GPU:", torch.cuda.get_device_name(0))
PY
```

检查动作和运动学，生成 USD，再运行少量仿真环境：

```bash
.venv-isaaclab3/bin/python tools/check_motion.py --robot e1
.venv-isaaclab3/bin/python tools/mjcf_to_usd.py --robot e1
.venv-isaaclab3/bin/python tools/test_env.py --robot e1 --num_envs 16
```

转换结果保存在 `data/e1_description/` 和 `data/e1_usd/`；`data/e1_usd/asset_path.txt` 记录实际 USD 路径。模型或仓库路径改变后重新转换。首次执行还会编译 Warp / PyTorch 内核，比后续启动耗时更长。

开始后台训练并查看日志：

```bash
bash run_e1_train.sh e1_paper
tail -f logs/e1_paper/stdout.log
```

启动器默认使用 4096 个环境和 320000 次迭代，检查点每 2000 次保存一次。输出目录为 `logs/e1_paper/`；选择新的运行名即可保留多次实验。启动器会拒绝覆盖已有训练输出。

资源有限时，可用启动器末尾的参数覆盖默认配置，例如：

```bash
bash run_e1_train.sh e1_small \
  --num_envs 1024 --buffer_steps 256 --wm_batch 512 --pi_batch 512
```

训练恢复使用包含优化器状态的 `last.pt`，并写入新的运行目录：

```bash
bash run_e1_train.sh e1_resume --resume logs/e1_paper/last.pt
```

完整动作库评估使用最新的编号检查点：

```bash
bash run_e1_eval.sh logs/e1_paper
```

第二个参数可以指定检查点，例如：

```bash
bash run_e1_eval.sh logs/e1_paper models/ckpt_168000.pt
```

评估会保存日志和 JSON 摘要。每段动作从第一帧运行一次，关闭随机化与噪声，以训练中的高度和倾斜阈值判断完成情况；这项指标描述仿真动作库表现。

## 5. 仅运行原生 MuJoCo 仿真与视频

这个环境适合已有检查点的 CPU 策略推理、原生 MuJoCo 仿真和 MP4 录制。创建独立虚拟环境，先安装 CPU 版 PyTorch，再安装项目所需包：

```bash
uv venv --python 3.12 .venv-sim2sim
uv pip install --python .venv-sim2sim/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
uv pip install --python .venv-sim2sim/bin/python -r requirements-sim2sim.txt
```

CPU 版 PyTorch 执行策略推理；视频渲染另行使用系统 OpenGL/EGL。这个环境无需安装 Isaac Lab 和 USD 导入器。

通过 `SUPERMIMIC_PYTHON` 为启动脚本指定解释器：

```bash
export SUPERMIMIC_PYTHON="$PWD/.venv-sim2sim/bin/python"

# 无窗口仿真，只输出测量结果。
bash run_e1_sim2sim.sh --checkpoint models/ckpt_168000.pt \
  --headless --motion dance1_subject2 --seconds 8 \
  --report results/sim2sim/dance_8s.json

# NVIDIA EGL 录制 960×540、25 fps 的 MP4。
MUJOCO_GL=egl bash run_e1_sim2sim.sh --checkpoint models/ckpt_168000.pt \
  --motion dance1_subject2 --seconds 8 \
  --video results/sim2sim/dance_8s.mp4 \
  --report results/sim2sim/dance_8s_video.json

# 打开交互窗口，需要可用的桌面显示环境。
bash run_e1_sim2sim.sh --checkpoint models/ckpt_168000.pt \
  --motion dance1_subject2
```

只运行 `--headless` 而不指定 `--video` 时，不创建渲染上下文。录制视频时，主机需要可用的 EGL/OpenGL 驱动；容器还需向进程提供相应 NVIDIA 图形库。交互窗口支持空格暂停、R 重置。

安装开发检查与绘图依赖后，检查 MuJoCo 状态约定与执行器映射：

```bash
uv pip install --python .venv-sim2sim/bin/python -r requirements-dev.txt
.venv-sim2sim/bin/python -m pytest -q tests/test_sim2sim.py
```

如果已有完整训练环境，也可以直接用它运行这些仿真命令。切回训练环境时，取消解释器覆盖：

```bash
unset SUPERMIMIC_PYTHON
```

## 6. 常见安装问题

- **训练找不到 USD**：确认动作和 E1 描述文件存在，先执行 `tools/mjcf_to_usd.py --robot e1`；不要复制其他机器生成的带绝对路径 USD 资产代替转换。
- **CUDA 不可用或 Warp 初始化失败**：分别检查 `nvidia-smi`、PyTorch 与 Warp 的可用性。确认解释器为 `.venv-isaaclab3/bin/python`，并检查版本字符串是否为上述 CUDA 12 兼容版本。
- **4096 个环境显存不足**：关闭其他占用显存的进程，或使用较小的环境数、缓冲区和训练批量；参数可追加到 `run_e1_train.sh` 后。
- **MP4 无法编码**：确认同一终端能执行 `ffmpeg` 和 `ffprobe`。
- **EGL 初始化失败**：先验证不带视频的 `--headless` 仿真，再检查主机 NVIDIA 图形驱动及容器的图形设备映射。EGL 错误需要处理渲染环境。

项目代码沿用上游复现的 MIT 许可及作者归属；Isaac Lab、Newton、机器人资产与动作数据分别适用其自身许可。相关来源和归属保留在 [README.md](README.md) 与 [LICENSE](LICENSE) 中。
