#!/bin/bash
# Evaluate one saved E1 checkpoint over every LaFAN clip using the training simulator.
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR=${1:-logs/e1_paper}
CHECKPOINT=${2:-latest}
PYTHON=${SUPERMIMIC_PYTHON:-"$PWD/.venv-isaaclab3/bin/python"}
if [ ! -x "$PYTHON" ]; then
  echo "Isaac Lab 3 interpreter not found: $PYTHON (see install.md)" >&2
  exit 1
fi
if [ "$CHECKPOINT" = latest ]; then
  shopt -s nullglob
  CHECKPOINTS=("$RUN_DIR"/ckpt_*.pt)
  if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "No saved checkpoints in $RUN_DIR" >&2
    exit 1
  fi
  CHECKPOINT=${CHECKPOINTS[-1]}
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi
VERSION=$(basename "$CHECKPOINT" .pt)
VERSION=${VERSION#ckpt_}
mkdir -p "$RUN_DIR" .cache/warp .cache/inductor .cache/mpl .cache/xdg
export WARP_CACHE_PATH="$PWD/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export MPLCONFIGDIR="$PWD/.cache/mpl"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
"$PYTHON" -u supermimic/evaluate.py --robot e1 --run_dir "$RUN_DIR" \
  --ckpt "$CHECKPOINT" --motion_dir "$PWD/datasets/e1_lafan" \
  --output "$RUN_DIR/eval_${VERSION}_summary.json" \
  2>&1 | tee "$RUN_DIR/eval_${VERSION}.log"
