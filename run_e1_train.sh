#!/bin/bash
# Run the original Isaac Lab 3 / Newton training loop with the local E1 assets.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-e1_paper}
if [ "$#" -gt 0 ]; then shift; fi
case "$RUN" in
  *[!a-zA-Z0-9_.-]*|''|.|..) echo "Run name must contain only letters, digits, _, . or -" >&2; exit 1 ;;
esac
PYTHON=${SUPERMIMIC_PYTHON:-"$PWD/.venv-isaaclab3/bin/python"}
if [ ! -x "$PYTHON" ]; then
  echo "Isaac Lab 3 interpreter not found: $PYTHON (see install.md)" >&2
  exit 1
fi
mkdir -p "logs/$RUN" .cache/warp .cache/inductor .cache/mpl .cache/xdg
if [ -f "logs/$RUN/train.pid" ] && kill -0 "$(cat "logs/$RUN/train.pid")" 2>/dev/null; then
  echo "Training is already running (PID $(cat "logs/$RUN/train.pid"))" >&2
  exit 1
fi
if [ -e "logs/$RUN/args.json" ] || [ -e "logs/$RUN/stdout.log" ]; then
  echo "Run directory already contains training output; choose a new run name: logs/$RUN" >&2
  exit 1
fi
export WARP_CACHE_PATH="$PWD/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export MPLCONFIGDIR="$PWD/.cache/mpl"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
nohup setsid "$PYTHON" -u supermimic/train.py --robot e1 --motion_dir "$PWD/datasets/e1_lafan" \
  --log_dir "$PWD/logs/$RUN" \
  --num_envs 4096 --buffer_steps 1000 --wm_window 8 --wm_batches 10 --wm_batch 2048 \
  --pi_batch 4096 --horizon 32 --critic_epochs 5 --critic_minibatches 4 \
  --lr 0.0002 --gamma 0.99 --lam 0.95 --utd 0.4 --action_noise 0.1 \
  --amp none --wm_only_iters 0 --lr_decay_iters 250000 --iterations 320000 \
  --ckpt_every 2000 --log_every 20 "$@" \
  >"logs/$RUN/stdout.log" 2>&1 </dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" >"logs/$RUN/train.pid"
echo "Started E1 training: PID=$TRAIN_PID, log=$PWD/logs/$RUN/stdout.log"
