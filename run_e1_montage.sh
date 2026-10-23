#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
DEFAULT_PYTHON="$PWD/.venv-isaaclab3/bin/python"
if [ ! -x "$DEFAULT_PYTHON" ]; then DEFAULT_PYTHON="$PWD/.venv-sim2sim/bin/python"; fi
PYTHON=${SUPERMIMIC_PYTHON:-"$DEFAULT_PYTHON"}
if [ ! -x "$PYTHON" ]; then
  echo "Python environment not found: $PYTHON (see install.md or set SUPERMIMIC_PYTHON)" >&2
  exit 1
fi
exec "$PYTHON" -u tools/lafan_montage.py "$@"
