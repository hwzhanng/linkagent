#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LINKAGENT_ISAACLAB_ROOT="$ROOT/simulation/.runtime/IsaacLab-3.0.0-beta2.patch1"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
cd "$ROOT"
case "${1:-smoke}" in
  smoke)
    shift || true
    exec "$PY" -m script.train_roto2_pressure --headless --smoke --num-envs 8 "$@"
    ;;
  train)
    shift
    exec "$PY" -m script.train_roto2_pressure --headless --num-envs 16384 --million-steps 100 --seed 42 --pressure-variant jacobian "$@"
    ;;
  evaluate)
    shift
    exec "$PY" -m script.train_roto2_pressure --headless --num-envs 64 --seed 2026 --pressure-variant jacobian "$@"
    ;;
  record)
    shift
    unset DISPLAY
    export __GL_SHADER_DISK_CACHE=0
    exec "$PY" -m script.train_roto2_pressure --headless --num-envs 64 --seed 2026 --pressure-variant jacobian --kit_args '--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0' "$@"
    ;;
  *) echo "Usage: $0 {smoke|train|evaluate|record} [extra arguments]" >&2; exit 2 ;;
esac
