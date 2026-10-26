#!/usr/bin/env bash
# Relocatable entry point. Keeps the tested invocation settings unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LINKAGENT_ISAACLAB_ROOT="${LINKAGENT_ISAACLAB_ROOT:-$ROOT/simulation/.runtime/IsaacLab-3.0.0-beta2.patch1}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
unset PYTHONPATH
PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
cd "$ROOT"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
 smoke) exec "$PY" -m script.train_roto2_pressure --headless --smoke --num-envs 8 "$@" ;;
 train) exec "$PY" -m script.train_roto2_pressure --headless --num-envs 8192 --seed 42 --pressure-variant plain "$@" ;;
 evaluate) exec "$PY" -m script.train_roto2_pressure --headless --num-envs 64 --seed 2026 --pressure-variant plain "$@" ;;
 record)
   unset DISPLAY
   export __GL_SHADER_DISK_CACHE=0
   exec "$PY" -m script.train_roto2_pressure --headless --num-envs 64 --seed 2026 --pressure-variant plain --kit_args '--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0' "$@" ;;
 *) echo 'Usage: bash script/run_roto2.sh {smoke|train|evaluate|record} [arguments]' >&2; exit 2 ;;
esac
