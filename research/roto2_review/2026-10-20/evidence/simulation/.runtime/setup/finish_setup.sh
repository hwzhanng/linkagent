#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export LINKAGENT_ISAACLAB_ROOT="$ROOT/simulation/.runtime/IsaacLab-3.0.0-beta2.patch1"
export PATH="/home/descfly/linkagent-ppo-tools/bin:/home/descfly/linkagent-ppo-tools/cmake/data/bin:$PATH"
export UV_CACHE_DIR="$ROOT/simulation/.runtime/setup/uv-cache"
export UV_HTTP_TIMEOUT=300
export PIP_NO_CACHE_DIR=1
export UV_INDEX_STRATEGY=unsafe-best-match
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH
PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
cd "$ROOT"
"$PY" -m pip show isaacsim
# Official core installation includes the PhysX, assets, environments and RL
# adapters required here. Optional teleoperation and alternative RL stacks
# are not used by the pinned multimodal_rl PPO.
(
  cd "$LINKAGENT_ISAACLAB_ROOT"
  source env_isaaclab/bin/activate
  ./isaaclab.sh --install core
)
bash simulation/.runtime/setup/install_roto2_verified.sh
"$PY" -m pytest -q tests/test_roto2_pressure_track.py
"$PY" -m pip freeze > simulation/.runtime/setup/requirements-installed.txt
"$PY" -m pip check > simulation/.runtime/setup/pip-check.txt 2>&1 || true
echo 'Dependency installation and CPU regression finished. GPU/PhysX smoke remains required.'
