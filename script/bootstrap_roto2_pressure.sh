#!/usr/bin/env bash
# On a clean, compatible Ubuntu x86_64 + NVIDIA workstation, create the
# EXACT Isaac Lab 3 beta / Isaac Sim 6 environment proven by LinkAgent.
# This script does not install a system NVIDIA driver or accept Omniverse EULA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="${LINKAGENT_ISAACLAB_ROOT:-$HOME/isaaclab/IsaacLab-3.0.0-beta2.patch1}"
ISAAC_SHA=ffff603eafc6b74264a5261cc0183d6a65390d78

fail() { echo "[training setup] $*" >&2; exit 2; }
[[ "$(uname -s)" == "Linux" ]] || fail "Requires Linux (recommend Ubuntu 24.04)."
[[ "$(uname -m)" == "x86_64" ]] || fail "This pinned recipe is for x86_64 NVIDIA GPUs only."
command -v nvidia-smi >/dev/null 2>&1 || fail "Missing NVIDIA driver/nvidia-smi."
nvidia-smi -L || fail "The NVIDIA GPU is not accessible."
command -v git >/dev/null 2>&1 || fail "Install git first."
command -v uv >/dev/null 2>&1 || fail "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1 || fail "Install git-lfs (Isaac Lab source includes large assets)."
command -v getconf >/dev/null 2>&1 || fail "GNU libc required."
GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
uvx=""
# Isaac Sim 6 pip wheels require glibc >= 2.35 (Ubuntu 22.04+).
if ! awk -v v="$GLIBC" 'BEGIN{split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=35))}'; then
  fail "glibc $GLIBC is too old for Isaac Sim 6 pip wheels (requires >=2.35)."
fi

mkdir -p "$(dirname "$ISAAC_ROOT")"
if [[ ! -d "$ISAAC_ROOT/.git" ]]; then
  [[ ! -e "$ISAAC_ROOT" ]] || fail "$ISAAC_ROOT exists but is not a Git checkout."
  git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$ISAAC_ROOT"
fi
git -C "$ISAAC_ROOT" fetch --depth=1 origin "$ISAAC_SHA"
git -C "$ISAAC_ROOT" checkout --detach "$ISAAC_SHA"
[[ "$(git -C "$ISAAC_ROOT" rev-parse HEAD)" == "$ISAAC_SHA" ]] || fail "Isaac Lab checkout SHA mismatch."
git -C "$ISAAC_ROOT" lfs pull

if [[ ! -f "$ISAAC_ROOT/env_isaaclab/bin/python" ]]; then
  (cd "$ISAAC_ROOT" && uv venv --python 3.12 --seed env_isaaclab)
fi
PY="$ISAAC_ROOT/env_isaaclab/bin/python"
"$PY" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
PY

# Follow the Isaac Lab v3.0-beta2.p1 pip-install recipe exactly:
# https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/pip_installation.html
if ! "$PY" -m pip show isaacsim >/dev/null 2>&1; then
  uv pip install --python "$PY" \
    "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow
fi
uv pip install --python "$PY" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Use the official 2026 Isaac Lab installer for its full compatible modules.
if ! "$PY" -m pip show isaaclab-physx >/dev/null 2>&1; then
  (cd "$ISAAC_ROOT" && (
     source env_isaaclab/bin/activate
     ./isaaclab.sh --install
  ))
fi

export LINKAGENT_ISAACLAB_ROOT="$ISAAC_ROOT"
echo "[training setup] Isaac Lab base at $ISAAC_ROOT"
echo "[training setup] Installing pinned RoTO2 PPO dependencies without changing Torch..."
bash "$ROOT/script/install_roto2_pressure.sh"
echo "[training setup] Ready to run the *real* CUDA/PhysX smoke:"
echo "  cd \"$ROOT\""
echo "  export LINKAGENT_ISAACLAB_ROOT=\"$ISAAC_ROOT\""
echo "  \"$PY\" -m script.train_roto2_pressure --headless --smoke --num-envs 8"
echo "The first Isaac Sim launch may require you to read and accept its NVIDIA license. This script does NOT accept it on your behalf."
