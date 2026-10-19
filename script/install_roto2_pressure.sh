#!/usr/bin/env bash
# Install minimal pinned 2026 RoTO PPO into a working Isaac Lab 3 beta venv.
# Never let the research package reinstall or upgrade Isaac/Torch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LINKAGENT_ISAACLAB_ROOT:?Set LINKAGENT_ISAACLAB_ROOT or run script/bootstrap_roto2_pressure.sh}"
PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
if [[ "$(uname -s)" != Linux ]]; then
  echo "Verified pipeline requires Linux + NVIDIA GPU, not $(uname -s)" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "Missing $PY. Install Isaac Lab 3.0.0-beta2.patch1 + Sim 6.0.1 first." >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Git required" >&2; exit 2
fi
"$PY" - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Requires Python 3.12, got "+sys.version)
for dep, required in (("isaacsim", "6.0.1"), ("torch", "2.10.0")):
    try:
        actual = version(dep)
    except PackageNotFoundError:
        raise SystemExit(f"Missing {dep} in this Isaac Lab environment")
    print(f"Base {dep}={actual}")
    if not actual.startswith(required):
        raise SystemExit(f"Expected {dep} {required} but got {actual}")
for dep in ("isaaclab", "isaaclab-physx"):
    try:
        print(f"Base {dep}="+version(dep))
    except PackageNotFoundError:
        raise SystemExit(f"Missing {dep}; run official Isaac Lab setup first")
PY
cd "$ROOT"
# Some checkout modes store vendor meshes in Git LFS; this is a no-op otherwise.
if git lfs version >/dev/null 2>&1; then
  git lfs pull --include="botSDK/urdf/01_vendor_original/02_extracted_packages/L20_8_r_s1/meshes/*"
fi
DEPS="$ROOT/.external_roto2"
mkdir -p "$DEPS"
fetch_verified() {
  local repo="$1" path="$2" commit="$3"
  if [[ ! -d "$DEPS/$path/.git" ]]; then
    git clone --no-checkout "https://github.com/$repo.git" "$DEPS/$path"
  fi
  git -C "$DEPS/$path" fetch --depth=1 origin "$commit"
  git -C "$DEPS/$path" checkout --detach "$commit"
  test "$(git -C "$DEPS/$path" rev-parse HEAD)" = "$commit"
}
# Upstream HEADs checked 2026-10-19; never silently follow main.
fetch_verified elle-miller/multimodal_rl multimodal_rl 4ac55e854b4c732534d443715a0322d4220b461c
fetch_verified elle-miller/roto roto 0fa344e6e224f141cb6ab3ed4c2fbbbd86cf1cc2

# We import the upstream PPO, but not the 2026 benchmark's original
# Allegro/Shadow assets. Skip unneeded optuna, dashboards, kornia etc.
"$PY" -m pip install --no-deps -e "$DEPS/multimodal_rl"

# Explicit torch constraint prevents pip from replacing Isaac's CUDA wheels.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
printf 'torch==2.10.0\n' > "$CONSTRAINTS"
"$PY" -m pip install -c "$CONSTRAINTS" \
    "gym>=0.26,<1" "gymnasium>=1,<2" "skrl==2.1.0" \
    "PyYAML>=6,<7" "tqdm>=4.66,<5" "matplotlib>=3.8,<4" \
    "tensorboard>=2.17,<3" "wandb>=0.28,<0.29"

"$PY" - <<'PY'
from importlib.metadata import version
import torch
import gymnasium, gym, yaml, matplotlib, skrl, wandb
from multimodal_rl.rl.ppo import PPO
from multimodal_rl.rl.trainer import Trainer
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper
print("RoTO2 PPO + Trainer + Encoder + wrapper: import PASS")
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access the NVIDIA GPU: no training")
PY
echo
echo "Installed; next run real GPU/PhysX smoke:"
echo "  \"$PY\" -m script.train_roto2_pressure --headless --smoke --num-envs 8"
echo "  \"$PY\" -m script.train_roto2_pressure --headless --num-envs 256 --million-steps 100 --seed 42 --pressure-variant jacobian"
