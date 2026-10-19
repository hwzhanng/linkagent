#!/usr/bin/env bash
# Install the EXACT peer-reviewed 2026 RoTO 2.0 baseline WITHOUT modifying PPO.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LINKAGENT_ISAACLAB_ROOT:?Set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout}"
PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing Isaac Lab Python at $PY" >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Git is required to fetch pinned 2026 RoTO 2.0 sources" >&2
  exit 2
fi
cd "$ROOT"
git lfs pull --include="botSDK/urdf/01_vendor_original/02_extracted_packages/L20_8_r_s1/meshes/*" 2>/dev/null || true

DEPS="$ROOT/.external_roto2"
mkdir -p "$DEPS"
fetch_verified () {
  local repo="$1" path="$2" commit="$3"
  if [[ ! -d "$DEPS/$path/.git" ]]; then
    git clone --no-checkout "https://github.com/$repo.git" "$DEPS/$path"
  fi
  git -C "$DEPS/$path" fetch --depth=1 origin "$commit"
  git -C "$DEPS/$path" checkout --detach "$commit"
  test "$(git -C "$DEPS/$path" rev-parse HEAD)" = "$commit"
}
# Upstream HEADs recorded on 2026-10-19; never silently chase main.
fetch_verified elle-miller/multimodal_rl multimodal_rl 4ac55e854b4c732534d443715a0322d4220b461c
fetch_verified elle-miller/roto roto 0fa344e6e224f141cb6ab3ed4c2fbbbd86cf1cc2

"$PY" -m pip install -e "$DEPS/multimodal_rl"
"$PY" -m pip install -e "$DEPS/roto"
"$PY" -m pip install PyYAML tensorboard "skrl>=2.1.0,<3"

"$PY" -c "import multimodal_rl.rl.ppo; import multimodal_rl.rl.trainer; import multimodal_rl.wrappers.isaaclab_wrapper; print('2026 RoTO PPO backend imports OK')"
echo "Pinned RoTO 2.0 backend installed. Verify real PhysX before long training:"
echo "  \"$PY\" -m script.train_roto2_pressure --headless --smoke --num-envs 8"
echo "  \"$PY\" -m script.train_roto2_pressure --headless --num-envs 256 --million-steps 100"
