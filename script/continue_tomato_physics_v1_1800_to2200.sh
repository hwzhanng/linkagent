#!/usr/bin/env bash
set -euo pipefail

project_root="/home/descfly/linkagent-physics-perf"
isaac_root="/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1"
python_bin="${isaac_root}/env_isaaclab/bin/python"
train_entry="${isaac_root}/scripts/reinforcement_learning/train.py"
source_pid=2671582
source_run="${project_root}/logs/rsl_rl/tomato_pick_physics_v1/2026-08-10_21-20-38_tomato_pick_physics_v1_resume1200_to1800_8192"
checkpoint="${source_run}/model_1799.pt"
driver_log="/tmp/tomato_physics_v1_convergence_driver_1800_to2200.log"
training_log="/tmp/tomato_physics_v1_formal_8192_perf_v7_resume1800_to2200.log"

exec >>"${driver_log}" 2>&1
echo "[$(date -Is)] waiting for source PID ${source_pid} to finish absolute iter1800"
while kill -0 "${source_pid}" 2>/dev/null; do
  sleep 5
done

echo "[$(date -Is)] source process exited; verifying atomic full-state checkpoint"
for _ in $(seq 1 12); do
  [[ -s "${checkpoint}" ]] && break
  sleep 5
done
[[ -s "${checkpoint}" ]] || {
  echo "missing checkpoint: ${checkpoint}"
  exit 1
}

"${python_bin}" -c '
import sys
import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu", weights_only=False)
required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"checkpoint missing keys: {missing}")
if int(payload["iter"]) != 1799:
    raise SystemExit("checkpoint iter mismatch: {}".format(payload["iter"]))
actor_keys = payload["actor_state_dict"]
critic_keys = payload["critic_state_dict"]
for key in ("obs_normalizer._mean", "obs_normalizer._var", "distribution.log_std_param"):
    if key not in actor_keys:
        raise SystemExit(f"actor checkpoint missing {key}")
for key in ("obs_normalizer._mean", "obs_normalizer._var"):
    if key not in critic_keys:
        raise SystemExit(f"critic checkpoint missing {key}")
if not payload["optimizer_state_dict"]:
    raise SystemExit("optimizer state is empty")
print("full-state checkpoint verified")
' "${checkpoint}"
sha256sum "${checkpoint}"

if pgrep -af "${train_entry}.*LinkAgent-TomatoPick-Physics-v1" >/dev/null; then
  echo "another Physics-v1 training process is still active; refusing concurrent Isaac launch"
  pgrep -af "${train_entry}.*LinkAgent-TomatoPick-Physics-v1"
  exit 1
fi

export LINKAGENT_ISAACLAB_ROOT="${isaac_root}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${project_root}"
export TOMATO_REWARD_ITERATION_OFFSET=1800
unset WANDB_RUN_ID WANDB_RESUME

echo "[$(date -Is)] launching full-state continuation absolute iter1800 to2200"
cd "${project_root}"
set -o pipefail
"${python_bin}" "${train_entry}" \
  --rl_library rsl_rl \
  --external_callback simulation.tomato_pick.isaac.physics_v1.register_task \
  --task LinkAgent-TomatoPick-Physics-v1 \
  --num_envs 8192 \
  --seed 42 \
  --max_iterations 400 \
  --logger wandb \
  --experiment_name tomato_pick_physics_v1 \
  --run_name tomato_pick_physics_v1_resume1800_to2200_8192 \
  --resume \
  --load_run '^2026-08-10_21-20-38_tomato_pick_physics_v1_resume1200_to1800_8192$' \
  --checkpoint '^model_1799\.pt$' \
  --viz none 2>&1 | tee -a "${training_log}"
