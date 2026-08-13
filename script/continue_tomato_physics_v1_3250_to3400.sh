#!/usr/bin/env bash
set -euo pipefail

project_root="/home/descfly/linkagent-physics-perf"
isaac_root="/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1"
python_bin="${isaac_root}/env_isaaclab/bin/python"
train_entry="${isaac_root}/scripts/reinforcement_learning/train.py"
source_run="${project_root}/logs/rsl_rl/tomato_pick_physics_v1/2026-08-11_08-52-10_tomato_pick_physics_v1_resume3000_to3400_8192"
checkpoint="${source_run}/model_3250.pt"
expected_sha256="2aab5363450944def8fb1a1311e423a0d34601ee5e6a6657e3b7e525debc29e5"
driver_log="/tmp/tomato_physics_v1_convergence_driver_3250_to3400.log"
training_log="/tmp/tomato_physics_v1_formal_8192_perf_v7_resume3250_to3400.log"

exec >>"${driver_log}" 2>&1
echo "[$(date -Is)] validating paused full-state checkpoint"
[[ -s "${checkpoint}" ]] || {
  echo "missing checkpoint: ${checkpoint}"
  exit 1
}
actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
[[ "${actual_sha256}" == "${expected_sha256}" ]] || {
  echo "checkpoint SHA256 mismatch: ${actual_sha256}"
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
if int(payload["iter"]) != 3250:
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

if pgrep -af "${train_entry}.*LinkAgent-TomatoPick-Physics-v1" >/dev/null; then
  echo "another Physics-v1 training process is active; refusing concurrent launch"
  pgrep -af "${train_entry}.*LinkAgent-TomatoPick-Physics-v1"
  exit 1
fi

export LINKAGENT_ISAACLAB_ROOT="${isaac_root}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${project_root}"
export TOMATO_REWARD_ITERATION_OFFSET=3250
unset WANDB_RUN_ID WANDB_RESUME

echo "[$(date -Is)] launching full-state continuation absolute iter3250 to3400"
echo "[$(date -Is)] model labels will run 3250..3399; model_3399.pt is the next atomic boundary"
cd "${project_root}"
set -o pipefail
"${python_bin}" "${train_entry}" \
  --rl_library rsl_rl \
  --external_callback simulation.tomato_pick.isaac.physics_v1.register_task \
  --task LinkAgent-TomatoPick-Physics-v1 \
  --num_envs 8192 \
  --seed 42 \
  --max_iterations 150 \
  --logger wandb \
  --experiment_name tomato_pick_physics_v1 \
  --run_name tomato_pick_physics_v1_resume3250_to3400_8192 \
  --resume \
  --load_run '^2026-08-11_08-52-10_tomato_pick_physics_v1_resume3000_to3400_8192$' \
  --checkpoint '^model_3250\.pt$' \
  --viz none 2>&1 | tee -a "${training_log}"
