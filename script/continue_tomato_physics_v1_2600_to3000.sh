#!/usr/bin/env bash
set -euo pipefail

project_root="/home/descfly/linkagent-physics-perf"
isaac_root="/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1"
python_bin="${isaac_root}/env_isaaclab/bin/python"
train_entry="${isaac_root}/scripts/reinforcement_learning/train.py"
source_pid=3082627
source_run="${project_root}/logs/rsl_rl/tomato_pick_physics_v1/2026-08-11_03-40-16_tomato_pick_physics_v1_resume2200_to2600_8192"
# The resumed RSL loop starts at label 2198 and executes 400 updates, so its
# last completed label and atomic checkpoint are 2597 (absolute boundary 2600).
checkpoint="${source_run}/model_2597.pt"
driver_log="/tmp/tomato_physics_v1_convergence_driver_2600_to3000.log"
training_log="/tmp/tomato_physics_v1_formal_8192_perf_v7_resume2600_to3000.log"
evaluation_dir="${project_root}/logs/qualification/tomato_physics_v1/convergence_eval_2600"

exec >>"${driver_log}" 2>&1
echo "[$(date -Is)] waiting for source PID ${source_pid} to finish absolute iter2600"
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
if int(payload["iter"]) != 2597:
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
export TOMATO_REWARD_ITERATION_OFFSET=2600
unset WANDB_RUN_ID WANDB_RESUME
mkdir -p "${evaluation_dir}"

echo "[$(date -Is)] running serial deterministic evaluation for absolute iter2600"
for seed in 17 42 2026; do
  output="${evaluation_dir}/model_2597_seed_${seed}.json"
  log="${evaluation_dir}/model_2597_seed_${seed}.log"
  "${python_bin}" "${project_root}/script/evaluate_tomato_checkpoint.py" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --num_envs 1024 \
    --steps 400 \
    --seed "${seed}" \
    --task LinkAgent-TomatoPick-Physics-v1 \
    --external_callback simulation.tomato_pick.isaac.physics_v1.register_task \
    --viz none >"${log}" 2>&1
  echo "[$(date -Is)] seed ${seed} evaluation complete"
done

jq -s '
  (map(.success_count) | add) as $successes |
  (map(.completed_episodes) | add) as $episodes |
  (map(.drop_count) | add) as $drops |
  {
    schema_version: 1,
    absolute_iteration: 2600,
    checkpoint: .[0].checkpoint,
    checkpoint_sha256: .[0].checkpoint_sha256,
    deterministic_policy: true,
    seeds: map(.seed),
    completed_episodes: $episodes,
    success_count: $successes,
    success_rate: ($successes / $episodes),
    drop_count: $drops,
    drop_rate: ($drops / $episodes),
    per_seed: map({seed, completed_episodes, success_count, success_rate, drop_count, drop_rate, detach_rate, retained_rate})
  }
' "${evaluation_dir}"/model_2597_seed_*.json >"${evaluation_dir}/aggregate.json"
jq '.' "${evaluation_dir}/aggregate.json"

# The 2200 evaluation was below 90%, so even a >=90% result at 2600 cannot
# satisfy the required two consecutive qualifying evaluations. Continue once
# the serial evaluation has completed.
echo "[$(date -Is)] launching full-state continuation absolute iter2600 to3000"
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
  --run_name tomato_pick_physics_v1_resume2600_to3000_8192 \
  --resume \
  --load_run '^2026-08-11_03-40-16_tomato_pick_physics_v1_resume2200_to2600_8192$' \
  --checkpoint '^model_2597\.pt$' \
  --viz none 2>&1 | tee -a "${training_log}"
