#!/usr/bin/env bash
set -euo pipefail

project_root="/home/descfly/linkagent-physics-perf"
isaac_root="/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1"
python_bin="${isaac_root}/env_isaaclab/bin/python"
train_entry="${isaac_root}/scripts/reinforcement_learning/train.py"
log_root="${project_root}/logs/rsl_rl/tomato_pick_physics_v1"
qualification_root="${project_root}/logs/qualification/tomato_physics_v1"
controller_log="/tmp/tomato_physics_v1_curve_convergence_from3800.log"
state_file="${qualification_root}/curve_convergence_controller_state.json"

# The already-armed 3400->3800 driver will create this source run. Starting
# from model_3399 and applying 400 updates yields the atomic model_3798.
boundary=3800
checkpoint_label=3798
source_run_glob="*_tomato_pick_physics_v1_resume3400_to3800_8192"
training_log="/tmp/tomato_physics_v1_formal_8192_perf_v7_resume3400_to3800_after_pause.log"

exec >>"${controller_log}" 2>&1
echo "[$(date -Is)] curve-convergence controller armed at absolute iter${boundary}"

latest_matching_run() {
  local pattern="$1"
  find "${log_root}" -maxdepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

wait_for_no_physics_v1_isaac() {
  while pgrep -af "${train_entry}.*LinkAgent-TomatoPick-Physics-v1" >/dev/null \
    || pgrep -af "${project_root}/script/evaluate_tomato_checkpoint.py" >/dev/null; do
    sleep 5
  done
}

verify_full_state_checkpoint() {
  local checkpoint="$1"
  local expected_iter="$2"
  "${python_bin}" -c '
import sys
import torch

path = sys.argv[1]
expected_iter = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"checkpoint missing keys: {missing}")
if int(payload["iter"]) != expected_iter:
    raise SystemExit("checkpoint iter mismatch: {} != {}".format(payload["iter"], expected_iter))
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
' "${checkpoint}" "${expected_iter}"
}

evaluate_boundary() {
  local checkpoint="$1"
  local logical_boundary="$2"
  local model_label="$3"
  local evaluation_dir="${qualification_root}/convergence_eval_${logical_boundary}"

  mkdir -p "${evaluation_dir}"
  export TOMATO_REWARD_ITERATION_OFFSET="${logical_boundary}"
  echo "[$(date -Is)] serial deterministic evaluation at absolute iter${logical_boundary}"
  for seed in 17 42 2026; do
    local output="${evaluation_dir}/model_${model_label}_seed_${seed}.json"
    local log="${evaluation_dir}/model_${model_label}_seed_${seed}.log"
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

  jq -s --argjson absolute_iteration "${logical_boundary}" '
    (map(.success_count) | add) as $successes |
    (map(.completed_episodes) | add) as $episodes |
    (map(.drop_count) | add) as $drops |
    {
      schema_version: 1,
      absolute_iteration: $absolute_iteration,
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
  ' "${evaluation_dir}"/model_"${model_label}"_seed_*.json >"${evaluation_dir}/aggregate.json"

  "${python_bin}" -c '
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
ansi = re.compile(r"\x1b\[[0-9;]*m")
iteration_pattern = re.compile(r"Learning iteration\s+(\d+)/\d+")
success_pattern = re.compile(r"Metrics/success_rate:\s+([0-9.]+)")
current_iteration = None
values = {}
for raw in log_path.read_text(errors="replace").splitlines():
    line = ansi.sub("", raw)
    match = iteration_pattern.search(line)
    if match:
        current_iteration = int(match.group(1))
        continue
    match = success_pattern.search(line)
    if match and current_iteration is not None:
        values[current_iteration] = float(match.group(1))
ordered = [values[key] for key in sorted(values)]
if len(ordered) < 200:
    raise SystemExit(f"need at least 200 success samples, found {len(ordered)}")
previous = ordered[-200:-100]
latest = ordered[-100:]
previous_mean = sum(previous) / len(previous)
latest_mean = sum(latest) / len(latest)
gain = latest_mean - previous_mean
payload = {
    "samples": len(ordered),
    "window_iterations": 100,
    "previous_window_mean_success": previous_mean,
    "latest_window_mean_success": latest_mean,
    "gain": gain,
    "absolute_gain": abs(gain),
    "plateau_at_one_percentage_point": abs(gain) <= 0.01,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n")
' "${training_log}" "${evaluation_dir}/curve_growth.json"
}

while true; do
  source_run=""
  checkpoint=""
  while true; do
    source_run="$(latest_matching_run "${source_run_glob}")"
    if [[ -n "${source_run}" ]]; then
      checkpoint="${source_run}/model_${checkpoint_label}.pt"
      [[ -s "${checkpoint}" ]] && break
    fi
    sleep 15
  done

  wait_for_no_physics_v1_isaac
  verify_full_state_checkpoint "${checkpoint}" "${checkpoint_label}"
  checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  echo "[$(date -Is)] boundary checkpoint ${checkpoint} ${checkpoint_sha256}"

  export LINKAGENT_ISAACLAB_ROOT="${isaac_root}"
  export OMNI_KIT_ACCEPT_EULA=YES
  export PYTHONPATH="${project_root}"
  unset WANDB_RUN_ID WANDB_RESUME
  evaluate_boundary "${checkpoint}" "${boundary}" "${checkpoint_label}"

  evaluation_dir="${qualification_root}/convergence_eval_${boundary}"
  current_rate="$(jq -r '.success_rate' "${evaluation_dir}/aggregate.json")"
  previous_boundary=$((boundary - 400))
  previous_aggregate="${qualification_root}/convergence_eval_${previous_boundary}/aggregate.json"
  if [[ -s "${previous_aggregate}" ]]; then
    previous_rate="$(jq -r '.success_rate' "${previous_aggregate}")"
  else
    previous_rate="0"
  fi
  plateau="$(jq -r '.plateau_at_one_percentage_point' "${evaluation_dir}/curve_growth.json")"
  previous_streak=0
  if [[ -s "${state_file}" ]]; then
    previous_streak="$(jq -r '.plateau_streak // 0' "${state_file}")"
  fi
  if [[ "${plateau}" == "true" ]]; then
    plateau_streak=$((previous_streak + 1))
  else
    plateau_streak=0
  fi

  converged="$(jq -nr \
    --argjson current "${current_rate}" \
    --argjson previous "${previous_rate}" \
    --argjson plateau_streak "${plateau_streak}" \
    '($current >= 0.9) and ($previous >= 0.9) and ((($current - $previous) | if . < 0 then -. else . end) <= 0.01) and ($plateau_streak >= 2)')"

  old_history='[]'
  if [[ -s "${state_file}" ]]; then
    old_history="$(jq -c '.history // []' "${state_file}")"
  fi
  jq -n \
    --arg status "$([[ "${converged}" == "true" ]] && echo converged || echo training)" \
    --arg updated_at_local "$(date -Is)" \
    --arg checkpoint "${checkpoint}" \
    --arg checkpoint_sha256 "${checkpoint_sha256}" \
    --argjson absolute_iteration "${boundary}" \
    --argjson checkpoint_iteration_field "${checkpoint_label}" \
    --argjson success_rate "${current_rate}" \
    --argjson previous_success_rate "${previous_rate}" \
    --argjson plateau_streak "${plateau_streak}" \
    --argjson converged "${converged}" \
    --argjson history "${old_history}" \
    --slurpfile curve "${evaluation_dir}/curve_growth.json" \
    '{
      schema_version: 1,
      status: $status,
      updated_at_local: $updated_at_local,
      hard_iteration_cap: null,
      absolute_iteration: $absolute_iteration,
      checkpoint_iteration_field: $checkpoint_iteration_field,
      checkpoint: $checkpoint,
      checkpoint_sha256: $checkpoint_sha256,
      success_rate: $success_rate,
      previous_success_rate: $previous_success_rate,
      plateau_streak: $plateau_streak,
      converged: $converged,
      history: ($history + [{
        absolute_iteration: $absolute_iteration,
        checkpoint_iteration_field: $checkpoint_iteration_field,
        checkpoint_sha256: $checkpoint_sha256,
        success_rate: $success_rate,
        curve: $curve[0],
        plateau_streak: $plateau_streak,
        converged: $converged
      }])
    }' >"${state_file}.tmp"
  mv "${state_file}.tmp" "${state_file}"
  jq '.' "${evaluation_dir}/aggregate.json"
  jq '.' "${evaluation_dir}/curve_growth.json"
  jq '.' "${state_file}"

  if [[ "${converged}" == "true" ]]; then
    echo "[$(date -Is)] successful curve-and-evaluation convergence confirmed at iter${boundary}"
    exit 0
  fi

  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( available_kb < 8 * 1024 * 1024 )); then
    echo "available memory below 8 GiB; refusing next large training segment"
    exit 1
  fi
  if pgrep -af "${train_entry}.*LinkAgent-TomatoPick-Physics-v1" >/dev/null; then
    echo "another Physics-v1 training process is active; refusing concurrent launch"
    exit 1
  fi

  target=$((boundary + 400))
  run_name="tomato_pick_physics_v1_resume${boundary}_to${target}_8192"
  training_log="/tmp/tomato_physics_v1_formal_8192_perf_v7_resume${boundary}_to${target}_auto.log"
  export TOMATO_REWARD_ITERATION_OFFSET="${boundary}"
  echo "[$(date -Is)] curve not yet converged; full-state continuation iter${boundary} to${target}"
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
    --run_name "${run_name}" \
    --resume \
    --load_run "^$(basename "${source_run}")$" \
    --checkpoint "^model_${checkpoint_label}\\.pt$" \
    --viz none 2>&1 | tee -a "${training_log}"

  boundary="${target}"
  checkpoint_label=$((checkpoint_label + 399))
  source_run_glob="*_${run_name}"
done
