#!/usr/bin/env bash
set -euo pipefail

confirmation="START_PHYSICS_V2_CURVE_CONTROLLER"
if [[ "${1:-}" != "--execute" || "${2:-}" != "${confirmation}" ]]; then
  echo "dry safety stop: use --execute ${confirmation} after external audit"
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_root="${LINKAGENT_ISAACLAB_ROOT:?set LINKAGENT_ISAACLAB_ROOT}"
python_bin="${isaac_root}/env_isaaclab/bin/python"
train_entry="${isaac_root}/scripts/reinforcement_learning/train.py"
experiment="tomato_pick_physics_v2_public_prior_v1"
task="LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1"
callback="simulation.tomato_pick.isaac.physics_v2.register_task"
log_root="${project_root}/logs/rsl_rl/${experiment}"
qualification_root="${project_root}/logs/qualification/tomato_physics_v2"
state_file="${qualification_root}/curve_controller_state.json"
history_file="${qualification_root}/curve_controller_history.jsonl"
controller_log="${qualification_root}/curve_controller.log"
lock_file="${qualification_root}/curve_controller.lock"
segment_updates=400

mkdir -p "${qualification_root}" "${log_root}"
exec >>"${controller_log}" 2>&1
echo "[$(date -Is)] Physics-v2 curve controller starting"

for command_name in flock jq sha256sum awk pgrep; do
  command -v "${command_name}" >/dev/null || {
    echo "required command is unavailable: ${command_name}"
    exit 1
  }
done
[[ -x "${python_bin}" ]] || { echo "managed Isaac Python is unavailable"; exit 1; }
[[ -f "${train_entry}" ]] || { echo "Isaac training entry is unavailable"; exit 1; }
exec 9>"${lock_file}"
flock -n 9 || { echo "another Physics-v2 curve controller owns the lock"; exit 1; }

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${project_root}"
unset WANDB_RUN_ID WANDB_RESUME

latest_matching_run() {
  local pattern="$1"
  find "${log_root}" -maxdepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

refuse_concurrent_v2() {
  if pgrep -af "${train_entry}.*${task}" >/dev/null \
    || pgrep -af "${project_root}/script/evaluate_tomato_checkpoint.py.*${task}" >/dev/null; then
    echo "another Physics-v2 Isaac process is active; refusing concurrent launch"
    exit 1
  fi
}

verify_full_state_checkpoint() {
  local checkpoint="$1"
  local expected_iter="$2"
  "${python_bin}" -c '
import sys
import torch

path = sys.argv[1]
expected = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"checkpoint missing keys: {missing}")
if int(payload["iter"]) != expected:
    raise SystemExit(f"checkpoint iter mismatch: {payload['"'"'iter'"'"']} != {expected}")
for key in ("obs_normalizer._mean", "obs_normalizer._var", "distribution.log_std_param"):
    if key not in payload["actor_state_dict"]:
        raise SystemExit(f"actor checkpoint missing {key}")
for key in ("obs_normalizer._mean", "obs_normalizer._var"):
    if key not in payload["critic_state_dict"]:
        raise SystemExit(f"critic checkpoint missing {key}")
if not payload["optimizer_state_dict"]:
    raise SystemExit("optimizer state is empty")
' "${checkpoint}" "${expected_iter}"
}

run_training_segment() {
  local start="$1"
  local target="$2"
  local source_run="${3:-}"
  local checkpoint_label="${4:-}"
  local run_name="tomato_pick_physics_v2_from${start}_to${target}_8192"
  local training_log="${qualification_root}/train_${start}_to_${target}.log"
  export TOMATO_REWARD_ITERATION_OFFSET="${start}"
  refuse_concurrent_v2
  if [[ -z "${source_run}" ]]; then
    "${python_bin}" "${train_entry}" \
      --rl_library rsl_rl \
      --external_callback "${callback}" \
      --task "${task}" \
      --num_envs 8192 \
      --seed 42 \
      --max_iterations "${segment_updates}" \
      --logger wandb \
      --experiment_name "${experiment}" \
      --run_name "${run_name}" \
      --viz none 2>&1 | tee -a "${training_log}"
  else
    "${python_bin}" "${train_entry}" \
      --rl_library rsl_rl \
      --external_callback "${callback}" \
      --task "${task}" \
      --num_envs 8192 \
      --seed 42 \
      --max_iterations "${segment_updates}" \
      --logger wandb \
      --experiment_name "${experiment}" \
      --run_name "${run_name}" \
      --resume \
      --load_run "^$(basename "${source_run}")$" \
      --checkpoint "^model_${checkpoint_label}\\.pt$" \
      --viz none 2>&1 | tee -a "${training_log}"
  fi
}

evaluate_boundary() {
  local checkpoint="$1"
  local boundary="$2"
  local checkpoint_label="$3"
  local training_log="$4"
  local evaluation_dir="${qualification_root}/convergence_eval_${boundary}"
  mkdir -p "${evaluation_dir}"
  export TOMATO_REWARD_ITERATION_OFFSET="${boundary}"
  for seed in 17 42 2026; do
    "${python_bin}" "${project_root}/script/evaluate_tomato_checkpoint.py" \
      --checkpoint "${checkpoint}" \
      --output "${evaluation_dir}/model_${checkpoint_label}_seed_${seed}.json" \
      --num_envs 1024 \
      --steps 400 \
      --seed "${seed}" \
      --task "${task}" \
      --external_callback "${callback}" \
      --viz none >"${evaluation_dir}/model_${checkpoint_label}_seed_${seed}.log" 2>&1
  done
  "${python_bin}" - "${evaluation_dir}" "${boundary}" "${training_log}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
boundary = int(sys.argv[2])
training_log = Path(sys.argv[3])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("model_*_seed_*.json"))]
episodes = sum(row["completed_episodes"] for row in rows)
success = sum(row["success_count"] for row in rows)
drop = sum(row["drop_count"] for row in rows)

def weighted_metric(name):
    return sum(
        row.get("physics_v2_episode_strategy_metrics", {}).get(name, 0.0)
        * row["completed_episodes"]
        for row in rows
    ) / episodes

aggregate = {
    "schema_version": 1,
    "absolute_iteration": boundary,
    "checkpoint": rows[0]["checkpoint"],
    "checkpoint_sha256": rows[0]["checkpoint_sha256"],
    "deterministic_policy": True,
    "seeds": [row["seed"] for row in rows],
    "completed_episodes": episodes,
    "success_count": success,
    "success_rate": success / episodes,
    "drop_count": drop,
    "drop_rate": drop / episodes,
    "paper_strategy_candidate_rate": weighted_metric("paper_strategy_candidate_rate"),
    "paper_strategy_success_rate": weighted_metric("paper_strategy_success_rate"),
    "twist_assisted_break_rate": weighted_metric("twist_assisted_break_rate"),
    "bend_assisted_break_rate": weighted_metric("bend_assisted_break_rate"),
    "moderate_pull_break_rate": weighted_metric("moderate_pull_break_rate"),
    "per_seed": [
        {
            key: row[key]
            for key in (
                "seed", "completed_episodes", "success_rate", "drop_rate",
                "detach_rate", "retained_rate"
            )
        }
        for row in rows
    ],
}
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

ansi = re.compile(r"\x1b\[[0-9;]*m")
iteration_pattern = re.compile(r"Learning iteration\s+(\d+)/\d+")
success_pattern = re.compile(r"Metrics/success_rate:\s+([0-9.]+)")
current = None
values = {}
for raw in training_log.read_text(errors="replace").splitlines():
    line = ansi.sub("", raw)
    match = iteration_pattern.search(line)
    if match:
        current = int(match.group(1))
        continue
    match = success_pattern.search(line)
    if match and current is not None:
        values[current] = float(match.group(1))
ordered = [values[key] for key in sorted(values)]
if len(ordered) < 200:
    raise SystemExit(f"need 200 training-curve samples, found {len(ordered)}")
previous = ordered[-200:-100]
latest = ordered[-100:]
previous_mean = sum(previous) / len(previous)
latest_mean = sum(latest) / len(latest)
gain = latest_mean - previous_mean
curve = {
    "samples": len(ordered),
    "window_iterations": 100,
    "previous_window_mean_success": previous_mean,
    "latest_window_mean_success": latest_mean,
    "gain": gain,
    "absolute_gain": abs(gain),
    "plateau_at_one_percentage_point": abs(gain) <= 0.01,
}
(root / "curve_growth.json").write_text(json.dumps(curve, indent=2) + "\n")
PY
}

initial_run_glob="*_tomato_pick_physics_v2_from0_to400_8192"
if [[ ! -s "${state_file}" ]]; then
  if [[ -n "$(latest_matching_run "${initial_run_glob}")" ]]; then
    echo "untracked pre-existing Physics-v2 run found; refusing implicit adoption"
    exit 1
  fi
  jq -n '{schema_version:1,status:"initialized",hard_iteration_cap:null,
    segment_updates:400,absolute_iteration:0,converged:false}' >"${state_file}.tmp"
  mv "${state_file}.tmp" "${state_file}"
fi

if [[ "$(jq -r '.status' "${state_file}")" == "converged" ]]; then
  echo "controller state is already converged"
  exit 0
fi

last_boundary="$(jq -r '.absolute_iteration // 0' "${state_file}")"
if (( last_boundary >= segment_updates )); then
  previous_checkpoint="$(jq -r '.checkpoint' "${state_file}")"
  previous_checkpoint_label="$(jq -r '.checkpoint_iteration_field' "${state_file}")"
  previous_run="$(dirname "${previous_checkpoint}")"
  verify_full_state_checkpoint "${previous_checkpoint}" "${previous_checkpoint_label}"
  boundary=$((last_boundary + segment_updates))
  checkpoint_label=$((previous_checkpoint_label + segment_updates - 1))
  source_run_glob="*_tomato_pick_physics_v2_from${last_boundary}_to${boundary}_8192"
  training_log="${qualification_root}/train_${last_boundary}_to_${boundary}.log"
  source_run="$(latest_matching_run "${source_run_glob}")"
  if [[ -z "${source_run}" || ! -s "${source_run}/model_${checkpoint_label}.pt" ]]; then
    run_training_segment \
      "${last_boundary}" "${boundary}" "${previous_run}" "${previous_checkpoint_label}"
  fi
else
  boundary=400
  checkpoint_label=399
  source_run_glob="${initial_run_glob}"
  training_log="${qualification_root}/train_0_to_400.log"
  source_run="$(latest_matching_run "${source_run_glob}")"
  if [[ -z "${source_run}" || ! -s "${source_run}/model_${checkpoint_label}.pt" ]]; then
    run_training_segment 0 400
  fi
fi

while true; do
  source_run="$(latest_matching_run "${source_run_glob}")"
  checkpoint="${source_run}/model_${checkpoint_label}.pt"
  verify_full_state_checkpoint "${checkpoint}" "${checkpoint_label}"
  checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  evaluate_boundary "${checkpoint}" "${boundary}" "${checkpoint_label}" "${training_log}"

  evaluation_dir="${qualification_root}/convergence_eval_${boundary}"
  current_rate="$(jq -r '.success_rate' "${evaluation_dir}/aggregate.json")"
  previous_rate=0
  previous_boundary=$((boundary - segment_updates))
  if [[ -s "${qualification_root}/convergence_eval_${previous_boundary}/aggregate.json" ]]; then
    previous_rate="$(jq -r '.success_rate' "${qualification_root}/convergence_eval_${previous_boundary}/aggregate.json")"
  fi
  plateau="$(jq -r '.plateau_at_one_percentage_point' "${evaluation_dir}/curve_growth.json")"
  plateau_streak="$(jq -r '.plateau_streak // 0' "${state_file}" 2>/dev/null || echo 0)"
  high_eval_streak="$(jq -r '.high_eval_streak // 0' "${state_file}" 2>/dev/null || echo 0)"
  if [[ "${plateau}" == "true" ]]; then plateau_streak=$((plateau_streak + 1)); else plateau_streak=0; fi
  if "${python_bin}" -c "raise SystemExit(0 if float('${current_rate}') >= 0.90 else 1)"; then
    high_eval_streak=$((high_eval_streak + 1))
  else
    high_eval_streak=0
  fi
  converged="$("${python_bin}" -c "print(str(float('${current_rate}') >= .90 and float('${previous_rate}') >= .90 and abs(float('${current_rate}') - float('${previous_rate}')) <= .01 and int('${plateau_streak}') >= 2).lower())")"
  jq -n \
    --arg status "$([[ "${converged}" == "true" ]] && echo converged || echo training)" \
    --arg checkpoint "${checkpoint}" \
    --arg checkpoint_sha256 "${checkpoint_sha256}" \
    --argjson absolute_iteration "${boundary}" \
    --argjson checkpoint_iteration_field "${checkpoint_label}" \
    --argjson success_rate "${current_rate}" \
    --argjson previous_success_rate "${previous_rate}" \
    --argjson plateau_streak "${plateau_streak}" \
    --argjson high_eval_streak "${high_eval_streak}" \
    --argjson converged "${converged}" \
    --slurpfile evaluation "${evaluation_dir}/aggregate.json" \
    --slurpfile curve "${evaluation_dir}/curve_growth.json" \
    '{schema_version:1,status:$status,hard_iteration_cap:null,segment_updates:400,
      absolute_iteration:$absolute_iteration,checkpoint_iteration_field:$checkpoint_iteration_field,
      checkpoint:$checkpoint,checkpoint_sha256:$checkpoint_sha256,success_rate:$success_rate,
      previous_success_rate:$previous_success_rate,plateau_streak:$plateau_streak,
      high_eval_streak:$high_eval_streak,converged:$converged,
      latest_evaluation:$evaluation[0],latest_curve:$curve[0]}' >"${state_file}.tmp"
  mv "${state_file}.tmp" "${state_file}"
  jq -c . "${state_file}" >>"${history_file}"
  if [[ "${converged}" == "true" ]]; then
    echo "[$(date -Is)] Physics-v2 convergence confirmed at ${boundary}"
    exit 0
  fi

  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( available_kb < 8 * 1024 * 1024 )); then
    echo "available memory below 8 GiB; refusing next segment"
    exit 1
  fi
  target=$((boundary + segment_updates))
  run_training_segment "${boundary}" "${target}" "${source_run}" "${checkpoint_label}"
  boundary="${target}"
  checkpoint_label=$((checkpoint_label + segment_updates - 1))
  source_run_glob="*_tomato_pick_physics_v2_from$((boundary - segment_updates))_to${boundary}_8192"
  training_log="${qualification_root}/train_$((boundary - segment_updates))_to_${boundary}.log"
done
