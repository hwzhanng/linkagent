"""Audit terminal failure modes of a deterministic tomato-picking policy.

This tool is intentionally read-only with respect to the task.  It instruments
terminal tensors at runtime, records episode-conditioned physics/randomization
data, and optionally applies an external action EMA for counterfactual tests.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config


ISAAC_ROOT = Path("/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1")
sys.path.insert(0, str(ISAAC_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402


ACTION_NAMES = (
    "approach",
    "lateral",
    "pull",
    "roll",
    "pitch",
    "twist",
    "global_flexion",
    "curl_shape",
    "radial_ulnar",
    "spread",
    "thumb_opposition",
)
OUTCOME_NAMES = ("success", "drop", "excess_force", "out_of_bounds", "timeout", "unknown")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default="LinkAgent-TomatoPick-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--external_callback", default="simulation.tomato_pick.isaac.register_task")
parser.add_argument("--action_ema_alpha", type=float, default=1.0)
parser.add_argument(
    "--action_ema_scope", choices=("none", "hand", "all"), default="none"
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

remaining_args_env_registration = None
if args_cli.external_callback:
    callback = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = callback()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args
installed_version = metadata.version("rsl-rl-lib")
torch = importlib.import_module("torch")


def _install_terminal_instrumentation() -> None:
    """Expose cloned pre-reset terminal tensors through ``extras``."""

    from simulation.tomato_pick.isaac.env import TomatoPickEnv

    if getattr(TomatoPickEnv, "_failure_audit_instrumented", False):
        return
    original = TomatoPickEnv._get_dones

    def instrumented(self):
        terminated, truncated = original(self)
        # Only clone tensors already computed by the environment.  Reading
        # additional lazy PhysX/Fabric views here can perturb synchronization
        # between done computation, reward computation, and automatic reset.
        self.extras["failure_audit"] = {
            "success": self._success.detach().clone(),
            "drop": self._dropped.detach().clone(),
            "excess_force": self._excess_force.detach().clone(),
            "out_of_bounds": self._out_of_bounds.detach().clone(),
            "detached": self._stem_detached.detach().clone(),
            "retained": self._retained.detach().clone(),
            "hold_streak": self._hold_streak.detach().clone(),
            "finger_force_n": self._last_physical_force_n.detach().clone(),
        }
        return terminated, truncated

    TomatoPickEnv._get_dones = instrumented
    TomatoPickEnv._failure_audit_instrumented = True


def _describe(values: torch.Tensor) -> dict[str, float | int]:
    if values.numel() == 0:
        return {"count": 0}
    values = values.float()
    quantiles = torch.quantile(
        values, torch.tensor((0.1, 0.5, 0.9), device=values.device)
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _binned_outcomes(
    values: torch.Tensor, outcome: torch.Tensor, bins: int = 5
) -> list[dict[str, float | int]]:
    if values.numel() == 0:
        return []
    values = values.float()
    edges = torch.quantile(values, torch.linspace(0.0, 1.0, bins + 1))
    result = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (values >= lower) & (
            values <= upper if index == bins - 1 else values < upper
        )
        count = int(mask.sum())
        if count == 0:
            continue
        item: dict[str, float | int] = {
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
        }
        for code, name in enumerate(OUTCOME_NAMES):
            item[f"{name}_rate"] = float((outcome[mask] == code).float().mean())
        result.append(item)
    return result


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if not 0.0 < args_cli.action_ema_alpha <= 1.0:
        raise ValueError("--action_ema_alpha must lie in (0, 1]")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    output = args_cli.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.steps <= 0 or args_cli.num_envs <= 0:
        raise ValueError("steps and num_envs must be positive")

    with launch_simulation(env_cfg, args_cli):
        # Import the task implementation only after SimulationApp exists.
        # Importing it before this point eagerly loads pxr and can crash Kit.
        _install_terminal_instrumentation()
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)

        configure_seed(args_cli.seed, True)
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = gym_env.unwrapped
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(retrieve_file_path(str(checkpoint))))
        print("[failure-audit] checkpoint loaded", flush=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        print("[failure-audit] rollout initialized", flush=True)

        num_envs = env.num_envs
        device = env.device
        ema_action = torch.zeros((num_envs, len(ACTION_NAMES)), device=device)
        ema_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_return = torch.zeros(num_envs, device=device)
        episode_discounted_return = torch.zeros(num_envs, device=device)
        episode_discount = torch.ones(num_envs, device=device)
        detach_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        max_hold_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        peak_force = torch.zeros(num_envs, device=device)
        peak_finger_force = torch.zeros((num_envs, 5), device=device)
        raw_abs_sum = torch.zeros((num_envs, len(ACTION_NAMES)), device=device)
        executed_abs_sum = torch.zeros_like(raw_abs_sum)
        raw_sum = torch.zeros_like(raw_abs_sum)
        executed_sum = torch.zeros_like(raw_abs_sum)
        raw_saturation_count = torch.zeros_like(raw_abs_sum)

        record_batches: dict[str, list[torch.Tensor]] = {
            "environment_id": [],
            "terminal_step": [],
            "outcome": [],
            "episode_length": [],
            "episode_return": [],
            "episode_discounted_return": [],
            "detach_step": [],
            "max_hold_streak": [],
            "peak_force_n": [],
            "peak_finger_force_n": [],
            "terminal_finger_force_n": [],
            "fruit_mass_kg": [],
            "fruit_axes_m": [],
            "break_force_n": [],
            "action_delay": [],
            "tactile_gain_mean": [],
            "tactile_bias_mean_n": [],
            "target_error_m": [],
            "axis_measurement_error": [],
            "raw_mean_abs_action": [],
            "executed_mean_abs_action": [],
            "raw_mean_action": [],
            "executed_mean_action": [],
            "terminal_executed_action": [],
            "raw_action_saturation_rate": [],
        }
        started = time.perf_counter()
        native_event_counts = {
            "success": 0,
            "drop": 0,
            "excess_force": 0,
            "out_of_bounds": 0,
            "timeout": 0,
        }
        logged_reset_count = 0.0
        logged_success_count = 0.0
        logged_drop_count = 0.0

        for step_index in range(args_cli.steps):
            with torch.inference_mode():
                raw_action = policy(obs)
                action = raw_action
                if args_cli.action_ema_scope != "none":
                    alpha = args_cli.action_ema_alpha
                    ema_action[~ema_initialized] = raw_action[~ema_initialized]
                    ema_initialized[~ema_initialized] = True
                    ema_action.mul_(1.0 - alpha).add_(raw_action, alpha=alpha)
                    action = raw_action.clone()
                    if args_cli.action_ema_scope == "all":
                        action.copy_(ema_action)
                    else:
                        action[:, 6:11] = ema_action[:, 6:11]

                executed = action.clamp(-1.0, 1.0)
                episode_length += 1
                raw_abs_sum += raw_action.abs()
                executed_abs_sum += executed.abs()
                raw_sum += raw_action
                executed_sum += executed
                raw_saturation_count += (raw_action.abs() >= 0.99).float()

                params_before = {
                    "fruit_mass_kg": base_env._fruit_mass_kg.detach().clone(),
                    "fruit_axes_m": base_env._fruit_axes_m.detach().clone(),
                    "break_force_n": base_env._break_force_n.detach().clone(),
                    "action_delay": base_env._action_delay.detach().clone(),
                    "tactile_gain_mean": base_env._tactile_gain.mean(dim=1).detach().clone(),
                    "tactile_bias_mean_n": base_env._tactile_bias_n.mean(dim=1).detach().clone(),
                    "target_error_m": torch.linalg.vector_norm(
                        base_env._perceived_fruit_position_error_w, dim=1
                    ).detach().clone(),
                    "axis_measurement_error": torch.linalg.vector_norm(
                        base_env._measured_fruit_axes_m - base_env._fruit_axes_m, dim=1
                    ).detach().clone(),
                }
                obs, rewards, dones, extras = env.step(action)
                policy.reset(dones)

            native_step = extras.get("step", {})
            native_event_counts["success"] += int(native_step.get("success_count", 0))
            native_event_counts["drop"] += int(native_step.get("drop_count", 0))
            native_event_counts["excess_force"] += int(
                native_step.get("excess_force_count", 0)
            )
            native_event_counts["out_of_bounds"] += int(
                native_step.get("out_of_bounds_count", 0)
            )
            timeouts = extras.get("time_outs")
            if timeouts is not None:
                native_event_counts["timeout"] += int(timeouts.sum().item())

            episode_return += rewards
            episode_discounted_return += episode_discount * rewards
            episode_discount *= 0.99

            if step_index == 0:
                print("[failure-audit] first environment step complete", flush=True)

            audit = extras["failure_audit"]
            forces = audit["finger_force_n"]
            peak_finger_force = torch.maximum(peak_finger_force, forces)
            peak_force = torch.maximum(peak_force, forces.amax(dim=1))
            max_hold_streak = torch.maximum(max_hold_streak, audit["hold_streak"])
            first_detach = (detach_step < 0) & audit["detached"]
            detach_step[first_detach] = episode_length[first_detach]

            done = dones.bool()
            done_ids = done.nonzero(as_tuple=False).squeeze(1)
            if done_ids.numel():
                timeout_mask = (
                    timeouts.bool() if timeouts is not None else torch.zeros_like(done)
                )
                log = extras.get("log", {})
                reset_count = float(log.get("Metrics/reset_count", done_ids.numel()))
                logged_reset_count += reset_count
                logged_success_count += (
                    float(log.get("Metrics/success_rate", 0.0)) * reset_count
                )
                logged_drop_count += (
                    float(log.get("Metrics/drop_rate", 0.0)) * reset_count
                )
                outcome = torch.full((num_envs,), 5, dtype=torch.long, device=device)
                outcome[audit["success"]] = 0
                outcome[audit["drop"]] = 1
                outcome[audit["excess_force"]] = 2
                outcome[audit["out_of_bounds"]] = 3
                outcome[timeout_mask] = 4

                length = episode_length[done_ids].clamp_min(1).float().unsqueeze(1)
                record_batches["environment_id"].append(done_ids.cpu())
                record_batches["terminal_step"].append(
                    torch.full_like(done_ids, step_index).cpu()
                )
                record_batches["outcome"].append(outcome[done_ids].cpu())
                record_batches["episode_length"].append(episode_length[done_ids].cpu())
                record_batches["episode_return"].append(episode_return[done_ids].cpu())
                record_batches["episode_discounted_return"].append(
                    episode_discounted_return[done_ids].cpu()
                )
                record_batches["detach_step"].append(detach_step[done_ids].cpu())
                record_batches["max_hold_streak"].append(max_hold_streak[done_ids].cpu())
                record_batches["peak_force_n"].append(peak_force[done_ids].cpu())
                record_batches["peak_finger_force_n"].append(
                    peak_finger_force[done_ids].cpu()
                )
                record_batches["terminal_finger_force_n"].append(forces[done_ids].cpu())
                for name, values in params_before.items():
                    record_batches[name].append(values[done_ids].cpu())
                record_batches["raw_mean_abs_action"].append(
                    (raw_abs_sum[done_ids] / length).cpu()
                )
                record_batches["executed_mean_abs_action"].append(
                    (executed_abs_sum[done_ids] / length).cpu()
                )
                record_batches["raw_mean_action"].append(
                    (raw_sum[done_ids] / length).cpu()
                )
                record_batches["executed_mean_action"].append(
                    (executed_sum[done_ids] / length).cpu()
                )
                record_batches["terminal_executed_action"].append(
                    executed[done_ids].cpu()
                )
                record_batches["raw_action_saturation_rate"].append(
                    (raw_saturation_count[done_ids] / length).cpu()
                )

                episode_length[done_ids] = 0
                episode_return[done_ids] = 0.0
                episode_discounted_return[done_ids] = 0.0
                episode_discount[done_ids] = 1.0
                detach_step[done_ids] = -1
                max_hold_streak[done_ids] = 0
                peak_force[done_ids] = 0.0
                peak_finger_force[done_ids] = 0.0
                raw_abs_sum[done_ids] = 0.0
                executed_abs_sum[done_ids] = 0.0
                raw_sum[done_ids] = 0.0
                executed_sum[done_ids] = 0.0
                raw_saturation_count[done_ids] = 0.0
                ema_action[done_ids] = 0.0
                ema_initialized[done_ids] = False

        elapsed = time.perf_counter() - started
        env.close()

        # SimulationApp terminates the process when its context exits, so all
        # CPU aggregation and file output must finish before leaving it.
        records = {
            name: torch.cat(parts, dim=0)
            for name, parts in record_batches.items()
            if parts
        }
        if not records or records["outcome"].numel() == 0:
            raise RuntimeError("failure audit completed no episodes")
        outcome = records["outcome"]
        recorded_counts = {
            name: int((outcome == code).sum().item())
            for code, name in enumerate(OUTCOME_NAMES[:5])
        }
        if recorded_counts != native_event_counts:
            raise RuntimeError(
                "terminal accounting mismatch: "
                f"recorded={recorded_counts}, native={native_event_counts}"
            )
        if abs(logged_reset_count - outcome.numel()) > 0.5:
            raise RuntimeError(
                "reset accounting mismatch: "
                f"log={logged_reset_count}, recorded={outcome.numel()}"
            )
        if abs(logged_success_count - recorded_counts["success"]) > 0.5:
            raise RuntimeError(
                "success accounting mismatch: "
                f"log={logged_success_count}, recorded={recorded_counts['success']}"
            )
        if abs(logged_drop_count - recorded_counts["drop"]) > 0.5:
            raise RuntimeError(
                "drop accounting mismatch: "
                f"log={logged_drop_count}, recorded={recorded_counts['drop']}"
            )
        summary: dict[str, object] = {
            "schema_version": 1,
            "checkpoint": str(checkpoint),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "completed_episodes": int(outcome.numel()),
            "native_event_counts": native_event_counts,
            "elapsed_seconds": elapsed,
            "action_ema_alpha": args_cli.action_ema_alpha,
            "action_ema_scope": args_cli.action_ema_scope,
            "outcome_rates": {
                name: float((outcome == code).float().mean())
                for code, name in enumerate(OUTCOME_NAMES)
            },
            "action_names": ACTION_NAMES,
            "global_raw_action_saturation_rate": records[
                "raw_action_saturation_rate"
            ].mean(dim=0).tolist(),
            "global_raw_mean_abs_action": records["raw_mean_abs_action"]
            .mean(dim=0)
            .tolist(),
            "global_executed_mean_abs_action": records[
                "executed_mean_abs_action"
            ].mean(dim=0).tolist(),
            "global_raw_mean_action": records["raw_mean_action"].mean(dim=0).tolist(),
            "global_executed_mean_action": records["executed_mean_action"]
            .mean(dim=0)
            .tolist(),
            "outcomes": {},
            "parameter_bins": {},
        }
        scalar_fields = (
            "episode_length",
            "episode_return",
            "episode_discounted_return",
            "detach_step",
            "max_hold_streak",
            "peak_force_n",
            "fruit_mass_kg",
            "break_force_n",
            "action_delay",
            "tactile_gain_mean",
            "tactile_bias_mean_n",
            "target_error_m",
            "axis_measurement_error",
        )
        outcomes = summary["outcomes"]
        assert isinstance(outcomes, dict)
        for code, name in enumerate(OUTCOME_NAMES):
            mask = outcome == code
            item: dict[str, object] = {
                field: _describe(records[field][mask]) for field in scalar_fields
            }
            item["peak_finger_force_n_mean"] = (
                records["peak_finger_force_n"][mask].float().mean(dim=0).tolist()
                if bool(mask.any())
                else []
            )
            item["terminal_finger_force_n_mean"] = (
                records["terminal_finger_force_n"][mask]
                .float()
                .mean(dim=0)
                .tolist()
                if bool(mask.any())
                else []
            )
            item["raw_mean_abs_action"] = (
                records["raw_mean_abs_action"][mask].float().mean(dim=0).tolist()
                if bool(mask.any())
                else []
            )
            item["raw_action_saturation_rate"] = (
                records["raw_action_saturation_rate"][mask]
                .float()
                .mean(dim=0)
                .tolist()
                if bool(mask.any())
                else []
            )
            item["raw_mean_action"] = (
                records["raw_mean_action"][mask].float().mean(dim=0).tolist()
                if bool(mask.any())
                else []
            )
            item["executed_mean_action"] = (
                records["executed_mean_action"][mask].float().mean(dim=0).tolist()
                if bool(mask.any())
                else []
            )
            item["terminal_executed_action"] = (
                records["terminal_executed_action"][mask]
                .float()
                .mean(dim=0)
                .tolist()
                if bool(mask.any())
                else []
            )
            outcomes[name] = item

        bin_fields = {
            "fruit_mass_kg": records["fruit_mass_kg"],
            "fruit_max_axis_m": records["fruit_axes_m"].amax(dim=1),
            "break_force_n": records["break_force_n"],
            "target_error_m": records["target_error_m"],
            "axis_measurement_error": records["axis_measurement_error"],
            "tactile_gain_mean": records["tactile_gain_mean"],
            "tactile_bias_mean_n": records["tactile_bias_mean_n"],
        }
        parameter_bins = summary["parameter_bins"]
        assert isinstance(parameter_bins, dict)
        for name, values in bin_fields.items():
            parameter_bins[name] = _binned_outcomes(values, outcome)

        output.parent.mkdir(parents=True, exist_ok=True)
        records_path = output.with_suffix(".records.pt")
        torch.save(records, records_path)
        summary["records_path"] = str(records_path)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("TOMATO_FAILURE_AUDIT=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
