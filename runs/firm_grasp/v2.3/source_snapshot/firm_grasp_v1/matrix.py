"""F4 固定 20 case / 4 env 批量评估入口。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from .disturbance import pulse_wrench


NUM_ENVS = 4
CASES_PER_ENV = 5
CASE_ORDER_ID = "continuation_interleaved_v1"
# v2.3 预登记新清单：8 quiet + 6 force(每轴±) + 6 torque(每轴±)，
# 交错排列，避免所有 quiet 先于扰动；case_id 与旧最终表不可混用。
CASE_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "force", "axis": 0, "sign": 1.0},
    {"mode": "torque", "axis": 0, "sign": 1.0},
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "force", "axis": 1, "sign": -1.0},
    {"mode": "torque", "axis": 1, "sign": -1.0},
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "force", "axis": 2, "sign": 1.0},
    {"mode": "torque", "axis": 2, "sign": 1.0},
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "force", "axis": 0, "sign": -1.0},
    {"mode": "torque", "axis": 0, "sign": -1.0},
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "force", "axis": 1, "sign": 1.0},
    {"mode": "torque", "axis": 1, "sign": 1.0},
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "force", "axis": 2, "sign": -1.0},
    {"mode": "torque", "axis": 2, "sign": -1.0},
    {"mode": "quiet", "axis": None, "sign": None},
    {"mode": "quiet", "axis": None, "sign": None},
)
CASE_COUNT = len(CASE_SCENARIOS)
RECOVERY_START_SUBSTEP = 252
RECOVERY_MAX_SPEED_M_S = 0.02
REQUIRED_HOLD_SUBSTEPS = 360


def _manifest(force_ratio: float, torque_ratio: float) -> list[dict[str, Any]]:
    rows = []
    for case_id, scenario in enumerate(CASE_SCENARIOS):
        rows.append(
            {
                "case_id": case_id,
                "scenario": dict(scenario),
                "force_ratio": force_ratio if scenario["mode"] == "force" else None,
                "torque_ratio": torque_ratio if scenario["mode"] == "torque" else None,
            }
        )
    return rows


def _source_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_metadata(case_indices: torch.Tensor) -> tuple[list[int | None], list[dict[str, Any] | None]]:
    indices = case_indices.detach().cpu().tolist()
    case_ids: list[int | None] = []
    scenarios: list[dict[str, Any] | None] = []
    for raw_index in indices:
        index = int(raw_index)
        if 0 <= index < CASE_COUNT:
            case_ids.append(index)
            scenarios.append(dict(CASE_SCENARIOS[index]))
        else:
            case_ids.append(None)
            scenarios.append(None)
    return case_ids, scenarios


def _matrix_env(base: type) -> type:
    """动态包装当前 env.py 类，避免导入或改写受保护环境源码。"""

    class MatrixEnv(base):
        def __init__(self, cfg: Any, **kwargs: Any) -> None:
            self._matrix_ready = False
            super().__init__(cfg, **kwargs)
            if int(self.num_envs) != NUM_ENVS:
                raise ValueError(f"F4 固定矩阵需要恰好 {NUM_ENVS} 个环境")
            protocol = self._f3_protocol
            if (
                protocol.get("goal") != "g2"
                or float(protocol.get("hold_seconds", 0.0)) != 3.0
                or protocol.get("disturbance_mode") != "quiet"
            ):
                raise ValueError("F4 需要 g2/3 秒，父级协议统一配置为 quiet")

            self._matrix_case_index = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self._matrix_assigned_per_env = [0] * self.num_envs
            self._matrix_retired = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._matrix_is_disturbed = torch.zeros_like(self._matrix_retired)
            self._matrix_active_cases: set[int] = set()
            self._matrix_force_ratio = float(protocol["force_ratio"])
            self._matrix_torque_ratio = float(protocol["torque_ratio"])
            self._matrix_ready = True
            self._identity = self._build_identity()

        def _build_identity(self) -> dict[str, Any]:
            identity = super()._build_identity()
            if not getattr(self, "_matrix_ready", False):
                return identity
            identity["f4_matrix"] = {
                "schema_version": "firm_grasp_f4_matrix.v1",
                "adapter": "research.firm_grasp_v1.matrix.MatrixEnv",
                "adapter_source_sha256": _source_sha256(),
                "num_envs": NUM_ENVS,
                "case_count": CASE_COUNT,
                "cases_per_env": CASES_PER_ENV,
                "assignment": "case_index = env_id + num_envs * per_env_ordinal",
                "retirement": "after_five_assigned_cases; retired env dones are masked",
                "parent_protocol": "g2_3s_strict_early_start_disturbed_false",
                "physics_hz": 120,
                "required_hold_substeps": REQUIRED_HOLD_SUBSTEPS,
                "strict_early_start_max_substep": 60,
                "pulse_substeps_inclusive": [180, 191],
                "required_pulse_substeps": 12,
                "recovery_start_substep": RECOVERY_START_SUBSTEP,
                "recovery_max_speed_m_s": RECOVERY_MAX_SPEED_M_S,
                "case_order_id": CASE_ORDER_ID,
                "cases": _manifest(
                    self._matrix_force_ratio, self._matrix_torque_ratio
                ),
                "cases_sha256": hashlib.sha256(
                    json.dumps(
                        _manifest(self._matrix_force_ratio, self._matrix_torque_ratio),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            return identity

        def _reset_idx(self, env_ids: Any) -> None:
            # Parent finalizes the preceding episode before this env receives its next case.
            super()._reset_idx(env_ids)
            if not getattr(self, "_matrix_ready", False):
                return
            if env_ids is None:
                ids = list(range(self.num_envs))
            elif isinstance(env_ids, torch.Tensor):
                ids = [int(value) for value in env_ids.detach().cpu().tolist()]
            else:
                ids = [int(value) for value in env_ids]
            for env_id in ids:
                previous_case = int(self._matrix_case_index[env_id].item())
                if previous_case >= 0:
                    self._matrix_active_cases.discard(previous_case)
                ordinal = self._matrix_assigned_per_env[env_id]
                if ordinal < CASES_PER_ENV:
                    case_index = env_id + self.num_envs * ordinal
                    if case_index >= CASE_COUNT:
                        raise RuntimeError("F4 per-env 队列越过固定 manifest")
                    self._matrix_case_index[env_id] = case_index
                    self._matrix_assigned_per_env[env_id] = ordinal + 1
                    self._matrix_retired[env_id] = False
                    self._matrix_is_disturbed[env_id] = (
                        CASE_SCENARIOS[case_index]["mode"] != "quiet"
                    )
                    self._matrix_active_cases.add(case_index)
                else:
                    self._matrix_case_index[env_id] = -1
                    self._matrix_is_disturbed[env_id] = False
                    self._matrix_retired[env_id] = True

        def _contact_geometry(self) -> dict[str, torch.Tensor]:
            geometry = super()._contact_geometry()
            if not getattr(self, "_matrix_ready", False):
                return geometry
            recovery_window = self._matrix_is_disturbed & (
                self._post_release_substep >= RECOVERY_START_SUBSTEP
            )
            recovered = (~recovery_window) | (
                self._firm_last_state.relative_speed_m_s <= RECOVERY_MAX_SPEED_M_S
            )
            geometry["qualified"] = geometry["qualified"] & recovered[:, None]
            return geometry

        @torch.inference_mode()
        def _apply_attachment_wrench(self) -> None:
            if not getattr(self, "_matrix_ready", False):
                return super()._apply_attachment_wrench()

            force, torque = self._unscaled_attachment_wrench()
            scale = self._support_release_scale() * (~self._stem_detached).float()
            force *= scale[:, None]
            torque *= scale[:, None]
            self._last_stem_force_w.copy_(force)
            self._last_stem_torque_w.copy_(torque)

            fresh = self._stem_detached & (self._release_start_counter < 0)
            self._release_start_counter[fresh] = self._sim_step_counter
            self._post_release_substep = torch.where(
                self._stem_detached,
                self._sim_step_counter - self._release_start_counter,
                -1,
            )

            self._test_force.zero_()
            self._test_torque.zero_()
            hand_quat = self._robot.data.body_pose_w.torch[:, self._ee_body_id, 3:7]
            for case_id in sorted(self._matrix_active_cases):
                scenario = CASE_SCENARIOS[case_id]
                mode = scenario["mode"]
                if mode == "quiet":
                    continue
                pulse_force, pulse_torque = pulse_wrench(
                    self._post_release_substep,
                    hand_quat,
                    self._fruit_mass_kg,
                    self._fruit_axes_m[:, 0],
                    mode=mode,
                    axis=int(scenario["axis"]),
                    sign=float(scenario["sign"]),
                    force_ratio=self._matrix_force_ratio,
                    torque_ratio=self._matrix_torque_ratio,
                )
                mask = (self._matrix_case_index == case_id) & self._stem_detached
                self._test_force.add_(pulse_force * mask[:, None])
                self._test_torque.add_(pulse_torque * mask[:, None])

            applied_pulse = (self._test_force != 0).any(1) | (self._test_torque != 0).any(1)
            self._pulse_counts += applied_pulse.long()
            # Keep the support wrench and per-env test pulse in the one actual PhysX setter.
            self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(force + self._test_force).unsqueeze(1),
                torques=(torque + self._test_torque).unsqueeze(1),
                is_global=True,
            )

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            terminated, truncated = super()._get_dones()
            if not getattr(self, "_matrix_ready", False):
                return terminated, truncated

            disturbed_success = self._terminal_success & self._matrix_is_disturbed
            missing_pulse = disturbed_success & (self._pulse_counts != 12)
            if bool(missing_pulse.any()):
                self._terminal_success[missing_pulse] = False
                self._terminal_reason[missing_pulse] = 0
                terminated[missing_pulse] = False
                at_timeout = self.episode_length_buf >= self.max_episode_length - 1
                truncated[missing_pulse & at_timeout] = True
                self._terminal_reason[missing_pulse & at_timeout] = 2

            inactive = self._matrix_retired | (self._matrix_case_index < 0)
            terminated[inactive] = False
            truncated[inactive] = False
            self._terminal_success[inactive] = False
            self._terminal_reason[inactive] = 0
            return terminated, truncated

        def _finalize_episodes(self, env_ids: Any) -> None:
            cursor = len(getattr(self, "completed_episodes", []))
            super()._finalize_episodes(env_ids)
            if not getattr(self, "_matrix_ready", False):
                return
            for row in self.completed_episodes[cursor:]:
                env_id = int(row["env_id"])
                case_id = int(self._matrix_case_index[env_id].item())
                if not 0 <= case_id < CASE_COUNT:
                    continue
                scenario = dict(CASE_SCENARIOS[case_id])
                row["case_id"] = case_id
                row["scenario"] = scenario
                row["matrix_case_id"] = case_id
                row["matrix_scenario"] = scenario
                protocol = dict(row.get("f3_protocol", {}))
                protocol.update(
                    disturbance_mode=scenario["mode"],
                    disturbance_axis=scenario["axis"],
                    disturbance_sign=scenario["sign"],
                    force_ratio=self._matrix_force_ratio,
                    torque_ratio=self._matrix_torque_ratio,
                )
                row["f3_protocol"] = protocol

        def _observe_physical_hold_substep(self, scene_dt_s: float) -> bool:
            if not getattr(self, "_matrix_ready", False):
                return super()._observe_physical_hold_substep(scene_dt_s)
            trace = getattr(self, "_pad_trace", None)
            cursor = len(trace) if isinstance(trace, list) else 0
            consumed = super()._observe_physical_hold_substep(scene_dt_s)
            trace = getattr(self, "_pad_trace", None)
            if isinstance(trace, list) and len(trace) > cursor:
                case_ids, scenarios = _case_metadata(self._matrix_case_index)
                for row in trace[cursor:]:
                    row["matrix_case_id"] = list(case_ids)
                    row["matrix_scenario"] = [
                        None if value is None else dict(value) for value in scenarios
                    ]
            return consumed

    MatrixEnv.__name__ = "MatrixEnv"
    MatrixEnv.__qualname__ = "MatrixEnv"
    return MatrixEnv


def _has_actor_seed(argv: list[str]) -> bool:
    return any(
        token == "--actor-seed-checkpoint"
        or token.startswith("--actor-seed-checkpoint=")
        for token in argv
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="以 4 个固定环境队列运行 F4 20 case G2 批量评估。",
        epilog="run.py 的权重、输出、设备和 reset 参数会原样透传。",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        required=True,
        help="明确选择固定 F4 20 case manifest",
    )
    own_args, run_args = parser.parse_known_args(argv)
    del own_args
    if not _has_actor_seed(run_args):
        parser.error("矩阵评估必须显式提供 --actor-seed-checkpoint frozen actor")

    run = importlib.import_module("research.firm_grasp_v1.run")
    original_run = run._run

    def run_with_matrix(args: Any, output: Path, stop: Any, phase: Any) -> int:
        # run.main has already created AppLauncher before dispatching to _run.
        env_module = importlib.import_module("research.firm_grasp_v1.env")
        original_env = env_module.FirmGraspEnv
        env_module.FirmGraspEnv = _matrix_env(original_env)
        try:
            return original_run(args, output, stop, phase)
        finally:
            env_module.FirmGraspEnv = original_env

    run._run = run_with_matrix
    forced_args = [
        "--mode", "evaluate",
        "--policy", "fresh",
        "--episodes", str(CASE_COUNT),
        "--num-envs", str(NUM_ENVS),
        "--goal", "g2",
        "--hold-seconds", "3",
        "--disturbance-mode", "quiet",
    ]
    try:
        return int(run.main(list(run_args) + forced_args) or 0)
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        run._run = original_run


if __name__ == "__main__":
    raise SystemExit(main())
