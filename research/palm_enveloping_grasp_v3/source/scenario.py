"""Pure-Python validation and hashing for audited v3 scenarios."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCENARIO_SCHEMA_VERSION = "palm_enveloping_scenario_v3.0"


class ScenarioError(ValueError):
    """Raised when a scenario is incomplete, unsafe, or ambiguous."""


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    """Return stable UTF-8 JSON bytes used as the scenario identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(mapping: Mapping[str, Any], path: str, expected_type: type | tuple[type, ...]) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ScenarioError(f"missing required scenario field: {path}")
        current = current[part]
    if not isinstance(current, expected_type):
        raise ScenarioError(f"scenario field {path} has invalid type {type(current).__name__}")
    return current


def validate_scenario(value: Mapping[str, Any]) -> dict[str, Any]:
    scenario = deepcopy(dict(value))
    if scenario.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ScenarioError(
            f"schema_version must be {SCENARIO_SCHEMA_VERSION!r}, got {scenario.get('schema_version')!r}"
        )
    if scenario.get("hardware_command_allowed") is not False:
        raise ScenarioError("v3 scenarios must set hardware_command_allowed=false")
    if _require(scenario, "task.stage", int) != 1:
        raise ScenarioError("v3_large90_nominal is a Stage-1 enclosure task")
    axes = _require(scenario, "object.semi_axes_m", list)
    if len(axes) != 3 or any(not isinstance(v, (int, float)) or v <= 0 for v in axes):
        raise ScenarioError("object.semi_axes_m must contain three positive values")
    if max(axes) - min(axes) > 1.0e-12:
        raise ScenarioError("the nominal v3 object must be isotropic")
    action = _require(scenario, "controller.target_action", list)
    if len(action) != 14 or any(not isinstance(v, (int, float)) or abs(v) > 1.0 for v in action):
        raise ScenarioError("controller.target_action must contain 14 values in [-1,1]")
    action_contract = _require(scenario, "controller.action_contract", str)
    if action_contract not in ("legacy_enveloping14", "direct_flexion14_v1"):
        raise ScenarioError("controller.action_contract is not supported")
    ceilings = _require(scenario, "safety.force_ceiling_n", list)
    if len(ceilings) != 5 or any(not isinstance(v, (int, float)) or v <= 0 for v in ceilings):
        raise ScenarioError("safety.force_ceiling_n must contain five positive values")
    coverage = _require(scenario, "strict.angular_coverage_min_deg", (int, float))
    if not 180.0 <= float(coverage) <= 360.0:
        raise ScenarioError("strict.angular_coverage_min_deg must be in [180,360]")
    if _require(scenario, "strict.require_current_physical_five_digit_contact", bool) is not True:
        raise ScenarioError("v3 strict success must require current physical five-digit contact")
    if "sequential_contact_allowed" in scenario["strict"]:
        if not isinstance(scenario["strict"]["sequential_contact_allowed"], bool):
            raise ScenarioError("strict.sequential_contact_allowed must be boolean")
    if _require(scenario, "strict.require_measured_distal_contact", bool) is not True:
        raise ScenarioError("v3 strict success must require measured distal contact")
    collision_source = _require(scenario, "contact_geometry.collision_source", str)
    if collision_source not in ("authored_proxy", "original_terminal_only", "full_imported"):
        raise ScenarioError("contact_geometry.collision_source is not supported")
    physical_tip_radius = _require(
        scenario, "contact_geometry.physical_tip_radius_m", (int, float)
    )
    if not 0.0 < float(physical_tip_radius) <= 0.012:
        raise ScenarioError("contact_geometry.physical_tip_radius_m must lie in (0, 0.012]")
    proxy_physical = _require(scenario, "contact_geometry.fingertip_proxy_physical", bool)
    proxy_only = _require(scenario, "contact_geometry.proxy_only_collision", bool)
    if collision_source == "authored_proxy" and not (proxy_physical and proxy_only):
        raise ScenarioError("authored_proxy requires a physical proxy-only collision scene")
    if collision_source != "authored_proxy" and (proxy_physical or proxy_only):
        raise ScenarioError("non-proxy collision sources must disable physical proxy-only collision")
    palm_share = _require(scenario, "strict.palm_load_share", list)
    if len(palm_share) != 2 or not 0.0 <= palm_share[0] <= palm_share[1] <= 1.0:
        raise ScenarioError("strict.palm_load_share must satisfy 0 <= min <= max <= 1")
    for key in ("diagnostic_hold_steps", "production_hold_steps"):
        if _require(scenario, f"strict.{key}", int) <= 0:
            raise ScenarioError(f"strict.{key} must be positive")
    if scenario["strict"]["diagnostic_hold_steps"] > scenario["strict"]["production_hold_steps"]:
        raise ScenarioError("diagnostic hold cannot exceed production hold")
    _require(scenario, "physics.urdf_terminal_mimic_enabled", bool)
    _require(scenario, "physics.direct_servo_zero_velocity", bool)
    _require(scenario, "physics.direct_position_servo_enabled", bool)
    _require(scenario, "physics.filter_fingertip_proxy_support_collision", bool)
    _require(scenario, "physics.disable_palm_cup_rim_collision", bool)
    drive_effort = _require(scenario, "physics.joint_drive_effort_limit_nm", (int, float))
    drive_kp = _require(scenario, "physics.joint_drive_stiffness_nm_rad", (int, float))
    drive_kd = _require(scenario, "physics.joint_drive_damping_nm_s_rad", (int, float))
    _require(scenario, "physics.hand_disable_gravity", bool)
    if _require(scenario, "physics.use_tomato_filtered_contact_force", bool) is not True:
        raise ScenarioError("v3 strict contact must use the tomato-filtered force matrix")
    history_substeps = _require(scenario, "physics.contact_history_substeps", int)
    if not 1 <= int(history_substeps) <= 12:
        raise ScenarioError("physics.contact_history_substeps must lie in [1,12]")
    _require(scenario, "physics.mimic_drive_effort_limit_nm", (int, float))
    _require(scenario, "physics.mimic_drive_stiffness_nm_rad", (int, float))
    _require(scenario, "physics.mimic_drive_damping_nm_s_rad", (int, float))
    mimic_armature = _require(scenario, "physics.mimic_drive_armature_kg_m2", (int, float))
    if not 1.0e-7 <= float(mimic_armature) <= 1.0e-2:
        raise ScenarioError("physics.mimic_drive_armature_kg_m2 must be within [1e-7, 1e-2]")
    if not 0.02 <= float(drive_effort) <= 1.0:
        raise ScenarioError("physics.joint_drive_effort_limit_nm must be within [0.02, 1.0]")
    if not 0.05 <= float(drive_kp) <= 8.0:
        raise ScenarioError("physics.joint_drive_stiffness_nm_rad must be within [0.05, 8.0]")
    if not 0.001 <= float(drive_kd) <= 3.0:
        raise ScenarioError("physics.joint_drive_damping_nm_s_rad must be within [0.001, 3.0]")
    mimic_rate = _require(scenario, "physics.urdf_mimic_servo_rate_rad_s", (int, float))
    if not 0.05 <= float(mimic_rate) <= 5.0:
        raise ScenarioError("physics.urdf_mimic_servo_rate_rad_s must be within [0.05, 5.0]")
    mimic = _require(scenario, "physics.urdf_mimic", dict)
    expected_mimic = {
        "straight_multiplier": 0.94,
        "straight_offset_rad": -0.04545,
        "thumb_multiplier": 1.11,
        "thumb_offset_rad": -0.04545,
    }
    for key, expected in expected_mimic.items():
        actual = float(_require(mimic, key, (int, float)))
        if abs(actual - expected) > 1.0e-9:
            raise ScenarioError(f"physics.urdf_mimic.{key} must retain the vendor URDF value {expected}")
    event = scenario["controller"].get("event_teacher")
    if event is not None:
        if not isinstance(event, dict):
            raise ScenarioError("controller.event_teacher must be a mapping")
        required = {
            "contact_threshold_n", "force_lower_n", "force_upper_n", "reacquire_step",
            "retreat_step", "trust_region", "thumb_reacquire_direction",
            "load_share_max", "search_period_steps", "fruit_speed_hold_m_s",
        }
        missing = sorted(required - event.keys())
        if missing:
            raise ScenarioError(f"controller.event_teacher is missing {missing}")
        for name in ("force_lower_n", "force_upper_n"):
            values = event[name]
            if not isinstance(values, list) or len(values) != 5 or any(float(v) <= 0 for v in values):
                raise ScenarioError(f"controller.event_teacher.{name} must contain five positive values")
        if any(float(hi) < float(lo) for lo, hi in zip(event["force_lower_n"], event["force_upper_n"])):
            raise ScenarioError("event teacher upper force must be >= lower force")
        if len(event["thumb_reacquire_direction"]) != 2:
            raise ScenarioError("event teacher thumb_reacquire_direction must contain two values")
        if "high_force_dwell_steps" in event and int(event["high_force_dwell_steps"]) <= 0:
            raise ScenarioError("event teacher high_force_dwell_steps must be positive")
        if "force_prediction_gain" in event and not 0.0 <= float(event["force_prediction_gain"]) <= 4.0:
            raise ScenarioError("event teacher force_prediction_gain must lie in [0,4]")
        if "balance_target_n" in event:
            values = event["balance_target_n"]
            if not isinstance(values, list) or len(values) != 5 or any(float(v) <= 0 for v in values):
                raise ScenarioError("event teacher balance_target_n must contain five positive values")
        if "digit_latch_start_step" in event and int(event["digit_latch_start_step"]) < 0:
            raise ScenarioError("event teacher digit_latch_start_step must be non-negative")
        if "palm_latch_threshold_n" in event and float(event["palm_latch_threshold_n"]) <= 0:
            raise ScenarioError("event teacher palm_latch_threshold_n must be positive")
    direct = _require(scenario, "controller.direct_teacher", dict)
    for key in ("coarse_steps", "distal_steps"):
        if not isinstance(direct.get(key), int) or int(direct[key]) <= 0:
            raise ScenarioError(f"controller.direct_teacher.{key} must be a positive integer")
    for key in ("force_lower_n", "force_upper_n"):
        values = direct.get(key)
        if not isinstance(values, list) or len(values) != 5 or any(float(v) <= 0 for v in values):
            raise ScenarioError(f"controller.direct_teacher.{key} must contain five positive values")
    if any(float(hi) < float(lo) for lo, hi in zip(direct["force_lower_n"], direct["force_upper_n"])):
        raise ScenarioError("direct teacher force upper values must be >= lower values")
    for key in ("precontact_fraction", "reacquire_action_step", "max_closure_extension"):
        if not isinstance(direct.get(key), (int, float)):
            raise ScenarioError(f"controller.direct_teacher.{key} must be numeric")
    if not 0.0 < float(direct["precontact_fraction"]) < 1.0:
        raise ScenarioError("direct teacher precontact_fraction must lie in (0,1)")
    if int(direct.get("contact_latch_steps", 0)) <= 0 or int(direct.get("lost_contact_grace_steps", -1)) < 0:
        raise ScenarioError("direct teacher latch timing is invalid")
    if int(direct.get("contact_latch_start_step", -1)) < 0:
        raise ScenarioError("direct teacher contact_latch_start_step must be non-negative")
    if int(direct.get("search_probe_steps", 0)) <= 0:
        raise ScenarioError("direct teacher search_probe_steps must be positive")
    if int(direct.get("high_force_dwell_steps", 0)) <= 0:
        raise ScenarioError("direct teacher high_force_dwell_steps must be positive")
    detect = direct.get("contact_detect_n")
    if not isinstance(detect, list) or len(detect) != 5 or any(float(value) <= 0.0 for value in detect):
        raise ScenarioError("direct teacher contact_detect_n must contain five positive values")
    return scenario


def apply_contact_collision_mode(cfg: Any, geometry: Mapping[str, Any]) -> None:
    """Apply one declared, mutually-exclusive fingertip collision source.

    ``original_terminal_only`` keeps only the five terminal collision bodies
    that existed when Isaac Lab initialized its contact views.  It is the
    audited comparison path for late-authored proxy reporting failures.
    """

    mode = str(geometry["collision_source"])
    attachment_mode = str(geometry.get("proxy_attachment_mode", "fixed_body"))
    if attachment_mode not in {"fixed_body", "attached_shape"}:
        raise ScenarioError("contact_geometry.proxy_attachment_mode is invalid")
    cfg.fingertip_collision_proxy_physical = bool(geometry["fingertip_proxy_physical"])
    cfg.proxy_only_collision = bool(geometry["proxy_only_collision"])
    cfg.debug_support_only_collision = mode == "original_terminal_only"
    cfg.debug_collision_digit = "all" if mode == "original_terminal_only" else ""
    if hasattr(cfg, "v3_proxy_attachment_mode"):
        cfg.v3_proxy_attachment_mode = attachment_mode
        event = cfg.events.get("v2_fingertip_collision_proxies")
        if event is not None:
            event.params["attachment_mode"] = attachment_mode
        terminal_names = ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")
        sensor_names = (
            "v3_thumb_pad_contact", "v3_index_pad_contact", "v3_middle_pad_contact",
            "v3_ring_pad_contact", "v3_little_pad_contact",
        )
        if attachment_mode == "attached_shape":
            for sensor_name, terminal_name in zip(sensor_names, terminal_names):
                getattr(cfg.scene, sensor_name).prim_path = f"{{ENV_REGEX_NS}}/Robot/{terminal_name}"


def load_scenario(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioError(f"cannot load scenario {source}: {exc}") from exc
    scenario = validate_scenario(raw)
    return scenario, canonical_hash(scenario)


def derive_run_scenario(
    scenario: Mapping[str, Any], *, seed: int, num_envs: int, steps: int, purpose: str
) -> tuple[dict[str, Any], str]:
    """Materialize runtime choices so every override changes the run hash."""

    if seed < 0 or num_envs <= 0 or steps <= 0:
        raise ScenarioError("seed must be non-negative and num_envs/steps must be positive")
    derived = deepcopy(dict(scenario))
    derived["runtime"] = {
        "seed": int(seed),
        "num_envs": int(num_envs),
        "steps": int(steps),
        "purpose": str(purpose),
    }
    return derived, canonical_hash(derived)


def merge_scenario_override(base: Mapping[str, Any], override: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Recursively apply a declared experiment fragment and revalidate it."""

    result = deepcopy(dict(base))

    def merge(target: dict[str, Any], patch: Mapping[str, Any], prefix: str) -> None:
        for key, value in patch.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in target:
                if path == "controller.event_teacher" and isinstance(value, Mapping):
                    target[key] = deepcopy(dict(value))
                    continue
                raise ScenarioError(f"override introduces unknown field: {path}")
            if isinstance(value, Mapping):
                if not isinstance(target[key], dict):
                    raise ScenarioError(f"override field {path} is not a mapping in the base scenario")
                merge(target[key], value, path)
            else:
                target[key] = deepcopy(value)

    merge(result, override, "")
    result = validate_scenario(result)
    return result, canonical_hash(result)
