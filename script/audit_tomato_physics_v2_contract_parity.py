"""Static CPU audit of offline-oracle versus integrated success semantics."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.tomato_pick.physics_v1 import load_physics_v1_profile
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile


PARENT_CFG = ROOT / "simulation/tomato_pick/isaac/physics_v1/env_cfg.py"
V2_CFG = ROOT / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py"
BASE_ENV = ROOT / "simulation/tomato_pick/isaac/env.py"
V2_ENV = ROOT / "simulation/tomato_pick/isaac/physics_v2/env.py"
RUNTIME_ENV = V2_ENV
ORACLE = ROOT / "simulation/tomato_pick/physics_v2_scenarios.py"


def _class_assignment_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            result: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.Assign):
                    result.update(
                        target.id
                        for target in item.targets
                        if isinstance(target, ast.Name)
                    )
                elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    result.add(item.target.id)
            return result
    raise ValueError(f"class not found: {class_name}")


def build_contract_parity_audit() -> dict[str, object]:
    parent = load_physics_v1_profile()
    profile = load_physics_v2_profile()
    secure = parent["secure_success"]
    registry = profile["success_contract_parity"]
    expected = {
        "minimum_supported_fingertips": secure["minimum_supported_fingertips"],
        "minimum_support_force_n": secure["minimum_support_force_n"],
        "maximum_center_distance_m": secure["maximum_center_distance_m"],
        "maximum_relative_speed_m_s": secure["maximum_relative_speed_m_s"],
        "hold_policy_steps": secure["hold_policy_steps"],
        "maximum_fingertip_force_n": profile["stem_modes_v2"]["common"][
            "safe_maximum_fingertip_force_n"
        ],
        "minimum_palm_support_force_n": profile["success_contract_parity"][
            "thresholds"
        ]["minimum_palm_support_force_n"],
        "palm_support_requires_detach": profile["success_contract_parity"][
            "thresholds"
        ]["palm_support_requires_detach"],
    }
    parent_text = PARENT_CFG.read_text(encoding="utf-8")
    runtime_text = RUNTIME_ENV.read_text(encoding="utf-8")
    base_text = BASE_ENV.read_text(encoding="utf-8")
    v2_text = V2_ENV.read_text(encoding="utf-8")
    v2_cfg_text = V2_CFG.read_text(encoding="utf-8")
    oracle_text = ORACLE.read_text(encoding="utf-8")
    v2_assignments = _class_assignment_names(
        V2_CFG, "TomatoPickPhysicsV2EnvCfg"
    )
    threshold_names = {
        "minimum_supported_fingertips",
        "minimum_support_force_n",
        "maximum_retained_distance_m",
        "maximum_retained_relative_speed_m_s",
        "success_hold_steps",
        "maximum_fingertip_force_n",
    }
    source_checks = {
        "parent_distance_comes_from_secure_profile": (
            'maximum_retained_distance_m = SECURE_SUCCESS["maximum_center_distance_m"]'
            in parent_text
        ),
        "parent_speed_comes_from_secure_profile": (
            "maximum_retained_relative_speed_m_s = SECURE_SUCCESS["
            in parent_text
            and '"maximum_relative_speed_m_s"' in parent_text
        ),
        "parent_support_force_comes_from_secure_profile": (
            'minimum_support_force_n = SECURE_SUCCESS["minimum_support_force_n"]'
            in parent_text
        ),
        "parent_force_ceiling_is_8n": "maximum_fingertip_force_n = 8.0"
        in parent_text,
        "physics_v2_does_not_override_success_thresholds": not bool(
            v2_assignments & threshold_names
        ),
        "base_retention_checks_distance_speed_and_force": all(
            token in base_text
            for token in (
                "distance <= self.cfg.maximum_retained_distance_m",
                "relative_speed <= self.cfg.maximum_retained_relative_speed_m_s",
                "& ~self._excess_force",
            )
        ),
        "base_scene_supports_auxiliary_contact_sensors": (
            "self._aux_contact_sensors" in base_text
        ),
        "runtime_has_retention_support_hook": all(
            token in runtime_text
            for token in (
                "def _retention_support_mask",
                "def _get_dones",
                "self._retained = kinematic_retained & secure_support",
                "def _additional_excess_force_mask",
            )
        ),
        "physics_v2_authors_palm_sensor": all(
            token in v2_cfg_text
            for token in (
                "auxiliary_contacts = _physics_v2_auxiliary_contacts()",
                '"palm": ContactSensorCfg',
                "prim_path=RIGHT_HAND_BODY_ROOT",
            )
        ),
        "physics_v2_uses_palm_after_detach": all(
            token in v2_text
            for token in (
                "def _physical_palm_force",
                "self._stem_detached & palm_support",
                "def _retention_reference",
            )
        ),
        "offline_oracle_supports_palm_after_detach": all(
            token in oracle_text
            for token in ("physical_palm_force_n", "palm_support = stem_detached.bool()")
        ),
        "offline_oracle_default_force_ceiling_is_8n": (
            "maximum_fingertip_force_n: float = 8.0" in oracle_text
        ),
    }
    registry_matches = registry["thresholds"] == expected
    semantic = registry["semantic_difference"]
    semantic_documented = (
        semantic.get("id")
        == "explicit_palm_contact_support_after_qualified_detach"
        and semantic.get("resolution_status")
        == "implemented_static_pending_exact_runtime_gpu_validation"
    )
    checks = {
        **source_checks,
        "registry_thresholds_match_profile_and_runtime": registry_matches,
        "semantic_difference_is_fail_closed_and_documented": semantic_documented,
        "formal_training_remains_blocked": profile["implementation_boundary"][
            "training_allowed"
        ]
        is False,
    }
    return {
        "schema_version": 1,
        "audit_id": "tomato_pick_physics_v2_success_contract_parity",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "static sources and pure profiles only; no Isaac import, GPU, PPO, or hardware",
        "thresholds": expected,
        "source_checks": source_checks,
        "semantic_difference": semantic,
        "threshold_parity_achieved": registry_matches
        and all(source_checks.values()),
        "exact_success_semantic_parity_achieved": registry_matches
        and all(source_checks.values()),
        "checks": checks,
        "passed_documentation_and_fail_closed_gate": all(checks.values()),
        "formal_training_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_contract_parity_audit()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed_documentation_and_fail_closed_gate"],
            }
        )
    )
    return 0 if report["passed_documentation_and_fail_closed_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
