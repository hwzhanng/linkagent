"""C2-0 cross-file gates for the tomato-pick simulator contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

from module.skills.tomato_pick import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    INTERFACE_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT_ID,
    PHASE_ORDER,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    ROOT
    / "botSDK/urdf/02_project_derived/simulation_baselines/tomato_pick_v1.yaml"
)
LEGACY_PATH = PROFILE_PATH.with_name("agricultural_grasp_no_equipment_proxy.yaml")
MOUNTS_PATH = (
    ROOT
    / "botSDK/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/config/mounts.json"
)
SIM_BASELINE_PATH = MOUNTS_PATH.with_name("simulation_baseline.yaml")
URDF_PATH = (
    ROOT
    / "botSDK/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/urdf/"
    "linkbot_bimanual_sdk_description.urdf"
)
CHECK_SCRIPT = (
    ROOT
    / "botSDK/lbot_arm_api-master/tools/isaac/"
    "check_agricultural_grasp_entry.py"
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _floats(value: object) -> tuple[float, ...]:
    return tuple(float(item) for item in value)  # type: ignore[arg-type]


class C2ZeroContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _json(PROFILE_PATH)

    def test_sim_profile_matches_frozen_policy_contract(self) -> None:
        policy = self.profile["policy"]
        self.assertIsInstance(policy, dict)
        self.assertEqual(policy["interface_version"], INTERFACE_VERSION)
        self.assertEqual(policy["observation_layout_id"], OBSERVATION_LAYOUT_ID)
        self.assertEqual(policy["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(policy["action_layout_id"], ACTION_LAYOUT_ID)
        self.assertEqual(policy["action_dim"], ACTION_DIM)
        self.assertEqual(policy["policy_rate_hz"], 20)
        self.assertEqual(tuple(policy["phases"]), PHASE_ORDER)
        self.assertEqual(policy["phase_step_limits"], [30, 30, 40, 20])
        self.assertEqual(
            policy["phase_semantics"],
            "observable_progress_labels_only_no_action_mask_or_timeout",
        )
        self.assertEqual(len(policy["action_scales"]), ACTION_DIM)
        environment = self.profile["environment"]
        self.assertEqual(
            environment["physics_rate_hz"] // environment["policy_decimation"],
            policy["policy_rate_hz"],
        )
        self.assertEqual(
            self.profile["entry_gates"]["ppo_training"],
            "allowed_after_113d_22d_physics_reward_and_optimizer_gates",
        )
        self.assertEqual(
            self.profile["entry_gates"]["rule_policy_acceptance"],
            "diagnostic_only_not_a_training_gate",
        )

    def test_source_paths_and_runtime_boundaries_are_explicit(self) -> None:
        for relative in self.profile["source_of_truth"].values():
            if isinstance(relative, str) and "/" in relative:
                self.assertTrue((ROOT / relative).exists(), relative)
                self.assertFalse(Path(relative).is_absolute(), relative)
        boundaries = self.profile["runtime_boundaries"]
        self.assertFalse(boundaries["agent_task"]["imports_vendor_sdk"])
        self.assertTrue(boundaries["sdk_gateway"]["process_isolation_from_agent_and_isaac"])
        self.assertFalse(boundaries["isaac_training"]["connects_real_can"])

    def test_mount_direction_matches_manifest_urdf_and_sim_baseline(self) -> None:
        profile_mounts = self.profile["robot"]["mounts_rpy_rad"]
        mounts = _json(MOUNTS_PATH)
        root = ET.parse(URDF_PATH).getroot()
        for side in ("left", "right"):
            expected = _floats(profile_mounts[side])
            self.assertEqual(_floats(mounts[f"{side}_hand"]["rpy"]), expected)
            joint = root.find(f"./joint[@name='{side}_hand_mount_joint']")
            self.assertIsNotNone(joint)
            origin = joint.find("origin")
            self.assertIsNotNone(origin)
            actual = tuple(float(item) for item in origin.attrib["rpy"].split())
            for left, right in zip(actual, expected):
                self.assertTrue(math.isclose(left, right, abs_tol=1e-12, rel_tol=0.0))

        baseline = SIM_BASELINE_PATH.read_text(encoding="utf-8")
        match = re.search(
            r"mounts:\s+right:.*?rpy:\s*\[([^]]+)\].*?left:.*?rpy:\s*\[([^]]+)\]",
            baseline,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        right = tuple(float(item.strip()) for item in match.group(1).split(","))
        left = tuple(float(item.strip()) for item in match.group(2).split(","))
        self.assertEqual(right, _floats(profile_mounts["right"]))
        self.assertEqual(left, _floats(profile_mounts["left"]))

    def test_tactile_contract_never_fabricates_newtons(self) -> None:
        tactile = self.profile["tactile"]
        matrix = tactile["verified_hardware_matrix"]
        force = tactile["calibrated_resultant_force_n"]
        proxy = tactile["simulation_proxy"]
        self.assertEqual(matrix["shape"], [5, 12, 6])
        self.assertEqual(matrix["range"], [0, 255])
        self.assertIsNone(matrix["unit"])
        self.assertFalse(force["raw_to_newton_linear_conversion_allowed"])
        self.assertEqual(force["before_calibration"], "force_valid_false_and_force_values_null")
        self.assertTrue(proxy["must_be_marked_simulated"])
        self.assertFalse(proxy["may_be_claimed_as_hardware_calibration"])

    def test_right_hand_and_legacy_profile_stay_blocked(self) -> None:
        hardware = self.profile["robot"]["hardware_validation"]
        self.assertEqual(hardware["right_g20"], "pending_direct_control_and_tactile")
        integration = _json(ROOT / "botSDK/INTEGRATION_MANIFEST.json")
        self.assertFalse(integration["observed_topology"]["right_g20"]["validated"])
        self.assertFalse(integration["observed_topology"]["simultaneous_bimanual_topology_validated"])
        legacy = _json(LEGACY_PATH)
        self.assertEqual(legacy["status"], "legacy_reference_only")
        self.assertFalse(legacy["current_training_entry"])

    def test_success_requires_detached_retained_and_not_dropped(self) -> None:
        task = self.profile["task"]
        self.assertTrue(task["stem"]["enabled"])
        self.assertEqual(
            task["success"],
            {
                "stem_detached": True,
                "fruit_retained_without_slip": True,
                "dropped": False,
                "verifier_valid": True,
            },
        )
        environment = self.profile["environment"]
        self.assertEqual(
            environment["termination"]["success"],
            "detached_and_retained_without_slip_and_not_dropped",
        )
        self.assertFalse(environment["reward"]["sim_ground_truth_in_actor"])
        self.assertTrue(environment["reward"]["sim_ground_truth_only_for_reward_or_critic"])

    def test_standalone_c2_entry_check_passes_without_isaac(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(CHECK_SCRIPT)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["next_gate"],
            "start_fresh_ppo_and_monitor_full_task_success",
        )


if __name__ == "__main__":
    unittest.main()
