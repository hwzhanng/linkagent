from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simulation.ppo_tomato_grasp.audit_left_hand_urdf import audit_urdf
from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
)


class LeftHandUrdfAuditTests(unittest.TestCase):
    def test_minimal_matching_urdf_passes_structural_contract(self):
        joints = []
        for name, lower, upper in zip(
            LEFT_ACTIVE_JOINT_ORDER, LEFT_SAFE_LOWER_RAD, LEFT_SAFE_UPPER_RAD
        ):
            joints.append(
                f'<joint name="{name}" type="revolute"><axis xyz="1 0 0"/>'
                f'<limit lower="{lower - 0.1}" upper="{upper + 0.1}" effort="1" velocity="10"/>'
                "</joint>"
            )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hand.urdf"
            path.write_text("<robot name=\"test\">" + "".join(joints) + "</robot>", encoding="utf-8")
            report = audit_urdf(path)
        self.assertTrue(report["structural_contract_passed"])
        self.assertTrue(report["physx_adapter_ready"])

    def test_passive_joint_without_mimic_blocks_physx_readiness(self):
        joints = []
        for name, lower, upper in zip(
            LEFT_ACTIVE_JOINT_ORDER, LEFT_SAFE_LOWER_RAD, LEFT_SAFE_UPPER_RAD
        ):
            joints.append(
                f'<joint name="{name}" type="revolute"><axis xyz="1 0 0"/>'
                f'<limit lower="{lower - 0.1}" upper="{upper + 0.1}" effort="1" velocity="10"/>'
                "</joint>"
            )
        joints.append(
            '<joint name="left_hand_index_joint3" type="revolute"><axis xyz="1 0 0"/>'
            '<limit lower="0" upper="1" effort="1" velocity="1"/></joint>'
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hand.urdf"
            path.write_text("<robot name=\"test\">" + "".join(joints) + "</robot>", encoding="utf-8")
            report = audit_urdf(path)
        self.assertTrue(report["structural_contract_passed"])
        self.assertFalse(report["physx_adapter_ready"])
        self.assertEqual(report["uncontrolled_without_mimic"][0]["joint"], "left_hand_index_joint3")


if __name__ == "__main__":
    unittest.main()
