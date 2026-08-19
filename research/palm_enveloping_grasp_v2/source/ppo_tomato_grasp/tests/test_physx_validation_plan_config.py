from __future__ import annotations

import json
import unittest
from pathlib import Path


PLAN = Path(__file__).resolve().parents[1] / "configs" / "physx_v1_validation_plan.json"


class PhysxValidationPlanConfigTests(unittest.TestCase):
    def test_plan_is_fail_closed_ordered_and_dimensionally_frozen(self):
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "planned_gpu_and_isaac_required_not_executed")
        self.assertFalse(plan["isaac_physx_release_allowed"])
        self.assertFalse(plan["formal_gpu_training_allowed"])
        self.assertFalse(plan["hardware_command_allowed"])
        gates = plan["gates"]
        ids = [gate["id"] for gate in gates]
        self.assertEqual(len(ids), len(set(ids)))
        seen: set[str] = set()
        for gate in gates:
            self.assertTrue(set(gate["requires"]).issubset(seen))
            self.assertTrue(gate["checks"])
            seen.add(gate["id"])
        vectorized = next(gate for gate in gates if gate["id"] == "P7_vectorized_smoke")
        self.assertEqual(vectorized["checks"]["actor_observation_dim"], 110)
        self.assertEqual(vectorized["checks"]["critic_state_dim"], 148)
        self.assertEqual(vectorized["checks"]["action_dim"], 10)

    def test_plan_precommits_velocity_contact_and_tactile_checks(self):
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        checks = {gate["id"]: gate["checks"] for gate in plan["gates"]}
        self.assertEqual(checks["P2_free_motion"]["maximum_commanded_velocity_rad_s"], 1.0)
        self.assertLessEqual(
            checks["P4_contact_force_units"]["known_mass_gravity_force_relative_error_maximum"],
            0.1,
        )
        self.assertTrue(checks["P5_tactile_projection"]["privileged_force_absent_from_actor"])


if __name__ == "__main__":
    unittest.main()
