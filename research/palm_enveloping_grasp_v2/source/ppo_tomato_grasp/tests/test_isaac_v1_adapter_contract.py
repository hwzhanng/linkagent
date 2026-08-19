from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    LEFT_SAFE_VELOCITY_RAD_S,
)
from simulation.ppo_tomato_grasp.isaac_v1_adapter_contract import (
    action_to_active_joint_command,
    canonical_joint_state,
    pack_deployable_actor_si,
    pack_physx_critic_si,
)


class IsaacV1AdapterContractTests(unittest.TestCase):
    def test_name_based_extraction_ignores_extra_joints_and_order(self):
        names = ("wrist_extra", *reversed(LEFT_ACTIVE_JOINT_ORDER))
        values = torch.arange(len(names), dtype=torch.float32).view(1, -1)
        canonical = canonical_joint_state(names, values)
        expected = torch.tensor([names.index(name) for name in LEFT_ACTIVE_JOINT_ORDER]).float()
        self.assertTrue(torch.equal(canonical[0], expected))

    def test_si_pack_has_exact_shape_and_expected_neutral_joint_normalization(self):
        batch = 3
        lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
        upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
        actor = pack_deployable_actor_si(
            fruit_center_in_palm_m=torch.zeros((batch, 3)),
            fruit_semi_axes_m=torch.full((batch, 3), 0.0325),
            fruit_orientation_6d=torch.tensor((1, 0, 0, 0, 1, 0)).float().expand(batch, -1),
            fruit_twist_in_palm_si=torch.zeros((batch, 6)),
            canonical_joint_position_rad=((lower + upper) / 2).expand(batch, -1),
            canonical_joint_velocity_rad_s=torch.zeros((batch, 16)),
            tactile_matrix_features=torch.zeros((batch, 50)),
            previous_action=torch.zeros((batch, 10)),
        )
        self.assertEqual(tuple(actor.shape), (batch, 110))
        self.assertTrue(torch.allclose(actor[:, 18:50], torch.zeros((batch, 32)), atol=1e-6))

    def test_action_command_is_rate_limited_and_targets_explicit_indices(self):
        names = (*reversed(LEFT_ACTIVE_JOINT_ORDER), "wrist_extra")
        current = torch.tensor(LEFT_SAFE_LOWER_RAD).view(1, 16)
        command = action_to_active_joint_command(torch.ones((1, 10)), current, names)
        velocity = torch.tensor(LEFT_SAFE_VELOCITY_RAD_S)
        maximum = torch.minimum(velocity * 0.5, torch.ones_like(velocity)) * 0.05
        self.assertTrue(torch.all(command.canonical_targets_rad - current <= maximum + 1e-7))
        fast_joint = LEFT_ACTIVE_JOINT_ORDER.index("left_hand_index_joint1")
        self.assertLessEqual(
            float(command.canonical_targets_rad[0, fast_joint] - current[0, fast_joint]),
            0.05 + 1e-7,
        )
        self.assertEqual(command.articulation_indices.numel(), 16)
        self.assertNotIn(len(names) - 1, command.articulation_indices.tolist())

    def test_physx_privileged_truth_is_kept_out_of_actor_and_packs_148d(self):
        batch = 2
        actor = torch.zeros((batch, 110))
        state = pack_physx_critic_si(
            actor,
            physical_fingertip_force_n=torch.ones((batch, 5)),
            physical_palm_force_n=torch.ones(batch),
            contact_points_in_palm_m=torch.zeros((batch, 5, 3)),
            object_mass_kg=torch.full((batch,), 0.1),
            static_friction=torch.full((batch,), 0.5),
            effective_stiffness_n_m=torch.full((batch,), 5000.0),
            peak_indentation_m=torch.zeros(batch),
            mean_force_exposure_n_s=torch.zeros(batch),
            mean_impact_exposure_n=torch.zeros(batch),
            damage_score=torch.zeros(batch),
            tactile_latent_gain=torch.ones((batch, 5)),
            tactile_latent_bias_normalized=torch.zeros((batch, 5)),
        )
        self.assertEqual(tuple(state.shape), (batch, 148))
        self.assertTrue(torch.equal(state[:, :110], actor))

    def test_missing_joint_and_rate_drift_fail_closed(self):
        names = LEFT_ACTIVE_JOINT_ORDER[:-1]
        with self.assertRaises(ValueError):
            canonical_joint_state(names, torch.zeros((1, 15)))
        with self.assertRaises(ValueError):
            action_to_active_joint_command(
                torch.zeros((1, 10)),
                torch.zeros((1, 16)),
                LEFT_ACTIVE_JOINT_ORDER,
                policy_dt_s=0.04,
            )


if __name__ == "__main__":
    unittest.main()
