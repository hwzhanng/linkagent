from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.hand_grasp_mdp import (
    FINGER_FORCE_CEILING_N,
    hand_grasp_transition,
    initial_hand_grasp_state,
)


class HandGraspMdpTests(unittest.TestCase):
    def transition(self, state, force, **overrides):
        batch = force.shape[0]
        values = {
            "surface_gap_m": torch.zeros((batch, 5)),
            "physical_finger_force_n": force,
            "physical_palm_force_n": torch.zeros(batch),
            "center_distance_m": torch.full((batch,), 0.02),
            "relative_speed_m_s": torch.full((batch,), 0.01),
            "slip_detected": torch.zeros(batch, dtype=torch.bool),
            "dropped": torch.zeros(batch, dtype=torch.bool),
            "self_collision": torch.zeros(batch, dtype=torch.bool),
            "action": torch.zeros((batch, 10)),
            "action_delta": torch.zeros((batch, 10)),
            "normalized_joint_limit_proximity": torch.zeros(batch),
        }
        values.update(overrides)
        return hand_grasp_transition(state, **values)

    def test_unequal_gentle_forces_can_succeed_without_balance_target(self):
        state = initial_hand_grasp_state(torch.zeros(1))
        force = torch.tensor([[1.1, 0.7, 0.4, 0.0, 0.0]])
        transition = None
        for _ in range(20):
            transition = self.transition(state, force)
            state = transition.state
        assert transition is not None
        self.assertTrue(transition.success.item())
        self.assertTrue(transition.metrics["finger_support"].item())
        self.assertLess(float(transition.metrics["damage_score"]), 0.1)

    def test_palm_enclosure_still_requires_thumb_and_an_opposing_finger(self):
        state = initial_hand_grasp_state(torch.zeros(3))
        force = torch.tensor(
            [
                [0.4, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.4, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        result = self.transition(state, force, physical_palm_force_n=torch.ones(3))
        self.assertEqual(result.metrics["palm_support"].tolist(), [True, False, False])

    def test_repeated_identical_contact_cannot_repeat_progress_reward(self):
        force = torch.tensor([[1.0, 0.7, 0.5, 0.0, 0.0]])
        state = initial_hand_grasp_state(torch.zeros(1))
        first = self.transition(state, force)
        second = self.transition(first.state, force)
        first_progress = sum(
            float(first.reward_terms[name])
            for name in (
                "proximity_progress",
                "contact_progress",
                "opposition_progress",
                "enclosure_progress",
                "retention_progress",
            )
        )
        second_progress = sum(
            float(second.reward_terms[name])
            for name in (
                "proximity_progress",
                "contact_progress",
                "opposition_progress",
                "enclosure_progress",
                "retention_progress",
            )
        )
        self.assertGreater(first_progress, 1.0)
        self.assertAlmostEqual(second_progress, 0.0, places=6)

    def test_soft_limits_reduce_reward_without_declaring_failure(self):
        state = initial_hand_grasp_state(torch.zeros(1))
        force = torch.tensor([[0.6, 0.5, 0.4, 0.0, 0.0]])
        interior = self.transition(state, force)
        boundary = self.transition(
            state,
            force,
            action=torch.full((1, 10), 0.99),
            normalized_joint_limit_proximity=torch.ones(1),
        )
        self.assertLess(float(boundary.reward), float(interior.reward))
        self.assertFalse(boundary.terminated.item())

    def test_small_transient_over_ceiling_is_not_immediate_terminal(self):
        ceilings = torch.tensor(FINGER_FORCE_CEILING_N)
        state = initial_hand_grasp_state(torch.zeros(1))
        force = (1.05 * ceilings).unsqueeze(0)
        result = self.transition(state, force)
        self.assertFalse(result.terminated.item())
        self.assertFalse(result.metrics["severe_overload"].item())
        self.assertGreater(float(result.metrics["damage_score"]), 0.0)

    def test_severe_overload_collision_and_drop_terminate(self):
        ceilings = torch.tensor(FINGER_FORCE_CEILING_N)
        state = initial_hand_grasp_state(torch.zeros(3))
        force = torch.zeros((3, 5))
        force[0] = 1.3 * ceilings
        result = self.transition(
            state,
            force,
            dropped=torch.tensor([False, True, False]),
            self_collision=torch.tensor([False, False, True]),
        )
        self.assertEqual(result.terminated.tolist(), [True, True, True])

    def test_slip_resets_hold_without_destroying_episode(self):
        force = torch.tensor([[1.0, 0.7, 0.5, 0.0, 0.0]])
        state = initial_hand_grasp_state(torch.zeros(1))
        for _ in range(5):
            result = self.transition(state, force)
            state = result.state
        slipped = self.transition(state, force, slip_detected=torch.ones(1, dtype=torch.bool))
        self.assertEqual(int(slipped.state.hold_streak), 0)
        self.assertFalse(slipped.terminated.item())
        self.assertLess(float(slipped.reward_terms["slip"]), 0.0)

    def test_waiting_is_negative_and_gentle_hold_outranks_overload(self):
        wait_state = initial_hand_grasp_state(torch.zeros(1))
        wait = self.transition(
            wait_state,
            torch.zeros((1, 5)),
            surface_gap_m=torch.full((1, 5), 0.08),
            center_distance_m=torch.full((1,), 0.08),
        )
        gentle = self.transition(
            initial_hand_grasp_state(torch.zeros(1)),
            torch.tensor([[1.0, 0.7, 0.5, 0.0, 0.0]]),
        )
        overload = self.transition(
            initial_hand_grasp_state(torch.zeros(1)),
            1.3 * torch.tensor(FINGER_FORCE_CEILING_N).unsqueeze(0),
        )
        self.assertLess(float(wait.reward), 0.0)
        self.assertGreater(float(gentle.reward), float(wait.reward))
        self.assertGreater(float(gentle.reward), float(overload.reward))


if __name__ == "__main__":
    unittest.main()
