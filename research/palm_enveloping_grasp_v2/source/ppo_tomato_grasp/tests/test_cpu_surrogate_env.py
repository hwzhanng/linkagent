from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.cpu_surrogate_env import CpuHandGraspEnv


class CpuSurrogateEnvTests(unittest.TestCase):
    def test_reset_and_step_cover_frozen_contract_without_gpu(self):
        env = CpuHandGraspEnv(32, seed=4, curriculum_fraction=0.5)
        observation, critic = env.reset()
        self.assertEqual(tuple(observation.shape), (32, 110))
        self.assertEqual(tuple(critic.shape), (32, 148))
        result = env.step(torch.zeros((32, 10)))
        self.assertEqual(tuple(result.reward.shape), (32,))
        self.assertTrue(torch.isfinite(result.observation).all())
        self.assertTrue(torch.isfinite(result.critic_state).all())
        self.assertTrue(torch.isfinite(result.reward).all())

    def test_closing_actions_generate_more_contact_than_open_actions(self):
        open_env = CpuHandGraspEnv(128, seed=8, curriculum_fraction=0.0, auto_reset=False)
        close_env = CpuHandGraspEnv(128, seed=8, curriculum_fraction=0.0, auto_reset=False)
        open_action = torch.zeros((128, 10))
        open_action[:, 0] = -1.0
        open_action[:, 4] = -1.0
        close_action = torch.zeros((128, 10))
        close_action[:, 0] = 0.7
        close_action[:, 4] = 0.6
        for _ in range(12):
            open_result = open_env.step(open_action)
            close_result = close_env.step(close_action)
        self.assertGreater(
            float(close_result.metrics["mean_force_n"].mean()),
            float(open_result.metrics["mean_force_n"].mean()) + 0.1,
        )
        self.assertGreater(
            float(close_result.metrics["secure_enclosure"].float().mean()),
            float(open_result.metrics["secure_enclosure"].float().mean()),
        )

    def test_seed_reproduces_profiles_and_transitions(self):
        left = CpuHandGraspEnv(16, seed=99, curriculum_fraction=0.7, auto_reset=False)
        right = CpuHandGraspEnv(16, seed=99, curriculum_fraction=0.7, auto_reset=False)
        actions = torch.linspace(-0.8, 0.8, 160).reshape(16, 10)
        for _ in range(5):
            left_step = left.step(actions)
            right_step = right.step(actions)
        self.assertTrue(torch.equal(left_step.observation, right_step.observation))
        self.assertTrue(torch.equal(left_step.reward, right_step.reward))
        self.assertTrue(torch.equal(left_step.terminated, right_step.terminated))

    def test_auto_reset_keeps_batch_finite_across_timeouts(self):
        env = CpuHandGraspEnv(8, seed=3, episode_steps=3, auto_reset=True)
        action = torch.zeros((8, 10))
        saw_timeout = False
        for _ in range(10):
            result = env.step(action)
            saw_timeout |= bool(result.truncated.any())
            self.assertTrue(torch.isfinite(result.observation).all())
        self.assertTrue(saw_timeout)

    def test_invalid_actions_fail_closed(self):
        env = CpuHandGraspEnv(4)
        with self.assertRaises(ValueError):
            env.step(torch.zeros((4, 5)))
        bad = torch.zeros((4, 10))
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            env.step(bad)


if __name__ == "__main__":
    unittest.main()
