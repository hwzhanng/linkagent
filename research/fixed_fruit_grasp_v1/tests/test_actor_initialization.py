"""固定果实 actor 先验初始化的 CPU 单元测试。"""

from __future__ import annotations

import json
import unittest

import torch
from tensordict import TensorDict
from torch import nn
from rsl_rl.models import MLPModel

from research.fixed_fruit_grasp_v1.run_utils import (
    initialize_actor_output_bias,
    parse_initial_action_bias_json,
)


class _ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(5, 9),
            nn.Tanh(),
            nn.Linear(9, 14),
        )
        self.critic = nn.Sequential(
            nn.Linear(7, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )


class ActorInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1729)
        self.bias = [round((index - 6) / 10, 1) for index in range(14)]

    def test_actor_critic_mean_is_constant_and_only_output_layer_changes(self) -> None:
        policy = _ActorCritic()
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}

        record = initialize_actor_output_bias(torch, policy, self.bias)

        input_a = torch.randn(3, 5)
        input_b = torch.randn(11, 5)
        mean_a = policy.actor(input_a)
        mean_b = policy.actor(input_b)
        expected_a = torch.tensor(self.bias).expand_as(mean_a)
        expected_b = torch.tensor(self.bias).expand_as(mean_b)
        torch.testing.assert_close(mean_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(mean_b, expected_b, rtol=0, atol=0)

        after = policy.state_dict()
        changed_names = {
            name for name, value in after.items() if not torch.equal(value, before[name])
        }
        self.assertEqual(changed_names, {"actor.2.weight", "actor.2.bias"})
        self.assertEqual(record["actor_output_layer"], "actor.2")
        self.assertEqual(record["kind"], "zero_output_weight_declared_action_bias_prior")
        self.assertEqual(record["bias"], self.bias)
        self.assertEqual(
            record["classification"],
            "declared_initialization_prior_not_learned_success",
        )
        self.assertNotEqual(record["weight_sha256_before"], record["weight_sha256_after"])
        self.assertEqual(torch.count_nonzero(policy.actor[2].weight).item(), 0)

    def test_direct_mlpmodel_policy_is_supported(self) -> None:
        # Exercise the installed RSL-RL MLPModel implementation directly.
        example_obs = TensorDict({"policy": torch.zeros(1, 4)}, batch_size=[1])
        policy = MLPModel(
            example_obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=14,
            hidden_dims=[6],
            activation="elu",
        )
        record = initialize_actor_output_bias(torch, policy, self.bias)
        torch.testing.assert_close(
            policy(TensorDict({"policy": torch.randn(4, 4)}, batch_size=[4])),
            torch.tensor(self.bias).expand(4, 14),
            rtol=0,
            atol=0,
        )
        self.assertEqual(record["actor_output_layer"], "mlp.2")

    def test_non_fourteen_actor_output_fails_before_mutating_policy(self) -> None:
        policy = nn.Module()
        policy.actor = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 13))
        policy.critic = nn.Linear(4, 1)
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "14 outputs"):
            initialize_actor_output_bias(torch, policy, self.bias)
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_invalid_bias_json_is_rejected(self) -> None:
        invalid_values = (
            json.dumps([0.0] * 13),
            "[NaN," + ",".join(["0"] * 13) + "]",
            json.dumps([1.01] + [0.0] * 13),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_initial_action_bias_json(value)

    def test_resume_and_bias_are_rejected_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "fresh run"):
            parse_initial_action_bias_json(json.dumps(self.bias), resume=True)


if __name__ == "__main__":
    unittest.main()
