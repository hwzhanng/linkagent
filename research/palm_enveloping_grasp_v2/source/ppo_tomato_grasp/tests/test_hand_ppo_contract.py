from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTION_ORDER,
    ACTOR_OBSERVATION_BLOCKS,
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_BLOCKS,
    CRITIC_STATE_DIM,
    block,
    pack_actor_observation,
    pack_critic_state,
)


class HandPpoContractTests(unittest.TestCase):
    def actor_blocks(self, batch: int = 4):
        return {
            name: torch.zeros((batch, item.stop - item.start))
            for name, item in ACTOR_OBSERVATION_BLOCKS.items()
        }

    def critic_blocks(self, batch: int = 4):
        return {
            name: torch.zeros((batch, item.stop - item.start))
            for name, item in CRITIC_PRIVILEGED_BLOCKS.items()
        }

    def test_dimensions_and_action_names_are_frozen(self):
        self.assertEqual(ACTION_DIM, 10)
        self.assertEqual(len(ACTION_ORDER), 10)
        self.assertEqual(ACTOR_OBSERVATION_DIM, 110)
        self.assertEqual(CRITIC_STATE_DIM, 148)

    def test_actor_and_critic_pack_exactly(self):
        actor_blocks = self.actor_blocks()
        actor_blocks["fruit_center_in_palm"][:] = 2.0
        actor = pack_actor_observation(**actor_blocks)
        self.assertEqual(tuple(actor.shape), (4, 110))
        self.assertTrue(torch.all(block(actor, "fruit_center_in_palm") == 1.0))
        critic = pack_critic_state(actor, **self.critic_blocks())
        self.assertEqual(tuple(critic.shape), (4, 148))
        self.assertTrue(torch.equal(block(critic, "previous_action"), actor[:, 100:110]))

    def test_missing_fields_nonfinite_values_and_shape_drift_fail_closed(self):
        actor_blocks = self.actor_blocks()
        actor_blocks.pop("previous_action")
        with self.assertRaises(ValueError):
            pack_actor_observation(**actor_blocks)

        actor_blocks = self.actor_blocks()
        actor_blocks["fruit_semi_axes"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            pack_actor_observation(**actor_blocks)

        actor = torch.zeros((4, ACTOR_OBSERVATION_DIM))
        privileged = self.critic_blocks()
        privileged["damage_score"] = torch.zeros((4, 2))
        with self.assertRaises(ValueError):
            pack_critic_state(actor, **privileged)

    def test_unknown_or_privileged_actor_lookup_is_rejected(self):
        actor = torch.zeros((2, ACTOR_OBSERVATION_DIM))
        with self.assertRaises(ValueError):
            block(actor, "damage_score")
        with self.assertRaises(KeyError):
            block(actor, "not_a_field")


if __name__ == "__main__":
    unittest.main()
