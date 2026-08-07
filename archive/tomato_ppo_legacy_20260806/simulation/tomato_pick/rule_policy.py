"""Non-learning policy used to validate C2 environment sequencing."""

from __future__ import annotations

from module.skills.tomato_pick import TomatoPickAction, TomatoPickObservation


class RuleTomatoPickPolicy:
    """Non-deployable pressure-driven smoke policy for the scalar contract."""

    policy_id = "c2_rule_acceptance_v1"
    deployable_checkpoint = False

    def reset(self) -> None:
        """The rule policy has no episode-local memory."""

        return None

    def act(self, observation: TomatoPickObservation) -> TomatoPickAction:
        """Exercise approach, flexion and pull without consuming hidden phase."""

        if not isinstance(observation, TomatoPickObservation):
            raise TypeError("observation must be TomatoPickObservation")
        pressure = max(observation.block(58, 63))
        action = [0.0] * 22
        if pressure < 0.20:
            action[2] = -0.60
        elif pressure < 0.50:
            for index in (0, 1, 2, 3, 4, 11, 12, 13, 14, 15):
                action[6 + index] = 0.80
        else:
            action[2] = 1.0
            for index in (0, 1, 2, 3, 4, 11, 12, 13, 14, 15):
                action[6 + index] = 0.20
        return TomatoPickAction(tuple(action))


__all__ = ["RuleTomatoPickPolicy"]
