"""Versioned v2 palm-supported enveloping-grasp experiment.

The v2 namespace is deliberately independent from ``simulation.ppo_tomato_grasp``.
The v1 source and checkpoints are treated as read-only baselines; v2 imports
their validated scene only through small, auditable adapters.
"""

from .contract import V2_CONTRACT_VERSION

__all__ = ["V2_CONTRACT_VERSION"]
