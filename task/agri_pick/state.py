"""Observable state machine for the five-tool tomato-picking workflow."""

from enum import Enum


class AgriPickState(str, Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    PREGRASP = "pregrasp"
    PICK = "pick"
    VERIFY = "verify"
    COMPLETE = "complete"
    ABORTED = "aborted"


__all__ = ["AgriPickState"]
