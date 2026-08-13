"""Policy-independent pick evidence sourced from live Isaac rigid-body state."""

from __future__ import annotations

from task.agri_pick.verification import PickEvidenceSnapshot

from .robotics import IsaacHarvestRoboticsBundle


class IsaacPickEvidenceSource:
    """Expose detachment, retention, and drop outcomes without policy access."""

    source_id = "isaac_clean_v5_physics_pick_evidence_v1"

    def __init__(self, bundle: IsaacHarvestRoboticsBundle) -> None:
        self._bundle = bundle
        self._active: set[str] = set()

    def begin(self, execution_id: str, target_id: str) -> None:
        del target_id
        self._active.add(execution_id)

    def sample(self, execution_id: str) -> PickEvidenceSnapshot:
        if execution_id not in self._active:
            raise KeyError(f"unknown Isaac evidence execution {execution_id!r}")
        state = self._bundle.payload_state()
        return PickEvidenceSnapshot(
            sample_monotonic_ns=self._bundle.clock.monotonic_ns(),
            source_sequence=self._bundle.sequence,
            detached=bool(state["detached"]),
            dropped=bool(state["dropped"]),
            retained=bool(state["retained"]),
            source_id=self.source_id,
            valid=True,
        )

    def finish(self, execution_id: str) -> None:
        self._active.discard(execution_id)


__all__ = ["IsaacPickEvidenceSource"]
