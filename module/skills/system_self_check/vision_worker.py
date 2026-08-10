"""Health adapters for the isolated tomato-vision worker."""

from __future__ import annotations

from pathlib import Path

from module.robotics import Clock
from module.skills.tomato_vision import TomatoVisionWorkerClient
from proto import VisionWorkerHealthSnapshot


class MockVisionWorkerHealthPort:
    def __init__(
        self,
        *,
        clock: Clock,
        ready: bool = True,
        worker_id: str = "mock-tomato-vision-worker",
        backend_id: str = "replay_v1",
    ) -> None:
        if not isinstance(ready, bool):
            raise TypeError("ready must be a bool")
        self._clock = clock
        self._ready = ready
        self._worker_id = worker_id
        self._backend_id = backend_id
        self._closed = False

    def snapshot(self) -> VisionWorkerHealthSnapshot:
        if self._closed:
            raise RuntimeError("vision worker health port is closed")
        return VisionWorkerHealthSnapshot(
            worker_id=self._worker_id,
            backend_id=self._backend_id,
            ready=self._ready,
            simulated=True,
            sample_monotonic_ns=self._clock.monotonic_ns(),
            evidence_sources=("mock:tomato_vision_worker_health",),
        )

    def close(self) -> None:
        self._closed = True


class LoopbackVisionWorkerHealthPort:
    def __init__(
        self,
        *,
        clock: Clock,
        endpoint: str,
        capture_root: str | Path,
        worker_id: str,
        timeout_s: float = 0.5,
        max_response_bytes: int = 64_000,
    ) -> None:
        self._clock = clock
        self._endpoint = endpoint
        self._client = TomatoVisionWorkerClient(
            clock=clock,
            endpoint=endpoint,
            capture_root=capture_root,
            worker_id=worker_id,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
            new_frame_wait_timeout_s=0.0,
        )

    def snapshot(self) -> VisionWorkerHealthSnapshot:
        payload = self._client.health()
        backend_id = payload.get("backend_id")
        simulated = payload.get("simulated")
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("vision worker health backend_id is invalid")
        if not isinstance(simulated, bool):
            raise ValueError("vision worker health simulated flag is invalid")
        return VisionWorkerHealthSnapshot(
            worker_id=str(payload["worker_id"]),
            backend_id=backend_id,
            ready=True,
            simulated=simulated,
            sample_monotonic_ns=self._clock.monotonic_ns(),
            evidence_sources=(
                f"vision_worker:{payload['worker_id']}",
                f"vision_worker_endpoint:{self._endpoint}",
            ),
        )

    def close(self) -> None:
        self._client.close()


__all__ = ["LoopbackVisionWorkerHealthPort", "MockVisionWorkerHealthPort"]
