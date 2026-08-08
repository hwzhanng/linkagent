"""Policy protocol plus offline and exported-policy implementations."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .contracts import PolicyManifest, TomatoPickAction, TomatoPickObservation


@runtime_checkable
class TomatoPickPolicy(Protocol):
    @property
    def manifest(self) -> PolicyManifest: ...
    def reset(self) -> None: ...
    def act(self, observation: TomatoPickObservation) -> TomatoPickAction: ...


class MockTomatoPickPolicy:
    """Simple deterministic stub; it is not a training expert or deployment policy."""

    def __init__(self, *, manifest: PolicyManifest | None = None) -> None:
        self._manifest = manifest or PolicyManifest.mock_default()

    @property
    def manifest(self) -> PolicyManifest:
        return self._manifest

    def reset(self) -> None:
        return None

    def act(self, observation: TomatoPickObservation) -> TomatoPickAction:
        if not isinstance(observation, TomatoPickObservation):
            raise TypeError("observation must be a TomatoPickObservation")
        pressure = observation.block("finger_pressure")
        # Keep the mock useful for the offline task without treating it as an
        # RL demonstration: approach, close, and pull are all constant outputs.
        pull = 0.35 if max(pressure) > 0.03 else -0.20
        hand = (0.45, 0.55, 0.10, 0.0, 0.0)
        return TomatoPickAction((0.0, 0.0, pull, 0.0, 0.0, 0.15, *hand))


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_skill_manifest(
    manifest_path: str | Path,
    *,
    allow_hardware: bool = False,
) -> tuple[Path, dict[str, Any], PolicyManifest]:
    """Validate a complete package and return its frozen policy contract."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("skill manifest must use schema_version 1")
    if not isinstance(payload.get("skill_id"), str) or not payload["skill_id"]:
        raise ValueError("skill manifest is missing skill_id")
    if not isinstance(payload.get("status"), str) or not payload["status"]:
        raise ValueError("skill manifest is missing status")
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("skill manifest is missing deployment metadata")
    if allow_hardware and deployment.get("hardware_approved") is not True:
        raise RuntimeError("this skill artifact is not approved for hardware")
    contract = payload.get("policy_manifest")
    if not isinstance(contract, dict):
        raise ValueError("skill manifest is missing policy_manifest")
    policy_manifest = PolicyManifest(**contract)

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("skill manifest is missing artifacts")
    package_root = path.parent
    for name, artifact in artifacts.items():
        if not isinstance(name, str) or not isinstance(artifact, dict):
            raise ValueError("skill artifacts must be named mappings")
        relative_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"{name}.path must be a non-empty string")
        if not isinstance(expected_sha, str) or not _SHA256_PATTERN.fullmatch(expected_sha):
            raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
        artifact_path = (package_root / relative_path).resolve()
        if not artifact_path.is_relative_to(package_root):
            raise ValueError(f"{name}.path must remain inside the skill package")
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"{name} SHA-256 does not match its manifest")
    source = artifacts.get("source_checkpoint")
    if isinstance(source, dict) and source.get("sha256") != policy_manifest.checkpoint_sha256:
        raise ValueError("policy checkpoint SHA-256 does not match source_checkpoint")
    return path, payload, policy_manifest


def load_skill_profile(profile_path: str | Path) -> dict[str, Any]:
    """Load a versioned policy profile and resolve its package path."""

    path = Path(profile_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("skill profile must use schema_version 1")
    skill = payload.get("skill")
    if not isinstance(skill, dict):
        raise ValueError("skill profile is missing skill configuration")
    values = dict(skill)
    if values.get("type") == "profile":
        raise ValueError("skill profiles cannot recursively load another profile")
    manifest_path = values.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        candidate = Path(manifest_path).expanduser()
        if not candidate.is_absolute():
            values["manifest_path"] = str((path.parent / candidate).resolve())
    return values


def _profile_policy(*, path: str | Path) -> TomatoPickPolicy:
    return build_policy(load_skill_profile(path))


class _TorchArtifactTomatoPickPolicy:
    """Common deterministic inference boundary for exported Torch artifacts.

    Torch is an optional deployment-profile dependency and is imported only
    when this policy type is selected.  The package manifest binds the model
    bytes to the frozen tomato-picking interface before any inference occurs.
    """

    artifact_name: str

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        device: str = "cpu",
        allow_hardware: bool = False,
    ) -> None:
        path, payload, self._manifest = validate_skill_manifest(
            manifest_path,
            allow_hardware=allow_hardware,
        )
        artifacts = payload.get("artifacts")
        actor = artifacts.get(self.artifact_name)
        if not isinstance(actor, dict):
            raise ValueError(f"skill manifest is missing {self.artifact_name}")
        artifact_path = (path.parent / actor["path"]).resolve()

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "exported tomato policy requires the optional torch runtime"
            ) from exc
        self._torch = torch
        self._device = torch.device(device)
        self._actor = self._load_actor(artifact_path)
        try:
            self._actor.eval()
        except NotImplementedError:
            # ExportedProgram graph modules are already frozen for inference
            # and deliberately reject train/eval mode transitions.
            pass
        self._validate_runtime_output()

    def _load_actor(self, artifact_path: Path) -> Any:
        raise NotImplementedError

    @property
    def manifest(self) -> PolicyManifest:
        return self._manifest

    def _infer(self, values: tuple[float, ...]) -> tuple[float, ...]:
        tensor = self._torch.tensor(
            (values,), dtype=self._torch.float32, device=self._device
        )
        with self._torch.inference_mode():
            raw = self._actor(tensor)
        flattened = raw.detach().to(device="cpu", dtype=self._torch.float32).reshape(-1)
        if flattened.numel() != 11:
            raise RuntimeError(
                f"deterministic Actor returned {flattened.numel()} values instead of 11"
            )
        values_out = tuple(float(value) for value in flattened.tolist())
        if not all(math.isfinite(value) for value in values_out):
            raise RuntimeError("deterministic Actor returned NaN or Inf")
        # RSL-RL training used clip_actions=1.0 and the Isaac environment
        # applied the same clamp before executing and feeding back actions.
        return tuple(max(-1.0, min(1.0, value)) for value in values_out)

    def _validate_runtime_output(self) -> None:
        self._infer((0.0,) * 66)

    def reset(self) -> None:
        reset = getattr(self._actor, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: TomatoPickObservation) -> TomatoPickAction:
        if not isinstance(observation, TomatoPickObservation):
            raise TypeError("observation must be a TomatoPickObservation")
        return TomatoPickAction(self._infer(observation.values))


class TorchScriptTomatoPickPolicy(_TorchArtifactTomatoPickPolicy):
    """Load the compatibility TorchScript artifact from a skill package."""

    artifact_name = "deterministic_actor"

    def _load_actor(self, artifact_path: Path) -> Any:
        return self._torch.jit.load(str(artifact_path), map_location=self._device)


class TorchExportTomatoPickPolicy(_TorchArtifactTomatoPickPolicy):
    """Load the preferred ``torch.export`` deterministic Actor artifact."""

    artifact_name = "deterministic_actor_export"

    def _load_actor(self, artifact_path: Path) -> Any:
        exported = self._torch.export.load(str(artifact_path))
        return exported.module().to(self._device)


PolicyBuilder = Callable[..., TomatoPickPolicy]
SKILL_REGISTRY: dict[str, PolicyBuilder] = {
    "mock": MockTomatoPickPolicy,
    "profile": _profile_policy,
    "torch_export": TorchExportTomatoPickPolicy,
    "torchscript": TorchScriptTomatoPickPolicy,
}


def build_policy(config: Mapping[str, Any]) -> TomatoPickPolicy:
    if not isinstance(config, Mapping):
        raise TypeError("skill config must be a mapping")
    values = dict(config)
    kind = values.pop("type", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("skill.type must be a non-empty string")
    try:
        builder = SKILL_REGISTRY[kind.strip()]
    except KeyError as exc:
        raise ValueError(f"unknown skill type {kind!r}") from exc
    policy = builder(**values)
    if not isinstance(policy, TomatoPickPolicy):
        raise TypeError("skill builder must return a TomatoPickPolicy")
    return policy


__all__ = [
    "MockTomatoPickPolicy",
    "PolicyBuilder",
    "SKILL_REGISTRY",
    "TorchExportTomatoPickPolicy",
    "TorchScriptTomatoPickPolicy",
    "TomatoPickPolicy",
    "build_policy",
    "load_skill_profile",
    "validate_skill_manifest",
]
