from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from research.small_fruit_pinch_v1.source.run_manifest import (
    ConfigValue,
    RunManifest,
    UnsupportedManifestValueError,
    build_run_manifest,
    collect_asset_paths,
    write_run_manifest,
)


def _sample_callable() -> None:
    pass


def _build_manifest(
    project_root: Path,
    *,
    env_config: Mapping[str, ConfigValue] | None = None,
    source_paths: Sequence[Path] | None = None,
    asset_paths: Sequence[Path] | None = None,
    checkpoint_path: Path | None = None,
) -> RunManifest:
    return build_run_manifest(
        project_root=project_root,
        cli_argv=("native_train.py", "--seed", "17"),
        cli_args={"seed": 17, "headless": True},
        env_config=env_config or {"episode_length": 640},
        agent_config={"learning_rate": 0.001},
        env_controls={
            "control_decimation": 2,
            "joint_names": ("thumb", "index"),
            "asset_root": project_root / "assets",
            "env_type": dict,
            "control_hook": _sample_callable,
        },
        source_paths=source_paths or (project_root / "source.py",),
        asset_paths=asset_paths or (project_root / "assets" / "fruit.usd",),
        checkpoint_path=checkpoint_path,
        task_version="small_fruit_pinch_v1",
        validation_contract_version="a-stage-v1",
        action_dimension=14,
        observation_dimension=96,
        runtime_versions={"isaac_runtime": "5.0.0-test"},
    )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "assets" / "meshes").mkdir(parents=True)
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "assets" / "fruit.usd").write_text(
        "@./meshes/fruit.obj@\n", encoding="utf-8"
    )
    (root / "assets" / "meshes" / "fruit.obj").write_text(
        "v 0 0 0\n", encoding="utf-8"
    )
    return root


def test_build_run_manifest_changes_hash_when_config_changes(project_root: Path) -> None:
    # Given
    original = _build_manifest(project_root, env_config={"episode_length": 640})

    # When
    changed = _build_manifest(project_root, env_config={"episode_length": 800})

    # Then
    assert original["resolved_config_sha256"] != changed["resolved_config_sha256"]
    assert original["run_sha256"] != changed["run_sha256"]


def test_build_run_manifest_changes_hash_when_source_changes(project_root: Path) -> None:
    # Given
    original = _build_manifest(project_root)

    # When
    (project_root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _build_manifest(project_root)

    # Then
    assert original["files"] != changed["files"]
    assert original["run_sha256"] != changed["run_sha256"]


def test_build_run_manifest_changes_hash_when_asset_changes(project_root: Path) -> None:
    # Given
    asset_paths = collect_asset_paths((project_root / "assets",))
    original = _build_manifest(project_root, asset_paths=asset_paths)

    # When
    (project_root / "assets" / "meshes" / "fruit.obj").write_text(
        "v 1 0 0\n", encoding="utf-8"
    )
    changed = _build_manifest(project_root, asset_paths=asset_paths)

    # Then
    assert original["files"] != changed["files"]
    assert original["run_sha256"] != changed["run_sha256"]


def test_build_run_manifest_hash_ignores_mapping_insertion_order(
    project_root: Path,
) -> None:
    # Given
    first_config = {"seed": 17, "nested": {"gain": 0.4, "enabled": True}}
    second_config = {"nested": {"enabled": True, "gain": 0.4}, "seed": 17}

    # When
    first = _build_manifest(project_root, env_config=first_config)
    second = _build_manifest(project_root, env_config=second_config)

    # Then
    assert first["resolved_config_sha256"] == second["resolved_config_sha256"]
    assert first["run_sha256"] == second["run_sha256"]


@pytest.mark.parametrize("missing_kind", ["source", "asset", "checkpoint"])
def test_build_run_manifest_rejects_missing_files(
    project_root: Path, missing_kind: str
) -> None:
    # Given
    source_paths = (project_root / "source.py",)
    asset_paths = (project_root / "assets" / "fruit.usd",)
    checkpoint_path = None
    if missing_kind == "source":
        source_paths = (project_root / "missing.py",)
    elif missing_kind == "asset":
        asset_paths = (project_root / "assets" / "missing.usd",)
    else:
        checkpoint_path = project_root / "missing.pt"

    # When / Then
    with pytest.raises(FileNotFoundError):
        _build_manifest(
            project_root,
            source_paths=source_paths,
            asset_paths=asset_paths,
            checkpoint_path=checkpoint_path,
        )


def test_build_run_manifest_rejects_unknown_config_type(project_root: Path) -> None:
    # Given
    unsupported = {"bad": {1, 2}}

    # When / Then
    with pytest.raises(UnsupportedManifestValueError, match="env_config.bad"):
        _build_manifest(project_root, env_config=unsupported)


def test_collect_asset_paths_includes_nested_dependencies(project_root: Path) -> None:
    # Given / When
    paths = collect_asset_paths((project_root / "assets",))

    # Then
    assert paths == (
        project_root / "assets" / "fruit.usd",
        project_root / "assets" / "meshes" / "fruit.obj",
    )


def test_write_run_manifest_exclusively_creates_json(project_root: Path) -> None:
    # Given
    output_path = project_root / "run_manifest.json"
    manifest = _build_manifest(project_root)

    # When
    write_run_manifest(output_path, manifest)

    # Then
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        write_run_manifest(output_path, manifest)


def test_unbounded_physics_limits_have_stable_json_encoding(project_root: Path) -> None:
    manifest = _build_manifest(project_root, env_config={"effort": float("inf"), "lower": -float("inf")})
    encoded = json.dumps(manifest, allow_nan=False)
    assert json.loads(encoded)["resolved_config"]["env"] == {
        "effort": {"float": "+inf"}, "lower": {"float": "-inf"}
    }


def test_nan_configuration_is_rejected(project_root: Path) -> None:
    with pytest.raises(UnsupportedManifestValueError):
        _build_manifest(project_root, env_config={"effort": float("nan")})
