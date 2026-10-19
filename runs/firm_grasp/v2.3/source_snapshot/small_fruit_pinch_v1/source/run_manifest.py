"""小果捏取实验的可复现运行清单。"""

from __future__ import annotations

import inspect
import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final, TypeAlias

from research.small_fruit_pinch_v1.source.size_evidence import (
    sha256_file,
    sha256_json,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ConfigValue: TypeAlias = (
    JsonScalar
    | Path
    | type
    | Callable[..., "ConfigValue"]
    | Mapping[str, "ConfigValue"]
    | Sequence["ConfigValue"]
)
RunManifest: TypeAlias = dict[str, JsonValue]

SCHEMA_VERSION: Final = "small_fruit_pinch_v1.run_manifest.v1"
_DISTRIBUTIONS: Final = {
    "torch": "torch",
    "isaaclab": "isaaclab",
    "isaacsim": "isaacsim",
    "rsl_rl": "rsl-rl-lib",
}


@dataclass(slots=True)
class UnsupportedManifestValueError(TypeError):
    """配置值无法确定性转换为 JSON。"""

    field_path: str
    value_type: str

    def __str__(self) -> str:
        return f"{self.field_path} 包含不支持的类型 {self.value_type}"


def _stable_path(path: Path, project_root: Path) -> str:
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _qualified_name(
    value: type | Callable[..., ConfigValue], field_path: str
) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    match module, qualname:
        case str() as module_name, str() as qualified_name:
            return f"{module_name}.{qualified_name}"
        case _:
            raise UnsupportedManifestValueError(
                field_path=field_path,
                value_type=type(value).__name__,
            )


def _normalize(value: ConfigValue, project_root: Path, field_path: str) -> JsonValue:
    match value:
        case None | bool() | int() | str():
            return value
        case float() as number:
            if math.isfinite(number):
                return number
            if math.isinf(number):
                return {"float": "+inf" if number > 0 else "-inf"}
            raise UnsupportedManifestValueError(
                field_path=field_path,
                value_type="non-finite float",
            )
        case Path() as path:
            return _stable_path(path, project_root)
        case type() as class_type:
            return {"python_type": _qualified_name(class_type, field_path)}
        case Mapping() as mapping:
            normalized: dict[str, JsonValue] = {}
            field_names: list[str] = []
            for key in mapping:
                match key:
                    case str() as field_name:
                        field_names.append(field_name)
                    case _:
                        raise UnsupportedManifestValueError(
                            field_path=field_path,
                            value_type=f"mapping key {type(key).__name__}",
                        )
            for field_name in sorted(field_names):
                normalized[field_name] = _normalize(
                    mapping[field_name],
                    project_root,
                    f"{field_path}.{field_name}",
                )
            return normalized
        case list() | tuple() as sequence:
            return [
                _normalize(item, project_root, f"{field_path}[{index}]")
                for index, item in enumerate(sequence)
            ]
        case _ if inspect.isroutine(value):
            return {"python_callable": _qualified_name(value, field_path)}
        case _:
            raise UnsupportedManifestValueError(
                field_path=field_path,
                value_type=type(value).__name__,
            )


def _resolve_file(path: Path, project_root: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if not resolved.is_file():
        raise IsADirectoryError(resolved)
    return resolved


def _hash_files(paths: Sequence[Path], project_root: Path) -> list[JsonValue]:
    resolved_paths = {_resolve_file(path, project_root) for path in paths}
    return [
        {
            "path": _stable_path(path, project_root),
            "sha256": sha256_file(path),
        }
        for path in sorted(
            resolved_paths,
            key=lambda item: _stable_path(item, project_root),
        )
    ]


def _runtime_versions(
    provided_versions: Mapping[str, str | None] | None,
) -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for key, distribution in _DISTRIBUTIONS.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    if provided_versions is not None:
        for key, value in provided_versions.items():
            if value is not None:
                versions[key] = value
    return {key: versions[key] for key in sorted(versions)}


def collect_asset_paths(asset_directories: Sequence[Path]) -> tuple[Path, ...]:
    """递归收集调用方明确给出的资产目录内全部文件。"""

    collected: set[Path] = set()
    for directory in asset_directories:
        resolved = directory.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        collected.update(path.resolve() for path in resolved.rglob("*") if path.is_file())
    return tuple(sorted(collected, key=lambda path: path.as_posix()))


def build_run_manifest(
    *,
    project_root: Path,
    cli_argv: Sequence[str],
    cli_args: Mapping[str, ConfigValue],
    env_config: Mapping[str, ConfigValue],
    agent_config: Mapping[str, ConfigValue],
    env_controls: Mapping[str, ConfigValue],
    source_paths: Sequence[Path],
    asset_paths: Sequence[Path],
    task_version: str,
    validation_contract_version: str,
    action_dimension: int,
    observation_dimension: int,
    critic_dimension: int | None = None,
    checkpoint_path: Path | None = None,
    runtime_versions: Mapping[str, str | None] | None = None,
) -> RunManifest:
    """构建无时间戳、可稳定哈希且可直接写为 JSON 的运行清单。"""

    root = project_root.resolve()
    resolved_config: JsonValue = {
        "cli_args": _normalize(cli_args, root, "cli_args"),
        "env": _normalize(env_config, root, "env_config"),
        "agent": _normalize(agent_config, root, "agent_config"),
        "env_controls": _normalize(env_controls, root, "env_controls"),
    }
    checkpoint: JsonValue = None
    if checkpoint_path is not None:
        checkpoint_file = _resolve_file(checkpoint_path, root)
        checkpoint = {
            "path": _stable_path(checkpoint_file, root),
            "sha256": sha256_file(checkpoint_file),
        }
    body: RunManifest = {
        "schema_version": SCHEMA_VERSION,
        "task_version": task_version,
        "validation_contract_version": validation_contract_version,
        "cli_argv": list(cli_argv),
        "dimensions": {
            "action": action_dimension,
            "observation": observation_dimension,
            "critic": critic_dimension,
        },
        "resolved_config": resolved_config,
        "resolved_config_sha256": sha256_json(resolved_config),
        "files": {
            "source": _hash_files(source_paths, root),
            "assets": _hash_files(asset_paths, root),
            "checkpoint": checkpoint,
        },
        "runtime_versions": _runtime_versions(runtime_versions),
    }
    return {**body, "run_sha256": sha256_json(body)}


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """以独占创建模式写入清单，已有路径时拒绝覆盖。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            manifest,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


__all__ = [
    "ConfigValue",
    "RunManifest",
    "UnsupportedManifestValueError",
    "build_run_manifest",
    "collect_asset_paths",
    "write_run_manifest",
]
