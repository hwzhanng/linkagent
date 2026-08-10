"""Create a CPU-only, read-only audit of Physics-v2 training artifacts.

This tool deliberately does not import Torch, Isaac Sim, or query a GPU.  It
turns an RSL-RL text log, its launch manifest, and local checkpoints into a
traceable inventory.  Training metrics can only nominate checkpoints for a
later independent evaluation; they are never treated as evidence that a model
is "best".
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
CHECKPOINT = re.compile(r"model_(\d+)\.pt$")
ERROR_TOKEN = re.compile(
    r"Traceback|CUDA out of memory|PhysX[^\n]*overflow|:\s*(?:nan|[-+]?inf)\b",
    flags=re.IGNORECASE,
)
METRICS = {
    "steps_per_second": "Steps per second",
    "value_loss": "Mean value loss",
    "surrogate_loss": "Mean surrogate loss",
    "entropy_loss": "Mean entropy loss",
    "mean_reward": "Mean reward",
    "action_std": "Mean action std",
    "success_rate": "Metrics/success_rate",
    "contact_rate": "Episode_Reward/contact",
    "drop_rate": "Metrics/drop_rate",
    "detach_rate": "Metrics/detach_rate",
    "retained_rate": "Metrics/retained_rate",
    "excess_force_failure": "Episode_Reward/excess_force_failure",
}


@dataclass(frozen=True)
class ParsedLog:
    iterations: list[dict[str, float | int | None]]
    requested_last_iteration: int | None
    errors: list[str]
    wandb_urls: list[str]
    simulation_context_cleared: bool


def _metric(block: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s+([-+0-9.eE]+)", block)
    return float(match.group(1)) if match else None


def parse_training_log(text: str) -> ParsedLog:
    """Parse completed iteration blocks without depending on a live process."""

    plain = ANSI.sub("", text)
    matches = list(ITERATION.finditer(plain))
    records: list[dict[str, float | int | None]] = []
    requested_last: int | None = None
    for index, match in enumerate(matches):
        requested_last = int(match.group(2))
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(plain)
        block = plain[match.start():stop]
        if "Iteration time:" not in block:
            continue
        record: dict[str, float | int | None] = {
            "iteration": int(match.group(1)),
        }
        record.update({name: _metric(block, label) for name, label in METRICS.items()})
        records.append(record)

    errors = sorted({match.group(0) for match in ERROR_TOKEN.finditer(plain)})
    urls = [url.rstrip(".,)") for url in re.findall(r"https://wandb\.ai/[^\s\x1b]+", plain)]
    return ParsedLog(
        iterations=records,
        requested_last_iteration=requested_last,
        errors=errors,
        wandb_urls=list(dict.fromkeys(urls)),
        simulation_context_cleared="SimulationContext cleared" in plain,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_record(record: dict[str, float | int | None]) -> bool:
    return all(
        value is None or isinstance(value, int) or math.isfinite(value)
        for value in record.values()
    )


def _candidate_key(item: dict[str, object]) -> tuple[float, ...]:
    """Prefer physical success/retention before reward; missing data ranks last."""

    metrics = item.get("training_metrics")
    values = metrics if isinstance(metrics, dict) else {}

    def present(name: str, default: float) -> float:
        value = values.get(name)
        return float(value) if isinstance(value, (int, float)) else default

    return (
        present("success_rate", -1.0),
        present("retained_rate", -1.0),
        -present("drop_rate", math.inf),
        -present("excess_force_failure", math.inf),
        present("mean_reward", -math.inf),
        float(item["iteration"]),
    )


def inventory_checkpoints(
    run_directory: Path,
    records: Iterable[dict[str, float | int | None]],
) -> list[dict[str, object]]:
    by_iteration = {int(item["iteration"]): item for item in records}
    inventory: list[dict[str, object]] = []
    for path in run_directory.glob("model_*.pt"):
        match = CHECKPOINT.match(path.name)
        if match is None or not path.is_file():
            continue
        iteration = int(match.group(1))
        stat = path.stat()
        inventory.append(
            {
                "iteration": iteration,
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
                "training_metrics": by_iteration.get(iteration),
            }
        )
    return sorted(inventory, key=lambda item: int(item["iteration"]))


def build_audit(
    *,
    log_path: Path,
    run_directory: Path,
    manifest_path: Path | None = None,
    candidate_count: int = 3,
) -> dict[str, object]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least one")
    parsed = parse_training_log(log_path.read_text(encoding="utf-8", errors="replace"))
    manifest = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoints = inventory_checkpoints(run_directory, parsed.iterations)
    ranked = sorted(checkpoints, key=_candidate_key, reverse=True)
    nominated = ranked[:candidate_count]

    latest = parsed.iterations[-1] if parsed.iterations else None
    expected_updates = manifest.get("max_updates") if isinstance(manifest, dict) else None
    reached_expected = (
        isinstance(expected_updates, int)
        and latest is not None
        and int(latest["iteration"]) + 1 >= expected_updates
    )
    clean_log = not parsed.errors and all(_finite_record(item) for item in parsed.iterations)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_id": "LinkAgent-TomatoPick-Physics-v2",
        "audit_boundary": {
            "cpu_only": True,
            "imports_torch_or_isaac": False,
            "queries_gpu": False,
            "connects_hardware": False,
            "mutates_training_artifacts": False,
        },
        "sources": {
            "training_log": str(log_path.resolve()),
            "training_log_sha256": sha256_file(log_path),
            "run_directory": str(run_directory.resolve()),
            "manifest": str(manifest_path.resolve()) if manifest_path else None,
            "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        },
        "training": {
            "completed_iteration_blocks": len(parsed.iterations),
            "latest_iteration": int(latest["iteration"]) if latest else None,
            "requested_last_iteration": parsed.requested_last_iteration,
            "expected_updates_from_manifest": expected_updates,
            "reached_expected_updates": reached_expected,
            "simulation_context_cleared": parsed.simulation_context_cleared,
            "clean_finite_log": clean_log,
            "error_tokens": parsed.errors,
            "latest_metrics": latest,
            "wandb_urls": parsed.wandb_urls,
        },
        "checkpoints": checkpoints,
        "nominated_for_independent_gpu_evaluation": [
            {
                "iteration": item["iteration"],
                "path": item["path"],
                "sha256": item["sha256"],
                "training_metrics": item["training_metrics"],
            }
            for item in nominated
        ],
        "selection_semantics": (
            "Training metrics nominate evaluation candidates only. Independent "
            "multi-seed evaluation is required before naming a best checkpoint."
        ),
        "passed_artifact_integrity": bool(checkpoints) and clean_log,
        "formal_training_or_release_authorized": False,
    }


def render_markdown(audit: dict[str, object]) -> str:
    training = audit["training"]
    assert isinstance(training, dict)
    candidates = audit["nominated_for_independent_gpu_evaluation"]
    assert isinstance(candidates, list)
    lines = [
        "# Physics-v2 training artifact audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "This is a CPU-only artifact audit. It does not evaluate policy behavior and does not authorize release.",
        "",
        "## Training record",
        "",
        f"- Completed iteration blocks: `{training['completed_iteration_blocks']}`",
        f"- Latest iteration: `{training['latest_iteration']}`",
        f"- Clean finite log: `{training['clean_finite_log']}`",
        f"- Expected update count reached: `{training['reached_expected_updates']}`",
        f"- Simulation context cleared: `{training['simulation_context_cleared']}`",
        "",
        "## Candidates for later independent evaluation",
        "",
    ]
    if not candidates:
        lines.append("No valid checkpoint was found.")
    else:
        lines.extend(("| Iteration | SHA-256 |", "|---:|---|"))
        for item in candidates:
            assert isinstance(item, dict)
            lines.append(f"| {item['iteration']} | `{item['sha256']}` |")
    lines.extend(
        (
            "",
            "> Training metrics only nominate these candidates. A multi-seed Isaac evaluation is still required before selecting a best model.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--candidate-count", type=int, default=3)
    args = parser.parse_args(argv)
    audit = build_audit(
        log_path=args.log,
        run_directory=args.run_dir,
        manifest_path=args.manifest,
        candidate_count=args.candidate_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = args.markdown or args.output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": audit["passed_artifact_integrity"]}))
    return 0 if audit["passed_artifact_integrity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
