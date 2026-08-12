#!/usr/bin/env python3
"""Build the audited Physics-v1 convergence datasets and portable report input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/\d+")
FLOAT_RE = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

FIELD_PATTERNS = {
    "total_steps": re.compile(r"Total steps:\s+(\d+)"),
    "steps_per_second": re.compile(r"Steps per second:\s+(\d+)"),
    "value_loss": re.compile(rf"Mean value loss:\s+{FLOAT_RE}"),
    "surrogate_loss": re.compile(rf"Mean surrogate loss:\s+{FLOAT_RE}"),
    "entropy_loss": re.compile(rf"Mean entropy loss:\s+{FLOAT_RE}"),
    "mean_reward": re.compile(rf"Mean reward:\s+{FLOAT_RE}"),
    "mean_episode_length": re.compile(rf"Mean episode length:\s+{FLOAT_RE}"),
    "action_std": re.compile(rf"Mean action std:\s+{FLOAT_RE}"),
    "success_rate": re.compile(rf"Metrics/success_rate:\s+{FLOAT_RE}"),
    "drop_rate": re.compile(rf"Metrics/drop_rate:\s+{FLOAT_RE}"),
    "detach_rate": re.compile(rf"Metrics/detach_rate:\s+{FLOAT_RE}"),
    "retained_rate": re.compile(rf"Metrics/retained_rate:\s+{FLOAT_RE}"),
    "mean_hold_steps": re.compile(rf"Metrics/mean_hold_steps:\s+{FLOAT_RE}"),
    "timeout_rate": re.compile(rf"Metrics/timeout_rate:\s+{FLOAT_RE}"),
}


SEGMENTS = [
    ("from_scratch", "/tmp/tomato_physics_v1_formal_8192_perf_v7.log", 1200),
    ("resume_1200_1800", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume1200.log", 1799),
    ("resume_1800_2200", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume1800_to2200.log", 2198),
    ("resume_2200_2600", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume2200_to2600.log", 2597),
    ("resume_2600_3000", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume2600_to3000.log", 2996),
    ("resume_3000_pause_3250", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume3000_to3400.log", 3250),
    ("resume_3250_3400", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume3250_to3400.log", 3399),
    ("resume_3400_3800", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume3400_to3800_after_pause.log", 3798),
    ("resume_3800_4200", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume3800_to4200_auto.log", 4197),
    ("resume_4200_4600", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume4200_to4600_auto.log", 4596),
    ("resume_4600_5000", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume4600_to5000_auto.log", 4995),
    ("resume_5000_pause_5150", "/tmp/tomato_physics_v1_formal_8192_perf_v7_resume5000_to5400_auto.log", 5150),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path, segment: str, cap: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)
        match = ITER_RE.search(line)
        if match:
            if current is not None and current["iteration"] <= cap:
                rows.append(current)
            current = {"iteration": int(match.group(1)), "segment": segment}
            continue
        if current is None:
            continue
        for field, pattern in FIELD_PATTERNS.items():
            match = pattern.search(line)
            if match:
                current[field] = int(match.group(1)) if field in {"total_steps", "steps_per_second"} else float(match.group(1))
                break
    if current is not None and current["iteration"] <= cap:
        rows.append(current)
    return rows


def load_training_csv(path: Path) -> list[dict[str, Any]]:
    """Load the packaged authoritative curve without requiring /tmp logs."""
    rows: list[dict[str, Any]] = []
    integer_fields = {"iteration", "total_steps", "steps_per_second"}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for field, value in raw.items():
                if value in {None, ""}:
                    continue
                if field == "segment":
                    row[field] = value
                elif field in integer_fields:
                    row[field] = int(value)
                else:
                    row[field] = float(value)
            rows.append(row)
    return rows


def rolling_mean(rows: list[dict[str, Any]], field: str, window: int = 25) -> list[float | None]:
    values: deque[float] = deque()
    total = 0.0
    result: list[float | None] = []
    for row in rows:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            result.append(None)
            continue
        values.append(float(value))
        total += float(value)
        if len(values) > window:
            total -= values.popleft()
        result.append(total / len(values))
    return result


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes <= 0:
        return (float("nan"), float("nan"))
    p = successes / episodes
    denom = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denom
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes)) / denom
    return center - half, center + half


def weighted_rate(payloads: list[dict[str, Any]], count_key: str, episodes: int) -> float | None:
    counts = [payload.get(count_key) for payload in payloads]
    if episodes <= 0 or any(not isinstance(value, (int, float)) for value in counts):
        return None
    return sum(float(value) for value in counts) / episodes


def load_evaluations(root: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("convergence_eval_*/aggregate.json"):
        match = re.search(r"convergence_eval_(\d+)(?:_pause)?$", path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))
    rows: list[dict[str, Any]] = []
    for boundary, aggregate_path in sorted(candidates):
        aggregate = json.loads(aggregate_path.read_text())
        seed_files = sorted(aggregate_path.parent.glob("model_*_seed_*.json"))
        seed_payloads = [json.loads(path.read_text()) for path in seed_files]
        episodes = int(aggregate["completed_episodes"])
        successes = int(aggregate["success_count"])
        seed_rates = [float(item["success_rate"]) for item in aggregate["per_seed"]]
        checkpoint_name = Path(aggregate["checkpoint"]).name
        low, high = wilson(successes, episodes)
        row = {
            "iteration": boundary,
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": aggregate["checkpoint_sha256"],
            "completed_episodes": episodes,
            "success_rate": float(aggregate["success_rate"]),
            "success_ci_low": low,
            "success_ci_high": high,
            "drop_rate": float(aggregate["drop_rate"]),
            "worst_seed_success": min(seed_rates),
            "best_seed_success": max(seed_rates),
            "seed_spread": max(seed_rates) - min(seed_rates),
            "detach_rate": weighted_rate(seed_payloads, "detach_count", episodes),
            "retained_rate": weighted_rate(seed_payloads, "retained_count", episodes),
            "excess_force_rate": weighted_rate(seed_payloads, "excess_force_count", episodes),
            "out_of_bounds_rate": weighted_rate(seed_payloads, "out_of_bounds_count", episodes),
            "timeout_rate": weighted_rate(seed_payloads, "timeout_count", episodes),
            "evaluation_directory": aggregate_path.parent.relative_to(root.parent.parent.parent).as_posix(),
            "per_seed": sorted(aggregate["per_seed"], key=lambda item: int(item["seed"])),
        }
        rows.append(row)
    return rows


def select_best(evaluations: list[dict[str, Any]], tie_pp: float = 0.1) -> tuple[dict[str, Any], dict[str, Any] | None]:
    leader = max(evaluations, key=lambda row: row["success_rate"])
    tie_rate = tie_pp / 100.0
    tied = [row for row in evaluations if leader["success_rate"] - row["success_rate"] <= tie_rate]
    selected = max(tied, key=lambda row: (-row["drop_rate"], row["worst_seed_success"], row["iteration"]))
    runner_up = max((row for row in evaluations if row is not selected), key=lambda row: row["success_rate"], default=None)
    return selected, runner_up


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    metric_definitions: list[str],
    sql: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "sql": sql,
            "description": description,
            "language": "sql",
            "tables_used": [path],
            "filters": ["Only iterations backed by an accepted atomic full-state checkpoint are retained."],
            "metric_definitions": metric_definitions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--training-csv",
        type=Path,
        help="Use the packaged combined curve instead of the original /tmp training logs.",
    )
    parser.add_argument(
        "--generated-at",
        help="Optional ISO-8601 timestamp for byte-reproducible artifact regeneration.",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "logs/qualification/tomato_physics_v1/convergence_report").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.training_csv:
        training_csv = args.training_csv.resolve()
        if not training_csv.is_file():
            raise SystemExit(f"missing packaged training CSV: {training_csv}")
        training_rows = load_training_csv(training_csv)
        inventory = [{
            "segment": "packaged_authoritative_curve",
            "source": training_csv.name,
            "cap_iteration": 5150,
            "parsed_rows": len(training_rows),
            "first_iteration": min((row["iteration"] for row in training_rows), default=None),
            "last_iteration": max((row["iteration"] for row in training_rows), default=None),
        }]
    else:
        merged: dict[int, dict[str, Any]] = {}
        inventory = []
        for segment, raw_path, cap in SEGMENTS:
            path = Path(raw_path)
            if not path.is_file():
                raise SystemExit(f"missing training log: {path}")
            parsed = parse_log(path, segment, cap)
            for row in parsed:
                merged[int(row["iteration"])] = row
            inventory.append({
                "segment": segment,
                "source": path.name,
                "cap_iteration": cap,
                "parsed_rows": len(parsed),
                "first_iteration": min((row["iteration"] for row in parsed), default=None),
                "last_iteration": max((row["iteration"] for row in parsed), default=None),
            })
        training_rows = [merged[key] for key in sorted(merged) if key <= 5150]
    if not training_rows or training_rows[-1]["iteration"] != 5150:
        raise SystemExit("combined curve does not end at authoritative checkpoint iter5150")

    rolling_fields = [
        "success_rate", "retained_rate", "drop_rate", "detach_rate", "timeout_rate",
        "mean_reward", "mean_hold_steps", "value_loss", "surrogate_loss", "entropy_loss",
        "action_std", "steps_per_second",
    ]
    for field in rolling_fields:
        values = rolling_mean(training_rows, field)
        for row, value in zip(training_rows, values):
            row[f"{field}_rolling25"] = value

    eval_root = project_root / "logs/qualification/tomato_physics_v1"
    evaluations = load_evaluations(eval_root)
    selected, runner_up = select_best(evaluations)
    selected_path = next(project_root.glob(f"logs/rsl_rl/tomato_pick_physics_v1/*/{selected['checkpoint']}"), None)
    if selected_path is None or sha256(selected_path) != selected["checkpoint_sha256"]:
        raise SystemExit("selected checkpoint path/hash could not be verified")

    fixed_eval_rows = []
    per_seed_rows = []
    failure_rows = []
    for row in evaluations:
        fixed_eval_rows.append({key: value for key, value in row.items() if key != "per_seed"})
        for seed_row in row["per_seed"]:
            per_seed_rows.append({
                "iteration": row["iteration"],
                "checkpoint": row["checkpoint"],
                "seed": str(seed_row["seed"]),
                "success_rate": float(seed_row["success_rate"]),
                "drop_rate": float(seed_row["drop_rate"]),
                "detach_rate": float(seed_row["detach_rate"]),
                "retained_rate": float(seed_row["retained_rate"]),
            })
        for label, field in (
            ("Drop", "drop_rate"),
            ("Excess force", "excess_force_rate"),
            ("Out of bounds", "out_of_bounds_rate"),
            ("Timeout", "timeout_rate"),
        ):
            if row.get(field) is not None:
                failure_rows.append({"iteration": row["iteration"], "failure_type": label, "rate": row[field]})

    previous100 = [row["success_rate"] for row in training_rows[-200:-100] if "success_rate" in row]
    latest100 = [row["success_rate"] for row in training_rows[-100:] if "success_rate" in row]
    curve_summary = {
        "previous_100_mean_success": sum(previous100) / len(previous100),
        "latest_100_mean_success": sum(latest100) / len(latest100),
    }
    curve_summary["gain"] = curve_summary["latest_100_mean_success"] - curve_summary["previous_100_mean_success"]

    training_fields = ["iteration", "segment"] + list(FIELD_PATTERNS) + [f"{field}_rolling25" for field in rolling_fields]
    write_csv(output_dir / "combined_training_metrics.csv", training_rows, training_fields)
    write_csv(
        output_dir / "checkpoint_evaluations.csv",
        fixed_eval_rows,
        [
            "iteration", "checkpoint", "checkpoint_sha256", "completed_episodes", "success_rate",
            "success_ci_low", "success_ci_high", "drop_rate", "worst_seed_success", "best_seed_success",
            "seed_spread", "detach_rate", "retained_rate", "excess_force_rate", "out_of_bounds_rate",
            "timeout_rate", "evaluation_directory",
        ],
    )
    write_csv(
        output_dir / "checkpoint_per_seed.csv",
        per_seed_rows,
        ["iteration", "checkpoint", "seed", "success_rate", "drop_rate", "detach_rate", "retained_rate"],
    )
    (output_dir / "source_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    selection = {
        "schema_version": 1,
        "status": "selected_after_user_requested_stop",
        "selected_checkpoint": selected_path.relative_to(project_root).as_posix(),
        "selected_checkpoint_iteration_field": selected["iteration"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_rule": "Highest fixed deterministic multi-seed success; candidates within 0.1 percentage point are treated as tied and broken by lower drop rate, then higher worst-seed success, then later iteration.",
        "fixed_evaluation": {
            "seeds": [17, 42, 2026],
            "num_envs_per_seed": 1024,
            "steps_per_seed": 400,
            "completed_episodes": selected["completed_episodes"],
            "success_rate": selected["success_rate"],
            "success_ci_95": [selected["success_ci_low"], selected["success_ci_high"]],
            "drop_rate": selected["drop_rate"],
            "worst_seed_success": selected["worst_seed_success"],
            "seed_spread": selected["seed_spread"],
        },
        "runner_up": None if runner_up is None else {
            "checkpoint": runner_up["checkpoint"],
            "iteration": runner_up["iteration"],
            "success_rate": runner_up["success_rate"],
            "drop_rate": runner_up["drop_rate"],
        },
        "original_success_target": 0.9,
        "formal_target_reached": selected["success_rate"] >= 0.9,
        "formal_curve_plateau_reached": abs(curve_summary["gain"]) <= 0.01,
        "curve": curve_summary,
        "interpretation": "Fixed deterministic evaluation is on an 85-86% plateau, while the latest stochastic training-curve comparison still shows modest growth. The selected checkpoint is the best tested candidate, but neither the original 90% target nor the prior combined formal stop rule was reached; stopping is user-authorized.",
    }
    (output_dir / "best_checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    sample_iterations = {0, 1200, 1800, 2200, 2600, 3000, 3250, 3400, 3800, 4200, 4600, 5000, 5150}
    training_curve = []
    for row in training_rows:
        if row["iteration"] % 5 != 0 and row["iteration"] not in sample_iterations:
            continue
        training_curve.append({
            "iteration": row["iteration"],
            "success_rate": row.get("success_rate_rolling25"),
            "retained_rate": row.get("retained_rate_rolling25"),
            "drop_rate": row.get("drop_rate_rolling25"),
            "detach_rate": row.get("detach_rate_rolling25"),
            "timeout_rate": row.get("timeout_rate_rolling25"),
            "mean_reward": row.get("mean_reward_rolling25"),
            "mean_hold_steps": row.get("mean_hold_steps_rolling25"),
            "value_loss": row.get("value_loss_rolling25"),
            "surrogate_loss": row.get("surrogate_loss_rolling25"),
            "entropy_loss": row.get("entropy_loss_rolling25"),
            "action_std": row.get("action_std_rolling25"),
            "steps_per_second": row.get("steps_per_second_rolling25"),
        })

    generated = args.generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_training = source(
        "training_logs",
        "Combined authoritative PPO training logs",
        "logs/qualification/tomato_physics_v1/convergence_report/combined_training_metrics.csv",
        "Parses the accepted training segments, removes iterations beyond atomic pause checkpoints, deduplicates overlapping resume labels, and computes 25-iteration rolling means.",
        [
            "Training success is Metrics/success_rate reported by RSL-RL for the stochastic training rollout.",
            "Rolling curves are arithmetic means over up to the latest 25 logged iterations.",
            "The x-axis uses the RSL-RL iteration label; resume boundaries can overlap by one label and are deduplicated in favor of the later segment.",
        ],
        "SELECT * FROM read_csv_auto('logs/qualification/tomato_physics_v1/convergence_report/combined_training_metrics.csv') ORDER BY iteration",
    )
    source_eval = source(
        "fixed_evaluations",
        "Deterministic multi-seed checkpoint evaluations",
        "logs/qualification/tomato_physics_v1/convergence_report/checkpoint_evaluations.csv",
        "Combines serial deterministic evaluations for seeds 17, 42, and 2026, each using 1024 environments for 400 steps.",
        [
            "Aggregate success rate equals total successes divided by total completed episodes across all three seeds.",
            "Drop and other failure rates use the same completed-episode denominator.",
            "Wilson 95% intervals describe binomial sampling uncertainty and do not capture simulator or curriculum misspecification.",
        ],
        "SELECT * FROM read_csv_auto('logs/qualification/tomato_physics_v1/convergence_report/checkpoint_evaluations.csv') ORDER BY iteration",
    )
    source_selection = source(
        "checkpoint_selection",
        "Checkpoint selection audit",
        "logs/qualification/tomato_physics_v1/convergence_report/best_checkpoint_selection.json",
        "Applies the documented success-first selection and robust tie-break rule to all fixed-evaluated candidates.",
        [
            "Candidates within 0.1 percentage point of the best success rate are treated as tied.",
            "Ties are broken by lower aggregate drop rate, then higher worst-seed success, then later iteration.",
        ],
        "SELECT fixed_evaluation.success_rate AS selected_success, runner_up.success_rate AS runner_up_success, fixed_evaluation.success_rate - runner_up.success_rate AS success_delta, fixed_evaluation.drop_rate AS selected_drop, runner_up.drop_rate AS runner_up_drop, fixed_evaluation.worst_seed_success, fixed_evaluation.seed_spread, selected_checkpoint_iteration_field AS checkpoint_iteration, original_success_target AS original_target FROM read_json_auto('logs/qualification/tomato_physics_v1/convergence_report/best_checkpoint_selection.json')",
    )
    source_per_seed = source(
        "per_seed_evaluations",
        "Per-seed deterministic checkpoint evaluations",
        "logs/qualification/tomato_physics_v1/convergence_report/checkpoint_per_seed.csv",
        "Loads the reviewed seed-level success, drop, detach, and retention rates for every fixed-evaluated checkpoint.",
        [
            "Each row is one checkpoint and one deterministic evaluation seed.",
            "Seed spread is assessed over seeds 17, 42, and 2026.",
        ],
        "SELECT * FROM read_csv_auto('logs/qualification/tomato_physics_v1/convergence_report/checkpoint_per_seed.csv') ORDER BY iteration, seed",
    )
    source_failures = source(
        "failure_evaluations",
        "Checkpoint failure-rate comparison",
        "logs/qualification/tomato_physics_v1/convergence_report/checkpoint_evaluations.csv",
        "Reshapes reviewed aggregate checkpoint failure rates into a failure-type series.",
        [
            "Failure rates divide each event count by completed episodes across the three evaluation seeds.",
            "Failure categories are logged independently and are not assumed mutually exclusive.",
        ],
        "WITH e AS (SELECT * FROM read_csv_auto('logs/qualification/tomato_physics_v1/convergence_report/checkpoint_evaluations.csv')) SELECT iteration, 'Drop' AS failure_type, drop_rate AS rate FROM e UNION ALL SELECT iteration, 'Excess force', excess_force_rate FROM e UNION ALL SELECT iteration, 'Out of bounds', out_of_bounds_rate FROM e UNION ALL SELECT iteration, 'Timeout', timeout_rate FROM e ORDER BY iteration, failure_type",
    )
    source_audit = source(
        "checkpoint_audit",
        "Checkpoint selection audit table",
        "logs/qualification/tomato_physics_v1/convergence_report/checkpoint_evaluations.csv",
        "Joins the fixed-evaluation comparison with the selected-checkpoint manifest and produces the exact audit table.",
        [
            "Selection is success-first with a 0.1 percentage-point tie band, then lower drop, higher worst-seed success, and later iteration.",
            "SHA256 is displayed as a 12-character prefix; the full digest is preserved in the source CSV and selection JSON.",
        ],
        "WITH e AS (SELECT * FROM read_csv_auto('logs/qualification/tomato_physics_v1/convergence_report/checkpoint_evaluations.csv')), s AS (SELECT selected_checkpoint_sha256 FROM read_json_auto('logs/qualification/tomato_physics_v1/convergence_report/best_checkpoint_selection.json')) SELECT e.iteration, e.checkpoint, e.success_rate, e.drop_rate, e.worst_seed_success, e.seed_spread, e.completed_episodes AS episodes, substr(e.checkpoint_sha256, 1, 12) AS sha256, CASE WHEN e.checkpoint_sha256 = s.selected_checkpoint_sha256 THEN 'BEST' ELSE '' END AS selected FROM e CROSS JOIN s ORDER BY e.iteration",
    )
    sources = [source_training, source_eval, source_per_seed, source_failures, source_selection, source_audit]

    headline = [{
        "selected_success": selected["success_rate"],
        "runner_up_success": runner_up["success_rate"] if runner_up else None,
        "success_delta": selected["success_rate"] - runner_up["success_rate"] if runner_up else None,
        "selected_drop": selected["drop_rate"],
        "runner_up_drop": runner_up["drop_rate"] if runner_up else None,
        "worst_seed_success": selected["worst_seed_success"],
        "seed_spread": selected["seed_spread"],
        "checkpoint_iteration": selected["iteration"],
        "original_target": 0.9,
    }]
    audit_rows = [
        {
            "iteration": row["iteration"],
            "checkpoint": row["checkpoint"],
            "success_rate": row["success_rate"],
            "drop_rate": row["drop_rate"],
            "worst_seed_success": row["worst_seed_success"],
            "seed_spread": row["seed_spread"],
            "episodes": row["completed_episodes"],
            "sha256": row["checkpoint_sha256"][:12],
            "selected": "BEST" if row["iteration"] == selected["iteration"] else "",
        }
        for row in evaluations
    ]

    cards = [
        {
            "id": "success_card",
            "description": "Best fixed deterministic multi-seed success and its nearest competitor.",
            "dataset": "headline",
            "sourceId": "checkpoint_selection",
            "metrics": [
                {"label": "Selected success", "field": "selected_success", "format": "percent"},
                {"label": "Runner-up", "field": "runner_up_success", "format": "percent"},
                {"label": "Delta", "field": "success_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "drop_card",
            "description": "Drop failures at the selected checkpoint versus the runner-up.",
            "dataset": "headline",
            "sourceId": "checkpoint_selection",
            "metrics": [
                {"label": "Selected drop", "field": "selected_drop", "format": "percent"},
                {"label": "Runner-up", "field": "runner_up_drop", "format": "percent"},
            ],
        },
        {
            "id": "worst_seed_card",
            "description": "The weakest evaluated seed and cross-seed spread for the selected checkpoint.",
            "dataset": "headline",
            "sourceId": "checkpoint_selection",
            "metrics": [
                {"label": "Worst-seed success", "field": "worst_seed_success", "format": "percent"},
                {"label": "Seed spread", "field": "seed_spread", "format": "percent"},
            ],
        },
        {
            "id": "checkpoint_card",
            "description": "Latest verified full-state pause checkpoint; original fixed-evaluation target remains 90%.",
            "dataset": "headline",
            "sourceId": "checkpoint_selection",
            "metrics": [
                {"label": "Checkpoint iteration", "field": "checkpoint_iteration", "format": "number"},
                {"label": "Original target", "field": "original_target", "format": "percent"},
            ],
        },
    ]

    charts = [
        {
            "id": "training_outcomes",
            "title": "Combined PPO outcome curve",
            "subtitle": "25-iteration rolling means across accepted full-state segments; RSL-RL iteration labels 0-5150",
            "intent": "trend",
            "question": "How did success, retention, drop, and detachment evolve across the complete accepted training history?",
            "rationale": "A multi-series line chart preserves the shared iteration axis and makes the plateau and failure trade-off visible.",
            "type": "line",
            "dataset": "training_curve",
            "sourceId": "training_logs",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "RSL-RL iteration label"},
                "y": {"fields": ["success_rate", "retained_rate", "drop_rate", "detach_rate"], "type": "quantitative", "label": "Rate", "format": "percent"},
                "tooltip": [
                    {"field": "success_rate", "type": "quantitative", "label": "Success", "format": "percent"},
                    {"field": "drop_rate", "type": "quantitative", "label": "Drop", "format": "percent"},
                ],
            },
            "valueFormat": "percent",
            "referenceLines": [{"axis": "y", "value": 0.9, "label": "90% fixed-eval target", "color": "orange", "lineStyle": "dashed"}],
            "layout": "full",
        },
        {
            "id": "checkpoint_success",
            "title": "Fixed evaluation success by checkpoint",
            "subtitle": "Three deterministic seeds; 1024 environments x 400 steps per seed",
            "intent": "trend",
            "question": "Which evaluated checkpoint has the highest aggregate deterministic success?",
            "rationale": "A checkpoint-indexed line shows the large gains, the seed-17 regression at 3000, and the later 85-86% plateau.",
            "type": "line",
            "dataset": "fixed_evaluations",
            "sourceId": "fixed_evaluations",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "Checkpoint iteration"},
                "y": {"field": "success_rate", "type": "quantitative", "label": "Aggregate success", "format": "percent"},
                "tooltip": [
                    {"field": "drop_rate", "type": "quantitative", "label": "Drop", "format": "percent"},
                    {"field": "completed_episodes", "type": "quantitative", "label": "Episodes"},
                ],
            },
            "valueFormat": "percent",
            "referenceLines": [{"axis": "y", "value": 0.9, "label": "Original target", "color": "orange", "lineStyle": "dashed"}],
            "layout": "half",
        },
        {
            "id": "per_seed_success",
            "title": "Per-seed success stability",
            "subtitle": "The selected checkpoint keeps every tested seed above 85.6%",
            "intent": "trend",
            "question": "Did improvement generalize across seeds or come from one favorable seed?",
            "rationale": "Three seed lines expose the checkpoint-3000 seed-17 failure and confirm the selected checkpoint is not seed-specific.",
            "type": "line",
            "dataset": "per_seed",
            "sourceId": "per_seed_evaluations",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "Checkpoint iteration"},
                "y": {"field": "success_rate", "type": "quantitative", "label": "Success", "format": "percent"},
                "color": {"field": "seed", "type": "nominal", "label": "Seed"},
            },
            "valueFormat": "percent",
            "layout": "half",
        },
        {
            "id": "failure_profile",
            "title": "Failure rates across evaluated checkpoints",
            "subtitle": "Drop remains the largest measured failure mode at the selected checkpoint",
            "intent": "trend",
            "question": "Which failure modes remain after the success plateau?",
            "rationale": "A multi-series trend separates drop, excess-force, out-of-bounds, and timeout failure rates without hiding their distinct trajectories.",
            "type": "line",
            "dataset": "failures",
            "sourceId": "failure_evaluations",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "Checkpoint iteration"},
                "y": {"field": "rate", "type": "quantitative", "label": "Failure rate", "format": "percent"},
                "color": {"field": "failure_type", "type": "nominal", "label": "Failure type"},
            },
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "reward_health",
            "title": "Reward and hold stability",
            "subtitle": "25-iteration rolling training reward and mean hold duration",
            "intent": "trend",
            "question": "Did reward or hold behavior destabilize while success flattened?",
            "rationale": "These similarly scaled operational measures reveal whether plateauing success coincided with a collapse in reward or retention duration.",
            "type": "line",
            "dataset": "training_curve",
            "sourceId": "training_logs",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "RSL-RL iteration label"},
                "y": {"fields": ["mean_reward", "mean_hold_steps"], "type": "quantitative", "label": "Rolling value"},
            },
            "layout": "half",
        },
        {
            "id": "critic_health",
            "title": "Critic loss",
            "subtitle": "25-iteration rolling mean value loss",
            "intent": "trend",
            "question": "Did the value estimator remain numerically finite through the stop point?",
            "rationale": "A dedicated loss axis avoids compressing policy losses that are orders of magnitude smaller.",
            "type": "line",
            "dataset": "training_curve",
            "sourceId": "training_logs",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "RSL-RL iteration label"},
                "y": {"field": "value_loss", "type": "quantitative", "label": "Value loss"},
            },
            "layout": "half",
        },
        {
            "id": "policy_health",
            "title": "Policy entropy and action dispersion",
            "subtitle": "25-iteration rolling entropy loss and mean action standard deviation",
            "intent": "trend",
            "question": "How did policy exploration change as evaluation success plateaued?",
            "rationale": "The two exploration indicators share direction and a manageable scale ratio, making a paired trend interpretable.",
            "type": "line",
            "dataset": "training_curve",
            "sourceId": "training_logs",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "RSL-RL iteration label"},
                "y": {"fields": ["entropy_loss", "action_std"], "type": "quantitative", "label": "Rolling value"},
            },
            "layout": "half",
        },
        {
            "id": "throughput_health",
            "title": "Training throughput",
            "subtitle": "25-iteration rolling simulation steps per second",
            "intent": "trend",
            "question": "Was the late plateau caused by a throughput collapse or runtime degradation?",
            "rationale": "A dedicated throughput series makes late-run performance degradation directly visible without mixing units.",
            "type": "line",
            "dataset": "training_curve",
            "sourceId": "training_logs",
            "encodings": {
                "x": {"field": "iteration", "type": "quantitative", "label": "RSL-RL iteration label"},
                "y": {"field": "steps_per_second", "type": "quantitative", "label": "Steps per second"},
            },
            "layout": "half",
        },
    ]

    tables = [{
        "id": "checkpoint_audit",
        "title": "Fixed-evaluated checkpoint audit",
        "subtitle": "Exact aggregate metrics for every boundary candidate and the user-authorized pause checkpoint",
        "dataset": "checkpoint_audit",
        "sourceId": "checkpoint_audit",
        "defaultSort": {"field": "iteration", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "iteration", "label": "Iter", "format": "number"},
            {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
            {"field": "success_rate", "label": "Success", "format": "percent"},
            {"field": "drop_rate", "label": "Drop", "format": "percent"},
            {"field": "worst_seed_success", "label": "Worst seed", "format": "percent"},
            {"field": "seed_spread", "label": "Seed spread", "format": "percent"},
            {"field": "episodes", "label": "Episodes", "format": "number"},
            {"field": "sha256", "label": "SHA256 prefix", "type": "text"},
            {"field": "selected", "label": "Selection", "type": "text"},
        ],
    }]

    selected_ci = selected["success_ci_low"], selected["success_ci_high"]
    delta_pp = (selected["success_rate"] - runner_up["success_rate"]) * 100 if runner_up else 0.0
    drop_delta_pp = (selected["drop_rate"] - runner_up["drop_rate"]) * 100 if runner_up else 0.0
    curve_gain_pp = curve_summary["gain"] * 100
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Physics-v1 PPO checkpoint selection and convergence review"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## The run is safely stopped; model_5150.pt is the best tested checkpoint\n\n"
                f"- **Selected checkpoint:** `model_5150.pt` at iter 5150, SHA256 `{selected['checkpoint_sha256'][:16]}...`; the Actor, Critic, both observation normalizers, policy standard deviation, and Adam state were verified before shutdown.\n"
                f"- **Fixed evaluation:** {selected['success_rate']:.2%} success across {selected['completed_episodes']:,} completed episodes (95% Wilson interval {selected_ci[0]:.2%}-{selected_ci[1]:.2%}), with {selected['drop_rate']:.2%} drop.\n"
                f"- **Why this checkpoint:** success is only {delta_pp:+.3f} percentage points versus the 5000 checkpoint, so they are effectively tied; iter 5150 wins the declared tie-break because drop is {drop_delta_pp:+.3f} percentage points lower and its worst tested seed is {selected['worst_seed_success']:.2%}.\n"
                f"- **Convergence interpretation:** the latest 100 training iterations averaged {curve_summary['latest_100_mean_success']:.2%} success versus {curve_summary['previous_100_mean_success']:.2%} for the preceding 100 ({curve_gain_pp:+.3f} percentage points). The deterministic evaluation is flat near 86%, but the stochastic training curve is still modestly rising; the original 90% target and prior combined stop rule were not reached. Shutdown is therefore an explicit user-authorized stop, not formal target convergence."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["success_card", "drop_card", "worst_seed_card", "checkpoint_card"]},
        {
            "id": "curve_finding",
            "type": "markdown",
            "body": (
                "## Training success still rose modestly, but fixed evaluation stopped following it\n\n"
                f"The combined curve below keeps only iterations supported by accepted atomic checkpoints and joins every full-state continuation segment. Detachment remained high while success and retention rose. The latest 100-iteration training mean is {curve_gain_pp:.2f} percentage points above the preceding 100, yet deterministic evaluation stayed near 86%; this divergence is why checkpoint choice relies on fixed evaluation rather than the noisy stochastic training curve. The 25-iteration mean reduces rollout noise without hiding segment-boundary behavior."
            ),
            "sourceId": "training_logs",
        },
        {"id": "training_outcomes_block", "type": "chart", "chartId": "training_outcomes", "layout": "full"},
        {
            "id": "evaluation_finding",
            "type": "markdown",
            "body": (
                "## Fixed evaluation confirms a stable 85-86% ceiling\n\n"
                "Boundary evaluations improved from 69.3% at iter 2200 to about 85% by iter 3400. The iter-3000 aggregate dip was a seed-17-specific regression; later checkpoints recovered across all three seeds. From iter 3400 onward, aggregate success remained in the 84.7-86.0% range, so the final selection is a robust tie-break rather than a large late breakthrough."
            ),
            "sourceId": "fixed_evaluations",
        },
        {"id": "checkpoint_success_block", "type": "chart", "chartId": "checkpoint_success", "layout": "half"},
        {"id": "per_seed_success_block", "type": "chart", "chartId": "per_seed_success", "layout": "half"},
        {
            "id": "failure_finding",
            "type": "markdown",
            "body": (
                "## Drop is lower at the selected checkpoint, but remains the largest measured failure mode\n\n"
                f"At iter 5150, drop is {selected['drop_rate']:.2%}, excess-force failure is {selected['excess_force_rate']:.2%}, out-of-bounds is {selected['out_of_bounds_rate']:.2%}, and timeout is {selected['timeout_rate']:.2%}. The practical next optimization target is therefore retention/drop behavior, not detachment itself. These categories are logged independently and should not be assumed mutually exclusive without episode-level labels."
            ),
            "sourceId": "failure_evaluations",
        },
        {"id": "failure_profile_block", "type": "chart", "chartId": "failure_profile", "layout": "full"},
        {
            "id": "health_finding",
            "type": "markdown",
            "body": (
                "## Runtime stayed finite while exploration continued to rise\n\n"
                "Reward, critic loss, entropy, action standard deviation, and throughput remained finite through the pause checkpoint. The late action standard deviation near 1.0 shows that the stochastic training policy was still broad even though deterministic fixed evaluation had plateaued. This supports checkpoint selection by deterministic evaluation rather than by a single noisy training iteration."
            ),
            "sourceId": "training_logs",
        },
        {"id": "reward_health_block", "type": "chart", "chartId": "reward_health", "layout": "half"},
        {"id": "critic_health_block", "type": "chart", "chartId": "critic_health", "layout": "half"},
        {"id": "policy_health_block", "type": "chart", "chartId": "policy_health", "layout": "half"},
        {"id": "throughput_health_block", "type": "chart", "chartId": "throughput_health", "layout": "half"},
        {
            "id": "definitions",
            "type": "markdown",
            "body": (
                "## Scope and metric definitions\n\n"
                "**Training curves** are stochastic on-policy rollout metrics from seed 42 with 8192 environments and 32 rollout steps. **Fixed evaluation** uses the deterministic policy for seeds 17, 42, and 2026, each with 1024 environments and 400 simulator steps. Success and failure rates divide event counts by completed episodes. The combined x-axis uses RSL-RL iteration labels; resume segments overlap at boundary labels, so later full-state segments replace earlier duplicates."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Selection method and robustness checks\n\n"
                "All candidate comparisons use saved serial evaluations from the same deterministic protocol. The selection rule ranks success first, treats differences within 0.1 percentage point as tied, then prefers lower drop, higher worst-seed success, and finally the later checkpoint. Wilson intervals quantify episode-count uncertainty. Full-state integrity was checked independently from performance: the selected checkpoint contains Actor, Critic, both normalizers, the policy log-standard-deviation parameter, and a populated Adam optimizer state."
            ),
            "sourceId": "checkpoint_selection",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## What this result does and does not establish\n\n"
                "The 5150-versus-5000 success difference is much smaller than the sampling interval and should not be interpreted as a statistically meaningful improvement. The evaluation covers three simulator seeds under one fixed physics/reward/MDP configuration; it does not establish sim-to-real transfer or hardware safety. No SDK, CAN, or real hardware was used. The original >=90% gate remains unmet."
            ),
        },
        {
            "id": "audit_table_intro",
            "type": "markdown",
            "body": (
                "## Audited checkpoint comparison\n\n"
                "The table provides exact selection inputs for every checkpoint that received the fixed multi-seed evaluation. The full SHA256 values and checkpoint paths are preserved in the companion selection JSON."
            ),
            "sourceId": "checkpoint_audit",
        },
        {"id": "checkpoint_audit_block", "type": "table", "tableId": "checkpoint_audit", "layout": "full"},
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next step\n\n"
                "Use `model_5150.pt` as the frozen best checkpoint for downstream simulator validation. Preserve `model_4995.pt` as the near-tied fallback. If optimization resumes later, start only from the verified full-state iter-5150 checkpoint and treat retention/drop failure as the primary diagnostic target; do not claim the 90% gate has passed without a new fixed multi-seed result."
            ),
            "sourceId": "checkpoint_selection",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Would a failure-stratified audit reveal a narrow retention condition that explains the remaining drop rate?\n"
                "- Does the selected deterministic policy remain stable under a broader held-out simulator seed set?\n"
                "- If training resumes, does reducing exploration after the plateau improve fixed success without changing physics, rewards, or acceptance thresholds?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Physics-v1 PPO checkpoint selection and convergence review",
            "description": "Audited full-history training curves, fixed multi-seed checkpoint comparison, and best-checkpoint selection after the user-authorized stop.",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "training_curve": training_curve,
                "fixed_evaluations": fixed_eval_rows,
                "per_seed": per_seed_rows,
                "failures": failure_rows,
                "checkpoint_audit": audit_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (output_dir / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")

    notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "required_structure_mapping": {
            "technical_summary": "technical_summary",
            "key_findings": ["curve_finding", "evaluation_finding", "failure_finding", "health_finding"],
            "scope_data_definitions": "definitions",
            "methodology": "methodology",
            "limitations_robustness": "limitations",
            "recommended_next_steps": "next_steps",
            "further_questions": "further_questions",
        },
        "chart_map": [
            {"chart": chart["id"], "question": chart["question"], "type": chart["type"], "supported_claim": chart["subtitle"]}
            for chart in charts
        ],
        "omissions": [
            "No real-hardware or sim-to-real claim is included because no hardware was connected.",
            "The 5150-versus-5000 difference is reported as an effective tie rather than a causal improvement.",
        ],
    }
    (output_dir / "report_source_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "training_rows": len(training_rows),
        "evaluation_candidates": len(evaluations),
        "selected_checkpoint": str(selected_path),
        "selected_success_rate": selected["success_rate"],
        "selected_drop_rate": selected["drop_rate"],
        "curve_gain": curve_summary["gain"],
    }, indent=2))


if __name__ == "__main__":
    main()
