#!/usr/bin/env python3
"""汇总 firm_grasp_v1 episode 结果；只读 evidence，不启动仿真。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTCOMES = ("SUCCESS", "FAILURE", "UNKNOWN")
SCENARIOS = ("quiet", "force", "torque")
F4_CASES = set(range(20))
F4_SCENARIO_COUNTS = {"quiet": 8, "force": 6, "torque": 6}
INITIALIZATION_IDENTITY_FIELDS = (
    "kind",
    "classification",
    "checkpoint_path",
    "checkpoint_sha256",
    "actor_output_layer",
    "weight_sha256_before",
    "weight_sha256_after",
    "bias_sha256_before",
    "bias_sha256_after",
    "new_task_identity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignment(value: str, description: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{description} 格式应为 LABEL=VALUE")
    label, assigned = value.split("=", 1)
    if not label or not assigned:
        raise argparse.ArgumentTypeError(f"{description} 的 LABEL 和 VALUE 均不能为空")
    return label, assigned


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return value


def read_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                rows.append({
                    "source_line": line_number,
                    "in_denominator": True,
                    "envelope": None,
                    "raw_episode": raw.rstrip("\n"),
                    "read_error": f"JSON 解析失败：{exc}",
                })
                continue
            if not isinstance(value, dict):
                rows.append({
                    "source_line": line_number,
                    "in_denominator": True,
                    "envelope": None,
                    "raw_episode": value,
                    "read_error": "episode 外层不是 JSON 对象",
                })
                continue
            rows.append({
                "source_line": line_number,
                "in_denominator": value.get("in_requested_denominator") is not False,
                "envelope": value,
                "raw_episode": value,
                "read_error": None,
            })
    return rows


def record_for(envelope: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        return {}
    nested = envelope.get("record")
    return nested if isinstance(nested, dict) else envelope


def scenario_for(envelope: dict[str, Any], record: dict[str, Any]) -> str | None:
    scenario = record.get("matrix_scenario", envelope.get("matrix_scenario"))
    if isinstance(scenario, dict):
        mode = scenario.get("mode")
        if mode in SCENARIOS:
            return mode
    protocol = record.get("f3_protocol", envelope.get("f3_protocol"))
    if isinstance(protocol, dict):
        mode = protocol.get("disturbance_mode")
        if mode in SCENARIOS:
            return mode
    for source in (record, envelope):
        mode = source.get("mode")
        if mode in SCENARIOS:
            return mode
    return None


def outcome_for(envelope: dict[str, Any], record: dict[str, Any]) -> str:
    value = envelope.get("outcome")
    if value in OUTCOMES:
        return value
    if value is not None:
        return "UNKNOWN"
    success = record.get("success")
    if isinstance(success, bool):
        return "SUCCESS" if success else "FAILURE"
    return "UNKNOWN"


def checkpoint_identity(metadata: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    initialization = metadata.get("initialization")
    init = initialization if isinstance(initialization, dict) else {}
    identity: dict[str, Any] = {
        "policy": metadata.get("policy"),
        "checkpoint_path": metadata.get("checkpoint_path", init.get("checkpoint_path")),
        "checkpoint_sha256": metadata.get("checkpoint_sha256", init.get("checkpoint_sha256")),
        "actor_seed_checkpoint_path": metadata.get(
            "actor_seed_checkpoint_path", init.get("actor_seed_checkpoint_path")
        ),
        "actor_warmstart_path": metadata.get(
            "actor_warmstart_path", init.get("actor_warmstart_path")
        ),
        "warmstart_action_offset": metadata.get(
            "warmstart_action_offset", init.get("warmstart_action_offset")
        ),
        "initialization": (
            {key: init.get(key) for key in INITIALIZATION_IDENTITY_FIELDS if key in init}
            if isinstance(initialization, dict)
            else None
        ),
        "declared_checkpoint_files": [],
    }
    declared: list[str] = []
    for source in (metadata, init):
        for key in ("checkpoint_path", "actor_seed_checkpoint_path", "actor_warmstart_path"):
            value = source.get(key)
            if isinstance(value, str) and value and value not in declared:
                declared.append(value)
    for value in declared:
        path = Path(value)
        if not path.is_absolute():
            path = metadata_path.parent / path
        file_entry: dict[str, Any] = {"declared_path": value, "sha256": None}
        if path.is_file():
            file_entry["sha256"] = sha256_file(path)
            file_entry["status"] = "HASHED_READ_ONLY"
        else:
            file_entry["status"] = "DECLARED_FILE_NOT_FOUND"
        identity["declared_checkpoint_files"].append(file_entry)
    return identity


def failure_reasons(envelope: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    fields = ("reason", "reason_code", "failure_phase", "record_status")
    return {
        key: record.get(key, envelope.get(key))
        for key in fields
        if key in record or key in envelope
    }


def summarize_one(
    label: str,
    output_dir: Path,
    expected_quiet: int | None,
) -> dict[str, Any]:
    episode_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "run_metadata.json"
    missing = [str(path) for path in (episode_path, summary_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label}: 缺少必需 evidence：{', '.join(missing)}")

    summary = read_json_object(summary_path)
    metadata = read_json_object(metadata_path)
    source_rows = read_episode_rows(episode_path)
    denominator: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    matrix_fields_seen = False
    for source in source_rows:
        envelope = source["envelope"]
        record = record_for(envelope)
        has_matrix_fields = isinstance(envelope, dict) and any(
            key in envelope or key in record for key in ("matrix_case_id", "matrix_scenario")
        )
        matrix_fields_seen = matrix_fields_seen or has_matrix_fields
        scenario = scenario_for(envelope, record) if isinstance(envelope, dict) else None
        case_id = record.get("matrix_case_id", envelope.get("matrix_case_id")) if isinstance(envelope, dict) else None
        env_id = record.get("env_id", envelope.get("env_id")) if isinstance(envelope, dict) else None
        matrix_scenario = (
            record.get("matrix_scenario", envelope.get("matrix_scenario"))
            if isinstance(envelope, dict) else None
        )
        protocol = (
            record.get("f3_protocol", envelope.get("f3_protocol"))
            if isinstance(envelope, dict) else None
        )
        outcome = outcome_for(envelope, record) if isinstance(envelope, dict) else "UNKNOWN"
        entry = {
            "source_line": source["source_line"],
            "in_denominator": source["in_denominator"],
            "case_id": case_id,
            "env_id": env_id,
            "scenario": scenario,
            "matrix_scenario": matrix_scenario,
            "f3_protocol": protocol,
            "outcome": outcome,
            "reason": failure_reasons(envelope, record) if isinstance(envelope, dict) else {},
            "read_error": source["read_error"],
            "raw_episode": source["raw_episode"],
        }
        if source["in_denominator"]:
            denominator.append(entry)
            if outcome != "SUCCESS":
                failures.append(entry)

    evaluation = summary.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    requested = evaluation.get("requested_denominator")
    reported_rows = evaluation.get("denominator_rows")
    counts: dict[str, dict[str, int]] = {
        name: {"denominator": 0, **{outcome: 0 for outcome in OUTCOMES}}
        for name in (*SCENARIOS, "unclassified")
    }
    for entry in denominator:
        group = entry["scenario"] if entry["scenario"] in SCENARIOS else "unclassified"
        counts[group]["denominator"] += 1
        counts[group][entry["outcome"]] += 1

    task_identity = metadata.get("task_identity")
    metadata_matrix = metadata.get("f4_matrix") is not None or (
        isinstance(task_identity, dict) and task_identity.get("f4_matrix") is not None
    )
    evaluation_name = f"{label} {output_dir.name}".lower()
    named_matrix = "matrix" in evaluation_name  # F4_open 等非矩阵也属于 F4 阶段，不能只按阶段名前缀判定
    is_matrix = metadata_matrix or matrix_fields_seen or named_matrix
    early_screen = bool(re.search(r"early[-_ ]window[-_ ]screen", evaluation_name))
    expected_kind = "F4_MATRIX" if is_matrix else "NON_MATRIX_EPISODES"
    if early_screen:
        expected_kind = "DIAGNOSTIC_EARLY_WINDOW_SCREEN"

    reasons: list[str] = []
    if early_screen:
        reasons.append("early-window_screen 是诊断筛查，不能计作完整 G2 evaluation")
    if is_matrix and expected_quiet is not None:
        reasons.append("--expect-quiet 仅用于非矩阵评估，不能套用于 F4 矩阵")
    if summary.get("status") != "EVALUATION_COMPLETE":
        reasons.append(f"summary.status 不是 EVALUATION_COMPLETE（实际：{summary.get('status')!r}）")
    if requested is None and not is_matrix and expected_quiet is None:
        reasons.append("没有 summary requested_denominator 或显式期望 quiet 分母")
    expected_denominator = 20 if is_matrix else expected_quiet
    if expected_denominator is None:
        expected_denominator = requested
    actual_denominator = len(denominator)
    if expected_denominator is not None and actual_denominator != expected_denominator:
        reasons.append(
            f"实际分母 {actual_denominator} 与期望分母 {expected_denominator} 不符"
        )
    if requested is not None and requested != actual_denominator:
        reasons.append(f"summary requested_denominator={requested} 与实际分母不符")
    if reported_rows is not None and reported_rows != actual_denominator:
        reasons.append(f"summary denominator_rows={reported_rows} 与实际分母不符")
    if any(entry["outcome"] == "UNKNOWN" for entry in denominator):
        reasons.append("分母中存在 UNKNOWN outcome")

    matrix_check: dict[str, Any] | None = None
    if is_matrix:
        case_ids = [entry["case_id"] for entry in denominator]
        integer_ids = [value for value in case_ids if isinstance(value, int) and not isinstance(value, bool)]
        id_counts = Counter(integer_ids)
        missing_ids = sorted(F4_CASES - set(integer_ids))
        duplicate_ids = sorted(case_id for case_id, count in id_counts.items() if count > 1)
        unexpected_ids = sorted(set(integer_ids) - F4_CASES)
        scenario_counts = {
            scenario: counts[scenario]["denominator"] for scenario in SCENARIOS
        }
        protocol_mismatches: list[dict[str, Any]] = []
        for entry in denominator:
            matrix_scenario = entry["matrix_scenario"]
            protocol = entry["f3_protocol"]
            if not isinstance(matrix_scenario, dict) or not isinstance(protocol, dict):
                protocol_mismatches.append({
                    "case_id": entry["case_id"],
                    "issue": "matrix_scenario 或 f3_protocol 缺失",
                })
                continue
            expected_protocol = {
                "disturbance_mode": matrix_scenario.get("mode"),
                "disturbance_axis": matrix_scenario.get("axis"),
                "disturbance_sign": matrix_scenario.get("sign"),
            }
            actual_protocol = {key: protocol.get(key) for key in expected_protocol}
            if actual_protocol != expected_protocol:
                protocol_mismatches.append({
                    "case_id": entry["case_id"],
                    "matrix_scenario": matrix_scenario,
                    "f3_protocol": actual_protocol,
                })
        matrix_check = {
            "expected_case_ids": list(range(20)),
            "unique_case_count": len(set(integer_ids)),
            "missing_case_ids": missing_ids,
            "duplicate_case_ids": duplicate_ids,
            "unexpected_case_ids": unexpected_ids,
            "null_or_noninteger_case_rows": sum(
                not isinstance(value, int) or isinstance(value, bool) for value in case_ids
            ),
            "scenario_counts": scenario_counts,
            "expected_scenario_counts": F4_SCENARIO_COUNTS,
            "scenario_protocol_mismatches": protocol_mismatches,
            "cases": [
                {
                    "case_id": entry["case_id"],
                    "env_id": entry["env_id"],
                    "matrix_scenario": entry["matrix_scenario"],
                    "f3_protocol": entry["f3_protocol"],
                    "outcome": entry["outcome"],
                }
                for entry in denominator
            ],
        }
        if actual_denominator != 20:
            reasons.append(f"F4 实际分母为 {actual_denominator}，应为 20")
        if missing_ids or duplicate_ids or unexpected_ids or matrix_check["null_or_noninteger_case_rows"]:
            reasons.append("F4 case_id 缺失、重复或超出 0..19")
        if scenario_counts != F4_SCENARIO_COUNTS:
            reasons.append(f"F4 实际场景分布不是 8/6/6：{scenario_counts}")
        if protocol_mismatches:
            reasons.append(f"F4 matrix_scenario 与实际 f3_protocol 不一致或字段缺失：{len(protocol_mismatches)} 行")
    elif expected_quiet is not None:
        non_quiet = sum(counts[name]["denominator"] for name in ("force", "torque", "unclassified"))
        if counts["quiet"]["denominator"] != expected_quiet or non_quiet:
            reasons.append(
                f"显式 quiet 评估需恰有 {expected_quiet} 条 quiet 且无其他场景"
            )

    metadata_identity = checkpoint_identity(metadata, metadata_path)
    return {
        "label": label,
        "input_dir": str(output_dir),
        "evaluation_kind": expected_kind,
        "completeness": "INCOMPLETE" if reasons else "COMPLETE",
        "completeness_reasons": reasons,
        "source": {
            "episodes_jsonl": str(episode_path),
            "summary_json": str(summary_path),
            "run_metadata_json": str(metadata_path),
            "summary_status": summary.get("status"),
            "requested_denominator": requested,
            "reported_denominator_rows": reported_rows,
            "observed_jsonl_records": len(source_rows),
            "observed_denominator_rows": actual_denominator,
            "outside_denominator_records": len(source_rows) - actual_denominator,
        },
        "checkpoint_identity": metadata_identity,
        "groups": counts,
        "f4_matrix_check": matrix_check,
        "failures_and_unknowns": failures,
    }


def md(value: Any) -> str:
    if value is None:
        return "null"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# F3/F4 Episode 汇总",
        "",
        "本表逐行读取各评估目录的 `episodes.jsonl`，按实际 episode 分母汇总。它记录观察到的结果，不证明 force closure、学习收益或可靠性显著性。诊断 early-window_screen 不计作完整 G2 evaluation。",
        "",
        f"生成时间（UTC）：{payload['generated_at_utc']}",
    ]
    for evaluation in payload["evaluations"]:
        lines.extend([
            "",
            f"## {md(evaluation['label'])}",
            "",
            f"类型：`{evaluation['evaluation_kind']}`；完整性：**{evaluation['completeness']}**",
            "",
            "| 场景 | 分母 | SUCCESS | FAILURE | UNKNOWN |",
            "|---|---:|---:|---:|---:|",
        ])
        for group in (*SCENARIOS, "unclassified"):
            item = evaluation["groups"][group]
            lines.append(
                f"| {group} | {item['denominator']} | {item['SUCCESS']} | {item['FAILURE']} | {item['UNKNOWN']} |"
            )
        lines.append(
            f"| 总计 | {evaluation['source']['observed_denominator_rows']} | "
            f"{sum(g['SUCCESS'] for g in evaluation['groups'].values())} | "
            f"{sum(g['FAILURE'] for g in evaluation['groups'].values())} | "
            f"{sum(g['UNKNOWN'] for g in evaluation['groups'].values())} |"
        )
        if evaluation["f4_matrix_check"] is not None:
            matrix = evaluation["f4_matrix_check"]
            lines.extend([
                "",
                f"F4 case 完整性：唯一 case {matrix['unique_case_count']}/20；缺失 `{matrix['missing_case_ids']}`；重复 `{matrix['duplicate_case_ids']}`；超范围 `{matrix['unexpected_case_ids']}`；非整数或空 ID 行 {matrix['null_or_noninteger_case_rows']}。",
                f"实际场景分布：`{json.dumps(matrix['scenario_counts'], ensure_ascii=False)}`（期望 quiet/force/torque = 8/6/6）。",
                f"scenario 与实际 f3_protocol 不一致/缺失：{len(matrix['scenario_protocol_mismatches'])} 行；逐 case 的 scenario 与协议见 JSON。",
            ])
        if evaluation["completeness_reasons"]:
            lines.extend(["", "完整性问题："])
            lines.extend(f"- {md(reason)}" for reason in evaluation["completeness_reasons"])
        lines.extend([
            "",
            "Checkpoint identity（取自 run_metadata / initialization；缺失字段为 null）：",
            "",
            "```json",
            json.dumps(evaluation["checkpoint_identity"], ensure_ascii=False, indent=2),
            "```",
            "",
            "失败与未知 episode：",
            "",
            "| 行 | case_id | env_id | 场景 | outcome | reason | reason_code | failure_phase |",
            "|---:|---:|---:|---|---|---|---|---|",
        ])
        if evaluation["failures_and_unknowns"]:
            for failure in evaluation["failures_and_unknowns"]:
                reason = failure["reason"]
                lines.append(
                    f"| {failure['source_line']} | {md(failure['case_id'])} | {md(failure['env_id'])} | "
                    f"{md(failure['scenario'])} | {failure['outcome']} | {md(reason.get('reason'))} | "
                    f"{md(reason.get('reason_code'))} | {md(reason.get('failure_phase'))} |"
                )
        else:
            lines.append("| — | — | — | — | — | — | — | — |")
        lines.extend(["", "JSON 中的 `failures_and_unknowns` 保留对应 episode 原始记录。"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval", action="append", required=True, metavar="LABEL=OUTPUTDIR",
        help="episode 评估目录；可重复",
    )
    parser.add_argument(
        "--expect-quiet", action="append", default=[], metavar="LABEL=COUNT",
        help="为指定非矩阵评估显式声明 quiet 分母，例如 initial=4",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="尚不存在的新输出目录")
    args = parser.parse_args()

    eval_specs = [parse_assignment(value, "--eval") for value in args.eval]
    labels = [label for label, _ in eval_specs]
    if len(set(labels)) != len(labels):
        parser.error("--eval 的 LABEL 必须唯一")
    quiet_expectations: dict[str, int] = {}
    for value in args.expect_quiet:
        label, raw_count = parse_assignment(value, "--expect-quiet")
        try:
            count = int(raw_count)
        except ValueError:
            parser.error(f"--expect-quiet 数量必须是正整数：{value}")
        if count < 1 or label in quiet_expectations:
            parser.error(f"--expect-quiet 数量需为正数且 LABEL 不重复：{value}")
        quiet_expectations[label] = count
    unknown_labels = sorted(set(quiet_expectations) - set(labels))
    if unknown_labels:
        parser.error(f"--expect-quiet 引用了未声明的 LABEL：{unknown_labels}")
    if args.output_dir.exists():
        parser.error(f"--output-dir 必须是全新目录：{args.output_dir}")

    evaluations = [
        summarize_one(label, Path(raw_dir).expanduser().resolve(), quiet_expectations.get(label))
        for label, raw_dir in eval_specs
    ]
    payload = {
        "schema_version": "firm_grasp_eval_table.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_count": len(evaluations),
        "scope_note": "Episode outcome 汇总；不构成 force-closure、学习收益或可靠性显著性证明。",
        "evaluations": evaluations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "EVAL_TABLE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "EVAL_TABLE.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"写入：{args.output_dir / 'EVAL_TABLE.json'}")
    print(f"写入：{args.output_dir / 'EVAL_TABLE.md'}")
    for evaluation in evaluations:
        print(
            f"{evaluation['label']}: {evaluation['completeness']} "
            f"({evaluation['source']['observed_denominator_rows']} denominator rows, "
            f"{len(evaluation['failures_and_unknowns'])} failure/unknown)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
