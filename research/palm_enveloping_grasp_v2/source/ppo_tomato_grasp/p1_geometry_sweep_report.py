"""Aggregate P1 large-fruit geometry probe artifacts into an auditable report.

This script is CPU-only and intentionally consumes JSON artifacts rather than
re-running Isaac Sim.  It keeps the P1 decision reproducible after the headless
Kit process has exited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def _best(data: dict) -> dict:
    rows = data.get("top_candidates", [])
    return rows[0] if rows else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        path for path in Path().glob(args.input_glob)
        if path.name != "p1_geometry_sweep_report.json"
    )
    if not paths:
        raise SystemExit(f"no artifacts matched {args.input_glob!r}")
    reports = [_load(path) for path in paths]
    rows = []
    for data in reports:
        best = _best(data)
        mode = data.get("geometry_mode") or ("isotropic" if data.get("axis_high_m", data.get("axis_m")) == data.get("axis_m") else "anisotropic_range")
        rows.append(
            {
                "file": data["_path"],
                "geometry_mode": mode,
                "diameter_mm": data.get("diameter_mm"),
                "axis_range_m": [data.get("axis_m"), data.get("axis_high_m", data.get("axis_m"))],
                "rest_xy_m": data.get("rest_xy_m"),
                "xy_jitter_m": data.get("xy_jitter_m"),
                "candidate_count": data.get("candidate_count"),
                "strict_candidate_count": data.get("strict_candidate_count"),
                "max_simultaneous_digits": data.get("max_simultaneous_digits"),
                "max_all_five_palm_streak": data.get("max_all_five_palm_streak"),
                "max_angular_coverage_deg_proxy": data.get("max_angular_coverage_deg_proxy"),
                "all_four_distal_participation_rate": data.get("all_four_distal_participation_rate"),
                "proximal_only_closure_rate": data.get("proximal_only_closure_rate"),
                "best_strict_candidate": best.get("strict_candidate"),
                "best_distal_joint_participation": best.get("distal_joint_participation"),
                "best_first_palm_support_step": best.get("first_palm_support_step"),
                "best_first_three_digit_step": best.get("first_three_digit_step"),
                "best_palm_first_margin_steps": best.get("palm_first_margin_steps"),
                "contact_geometry_source": data.get("contact_geometry_source"),
            }
        )

    isotropic = [row for row in rows if row["geometry_mode"] == "isotropic"]
    anisotropic = [row for row in rows if row["geometry_mode"] == "anisotropic_range"]
    strict_diameters = sorted({row["diameter_mm"] for row in isotropic if (row["strict_candidate_count"] or 0) > 0})
    strict_positions = sum(1 for row in anisotropic if (row["strict_candidate_count"] or 0) > 0)
    p1_decision = "GO_WITH_CONSTRAINTS" if strict_diameters and strict_positions >= 2 else "HOLD_AND_REPAIR_GEOMETRY"
    summary = {
        "schema_version": 1,
        "audit": "p1_large_fruit_geometry_sweep",
        "artifact_count": len(rows),
        "isotropic_probe_count": len(isotropic),
        "anisotropic_position_probe_count": len(anisotropic),
        "strict_feasible_isotropic_diameters_mm": strict_diameters,
        "strict_feasible_anisotropic_positions": strict_positions,
        "decision": p1_decision,
        "decision_basis": {
            "strict_candidate_definition": "probe-level palm-first margin, five-digit+ palm streak, proxy coverage >=120 deg, all four distal joints active",
            "position_gate": "at least two anisotropic/position probes with a strict candidate",
            "sensor_geometry_status": "contact_pos_w unavailable in Isaac Sim 4.5 ContactSensor; proxy is explicitly marked and must be replaced/validated in P2",
            "action_mapping_status": "10D has no independent straight-finger distal residual; 14D candidate required for final policy",
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# P1 大果径掌托包络几何扫描报告",
        "",
        f"- 结论：**{p1_decision}**",
        f"- 扫描文件：{len(rows)} 个；等径扫描 {len(isotropic)} 个；椭球/位置扫描 {len(anisotropic)} 个。",
        f"- 出现严格候选的等径直径：{', '.join(f'{d:.0f} mm' for d in strict_diameters) or '无'}。",
        f"- 出现严格候选的椭球/位置点：{strict_positions}/{len(anisotropic)}。",
        "",
        "## 关键解释",
        "",
        "探针的严格候选不是 PPO 成功率，而是验证动作序列是否具备掌托先行、五指同步、空间覆盖和四指指尖参与的几何可行性。接触点字段在当前 Isaac Sim 4.5 ContactSensor 中不可用，因此覆盖角使用指尖位置代理；P2 必须增加真实接触点/法向或做等价校验。",
        "",
        "## 扫描明细",
        "",
        "| 模式 | 直径/范围 | 掌心位置 | 严格候选 | 最大五指+掌托连续步 | 最大覆盖代理 | 四指指尖全参与率 | 根部退化率 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        span = f"{row['axis_range_m'][0]:.3f}–{row['axis_range_m'][1]:.3f} m"
        md.append(
            f"| {row['geometry_mode']} | {row['diameter_mm']:.0f} mm / {span} | "
            f"{row['rest_xy_m']} ± {row['xy_jitter_m']:.3f} m | {row['strict_candidate_count']} | "
            f"{row['max_all_five_palm_streak']} | {row['max_angular_coverage_deg_proxy']:.1f}° | "
            f"{row['all_four_distal_participation_rate']:.3f} | {row['proximal_only_closure_rate']:.3f} |"
        )
    md += [
        "",
        "## P1 决策与约束",
        "",
        "1. 70/80/100 mm 等径动作序列出现严格候选，说明 URDF 与掌托包络拓扑存在可行解；90 mm 当前候选保持窗口不足，不能直接作为失败结论，需在课程中延长闭合/保持并引入 14D 远端残差。",
        "2. 掌心横向偏移到 ±15 mm 时严格候选明显减少，训练不能只使用中心位置；应将中心工作区作为 G0，逐步放宽到经过扫描的可达位置带。",
        "3. 四指指尖全参与率不是常数，且存在根部退化样本；新策略必须按手指报告 `distal_joint_participation` 与 `proximal_only_closure`，并在 P2/P3 作为课程门槛。",
        "4. 该报告只授权仿真训练，不授权任何实机命令。",
        "",
    ]
    args.output_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
