"""尺寸扩域首回合证据的纯 Python 汇总与完整性工具。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


TERMINATION_LABELS = {
    0: "timeout",
    1: "success",
    2: "drop",
    3: "overload",
    4: "self_collision",
    5: "forbidden_first_contact",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """对规范化 JSON 计算稳定 SHA-256。"""

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """返回二项比例 Wilson score 区间；空样本返回 ``(0, 1)``。"""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes/total 无效")
    if total == 0:
        return 0.0, 1.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_first_episode_rows(
    rows: Sequence[dict[str, Any]],
    diameter_bins_mm: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    """按桶汇总互斥终止原因，并计算严格成功 Wilson 95% 区间。"""

    summaries: list[dict[str, Any]] = []
    for bin_index, (low, high) in enumerate(diameter_bins_mm):
        selected = [row for row in rows if int(row["bin_index"]) == bin_index]
        counts = {
            label: sum(row["termination_reason"] == label for row in selected)
            for label in TERMINATION_LABELS.values()
        }
        total = len(selected)
        successes = counts["success"]
        lower, upper = wilson_interval(successes, total)
        summaries.append(
            {
                "bin_index": bin_index,
                "diameter_mm": [float(low), float(high)],
                "first_episode_count": total,
                "termination_counts": counts,
                "strict_success_rate": successes / total if total else None,
                "strict_success_wilson95": [lower, upper],
            }
        )
    return summaries


__all__ = [
    "TERMINATION_LABELS",
    "sha256_file",
    "sha256_json",
    "summarize_first_episode_rows",
    "wilson_interval",
]
