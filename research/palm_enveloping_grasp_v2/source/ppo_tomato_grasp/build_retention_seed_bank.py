"""Build a retention-course joint-state bank from pre-break rollouts.

Only states that already show five fingertip contacts, palm support and no
drop are admitted.  The bank is a reset distribution for a later retention
curriculum; it is not a success label and never changes the strict verifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_bank(
    replay_path: Path,
    output_path: Path,
    *,
    minimum_force_n: float = 0.20,
    minimum_palm_force_n: float = 0.30,
    maximum_rows: int = 256,
    seed: int = 0,
) -> dict[str, object]:
    replay = torch.load(replay_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    required = ("joint_pos", "contact_forces_n", "palm_force_n", "stem_broke", "dropped")
    missing = [key for key in required if key not in replay]
    if missing:
        raise ValueError(f"replay is missing fields: {missing}")
    force = replay["contact_forces_n"].float()
    mask = (
        replay["stem_broke"].bool()
        & (~replay["dropped"].bool())
        & (force >= float(minimum_force_n)).all(-1)
        & (replay["palm_force_n"].float() >= float(minimum_palm_force_n))
    )
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("replay contains no valid five-finger pre/post-break seed states")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(indices.numel(), generator=generator)
    indices = indices[permutation[: min(int(maximum_rows), indices.numel())]]
    bank = {
        "version": "retention_seed_bank_v1",
        "joint_pos": replay["joint_pos"][indices].float().contiguous(),
        "tomato_axes_m": replay.get("tomato_axes_m", torch.zeros((indices.numel(), 3)))[indices].float().contiguous(),
        "contact_forces_n": force[indices].contiguous(),
        "palm_force_n": replay["palm_force_n"][indices].float().contiguous(),
        "source_row": indices.to(torch.long),
        "source_replay": str(replay_path.expanduser().resolve()),
        "minimum_force_n": float(minimum_force_n),
        "minimum_palm_force_n": float(minimum_palm_force_n),
    }
    output_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path.expanduser().resolve())
    return {
        "version": bank["version"],
        "source_replay": bank["source_replay"],
        "rows_seen": int(force.shape[0]),
        "rows_selected": int(indices.numel()),
        "minimum_force_n": bank["minimum_force_n"],
        "minimum_palm_force_n": bank["minimum_palm_force_n"],
        "mean_digit_force_n": [float(v) for v in bank["contact_forces_n"].mean(0)],
        "mean_palm_force_n": float(bank["palm_force_n"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-force", type=float, default=0.20)
    parser.add_argument("--minimum-palm-force", type=float, default=0.30)
    parser.add_argument("--maximum-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(build_bank(
        args.replay,
        args.output,
        minimum_force_n=args.minimum_force,
        minimum_palm_force_n=args.minimum_palm_force,
        maximum_rows=args.maximum_rows,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
