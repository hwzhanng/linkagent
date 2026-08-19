"""Render a compact pressure/action replay figure from a captured NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--replay", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data = np.load(args.replay)
steps = np.arange(data["digit_force_n"].shape[0])
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].plot(steps, data["digit_force_n"])
axes[0].plot(steps, data["palm_force_n"], "k--")
axes[0].set_ylabel("force (N)")
axes[0].legend(["thumb", "index", "middle", "ring", "little", "palm"], ncol=3)
axes[1].plot(steps, data["action"])
axes[1].set_ylabel("10D action")
axes[2].plot(steps, data["stable_steps"], label="stable steps")
axes[2].plot(steps, data["stem_damage"], label="stem damage")
axes[2].set_ylabel("state")
axes[2].set_xlabel("policy step (20 Hz)")
axes[2].legend()
fig.suptitle(f"{args.replay.name} — success={bool(data['success'])}")
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=150)
plt.close(fig)
