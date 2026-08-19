"""Compact asymmetric PPO trainer for the v2 palm-enveloping task.

The trainer is intentionally self-contained so importing it never parses the
legacy v1 CLI or mutates the v1 artifact directory.  It supports staged runs:
stage 1 learns palm-supported enclosure and stage 2 adds the stem pick and
post-break retention objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
parser.add_argument("--profile", choices=("v1_training_small", "v1_nominal_reliable", "v2_band1_match", "v2_band2_shape", "v2_band3_sensor_actuator", "v2_band4_stress"), default="v1_nominal_reliable")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--horizon", type=int, default=64)
parser.add_argument("--updates", type=int, default=200)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch-size", type=int, default=1024)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--entropy-coef", type=float, default=1.0e-3)
parser.add_argument("--value-coef", type=float, default=0.5)
parser.add_argument("--clip-eps", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--initial-std", type=float, default=0.35)
parser.add_argument("--minimum-std", type=float, default=0.08)
parser.add_argument("--teacher-bc-coef", type=float, default=0.20,
                    help="weak online staged-teacher behavior-cloning coefficient")
parser.add_argument("--teacher-bc-updates", type=int, default=80)
parser.add_argument("--teacher-blend", type=float, default=0.65,
                    help="early rollout blend toward the weak staged teacher")
parser.add_argument("--teacher-blend-updates", type=int, default=80)
parser.add_argument("--bc-only-updates", type=int, default=0,
                    help="updates used only to fit the observation-conditioned teacher before PPO")
parser.add_argument("--teacher-root-scale", type=float, default=1.0)
parser.add_argument("--teacher-distal-scale", type=float, default=1.0)
parser.add_argument("--teacher-distal-base", type=float, default=0.15)
parser.add_argument("--teacher-prior-ramp-steps", type=int, default=0,
                    help="simulation-only complete-prior ramp duration; 0 keeps the staged teacher")
parser.add_argument("--teacher-prior-action", type=str, default=None,
                    help="optional 14-value simulation-only weak action-manifold prior")
parser.add_argument("--teacher-force-band", action="store_true",
                    help="use raw link4 force-band feedback in the simulation-only teacher warm start")
parser.add_argument("--teacher-force-lower", type=str, default="0.10,0.10,0.10,0.10",
                    help="four lower raw link4 force bands for index,middle,ring,little")
parser.add_argument("--teacher-force-upper", type=str, default="4.0,3.0,3.0,2.5",
                    help="four upper raw link4 force bands for index,middle,ring,little")
parser.add_argument("--teacher-force-step", type=float, default=0.02)
parser.add_argument("--teacher-force-retreat", type=float, default=0.02)
parser.add_argument("--strict-coverage-deg", type=float, default=None,
                    help="temporary curriculum coverage threshold; final runs should use the config default")
parser.add_argument("--strict-hold-steps", type=int, default=None,
                    help="temporary curriculum hold threshold; final runs should use the config default")
parser.add_argument("--pregrasp-drop-grace-steps", type=int, default=None)
parser.add_argument("--assist-gain", type=float, default=None,
                    help="v2-only distal-to-proximal reach coupling gain")
parser.add_argument("--assist-per-digit", type=str, default=None,
                    help="v2-only four per-digit distal-to-proximal reach gains: index,middle,ring,little")
parser.add_argument("--per-digit-root-span", type=float, default=None,
                    help="optional v2-only root residual span; leave unset to preserve the historical mapper")
parser.add_argument("--measured-distal-reward", type=float, default=0.0,
                    help="add a bounded raw link4-contact reward during training")
parser.add_argument("--measured-distal-force-reward", type=float, default=0.0,
                    help="add continuous raw link4-force shaping normalized by contact threshold")
parser.add_argument("--smoothing-alpha", type=float, default=None,
                    help="v2-only coarse action smoothing alpha")
parser.add_argument("--distal-filter-alpha", type=float, default=None,
                    help="v2-only distal residual command filter alpha")
parser.add_argument("--distal-filter-alpha-per-digit", type=str, default=None,
                    help="optional four index,middle,ring,little distal filter alphas")
parser.add_argument("--teacher-tactile-gain", type=float, default=0.08)
parser.add_argument("--teacher-max-feedback-step", type=float, default=0.03)
parser.add_argument("--teacher-thumb-first-steps", type=int, default=0,
                    help="simulation-only thumb/coverage course duration before straight-finger release")
parser.add_argument("--teacher-thumb-first-global-scale", type=float, default=0.35,
                    help="bounded global-curl fraction during the thumb-first course")
parser.add_argument("--teacher-coverage-delta", type=str, default="0.0,0.0,0.0,0.0",
                    help="signed bounded coverage delta for balance, spread, opposition, thumb residual")
parser.add_argument("--teacher-motion-smoothing-alpha", type=float, default=1.0,
                    help="simulation-only teacher action slew fraction")
parser.add_argument("--teacher-thumb-weight", type=float, default=1.0,
                    help="behavior-cloning weight for the thumb-opposition channel")
parser.add_argument("--teacher-distal-weight", type=float, default=1.0,
                    help="behavior-cloning weight for the four independent distal channels")
parser.add_argument(
    "--physical-fingertip-proxy",
    action="store_true",
    help="enable the authored v2 fingertip collision pad so ContactSensor receives physical link4 contacts",
)
parser.add_argument(
    "--proxy-only-collision",
    action="store_true",
    help="simulation-only: disable uncalibrated vendor hand colliders and keep only v2 fingertip proxies plus palm support",
)
parser.add_argument(
    "--disable-virtual-distal-contact",
    action="store_true",
    help="disable the distance-only distal tactile proxy for a measured-contact training run",
)
parser.add_argument("--fingertip-proxy-radius", type=float, default=None)
parser.add_argument("--strict-require-measured-distal-contact", action="store_true")
parser.add_argument("--palm-first-gate", action="store_true",
                    help="gate distal residual/assist until palm support is observed")
parser.add_argument("--palm-first-coarse-scale", type=float, default=1.0,
                    help="scale global/proximal action while palm support is absent")
parser.add_argument("--palm-first-coarse-max-scale", type=float, default=1.0,
                    help="maximum coarse scale after palm support; use <1 for a conservative distal-contact course")
parser.add_argument("--palm-first-ramp-steps", type=int, default=None)
parser.add_argument("--palm-first-motion-hold", action="store_true",
                    help="damp fresh coarse commands when supported fruit is moving outward")
parser.add_argument("--palm-first-motion-hold-scale", type=float, default=0.25)
parser.add_argument("--palm-first-motion-hold-speed", type=float, default=0.08)
parser.add_argument("--palm-first-motion-hold-offset", type=float, default=0.55)
parser.add_argument("--per-digit-distal-gate", action="store_true",
                    help="release each distal residual only after its own link3 contact dwell")
parser.add_argument("--per-digit-link3-threshold", type=float, default=None)
parser.add_argument("--per-digit-link3-stable-steps", type=int, default=None)
parser.add_argument("--per-digit-distal-min-episode-step", type=int, default=None)
parser.add_argument("--per-digit-near-surface-gate", action="store_true",
                    help="release each distal residual after palm dwell and that fingertip is near the fruit")
parser.add_argument("--per-digit-near-surface-distance", type=float, default=0.043)
parser.add_argument("--per-digit-near-surface-min-episode-step", type=int, default=20)
parser.add_argument("--per-digit-distal-probe", action="store_true",
                    help="enable bounded post-palm distal contact bootstrap")
parser.add_argument("--per-digit-distal-probe-distance", type=float, default=0.064)
parser.add_argument("--per-digit-distal-probe-action", type=float, default=0.14)
parser.add_argument("--per-digit-distal-probe-action-per-digit", type=str, default=None,
                    help="optional four-value index,middle,ring,little probe action vector")
parser.add_argument("--per-digit-distal-probe-start-step", type=int, default=20)
parser.add_argument("--per-digit-distal-probe-ramp-steps", type=int, default=80)
parser.add_argument("--per-digit-distal-probe-staggered", action="store_true",
                    help="stagger the simulation-only distal bootstrap across the four straight fingers")
parser.add_argument("--per-digit-distal-probe-digit-interval", type=int, default=20)
parser.add_argument("--tomato-contact-offset", type=float, default=None)
parser.add_argument("--tomato-rest-offset", type=float, default=None)
parser.add_argument("--proxy-contact-offset", type=float, default=None)
parser.add_argument("--proxy-rest-offset", type=float, default=None)
parser.add_argument("--palm-support-contact-offset", type=float, default=None)
parser.add_argument("--palm-support-rest-offset", type=float, default=None)
parser.add_argument("--reset-clearance", type=float, default=None)
parser.add_argument("--cup-half-extent", type=str, default=None,
                    help="simulation-only palm cup inner half extents x,y in metres")
parser.add_argument("--rim-height", type=float, default=None,
                    help="simulation-only palm rim height in metres")
parser.add_argument("--debug-collision-digit", choices=("", "palm", "all", "index", "middle", "ring", "little"), default="",
                    help="simulation-only support collision filter; never a hardware setting")
parser.add_argument("--command-rate-rad-s", type=float, default=None)
parser.add_argument("--mass-min", type=float, default=None)
parser.add_argument("--mass-max", type=float, default=None)
parser.add_argument("--density-min", type=float, default=None)
parser.add_argument("--density-max", type=float, default=None)
parser.add_argument("--axis-range-m", type=str, default=None,
                    help="override isotropic tomato semi-axis range as one or two metres")
parser.add_argument("--contact-stiffness-n-m", type=float, default=None,
                    help="override authored tomato compliant-contact stiffness for this simulation run")
parser.add_argument("--v1-teacher-checkpoint", type=Path,
                    default=ROOT / "simulation/ppo_tomato_grasp/artifacts/stage4_band1_long/best.pt",
                    help="frozen v1 actor used only as an observation-conditioned warm-start prior")
parser.add_argument("--disable-v1-teacher", action="store_true")
parser.add_argument(
    "--residual-base-controller", action="store_true",
    help="simulation-only: execute a staged base controller plus residual_scale*tanh(actor); PPO learns residuals",
)
parser.add_argument(
    "--residual-simultaneous-reach", action="store_true",
    help="ablation: ramp distal channels during the coarse approach; default is palm/root/thumb first",
)
parser.add_argument(
    "--residual-scale", type=float, default=0.15,
    help="bounded action residual magnitude around the staged base controller",
)
parser.add_argument(
    "--residual-teacher-coverage-blend", type=float, default=0.0,
    help="bounded simulation-only thumb-first teacher delta on channels 2:14 after base approach",
)
parser.add_argument(
    "--residual-mask", choices=("all", "thumb_distal", "distal", "coverage"), default="thumb_distal",
    help="residual channels exposed to PPO; default protects coarse palm/root actions",
)
parser.add_argument(
    "--residual-no-phase-gate", action="store_true",
    help="ablation: allow residuals during coarse approach; default enables them after the base approach",
)
parser.add_argument("--residual-target-action", type=str, default=None,
                    help="14 comma-separated base-action values in [-1,1]")
parser.add_argument("--residual-approach-steps", type=int, default=240)
parser.add_argument("--residual-per-digit-steps", type=int, default=60)
parser.add_argument("--residual-hold-steps", type=int, default=80,
                    help="extra post-insertion simulation steps reserved for force-band settling")
parser.add_argument("--residual-approach-smoothing", type=float, default=1.0,
                    help="fraction of the staged coarse ramp applied per policy step")
parser.add_argument("--residual-thumb-delay-steps", type=int, default=0)
parser.add_argument("--residual-thumb-ramp-steps", type=int, default=60)
parser.add_argument("--residual-thumb-first-steps", type=int, default=0,
                    help="thumb-only coarse sub-phase before straight-finger roots ramp")
parser.add_argument("--residual-thumb-first-global-scale", type=float, default=0.0,
                    help="fraction of global flexion allowed during the thumb-first sub-phase")
parser.add_argument("--residual-lower-force-per-digit", type=str, default=None,
                    help="optional four lower link4 force bands for index,middle,ring,little")
parser.add_argument("--residual-upper-force-per-digit", type=str, default=None,
                    help="optional four upper link4 force bands for index,middle,ring,little")
parser.add_argument(
    "--residual-insertion-step-per-digit", type=str, default=None,
    help="optional four index,middle,ring,little insertion fractions",
)
parser.add_argument(
    "--residual-retreat-per-digit", type=str, default=None,
    help="optional four index,middle,ring,little retreat fractions",
)
parser.add_argument("--residual-thumb-hold-offset", type=str, default="0.0,0.0",
                    help="two bounded thumb action offsets used only in residual hold feedback")
parser.add_argument("--residual-thumb-lower-force", type=float, default=0.10)
parser.add_argument("--residual-thumb-upper-force", type=float, default=0.80)
parser.add_argument("--residual-thumb-search-amplitude", type=str, default="0.0,0.0",
                    help="optional bounded two-sided thumb search amplitude during low-force hold")
parser.add_argument("--residual-thumb-search-period", type=int, default=24,
                    help="low-force frames before the signed thumb search offset flips")
parser.add_argument("--residual-thumb-contact-gate-steps", type=int, default=0,
                    help="hold straight-finger insertion until thumb force is in-band for N frames")
parser.add_argument("--residual-thumb-contact-latch-force", type=float, default=0.0,
                    help="measured thumb force that starts a bounded action latch")
parser.add_argument("--residual-thumb-contact-latch-steps", type=int, default=0,
                    help="policy frames to hold the thumb action after contact; zero disables")
parser.add_argument("--residual-thumb-contact-latch-freeze-coarse", action="store_true",
                    help="freeze coarse palm/root channels during the thumb contact latch")
parser.add_argument("--seed", type=int, default=20260817)
parser.add_argument("--output-dir", type=Path, default=ROOT / "simulation/ppo_tomato_grasp/artifacts/enveloping_grasp_v2/t0_static_enclosure")
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--resume-weights-only", action="store_true",
                    help="load only model weights from a checkpoint (useful for offline-BC warm starts)")
parser.add_argument("--freeze-coarse-policy", action="store_true",
                    help="freeze the shared actor and coarse 10 action rows; train only distal rows/log-std")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile, profile_definition
from simulation.ppo_tomato_grasp_v2.contract import (
    OBSERVATION_BLOCKS,
    V2_ACTOR_OBSERVATION_DIM,
    V2_ACTION_DIM,
    V2_CRITIC_STATE_DIM,
)
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v2.residual_controller import ResidualBaseController, ResidualBaseControllerCfg
from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg


class V2ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_body = nn.Sequential(nn.Linear(V2_ACTOR_OBSERVATION_DIM, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.critic_body = nn.Sequential(nn.Linear(V2_CRITIC_STATE_DIM, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.actor_mean = nn.Linear(128, V2_ACTION_DIM)
        self.value = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((V2_ACTION_DIM,), math.log(args_cli.initial_std)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        std = self.log_std.clamp(math.log(args_cli.minimum_std), 1.0).exp()
        mean = self.actor_mean(self.actor_body(obs))
        return Normal(torch.nan_to_num(mean, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0), std)

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        value = self.value(self.critic_body(state)).squeeze(-1)
        return torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)


class FrozenV1Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_body = nn.Sequential(nn.Linear(110, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.actor_mean = nn.Linear(128, 14)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(self.actor_body(observation)))


def v2_to_v1_observation(env, v2_observation: torch.Tensor) -> torch.Tensor:
    """Project the v2 live state into the frozen v1 110D interface."""

    axes = ((env.tomato_axes - 0.025) / 0.012).clamp(-1.0, 1.0)
    tactile_v2 = v2_observation[:, OBSERVATION_BLOCKS["tactile_per_digit"]].reshape(env.num_envs, 5, 7)
    index = torch.arange(5, device=v2_observation.device, dtype=v2_observation.dtype).view(1, 5)
    # The frozen checkpoint is a stage-4 v1 actor, whose tactile channels 3:7
    # are reused for the four phase values.  Supplying constants there made
    # the warm-start act as an out-of-distribution open-loop prior.  Preserve
    # the live v2 force/rate/validity channels and broadcast the causal v2
    # phase block into the legacy tactile layout instead.
    phase = v2_observation[:, 100:104].unsqueeze(1).expand(-1, 5, -1)
    force = tactile_v2[:, :, 0]
    old_tactile = torch.cat(
        (
            force.unsqueeze(-1), force.unsqueeze(-1), (force / 0.15).clamp(0.0, 1.0).unsqueeze(-1),
            phase,
            tactile_v2[:, :, 1].unsqueeze(-1), tactile_v2[:, :, 2].unsqueeze(-1),
            (1.0 - tactile_v2[:, :, 5]).unsqueeze(-1),
        ), dim=-1
    ).reshape(env.num_envs, 50)
    return torch.cat((v2_observation[:, 0:3], axes, v2_observation[:, 6:50], old_tactile, v2_observation[:, 108:118]), dim=-1).clamp(-1.0, 1.0)


def squashed_log_prob(dist: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return (dist.log_prob(raw) - torch.log(1.0 - action.square() + 1.0e-6)).sum(-1)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "simulation" / "ppo_tomato_grasp_v2").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_teacher_prior(value: str | None) -> list[float] | None:
    if value is None:
        return None
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != V2_ACTION_DIM:
        raise ValueError("--teacher-prior-action requires 14 comma-separated values")
    if any(abs(item) > 1.000001 for item in values):
        raise ValueError("--teacher-prior-action values must lie in [-1, 1]")
    return values


def residual_action_mask(name: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(V2_ACTION_DIM, device=device)
    if name == "all":
        mask[:] = 1.0
    elif name == "thumb_distal":
        mask[5] = 1.0
        mask[10:] = 1.0
    elif name == "distal":
        mask[10:] = 1.0
    elif name == "coverage":
        # Keep global flexion/curl fixed for palm support; learn only the
        # radial-balance/spread/opposition/root/distal subspace that can
        # change coverage without collapsing the support manifold.
        mask[2:] = 1.0
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown residual mask {name!r}")
    return mask


def parse_residual_target(value: str | None) -> list[float]:
    """Parse the explicit simulation-only base action for residual PPO."""

    if value is None:
        # Candidate297 is a weak, auditable geometry prior, not a terminal
        # pose: the actor remains responsible for force/geometry residuals.
        return [
            1.0, -1.0, 0.5817804336547852, -0.009242807142138481,
            1.0, -0.01167266070842743, 0.58189457654953, 0.9941583275794985,
            0.11048947274684906, 0.05933154374361038, -0.6171775460243225,
            0.6542128324508667, -0.24184483289718628, 0.5135747194290161,
        ]
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != V2_ACTION_DIM or any(abs(item) > 1.000001 for item in values):
        raise ValueError("--residual-target-action requires 14 comma-separated values in [-1,1]")
    return values


def parse_teacher_four(value: str, name: str) -> tuple[float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 4 or any(item < 0.0 for item in values):
        raise ValueError(f"{name} requires four non-negative comma-separated values")
    return tuple(values)  # type: ignore[return-value]


def parse_teacher_signed_four(value: str, name: str) -> tuple[float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 4 or any(abs(item) > 0.30 for item in values):
        raise ValueError(f"{name} requires four values in [-0.30, 0.30]")
    return tuple(values)  # type: ignore[return-value]


def parse_force_four(value: str | None, name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value is None:
        return default
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 4 or any(item < 0.0 for item in values):
        raise ValueError(f"{name} requires four non-negative values")
    return tuple(values)  # type: ignore[return-value]


def parse_four_positive(value: str | None, name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 4 or any(not 0.0 < item <= 1.0 for item in values):
        raise ValueError(f"{name} requires four values in (0,1]")
    return tuple(values)  # type: ignore[return-value]


def parse_signed_pair(value: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 2 or any(abs(item) > 1.0 for item in values):
        raise ValueError(f"{name} requires two values in [-1,1]")
    return float(values[0]), float(values[1])


def parse_pair(value: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 2 or any(item <= 0.0 for item in values):
        raise ValueError(f"{name} requires two positive comma-separated metres")
    return float(values[0]), float(values[1])


def parse_axis_range(value: str, name: str = "--axis-range-m") -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) == 1:
        values = [values[0], values[0]]
    if len(values) != 2 or any(item <= 0.0 for item in values) or values[0] > values[1]:
        raise ValueError(f"{name} requires one positive value or min,max with 0 < min <= max")
    return float(values[0]), float(values[1])


def make_cfg():
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.training_stage = args_cli.stage
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = args_cli.stage >= 2
    cfg.episode_length_s = 12.0
    if args_cli.residual_base_controller:
        # Keep one residual episode alive through approach, all four
        # insertion windows, and the initial hold; otherwise the controller
        # would be reset at the old 240-step v2 episode boundary.
        cfg.episode_length_s = max(
            24.0,
            (int(args_cli.residual_approach_steps)
             + 4 * int(args_cli.residual_per_digit_steps)
             + int(args_cli.residual_hold_steps) + 20) * cfg.policy_dt_s,
        )
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg = apply_profile(cfg, args_cli.profile, num_envs=args_cli.num_envs, seed=args_cli.seed)
    if args_cli.axis_range_m is not None:
        axis_range = parse_axis_range(args_cli.axis_range_m)
        cfg.axis_range_m = axis_range
        cfg.isotropic_axes = True
        if axis_range[1] > 0.040:
            cfg.accepted_mass_kg = (0.050, 0.500)
        cfg.events["tomato_geometry"].params["axis_range_m"] = axis_range
        cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = cfg.accepted_mass_kg
    if args_cli.contact_stiffness_n_m is not None:
        if args_cli.contact_stiffness_n_m <= 0.0:
            raise ValueError("--contact-stiffness-n-m must be positive")
        cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness = float(args_cli.contact_stiffness_n_m)
    if args_cli.strict_coverage_deg is not None:
        cfg.strict_angular_coverage_min_deg = float(args_cli.strict_coverage_deg)
    if args_cli.strict_hold_steps is not None:
        cfg.strict_hold_steps = int(args_cli.strict_hold_steps)
    if args_cli.pregrasp_drop_grace_steps is not None:
        cfg.pregrasp_drop_grace_steps = int(args_cli.pregrasp_drop_grace_steps)
    if args_cli.assist_gain is not None:
        cfg.distal_proximal_reach_assist_gain = float(args_cli.assist_gain)
        cfg.distal_proximal_reach_assist_gain_per_digit = None
    if args_cli.assist_per_digit is not None:
        cfg.distal_proximal_reach_assist_gain_per_digit = parse_teacher_four(
            args_cli.assist_per_digit, "--assist-per-digit"
        )
        cfg.distal_proximal_reach_assist_gain = 0.0
    if args_cli.per_digit_root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args_cli.per_digit_root_span)
    cfg.v2_measured_distal_contact_reward = max(0.0, float(args_cli.measured_distal_reward))
    cfg.v2_measured_distal_force_reward = max(0.0, float(args_cli.measured_distal_force_reward))
    if args_cli.smoothing_alpha is not None:
        cfg.coarse_action_smoothing_alpha = float(args_cli.smoothing_alpha)
    if args_cli.distal_filter_alpha is not None:
        if not 0.0 < args_cli.distal_filter_alpha < 1.0:
            raise ValueError("--distal-filter-alpha must lie in (0,1)")
        cfg.distal_action_filter_alpha = float(args_cli.distal_filter_alpha)
    if args_cli.distal_filter_alpha_per_digit is not None:
        values = tuple(float(item.strip()) for item in args_cli.distal_filter_alpha_per_digit.split(",") if item.strip())
        if len(values) != 4 or any(value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("--distal-filter-alpha-per-digit requires four values in [0,1)")
        cfg.distal_action_filter_alpha = values
    # A proxy-only run is explicitly a physical-contact audit. Treating the
    # switch as merely a filter while leaving the authored pads non-physical
    # produces a misleading zero-contact training signal.
    cfg.fingertip_collision_proxy_physical = bool(
        args_cli.physical_fingertip_proxy or args_cli.proxy_only_collision
    )
    cfg.proxy_only_collision = bool(args_cli.proxy_only_collision)
    if args_cli.fingertip_proxy_radius is not None:
        cfg.fingertip_collision_proxy_radius_m = float(args_cli.fingertip_proxy_radius)
    if args_cli.disable_virtual_distal_contact:
        cfg.virtual_distal_contact_enabled = False
    cfg.strict_require_measured_distal_contact = bool(args_cli.strict_require_measured_distal_contact)
    cfg.palm_first_distal_gate = bool(args_cli.palm_first_gate)
    cfg.palm_first_coarse_scale = float(args_cli.palm_first_coarse_scale)
    cfg.palm_first_coarse_max_scale = float(args_cli.palm_first_coarse_max_scale)
    if args_cli.palm_first_ramp_steps is not None:
        cfg.palm_first_ramp_steps = int(args_cli.palm_first_ramp_steps)
    cfg.palm_first_motion_hold_enabled = bool(args_cli.palm_first_motion_hold)
    cfg.palm_first_motion_hold_scale = float(args_cli.palm_first_motion_hold_scale)
    cfg.palm_first_motion_hold_speed_m_s = float(args_cli.palm_first_motion_hold_speed)
    cfg.palm_first_motion_hold_offset_ratio = float(args_cli.palm_first_motion_hold_offset)
    cfg.per_digit_distal_gate = bool(args_cli.per_digit_distal_gate)
    if args_cli.per_digit_link3_threshold is not None:
        cfg.per_digit_distal_link3_threshold_n = float(args_cli.per_digit_link3_threshold)
    if args_cli.per_digit_link3_stable_steps is not None:
        cfg.per_digit_distal_link3_stable_steps = int(args_cli.per_digit_link3_stable_steps)
    if args_cli.per_digit_distal_min_episode_step is not None:
        cfg.per_digit_distal_min_episode_step = int(args_cli.per_digit_distal_min_episode_step)
    cfg.per_digit_distal_near_surface_enabled = bool(args_cli.per_digit_near_surface_gate)
    cfg.per_digit_distal_near_surface_distance_m = float(args_cli.per_digit_near_surface_distance)
    cfg.per_digit_distal_near_surface_min_episode_step = int(args_cli.per_digit_near_surface_min_episode_step)
    cfg.per_digit_distal_probe_enabled = bool(args_cli.per_digit_distal_probe)
    cfg.per_digit_distal_probe_distance_m = float(args_cli.per_digit_distal_probe_distance)
    cfg.per_digit_distal_probe_action = float(args_cli.per_digit_distal_probe_action)
    if args_cli.per_digit_distal_probe_action_per_digit is not None:
        values = tuple(float(item.strip()) for item in args_cli.per_digit_distal_probe_action_per_digit.split(",") if item.strip())
        if len(values) != 4 or any(abs(item) > 1.000001 for item in values):
            raise ValueError("--per-digit-distal-probe-action-per-digit requires four values in [-1,1]")
        cfg.per_digit_distal_probe_action_per_digit = values
    cfg.per_digit_distal_probe_start_episode_step = int(args_cli.per_digit_distal_probe_start_step)
    cfg.per_digit_distal_probe_ramp_steps = int(args_cli.per_digit_distal_probe_ramp_steps)
    cfg.per_digit_distal_probe_staggered = bool(args_cli.per_digit_distal_probe_staggered)
    cfg.per_digit_distal_probe_digit_interval_steps = int(args_cli.per_digit_distal_probe_digit_interval)
    if args_cli.tomato_contact_offset is not None:
        cfg.tomato_contact_offset_m = float(args_cli.tomato_contact_offset)
    if args_cli.tomato_rest_offset is not None:
        cfg.tomato_rest_offset_m = float(args_cli.tomato_rest_offset)
    if args_cli.proxy_contact_offset is not None:
        cfg.fingertip_proxy_contact_offset_m = float(args_cli.proxy_contact_offset)
    if args_cli.proxy_rest_offset is not None:
        cfg.fingertip_proxy_rest_offset_m = float(args_cli.proxy_rest_offset)
    if args_cli.palm_support_contact_offset is not None:
        cfg.palm_support_contact_offset_m = float(args_cli.palm_support_contact_offset)
    if args_cli.palm_support_rest_offset is not None:
        cfg.palm_support_rest_offset_m = float(args_cli.palm_support_rest_offset)
    if args_cli.reset_clearance is not None:
        cfg.tomato_reset_clearance_m = float(args_cli.reset_clearance)
    if args_cli.cup_half_extent is not None:
        cfg.palm_cup_inner_half_extent_m = parse_pair(args_cli.cup_half_extent, "--cup-half-extent")
    if args_cli.rim_height is not None:
        cfg.palm_cup_rim_height_m = float(args_cli.rim_height)
    cfg.debug_support_only_collision = bool(args_cli.debug_collision_digit)
    cfg.debug_collision_digit = args_cli.debug_collision_digit
    if args_cli.command_rate_rad_s is not None:
        cfg.command_rate_limit_rad_s = float(args_cli.command_rate_rad_s)
    if args_cli.mass_min is not None or args_cli.mass_max is not None:
        low = float(args_cli.mass_min if args_cli.mass_min is not None else cfg.accepted_mass_kg[0])
        high = float(args_cli.mass_max if args_cli.mass_max is not None else cfg.accepted_mass_kg[1])
        if not (0.0 < low <= high):
            raise ValueError("mass range must satisfy 0 < min <= max")
        cfg.accepted_mass_kg = (low, high)
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (low, high)
    if args_cli.density_min is not None or args_cli.density_max is not None:
        low = float(args_cli.density_min if args_cli.density_min is not None else cfg.density_range_kg_m3[0])
        high = float(args_cli.density_max if args_cli.density_max is not None else cfg.density_range_kg_m3[1])
        if not (0.0 < low <= high):
            raise ValueError("density range must satisfy 0 < min <= max")
        cfg.density_range_kg_m3 = (low, high)
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (low, high)
    return cfg


def save_checkpoint(path: Path, policy: V2ActorCritic, optimizer, update: int, total_steps: int, metrics: dict, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "palm_enveloping_grasp_v2",
        "task": "palm_supported_five_finger_enveloping",
        "stage": args_cli.stage,
        "profile": args_cli.profile,
        "actor_observation_dim": V2_ACTOR_OBSERVATION_DIM,
        "critic_state_dim": V2_CRITIC_STATE_DIM,
        "action_dim": V2_ACTION_DIM,
        "contract_version": "palm_enveloping_grasp_v2.0",
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "total_steps": total_steps,
        "metrics": metrics,
        "resolved_config": {
            "profile_values": profile_definition(args_cli.profile),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "horizon": args_cli.horizon,
            "gamma": args_cli.gamma,
            "gae_lambda": args_cli.gae_lambda,
            "learning_rate": args_cli.learning_rate,
            "stage": args_cli.stage,
            "safe_force_n": cfg.safe_force_n,
            "strict_hold_steps": cfg.strict_hold_steps,
            "teacher_bc_coef": args_cli.teacher_bc_coef,
            "teacher_blend": args_cli.teacher_blend,
            "bc_only_updates": args_cli.bc_only_updates,
            "teacher_root_scale": args_cli.teacher_root_scale,
            "teacher_distal_scale": args_cli.teacher_distal_scale,
            "teacher_distal_base": args_cli.teacher_distal_base,
            "teacher_prior_ramp_steps": args_cli.teacher_prior_ramp_steps,
            "teacher_prior_action": parse_teacher_prior(args_cli.teacher_prior_action),
            "teacher_force_band": args_cli.teacher_force_band,
            "teacher_force_lower_n": parse_teacher_four(args_cli.teacher_force_lower, "--teacher-force-lower"),
            "teacher_force_upper_n": parse_teacher_four(args_cli.teacher_force_upper, "--teacher-force-upper"),
            "teacher_force_step": args_cli.teacher_force_step,
            "teacher_force_retreat": args_cli.teacher_force_retreat,
            "teacher_thumb_first_steps": args_cli.teacher_thumb_first_steps,
            "teacher_thumb_first_global_scale": args_cli.teacher_thumb_first_global_scale,
            "teacher_coverage_delta": parse_teacher_signed_four(args_cli.teacher_coverage_delta, "--teacher-coverage-delta"),
            "teacher_motion_smoothing_alpha": args_cli.teacher_motion_smoothing_alpha,
            "strict_coverage_deg": cfg.strict_angular_coverage_min_deg,
            "strict_hold_steps": cfg.strict_hold_steps,
            "pregrasp_drop_grace_steps": cfg.pregrasp_drop_grace_steps,
            "distal_proximal_reach_assist_gain": cfg.distal_proximal_reach_assist_gain,
            "distal_proximal_reach_assist_gain_per_digit": cfg.distal_proximal_reach_assist_gain_per_digit,
            "per_digit_root_residual_span_fraction": cfg.per_digit_root_residual_span_fraction,
            "v2_measured_distal_contact_reward": cfg.v2_measured_distal_contact_reward,
            "v2_measured_distal_force_reward": cfg.v2_measured_distal_force_reward,
            "coarse_action_smoothing_alpha": cfg.coarse_action_smoothing_alpha,
            "distal_action_filter_alpha": (
                [float(v) for v in cfg.distal_action_filter_alpha]
                if isinstance(cfg.distal_action_filter_alpha, (tuple, list))
                else float(cfg.distal_action_filter_alpha)
            ),
            "teacher_tactile_gain": args_cli.teacher_tactile_gain,
            "teacher_max_feedback_step": args_cli.teacher_max_feedback_step,
            "teacher_thumb_weight": args_cli.teacher_thumb_weight,
            "teacher_distal_weight": args_cli.teacher_distal_weight,
            "fingertip_collision_proxy_physical": cfg.fingertip_collision_proxy_physical,
            "proxy_only_collision": cfg.proxy_only_collision,
            "virtual_distal_contact_enabled": cfg.virtual_distal_contact_enabled,
            "fingertip_collision_proxy_radius_m": cfg.fingertip_collision_proxy_radius_m,
            "strict_require_measured_distal_contact": cfg.strict_require_measured_distal_contact,
            "palm_first_distal_gate": cfg.palm_first_distal_gate,
            "palm_first_coarse_scale": cfg.palm_first_coarse_scale,
            "palm_first_coarse_max_scale": cfg.palm_first_coarse_max_scale,
            "palm_first_ramp_steps": cfg.palm_first_ramp_steps,
            "freeze_coarse_policy": args_cli.freeze_coarse_policy,
        "residual_base_controller": bool(args_cli.residual_base_controller),
        "residual_scale": float(args_cli.residual_scale),
        "residual_teacher_coverage_blend": float(args_cli.residual_teacher_coverage_blend),
        "residual_mask": args_cli.residual_mask,
        "residual_phase_gate": not bool(args_cli.residual_no_phase_gate),
            "residual_target_action": parse_residual_target(args_cli.residual_target_action),
            "residual_approach_smoothing": float(args_cli.residual_approach_smoothing),
            "residual_approach_steps": int(args_cli.residual_approach_steps),
            "residual_per_digit_steps": int(args_cli.residual_per_digit_steps),
            "residual_hold_steps": int(args_cli.residual_hold_steps),
            "residual_thumb_delay_steps": int(args_cli.residual_thumb_delay_steps),
            "residual_thumb_ramp_steps": int(args_cli.residual_thumb_ramp_steps),
            "residual_thumb_first_steps": int(args_cli.residual_thumb_first_steps),
            "residual_thumb_first_global_scale": float(args_cli.residual_thumb_first_global_scale),
            "residual_lower_force_per_digit": list(parse_force_four(
                args_cli.residual_lower_force_per_digit,
                "--residual-lower-force-per-digit",
                (0.18, 0.18, 0.18, 0.18),
            )),
            "residual_upper_force_per_digit": list(parse_force_four(
                args_cli.residual_upper_force_per_digit,
                "--residual-upper-force-per-digit",
                (0.80, 0.80, 0.80, 0.80),
            )),
            "residual_insertion_step_per_digit": parse_four_positive(
                args_cli.residual_insertion_step_per_digit,
                "--residual-insertion-step-per-digit",
            ),
            "residual_retreat_per_digit": parse_four_positive(
                args_cli.residual_retreat_per_digit,
                "--residual-retreat-per-digit",
            ),
            "residual_thumb_hold_target_offset": list(parse_signed_pair(
                args_cli.residual_thumb_hold_offset, "--residual-thumb-hold-offset"
            )),
            "residual_thumb_lower_force_n": float(args_cli.residual_thumb_lower_force),
            "residual_thumb_upper_force_n": float(args_cli.residual_thumb_upper_force),
            "residual_thumb_search_amplitude": list(parse_signed_pair(
                args_cli.residual_thumb_search_amplitude, "--residual-thumb-search-amplitude"
            )),
            "residual_thumb_search_period_steps": int(args_cli.residual_thumb_search_period),
            "residual_thumb_contact_gate_steps": int(args_cli.residual_thumb_contact_gate_steps),
            "residual_thumb_contact_latch_force_n": float(args_cli.residual_thumb_contact_latch_force),
            "residual_thumb_contact_latch_steps": int(args_cli.residual_thumb_contact_latch_steps),
            "residual_thumb_contact_latch_freeze_coarse": bool(args_cli.residual_thumb_contact_latch_freeze_coarse),
            "palm_first_motion_hold_enabled": cfg.palm_first_motion_hold_enabled,
            "palm_first_motion_hold_scale": cfg.palm_first_motion_hold_scale,
            "palm_first_motion_hold_speed_m_s": cfg.palm_first_motion_hold_speed_m_s,
            "palm_first_motion_hold_offset_ratio": cfg.palm_first_motion_hold_offset_ratio,
            "per_digit_distal_gate": cfg.per_digit_distal_gate,
            "per_digit_distal_link3_threshold_n": cfg.per_digit_distal_link3_threshold_n,
            "per_digit_distal_link3_stable_steps": cfg.per_digit_distal_link3_stable_steps,
            "per_digit_distal_min_episode_step": cfg.per_digit_distal_min_episode_step,
            "per_digit_distal_near_surface_enabled": cfg.per_digit_distal_near_surface_enabled,
            "per_digit_distal_near_surface_distance_m": cfg.per_digit_distal_near_surface_distance_m,
            "per_digit_distal_near_surface_min_episode_step": cfg.per_digit_distal_near_surface_min_episode_step,
            "per_digit_distal_probe_enabled": cfg.per_digit_distal_probe_enabled,
            "per_digit_distal_probe_distance_m": cfg.per_digit_distal_probe_distance_m,
            "per_digit_distal_probe_action": cfg.per_digit_distal_probe_action,
            "per_digit_distal_probe_action_per_digit": cfg.per_digit_distal_probe_action_per_digit,
            "per_digit_distal_probe_start_episode_step": cfg.per_digit_distal_probe_start_episode_step,
            "per_digit_distal_probe_ramp_steps": cfg.per_digit_distal_probe_ramp_steps,
            "per_digit_distal_probe_staggered": cfg.per_digit_distal_probe_staggered,
            "per_digit_distal_probe_digit_interval_steps": cfg.per_digit_distal_probe_digit_interval_steps,
            "tomato_contact_offset_m": cfg.tomato_contact_offset_m,
            "tomato_rest_offset_m": cfg.tomato_rest_offset_m,
            "fingertip_proxy_contact_offset_m": cfg.fingertip_proxy_contact_offset_m,
            "fingertip_proxy_rest_offset_m": cfg.fingertip_proxy_rest_offset_m,
            "palm_support_contact_offset_m": cfg.palm_support_contact_offset_m,
            "palm_support_rest_offset_m": cfg.palm_support_rest_offset_m,
            "tomato_reset_clearance_m": cfg.tomato_reset_clearance_m,
            "palm_cup_inner_half_extent_m": cfg.palm_cup_inner_half_extent_m,
            "palm_cup_rim_height_m": cfg.palm_cup_rim_height_m,
            "debug_support_only_collision": cfg.debug_support_only_collision,
            "debug_collision_digit": cfg.debug_collision_digit,
            "command_rate_limit_rad_s": cfg.command_rate_limit_rad_s,
            "accepted_mass_kg": cfg.accepted_mass_kg,
            "density_range_kg_m3": cfg.density_range_kg_m3,
            "axis_range_m": cfg.axis_range_m,
            "isotropic_axes": cfg.isotropic_axes,
            "contact_stiffness_n_m": cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness,
            "palm_first_stable_steps": cfg.palm_first_stable_steps,
            "palm_first_ramp_steps": cfg.palm_first_ramp_steps,
        },
        "source_sha256": source_hash(),
        "command": " ".join(sys.argv),
        "hardware_command_allowed": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    device = torch.device(args_cli.device)
    cfg = make_cfg()
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    obs, _ = env.reset(seed=args_cli.seed)
    residual_controller = None
    residual_target = parse_residual_target(args_cli.residual_target_action)
    if args_cli.residual_base_controller:
        if not 0.0 < float(args_cli.residual_scale) <= 1.0:
            raise ValueError("--residual-scale must lie in (0,1]")
        residual_lower_force = parse_force_four(
            args_cli.residual_lower_force_per_digit,
            "--residual-lower-force-per-digit",
            (0.18, 0.18, 0.18, 0.18),
        )
        residual_upper_force = parse_force_four(
            args_cli.residual_upper_force_per_digit,
            "--residual-upper-force-per-digit",
            (0.80, 0.80, 0.80, 0.80),
        )
        if any(upper < lower for lower, upper in zip(residual_lower_force, residual_upper_force)):
            raise ValueError("each residual force upper band must be >= its lower band")
        residual_controller = ResidualBaseController(
            args_cli.num_envs,
            device,
            tuple(residual_target),
            ResidualBaseControllerCfg(
                approach_steps=max(int(args_cli.residual_approach_steps), 1),
                per_digit_steps=max(int(args_cli.residual_per_digit_steps), 1),
                approach_smoothing=float(args_cli.residual_approach_smoothing),
                thumb_delay_steps=max(int(args_cli.residual_thumb_delay_steps), 0),
                thumb_ramp_steps=max(int(args_cli.residual_thumb_ramp_steps), 1),
                thumb_first_steps=max(int(args_cli.residual_thumb_first_steps), 0),
                thumb_first_global_scale=float(args_cli.residual_thumb_first_global_scale),
                simultaneous_reach=bool(args_cli.residual_simultaneous_reach),
                lower_force_n=residual_lower_force,
                upper_force_n=residual_upper_force,
                insertion_step_fraction_per_digit=parse_four_positive(
                    args_cli.residual_insertion_step_per_digit,
                    "--residual-insertion-step-per-digit",
                ),
                retreat_fraction_per_digit=parse_four_positive(
                    args_cli.residual_retreat_per_digit,
                    "--residual-retreat-per-digit",
                ),
                thumb_lower_force_n=float(args_cli.residual_thumb_lower_force),
                thumb_upper_force_n=float(args_cli.residual_thumb_upper_force),
                thumb_hold_target_offset=parse_signed_pair(
                    args_cli.residual_thumb_hold_offset, "--residual-thumb-hold-offset"
                ),
                thumb_search_amplitude=parse_signed_pair(
                    args_cli.residual_thumb_search_amplitude, "--residual-thumb-search-amplitude"
                ),
                thumb_search_period_steps=max(int(args_cli.residual_thumb_search_period), 1),
                thumb_first_contact_gate_steps=max(int(args_cli.residual_thumb_contact_gate_steps), 0),
                thumb_contact_latch_force_n=max(float(args_cli.residual_thumb_contact_latch_force), 0.0),
                thumb_contact_latch_steps=max(int(args_cli.residual_thumb_contact_latch_steps), 0),
                thumb_contact_latch_freeze_coarse=bool(args_cli.residual_thumb_contact_latch_freeze_coarse),
            ),
        )
        print(
            "[v2-ppo] residual base controller enabled: "
            f"scale={args_cli.residual_scale:.3f}, "
            f"approach={args_cli.residual_approach_steps}, "
            f"per_digit={args_cli.residual_per_digit_steps}",
            flush=True,
        )
    policy = V2ActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)
    v1_teacher = None
    if not args_cli.disable_v1_teacher and args_cli.v1_teacher_checkpoint is not None:
        v1_path = args_cli.v1_teacher_checkpoint.expanduser().resolve()
        if v1_path.is_file():
            frozen = torch.load(v1_path, map_location=device, weights_only=False)
            if (frozen.get("actor_observation_dim"), frozen.get("action_dim")) == (110, 14):
                v1_teacher = FrozenV1Actor().to(device)
                v1_teacher.load_state_dict({
                    "actor_body.0.weight": frozen["model"]["actor_body.0.weight"],
                    "actor_body.0.bias": frozen["model"]["actor_body.0.bias"],
                    "actor_body.2.weight": frozen["model"]["actor_body.2.weight"],
                    "actor_body.2.bias": frozen["model"]["actor_body.2.bias"],
                    "actor_body.4.weight": frozen["model"]["actor_body.4.weight"],
                    "actor_body.4.bias": frozen["model"]["actor_body.4.bias"],
                    "actor_mean.weight": frozen["model"]["actor_mean.weight"],
                    "actor_mean.bias": frozen["model"]["actor_mean.bias"],
                })
                v1_teacher.eval()
                for parameter in v1_teacher.parameters():
                    parameter.requires_grad_(False)
                print(f"[v2-ppo] loaded frozen v1 observation-conditioned teacher: {v1_path}", flush=True)
    if args_cli.resume is not None:
        checkpoint = torch.load(args_cli.resume.expanduser().resolve(), map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer") and not args_cli.resume_weights_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args_cli.learning_rate
    if args_cli.freeze_coarse_policy:
        # Keep the audited palm/root enclosure fixed while the four distal
        # output rows learn from measured link4 contact.  Hooks mask coarse
        # row gradients without changing the checkpoint architecture.
        for parameter in policy.actor_body.parameters():
            parameter.requires_grad_(False)
        policy.actor_mean.weight.register_hook(
            lambda grad: torch.cat((torch.zeros_like(grad[:10]), grad[10:]), dim=0)
        )
        policy.actor_mean.bias.register_hook(
            lambda grad: torch.cat((torch.zeros_like(grad[:10]), grad[10:]), dim=0)
        )
        policy.log_std.register_hook(
            lambda grad: torch.cat((torch.zeros_like(grad[:10]), grad[10:]), dim=0)
        )
    # Rollout storage is allocated once; all tensors remain on the GPU.
    n, h = args_cli.num_envs, args_cli.horizon
    obs_buf = torch.zeros(h, n, V2_ACTOR_OBSERVATION_DIM, device=device)
    state_buf = torch.zeros(h, n, V2_CRITIC_STATE_DIM, device=device)
    raw_buf = torch.zeros(h, n, V2_ACTION_DIM, device=device)
    action_buf = torch.zeros_like(raw_buf)
    logp_buf = torch.zeros(h, n, device=device)
    value_buf = torch.zeros(h, n, device=device)
    reward_buf = torch.zeros(h, n, device=device)
    done_buf = torch.zeros(h, n, device=device)
    teacher_buf = torch.zeros(h, n, V2_ACTION_DIM, device=device)
    teacher_prior = parse_teacher_prior(args_cli.teacher_prior_action)
    teacher_cfg = EnvelopingTeacherCfg(
        distal_base_action=args_cli.teacher_distal_base,
        distal_prior_action=None if teacher_prior is None else tuple(teacher_prior[10:]),
        tactile_gain=args_cli.teacher_tactile_gain,
        max_feedback_step=args_cli.teacher_max_feedback_step,
        thumb_first_phase_steps=max(0, int(args_cli.teacher_thumb_first_steps)),
        thumb_first_global_scale=float(args_cli.teacher_thumb_first_global_scale),
        thumb_first_coverage_delta=parse_teacher_signed_four(args_cli.teacher_coverage_delta, "--teacher-coverage-delta"),
        motion_smoothing_alpha=float(args_cli.teacher_motion_smoothing_alpha),
        prior_ramp_steps=max(0, int(args_cli.teacher_prior_ramp_steps)),
        force_band_enabled=bool(args_cli.teacher_force_band),
        force_band_lower_n=parse_teacher_four(args_cli.teacher_force_lower, "--teacher-force-lower"),
        force_band_upper_n=parse_teacher_four(args_cli.teacher_force_upper, "--teacher-force-upper"),
        force_band_step=float(args_cli.teacher_force_step),
        force_band_retreat=float(args_cli.teacher_force_retreat),
    )
    teacher = EnvelopingTeacher(n, device, teacher_cfg)
    residual_mask = residual_action_mask(args_cli.residual_mask, device)
    # The weak teacher contains three deliberately varied branches.  A branch
    # id is not part of the deployable observation, so fitting all branches at
    # once would make behavior cloning non-Markov.  Use one fixed branch for
    # the PPO warm start; branch diversity is reintroduced only after the
    # policy has a stable single-branch solution.
    teacher.root_prior[:] = teacher.root_prior[0:1]
    teacher.residual_prior[:] = teacher.residual_prior[0:1]
    teacher.root_prior[:, :5] = (teacher.root_prior[:, :5] * args_cli.teacher_root_scale).clamp(-1.0, 1.0)
    teacher.residual_prior = (teacher.residual_prior * args_cli.teacher_root_scale).clamp(-1.0, 1.0)
    if teacher_prior is not None:
        prior_tensor = torch.tensor(teacher_prior, device=device, dtype=torch.float32)
        # The first five channels are coarse/root opposition; channels 5:10
        # are the per-finger proximal residuals.  Distal channels are passed
        # through EnvelopingTeacherCfg as an independent 4D target.
        teacher.root_prior[:, :10] = prior_tensor[:10]
        teacher.residual_prior[:] = prior_tensor[5:10]
    best_score = -float("inf")
    total_steps = 0
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args_cli.output_dir / "metrics.jsonl"
    for update in range(1, args_cli.updates + 1):
        teacher_abs_sum = torch.zeros((), device=device)
        teacher_thumb_sum = torch.zeros((), device=device)
        teacher_active_count = torch.zeros((), device=device)
        for step in range(h):
            # Isaac Lab returns inference tensors from its environment hooks;
            # clone before feeding them to a graph-tracked PPO network.
            actor_obs = obs["policy"].clone()
            critic_state = env._get_states().clone()
            teacher_action_now = teacher.command(
                env.contact_forces,
                env.palm_force,
                episode_step=env.episode_length_buf,
                measured_digit_force_n=env._v2_measured_link4_force,
            ).clone()
            teacher_action_now[:, 10:] = (teacher_action_now[:, 10:] * args_cli.teacher_distal_scale).clamp(-1.0, 1.0)
            if v1_teacher is not None:
                v1_action_now = v1_teacher(v2_to_v1_observation(env, actor_obs))
                teacher_action_now = 0.35 * teacher_action_now + 0.65 * v1_action_now
            dist = policy.distribution(actor_obs)
            sampled_raw = dist.rsample()
            sampled_action = torch.tanh(sampled_raw)
            # Linearly hand control from the staged teacher to PPO.  The old
            # hard switch (1.0 until N, then 0.0) caused the repeated late
            # value/return collapses observed in T0: the actor never sampled
            # its own corrective actions during warm-up and was then exposed
            # to an abrupt distribution shift.  The teacher remains only a
            # bounded warm-start prior; it cannot be active after the window.
            if args_cli.teacher_blend_updates > 0:
                # A behavior-cloning warm-up must actually collect the
                # teacher trajectory.  The previous schedule started
                # decaying on update 1 even when ``bc_only_updates`` was
                # enabled, so the nominal "BC-only" phase quietly injected
                # more and more untrained PPO actions and became a moving
                # target.  Hold the requested teacher blend for the explicit
                # BC window, then decay it over the remaining teacher window.
                if update <= args_cli.bc_only_updates:
                    blend = float(args_cli.teacher_blend)
                else:
                    decay_step = max(update - args_cli.bc_only_updates - 1, 0)
                    decay_span = max(
                        int(args_cli.teacher_blend_updates) - int(args_cli.bc_only_updates),
                        1,
                    )
                    blend = args_cli.teacher_blend * max(
                        0.0, 1.0 - float(decay_step) / float(decay_span)
                    )
            else:
                blend = 0.0
            # In residual mode the staged controller owns the physical action
            # manifold.  PPO samples a zero-centred, bounded residual and the
            # likelihood stored below is for that residual—not for the
            # translated action sent to PhysX.  Mixing the old teacher here
            # would silently turn the residual into a second open-loop pose.
            if residual_controller is not None:
                base_action = residual_controller.command(
                    env.episode_length_buf,
                    env._v2_measured_link4_force,
                )
                phase_gate = (
                    (env.episode_length_buf >= int(args_cli.residual_approach_steps)).float()
                    if not args_cli.residual_no_phase_gate
                    else torch.ones(n, device=device)
                )
                bounded_residual = sampled_action * residual_mask.unsqueeze(0) * phase_gate.unsqueeze(-1)
                action = (
                    base_action + float(args_cli.residual_scale) * bounded_residual
                ).clamp(-0.995, 0.995)
                if float(args_cli.residual_teacher_coverage_blend) > 0.0:
                    teacher_gate = phase_gate * (teacher.phase >= 1).float()
                    teacher_delta = (teacher_action_now[:, 2:] - base_action[:, 2:]).clamp(-0.30, 0.30)
                    action[:, 2:] = (
                        action[:, 2:]
                        + float(args_cli.residual_teacher_coverage_blend)
                        * teacher_gate.unsqueeze(-1)
                        * teacher_delta
                    ).clamp(-0.995, 0.995)
                likelihood_action = sampled_action
            elif args_cli.freeze_coarse_policy:
                # Distal-only warm start: the frozen coarse enclosure must be
                # sampled from the checkpoint even while the scripted prior
                # explores link4.  Mixing teacher root actions here would
                # invalidate the safety-preserving ablation.
                action = sampled_action.clone()
                action[:, 10:] = ((1.0 - blend) * sampled_action[:, 10:] + blend * teacher_action_now[:, 10:])
                action = action.clamp(-0.995, 0.995)
                likelihood_action = action
            else:
                action = ((1.0 - blend) * sampled_action + blend * teacher_action_now).clamp(-0.995, 0.995)
                likelihood_action = action
            teacher_abs_sum += teacher_action_now.abs().mean()
            teacher_thumb_sum += teacher_action_now[:, 4].mean()
            teacher_active_count += (teacher_action_now.abs().mean(-1) > 0.05).float().mean()
            # Evaluate the likelihood of the action actually applied to PhysX;
            # the clipped inverse tanh keeps the PPO ratio numerically finite.
            # ``likelihood_action`` is the action represented by the policy
            # distribution.  In ordinary PPO it equals the environment
            # action; in residual PPO it remains the unshifted residual.
            raw = torch.atanh(likelihood_action)
            value = policy.critic(critic_state)
            next_obs, reward, terminated, truncated, _info = env.step(action.detach())
            done = (terminated | truncated).float()
            obs_buf[step] = actor_obs
            state_buf[step] = critic_state
            raw_buf[step] = raw
            action_buf[step] = likelihood_action
            logp_buf[step] = squashed_log_prob(dist, raw, likelihood_action)
            value_buf[step] = value
            reward_buf[step] = reward.clamp(-20.0, 20.0)
            done_buf[step] = done
            teacher_buf[step] = (
                torch.zeros_like(teacher_action_now)
                if residual_controller is not None else teacher_action_now
            )
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if done_ids.numel() > 0:
                teacher.reset(done_ids)
                if residual_controller is not None:
                    residual_controller.reset(done_ids)
            obs = next_obs
            total_steps += n
        with torch.no_grad():
            next_value = policy.critic(env._get_states().clone())
            advantages = torch.zeros_like(reward_buf)
            gae = torch.zeros(n, device=device)
            for step in reversed(range(h)):
                nonterminal = 1.0 - done_buf[step]
                next_v = next_value if step == h - 1 else value_buf[step + 1]
                delta = reward_buf[step] + args_cli.gamma * nonterminal * next_v - value_buf[step]
                gae = delta + args_cli.gamma * args_cli.gae_lambda * nonterminal * gae
                advantages[step] = gae
            returns = advantages + value_buf
        # Rollout tensors are targets for PPO; never retain the policy graph
        # across minibatches or epochs.
        flat_obs = obs_buf.detach().reshape(-1, V2_ACTOR_OBSERVATION_DIM)
        flat_state = state_buf.detach().reshape(-1, V2_CRITIC_STATE_DIM)
        flat_raw = raw_buf.detach().reshape(-1, V2_ACTION_DIM)
        flat_action = action_buf.detach().reshape(-1, V2_ACTION_DIM)
        flat_old_logp = logp_buf.detach().reshape(-1)
        flat_adv = advantages.detach().reshape(-1)
        flat_ret = returns.detach().clamp(-100.0, 100.0).reshape(-1)
        flat_teacher = teacher_buf.detach().reshape(-1, V2_ACTION_DIM)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1.0e-8)
        batch = flat_obs.shape[0]
        skipped_optimizer_steps = 0
        for _epoch in range(args_cli.epochs):
            for start in range(0, batch, args_cli.minibatch_size):
                indices = torch.arange(start, min(start + args_cli.minibatch_size, batch), device=device)
                dist = policy.distribution(flat_obs[indices])
                new_logp = squashed_log_prob(dist, flat_raw[indices], flat_action[indices])
                ratio = (new_logp - flat_old_logp[indices]).exp()
                policy_loss = -torch.minimum(ratio * flat_adv[indices], ratio.clamp(1.0 - args_cli.clip_eps, 1.0 + args_cli.clip_eps) * flat_adv[indices]).mean()
                value_loss = 0.5 * (policy.critic(flat_state[indices]) - flat_ret[indices]).square().mean()
                entropy = dist.entropy().sum(-1).mean()
                teacher_coef = (
                    0.0
                    if residual_controller is not None
                    else (args_cli.teacher_bc_coef if update <= args_cli.teacher_bc_updates else 0.0)
                )
                teacher_pred = torch.tanh(policy.actor_mean(policy.actor_body(flat_obs[indices])))
                teacher_error = (teacher_pred - flat_teacher[indices]).square()
                # Thumb opposition and distal link4 outputs are the rare,
                # high-value labels in the staged teacher.  A plain mean is
                # dominated by the many open/near-zero channels and lets the
                # actor settle into a four-finger/one-finger local minimum.
                teacher_weights = teacher_error.new_ones((V2_ACTION_DIM,))
                teacher_weights[4] = max(float(args_cli.teacher_thumb_weight), 0.0)
                teacher_weights[10:] = max(float(args_cli.teacher_distal_weight), 0.0)
                teacher_loss = (teacher_error * teacher_weights).mean()
                if update <= args_cli.bc_only_updates:
                    # A short supervised warm start prevents the first sparse
                    # PPO updates from destroying a useful tactile/phase
                    # prior.  Critic fitting is retained with a small weight
                    # so the subsequent advantage estimates are finite.
                    loss = teacher_coef * teacher_loss + 0.10 * value_loss
                else:
                    loss = policy_loss + args_cli.value_coef * value_loss - args_cli.entropy_coef * entropy + teacher_coef * teacher_loss
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    skipped_optimizer_steps += 1
                    continue
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=False)
                if not torch.isfinite(grad_norm):
                    # Do not convert a bad gradient into an all-zero policy.
                    # The previous implementation silently applied
                    # nan_to_num(parameter)=0 after optimizer.step(), which
                    # produced apparently healthy teacher returns while the
                    # deployable actor had been erased.
                    skipped_optimizer_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                backup = [parameter.detach().clone() for parameter in policy.parameters()]
                optimizer.step()
                invalid_parameter = any(not torch.isfinite(parameter).all() for parameter in policy.parameters())
                if invalid_parameter:
                    skipped_optimizer_steps += 1
                    for parameter, previous in zip(policy.parameters(), backup):
                        parameter.data.copy_(previous)
                    # Clear any non-finite Adam moments without touching a
                    # finite parameter trajectory; the next minibatch can
                    # continue from the last valid update.
                    for state in optimizer.state.values():
                        for value in state.values():
                            if torch.is_tensor(value) and not torch.isfinite(value).all():
                                value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    for parameter in policy.parameters():
                        parameter.data.clamp_(-10.0, 10.0)
                # Keep exploration in the physically useful range.  Letting
                # log_std drift to zero or above after the teacher phase
                # injects multi-radian action noise and recreates the late
                # value-collapse seen in the first T0 attempts.
                policy.log_std.data.clamp_(
                    math.log(args_cli.minimum_std), math.log(args_cli.initial_std)
                )
        metrics = {
            "update": update,
            "total_steps": total_steps,
            "rollout_return_mean": float(reward_buf.sum(0).mean()),
            "rollout_reward_mean": float(reward_buf.mean()),
            "advantage_std": float(advantages.std()),
            "value_mean": float(value_buf.mean()),
            "policy_std_mean": float(policy.log_std.exp().mean()),
            "skipped_optimizer_steps": int(skipped_optimizer_steps),
            "teacher_action_abs_mean": float(teacher_abs_sum / max(h, 1)),
            "teacher_thumb_action_mean": float(teacher_thumb_sum / max(h, 1)),
            "teacher_nontrivial_fraction": float(teacher_active_count / max(h, 1)),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        # During the teacher/BC warm-up, the rollout is deliberately a blend
        # of the frozen scripted prior and the policy.  Selecting `best.pt`
        # from that mixed rollout can therefore save a checkpoint that looks
        # good only while the teacher is still active; once the blend decays,
        # the same actor may fail immediately.  Keep `latest.pt` for every
        # update, but only allow the deployable `best.pt` to be selected after
        # teacher control has fully ended.  This makes checkpoint comparison
        # match independent actor-only evaluation.
        teacher_active_until = max(
            int(args_cli.bc_only_updates),
            int(args_cli.teacher_blend_updates) if float(args_cli.teacher_blend) > 0.0 else 0,
            int(args_cli.teacher_bc_updates) if float(args_cli.teacher_bc_coef) > 0.0 else 0,
        )
        score = metrics["rollout_return_mean"] if update > teacher_active_until else -float("inf")
        save_checkpoint(args_cli.output_dir / "latest.pt", policy, optimizer, update, total_steps, metrics, cfg)
        if score > best_score:
            best_score = score
            save_checkpoint(args_cli.output_dir / "best.pt", policy, optimizer, update, total_steps, metrics, cfg)
        if update == 1 or update % 10 == 0:
            print(json.dumps(metrics), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
