"""Analytic isolated-physics checks for TomatoPick Physics-v2.

These helpers intentionally do not import Isaac Sim.  They qualify equations,
units, parameter ordering, and numerical guards before the same cases are run
inside the frozen Isaac/PhysX runtime.  Passing them is necessary but is not an
integrated simulator qualification.
"""

from __future__ import annotations

import math

import torch


def semi_implicit_free_fall(
    *,
    height_m: float,
    mass_kg: float,
    dt_s: float,
    steps: int,
    gravity_m_s2: float = 9.81,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Integrate unconstrained vertical motion with semi-implicit Euler."""

    if height_m <= 0.0 or mass_kg <= 0.0 or dt_s <= 0.0 or steps < 1:
        raise ValueError("height, mass, dt, and steps must be positive")
    if gravity_m_s2 <= 0.0:
        raise ValueError("gravity magnitude must be positive")
    position = torch.empty(steps + 1, device=device, dtype=dtype)
    velocity = torch.empty_like(position)
    position[0] = height_m
    velocity[0] = 0.0
    for index in range(steps):
        velocity[index + 1] = velocity[index] - gravity_m_s2 * dt_s
        position[index + 1] = position[index] + velocity[index + 1] * dt_s
    energy = mass_kg * gravity_m_s2 * position + 0.5 * mass_kg * velocity.square()
    return {"position_m": position, "velocity_m_s": velocity, "energy_j": energy}


def incline_will_slide(static_friction: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Return the ideal Coulomb incline threshold decision."""

    if static_friction.ndim != 1 or not bool(torch.isfinite(static_friction).all()):
        raise ValueError("static_friction must be a finite one-dimensional tensor")
    if bool((static_friction < 0.0).any()) or not 0.0 <= angle_deg < 90.0:
        raise ValueError("invalid friction or incline angle")
    return math.tan(math.radians(angle_deg)) > static_friction


def simulate_damped_compression(
    *,
    mass_kg: float,
    stiffness_n_m: float,
    damping_n_s_m: float,
    initial_indentation_m: float,
    initial_closing_speed_m_s: float,
    dt_s: float,
    steps: int,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Integrate a bilateral linear spring-damper while contact is compressed.

    The bilateral form is used only as a conservative numerical energy check;
    the production force law remains unilateral and is checked separately.
    """

    if (
        mass_kg <= 0.0
        or stiffness_n_m <= 0.0
        or damping_n_s_m < 0.0
        or initial_indentation_m < 0.0
        or dt_s <= 0.0
        or steps < 1
    ):
        raise ValueError("invalid damped-compression parameters")
    indentation = torch.empty(steps + 1, dtype=dtype)
    speed = torch.empty_like(indentation)
    indentation[0] = initial_indentation_m
    speed[0] = initial_closing_speed_m_s
    for index in range(steps):
        acceleration = (
            -stiffness_n_m * indentation[index]
            - damping_n_s_m * speed[index]
        ) / mass_kg
        speed[index + 1] = speed[index] + acceleration * dt_s
        indentation[index + 1] = indentation[index] + speed[index + 1] * dt_s
    energy = (
        0.5 * mass_kg * speed.square()
        + 0.5 * stiffness_n_m * indentation.square()
    )
    return {
        "indentation_m": indentation,
        "closing_speed_m_s": speed,
        "energy_j": energy,
    }


def restitution_energy_fraction(restitution: float) -> float:
    """Return post/pre translational kinetic-energy ratio for a normal impact."""

    if not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must lie in [0, 1]")
    return restitution * restitution


__all__ = [
    "incline_will_slide",
    "restitution_energy_fraction",
    "semi_implicit_free_fall",
    "simulate_damped_compression",
]
