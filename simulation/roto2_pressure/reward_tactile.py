"""Optional task costs for the stage2 rotation/contact tradeoff.

These costs never gate motion, redefine support, change termination thresholds,
or add observations. All coefficients default to zero for stage1 parity.
"""
import torch


def tactile_constraint_costs(pad_quality, pad_forces, distal_forces, peak_pressure,
                             *, quality_weight=0., offpad_weight=0., pressure_weight=0.,
                             force_limit_n=8., pressure_limit_pa=18000.):
    terms = {}
    if quality_weight:
        # Reward quality only at measured effective pad contacts. Top three
        # permits a fourth contacting finger to release without a hard gate.
        quality = pad_quality.clamp(0, 1) * (pad_forces > .15)
        terms['tactile_quality_cost'] = -quality_weight * (
            1 - quality.topk(3, dim=-1).values.mean(-1))
    if offpad_weight:
        # Only measured distal-link normal forces outside the active pad.
        # Palm/proximal support is deliberately not classified as bad contact.
        outside = (distal_forces - pad_forces).clamp_min(0)
        terms['tactile_offpad_cost'] = -offpad_weight * (
            outside / force_limit_n).clamp(max=1).mean(-1)
    if pressure_weight:
        # Anticipatory soft cost from half of the unchanged physical limit;
        # releasing a contact cannot generate a positive relief reward.
        excess = (peak_pressure / pressure_limit_pa - .5).clamp(0, 4)
        terms['tactile_pressure_cost'] = -pressure_weight * excess.square()
    return terms


def contact_motion_discount(rotation_reward, pad_quality, pad_forces, *, floor=1.):
    """Discount positive rotation without effective pads; never gate motion.

    Three good pads receive full credit, two and one receive partial credit,
    and zero pads retain a strictly positive floor for temporary release.
    Reverse-motion penalties and all physical/task acceptance stay unchanged.
    The returned adjustment is nonpositive and zero when there is no motion.
    """
    if not 0. < floor <= 1.:
        raise ValueError('contact motion floor must be in (0, 1]')
    quality = pad_quality.clamp(0, 1) * (pad_forces > .15)
    measured = quality.topk(3, dim=-1).values.mean(-1)
    return -(1. - floor) * (1. - measured) * rotation_reward.clamp_min(0.)
