"""Build the same compact Actor observation in simulation and deployment."""

from __future__ import annotations

from dataclasses import dataclass

from proto import ArmState, HandState, TactileFrame, TomatoGraspPlan, TomatoTarget

from .contracts import PolicyManifest, TomatoPickAction, TomatoPickObservation
from .geometry import clamp, quaternion_conjugate, rotate_vector


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    observation: TomatoPickObservation
    contact_count: int
    max_normal_force_n: float
    slip_detected: bool
    target_valid: bool
    arm_valid: bool
    hand_valid: bool
    force_valid: tuple[bool, ...]
    matrix_valid: tuple[bool, ...]
    normal_force_n: tuple[float | None, ...]
    tangential_force_n: tuple[float | None, ...]


class TactileFeatureExtractor:
    """Return only the five calibrated pressure resultants used by the Actor."""

    feature_extractor_id = "g20_pressure_5d_v1"

    def __init__(self, *, contact_threshold_n: float = 0.6) -> None:
        if contact_threshold_n <= 0.0:
            raise ValueError("contact_threshold_n must be positive")
        self.contact_threshold_n = float(contact_threshold_n)

    def reset(self) -> None:
        return None

    def extract(
        self,
        frame: TactileFrame,
        *,
        now_ns: int,
        manifest: PolicyManifest,
    ) -> tuple[
        tuple[float, ...],
        tuple[bool, ...],
        tuple[bool, ...],
        tuple[float | None, ...],
        tuple[float | None, ...],
        int,
        float,
        bool,
    ]:
        pressure: list[float] = []
        force_valid: list[bool] = []
        matrix_valid: list[bool] = []
        normal_values: list[float | None] = []
        tangential_values: list[float | None] = []
        contact_count = 0
        max_normal = 0.0
        slip_detected = False
        full_scale = float(
            manifest.normalization_stats.get("pressure_full_scale_n", 20.0)
        )
        for finger in frame.fingers:
            age_s = (
                max(0.0, (now_ns - finger.force_sample_monotonic_ns) / 1.0e9)
                if finger.force_sample_monotonic_ns is not None
                else float("inf")
            )
            force_ok = finger.force_valid and age_s <= manifest.max_force_age_s
            normal = finger.normal_force_n if force_ok else None
            tangent = finger.tangential_force_n if force_ok else None
            normal_values.append(normal)
            tangential_values.append(tangent)
            force_valid.append(force_ok)
            matrix_valid.append(finger.matrix_valid)
            pressure.append(clamp((normal or 0.0) / full_scale, 0.0, 1.0))
            if normal is not None and normal >= self.contact_threshold_n:
                contact_count += 1
                shear_ratio = (tangent or 0.0) / max(normal, self.contact_threshold_n)
                slip_detected = slip_detected or shear_ratio > 1.4
            max_normal = max(max_normal, normal or 0.0)
        return (
            tuple(pressure),
            tuple(force_valid),
            tuple(matrix_valid),
            tuple(normal_values),
            tuple(tangential_values),
            contact_count,
            max_normal,
            slip_detected,
        )


class ObservationBuilder:
    """Build the frozen 66D vector from deployment-available measurements."""

    def __init__(self, manifest: PolicyManifest, extractor: TactileFeatureExtractor) -> None:
        if extractor.feature_extractor_id != manifest.tactile_feature_extractor_id:
            raise ValueError("tactile extractor does not match the policy manifest")
        self.manifest = manifest
        self.extractor = extractor

    @staticmethod
    def _age(now_ns: int, sample_ns: int, limit_s: float) -> bool:
        return 0 <= now_ns - sample_ns <= limit_s * 1.0e9

    def build(
        self,
        *,
        now_ns: int,
        target: TomatoTarget,
        plan: TomatoGraspPlan,
        episode_progress: float,
        arm: ArmState,
        hand: HandState,
        tactile: TactileFrame,
        previous_action: TomatoPickAction,
    ) -> ObservationSnapshot:
        del episode_progress
        target_valid = (
            self._age(now_ns, target.sample_monotonic_ns, self.manifest.max_state_age_s[0])
            and target.observation_sequence == plan.source_observation_sequence
        )
        arm_valid = (
            arm.enabled
            and arm.stamp.valid
            and self._age(
                now_ns,
                arm.stamp.sample_monotonic_ns,
                self.manifest.max_state_age_s[1],
            )
        )
        hand_valid = (
            hand.stamp.valid
            and not hand.fault_codes
            and self._age(
                now_ns,
                hand.stamp.sample_monotonic_ns,
                self.manifest.max_state_age_s[2],
            )
        )
        (
            pressure,
            force_valid,
            matrix_valid,
            normal_force,
            tangential_force,
            contact_count,
            max_normal,
            slip_detected,
        ) = self.extractor.extract(tactile, now_ns=now_ns, manifest=self.manifest)

        palm_inverse = quaternion_conjugate(arm.ee_pose.quaternion_wxyz)
        fruit_delta_world = tuple(
            fruit - palm
            for fruit, palm in zip(target.center_pose.position_m, arm.ee_pose.position_m)
        )
        relative_scale = float(
            self.manifest.normalization_stats.get("relative_position_scale_m", 0.15)
        )
        fruit_center_in_palm = tuple(
            clamp(value / relative_scale, -1.0, 1.0)
            for value in rotate_vector(palm_inverse, fruit_delta_world)
        )
        pull_axis_world = rotate_vector(
            plan.grasp_frame.quaternion_wxyz, (0.0, 0.0, 1.0)
        )
        pull_axis_in_palm = rotate_vector(palm_inverse, pull_axis_world)
        linear_scale = float(
            self.manifest.normalization_stats.get("linear_velocity_scale_m_s", 0.5)
        )
        angular_scale = float(
            self.manifest.normalization_stats.get("angular_velocity_scale_rad_s", 2.0)
        )
        twist_in_palm = (
            *rotate_vector(palm_inverse, tuple(arm.ee_twist[:3])),
            *rotate_vector(palm_inverse, tuple(arm.ee_twist[3:])),
        )
        twist_norm = tuple(
            clamp(value / (linear_scale if index < 3 else angular_scale), -1.0, 1.0)
            for index, value in enumerate(twist_in_palm)
        )
        fruit_axes_norm = tuple(
            clamp(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0)
            if target_valid
            else 0.0
            for value, lower, upper in zip(
                target.semi_axes_m,
                self.manifest.fruit_axis_lower_m,
                self.manifest.fruit_axis_upper_m,
            )
        )
        hand_position_norm = tuple(
            clamp(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0)
            for value, lower, upper in zip(
                hand.joint_positions_rad,
                self.manifest.hand_position_lower_rad,
                self.manifest.hand_position_upper_rad,
            )
        )
        hand_velocity_norm = tuple(
            clamp(value / scale, -1.0, 1.0)
            for value, scale in zip(
                hand.joint_velocities_rad_s,
                self.manifest.hand_velocity_scale_rad_s,
            )
        )
        gravity_in_palm = rotate_vector(palm_inverse, (0.0, 0.0, -1.0))
        observation = TomatoPickObservation.pack(
            fruit_center_in_palm=fruit_center_in_palm,
            fruit_semi_axes_norm=fruit_axes_norm,
            pull_axis_in_palm=pull_axis_in_palm,
            ee_twist_in_palm_norm=twist_norm,
            hand_joint_position_norm=hand_position_norm,
            hand_joint_velocity_norm=hand_velocity_norm,
            finger_pressure_norm=pressure,
            gravity_in_palm=gravity_in_palm,
            previous_action=previous_action.values,
        )
        return ObservationSnapshot(
            observation=observation,
            contact_count=contact_count,
            max_normal_force_n=max_normal,
            slip_detected=slip_detected,
            target_valid=target_valid,
            arm_valid=arm_valid,
            hand_valid=hand_valid,
            force_valid=force_valid,
            matrix_valid=matrix_valid,
            normal_force_n=normal_force,
            tangential_force_n=tangential_force,
        )


__all__ = ["ObservationBuilder", "ObservationSnapshot", "TactileFeatureExtractor"]
