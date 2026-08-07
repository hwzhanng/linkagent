"""Deployable pressure features and the frozen 113D observation builder."""

from __future__ import annotations

from dataclasses import dataclass

from proto import ArmState, HandState, TactileFrame, TomatoGraspPlan, TomatoTarget

from .contracts import PolicyManifest, TomatoPickAction, TomatoPickObservation
from .geometry import (
    clamp,
    pose_error_in_grasp_frame,
    quaternion_conjugate,
    rotate_vector,
    twist_in_grasp_frame,
)


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
    """Produce pressure and pressure-rate features available in sim and real."""

    feature_extractor_id = "g20_pressure_features"

    def __init__(self, *, contact_threshold_n: float = 0.6, slip_score_threshold: float = 0.8) -> None:
        if contact_threshold_n <= 0.0:
            raise ValueError("contact_threshold_n must be > 0")
        if not 0.0 < slip_score_threshold <= 1.0:
            raise ValueError("slip_score_threshold must be in (0,1]")
        self.contact_threshold_n = float(contact_threshold_n)
        self.slip_score_threshold = float(slip_score_threshold)
        self._previous_pressure_norm = (0.0,) * 5

    def reset(self) -> None:
        self._previous_pressure_norm = (0.0,) * 5

    def extract(
        self,
        frame: TactileFrame,
        *,
        now_ns: int,
        manifest: PolicyManifest,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...],
        tuple[bool, ...],
        tuple[bool, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[float | None, ...],
        tuple[float | None, ...],
        int,
        float,
        bool,
    ]:
        pressure_values: list[float] = []
        force_valid_values: list[bool] = []
        matrix_valid_values: list[bool] = []
        force_age_values: list[float] = []
        matrix_age_values: list[float] = []
        normal_values: list[float | None] = []
        tangential_values: list[float | None] = []
        contact_count = 0
        max_normal = 0.0
        slip_detected = False
        for finger in frame.fingers:
            force_age_s = (
                max(0.0, (now_ns - finger.force_sample_monotonic_ns) / 1e9)
                if finger.force_sample_monotonic_ns is not None
                else float("inf")
            )
            matrix_age_s = (
                max(0.0, (now_ns - finger.matrix_sample_monotonic_ns) / 1e9)
                if finger.matrix_sample_monotonic_ns is not None
                else float("inf")
            )
            force_ok = finger.force_valid and force_age_s <= manifest.max_force_age_s
            matrix_ok = finger.matrix_valid and matrix_age_s <= manifest.max_matrix_age_s
            force_valid_values.append(force_ok)
            matrix_valid_values.append(matrix_ok)
            force_age_values.append(clamp(force_age_s / manifest.max_force_age_s, 0.0, 1.0))
            matrix_age_values.append(clamp(matrix_age_s / manifest.max_matrix_age_s, 0.0, 1.0))

            normal = finger.normal_force_n if force_ok else None
            tangent = finger.tangential_force_n if force_ok else None
            normal_values.append(normal)
            tangential_values.append(tangent)
            contact = bool(normal is not None and normal >= self.contact_threshold_n)
            if contact:
                contact_count += 1
            max_normal = max(max_normal, normal or 0.0)
            shear_ratio = (tangent or 0.0) / max(normal or 0.0, self.contact_threshold_n)
            slip_score = clamp((shear_ratio - 0.20) / 1.20, 0.0, 1.0) if contact else 0.0
            slip_detected = slip_detected or slip_score >= self.slip_score_threshold
            pressure_values.append(
                clamp((normal or 0.0) / 20.0, 0.0, 1.0) if force_ok else 0.0
            )
        pressure = tuple(pressure_values)
        pressure_rate = tuple(
            clamp(current - previous, -1.0, 1.0)
            for current, previous in zip(pressure, self._previous_pressure_norm)
        )
        self._previous_pressure_norm = pressure
        return (
            pressure,
            pressure_rate,
            tuple(force_valid_values),
            tuple(matrix_valid_values),
            tuple(force_age_values),
            tuple(matrix_age_values),
            tuple(normal_values),
            tuple(tangential_values),
            contact_count,
            max_normal,
            slip_detected,
        )


class ObservationBuilder:
    """Build exactly the deployable actor vector and health summary."""

    def __init__(self, manifest: PolicyManifest, extractor: TactileFeatureExtractor) -> None:
        if not isinstance(manifest, PolicyManifest):
            raise TypeError("manifest must be a PolicyManifest")
        if extractor.feature_extractor_id != manifest.tactile_feature_extractor_id:
            raise ValueError("tactile feature extractor does not match policy manifest")
        self.manifest = manifest
        self.extractor = extractor

    @staticmethod
    def _age(now_ns: int, sample_ns: int, limit_s: float) -> tuple[float, bool]:
        age_s = max(0.0, (now_ns - sample_ns) / 1e9)
        return clamp(age_s / limit_s, 0.0, 1.0), age_s <= limit_s

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
        target_age, target_valid = self._age(
            now_ns,
            target.sample_monotonic_ns,
            self.manifest.max_state_age_s[0],
        )
        arm_age, arm_fresh = self._age(
            now_ns,
            arm.stamp.sample_monotonic_ns,
            self.manifest.max_state_age_s[1],
        )
        hand_age, hand_fresh = self._age(
            now_ns,
            hand.stamp.sample_monotonic_ns,
            self.manifest.max_state_age_s[2],
        )
        arm_valid = arm.stamp.valid and arm_fresh and arm.enabled
        hand_valid = hand.stamp.valid and hand_fresh and not hand.fault_codes
        target_valid = (
            target_valid
            and target.observation_sequence == plan.source_observation_sequence
        )
        tactile_values = self.extractor.extract(
            tactile,
            now_ns=now_ns,
            manifest=self.manifest,
        )
        (
            finger_pressure,
            finger_pressure_rate,
            force_valid,
            matrix_valid,
            force_age,
            matrix_age,
            normal_force,
            tangential_force,
            contact_count,
            max_normal,
            slip_detected,
        ) = tactile_values

        joint_position_norm = tuple(
            clamp(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0)
            for value, lower, upper in zip(
                arm.joint_positions_rad,
                self.manifest.arm_position_lower_rad,
                self.manifest.arm_position_upper_rad,
            )
        )
        joint_velocity_norm = tuple(
            clamp(value / scale, -1.0, 1.0)
            for value, scale in zip(
                arm.joint_velocities_rad_s,
                self.manifest.arm_velocity_scale_rad_s,
            )
        )
        hand_joint_position_norm = tuple(
            clamp(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0)
            for value, lower, upper in zip(
                hand.joint_positions_rad,
                self.manifest.hand_position_lower_rad,
                self.manifest.hand_position_upper_rad,
            )
        )
        hand_joint_velocity_norm = tuple(
            clamp(value / scale, -1.0, 1.0)
            for value, scale in zip(
                hand.joint_velocities_rad_s,
                self.manifest.hand_velocity_scale_rad_s,
            )
        )
        gravity_direction = rotate_vector(
            quaternion_conjugate(plan.grasp_frame.quaternion_wxyz),
            (0.0, 0.0, -1.0),
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
        observation = TomatoPickObservation.pack(
            target_pose_error=pose_error_in_grasp_frame(
                reference=plan.contact_pose,
                current=arm.ee_pose,
                grasp_frame=plan.grasp_frame,
            ),
            ee_twist=twist_in_grasp_frame(
                arm.ee_twist,
                grasp_frame=plan.grasp_frame,
            ),
            arm_joint_position_norm=joint_position_norm,
            arm_joint_velocity_norm=joint_velocity_norm,
            hand_joint_position_norm=hand_joint_position_norm,
            hand_joint_velocity_norm=hand_joint_velocity_norm,
            finger_pressure_norm=finger_pressure,
            finger_pressure_rate_norm=finger_pressure_rate,
            gravity_direction=gravity_direction,
            previous_action=previous_action.values,
            target_arm_hand_valid=(
                float(target_valid),
                float(arm_valid),
                float(hand_valid),
            ),
            force_valid=tuple(float(value) for value in force_valid),
            target_arm_hand_age_norm=(target_age, arm_age, hand_age),
            force_age_norm=force_age,
            fruit_axes_norm=fruit_axes_norm,
            episode_progress=clamp(episode_progress, 0.0, 1.0),
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


__all__ = [
    "ObservationBuilder",
    "ObservationSnapshot",
    "TactileFeatureExtractor",
]
