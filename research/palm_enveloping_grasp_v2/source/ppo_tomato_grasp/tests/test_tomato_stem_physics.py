"""CPU tests for the pure-torch tomato stem physics and success verifier."""

from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    TOMATO_CRITIC_STATE_DIM,
    block,
    pack_actor_observation,
    pack_critic_state,
    pack_tomato_critic_state,
)
from simulation.ppo_tomato_grasp.tomato_stem_physics import (
    evaluate_stage1_hold_conditions,
    evaluate_success_conditions,
    quat_box_minus_wxyz,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    quat_rotate_vector_wxyz,
    quat_to_rot_matrix_6d,
    stem_attachment_wrench,
    stem_load_components,
    update_stem_damage,
)


def _actor_observation(num: int = 1) -> torch.Tensor:
    return torch.zeros((num, ACTOR_OBSERVATION_DIM))


class TestQuaternionHelpers(unittest.TestCase):
    def test_conjugate_and_mul_identity(self):
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # 180 deg about z
        product = quat_mul_wxyz(quat, quat_conjugate_wxyz(quat))
        torch.testing.assert_close(product, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    def test_rotate_vector(self):
        quat = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])  # 90 deg about z
        rotated = quat_rotate_vector_wxyz(quat, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5, rtol=1e-5)

    def test_box_minus_recovers_angle(self):
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        rotated = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])  # 90 deg about z
        error = quat_box_minus_wxyz(rotated, identity)
        torch.testing.assert_close(
            error, torch.tensor([[0.0, 0.0, 0.5 * torch.pi]]), atol=1e-4, rtol=1e-4
        )

    def test_quat_to_rot_matrix_6d_identity(self):
        six = quat_to_rot_matrix_6d(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        torch.testing.assert_close(six, torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))

    def test_quat_to_rot_matrix_6d_rotation(self):
        six = quat_to_rot_matrix_6d(
            torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])  # 90 deg about z
        )
        expected = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(six, expected, atol=1e-5, rtol=1e-5)


class TestStemWrench(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.mass = torch.tensor([0.100, 0.100])
        self.inertia = torch.tensor([3.6e-5, 3.6e-5])
        self.axes = torch.tensor([[0.030, 0.030, 0.030], [0.030, 0.030, 0.030]])
        self.pos = torch.tensor([[0.0, 0.0, 0.20], [0.0, 0.0, 0.20]])
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        self.vel = torch.zeros((self.n, 3))
        self.anchor = self.pos + torch.tensor([[0.0, 0.0, 0.030], [0.0, 0.0, 0.030]])

    def test_zero_force_at_rest(self):
        force, torque, attachment, axis = stem_attachment_wrench(
            mass_kg=self.mass,
            inertia_kg_m2=self.inertia,
            axes_m=self.axes,
            tomato_pos_w=self.pos,
            tomato_quat_wxyz=self.quat,
            tomato_lin_vel_w=self.vel,
            tomato_ang_vel_w=self.vel,
            anchor_pos_w=self.anchor,
            anchor_quat_wxyz=self.quat,
            omega_linear_rad_s=30.0,
            zeta_linear=1.0,
            omega_angular_rad_s=12.0,
            zeta_angular=1.0,
            maximum_force_n=120.0,
            maximum_torque_nm=0.8,
        )
        # No gravity compensation: the palm carries the weight at rest.
        torch.testing.assert_close(force, torch.zeros((self.n, 3)), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torque, torch.zeros((self.n, 3)), atol=1e-6, rtol=1e-6)

    def test_tension_when_anchor_rises(self):
        raised = self.anchor + torch.tensor([[0.0, 0.0, 0.010], [0.0, 0.0, 0.010]])
        force, _, attachment, axis = stem_attachment_wrench(
            mass_kg=self.mass,
            inertia_kg_m2=self.inertia,
            axes_m=self.axes,
            tomato_pos_w=self.pos,
            tomato_quat_wxyz=self.quat,
            tomato_lin_vel_w=self.vel,
            tomato_ang_vel_w=self.vel,
            anchor_pos_w=raised,
            anchor_quat_wxyz=self.quat,
            omega_linear_rad_s=30.0,
            zeta_linear=1.0,
            omega_angular_rad_s=12.0,
            zeta_angular=1.0,
            maximum_force_n=120.0,
            maximum_torque_nm=0.8,
        )
        expected = self.mass * 900.0 * 0.010
        torch.testing.assert_close(force[:, 2], expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(force[:, :2], torch.zeros((self.n, 2)), atol=1e-6, rtol=1e-6)
        # Pull axis points from anchor toward the tomato.
        torch.testing.assert_close(axis[:, 2], torch.full((self.n,), -1.0), atol=1e-6, rtol=1e-6)

    def test_force_clamp(self):
        raised = self.anchor + torch.tensor([[0.0, 0.0, 0.50], [0.0, 0.0, 0.50]])
        force, _, _, _ = stem_attachment_wrench(
            mass_kg=self.mass,
            inertia_kg_m2=self.inertia,
            axes_m=self.axes,
            tomato_pos_w=self.pos,
            tomato_quat_wxyz=self.quat,
            tomato_lin_vel_w=self.vel,
            tomato_ang_vel_w=self.vel,
            anchor_pos_w=raised,
            anchor_quat_wxyz=self.quat,
            omega_linear_rad_s=30.0,
            zeta_linear=1.0,
            omega_angular_rad_s=12.0,
            zeta_angular=1.0,
            maximum_force_n=10.0,
            maximum_torque_nm=0.8,
        )
        self.assertTrue(bool((force.norm(dim=-1) <= 10.0 + 1e-6).all()))


class TestStemLoadComponents(unittest.TestCase):
    def setUp(self):
        self.axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        self.limits = {
            "tensile_limit_n": torch.tensor([2.0, 2.0]),
            "shear_limit_n": torch.tensor([1.6, 1.6]),
            "bending_limit_nm": torch.tensor([0.04, 0.04]),
            "torsional_limit_nm": torch.tensor([0.02, 0.02]),
        }

    def test_pure_tension(self):
        force = torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, -2.0]])
        components = stem_load_components(force, torch.zeros_like(force), self.axis, **self.limits)
        torch.testing.assert_close(components["tension"], torch.ones(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components["shear"], torch.zeros(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components["torsion"], torch.zeros(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components["bending"], torch.zeros(2), atol=1e-6, rtol=1e-6)

    def test_pure_shear(self):
        force = torch.tensor([[1.6, 0.0, 0.0], [1.6, 0.0, 0.0]])
        components = stem_load_components(force, torch.zeros_like(force), self.axis, **self.limits)
        torch.testing.assert_close(components["shear"], torch.ones(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components["tension"], torch.zeros(2), atol=1e-6, rtol=1e-6)

    def test_pure_torsion(self):
        torque = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]])
        components = stem_load_components(torch.zeros((2, 3)), torque, self.axis, **self.limits)
        torch.testing.assert_close(components["torsion"], torch.ones(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components["bending"], torch.zeros(2), atol=1e-6, rtol=1e-6)

    def test_pure_bending(self):
        torque = torch.tensor([[0.04, 0.0, 0.0], [0.04, 0.0, 0.0]])
        components = stem_load_components(torch.zeros((2, 3)), torque, self.axis, **self.limits)
        torch.testing.assert_close(components["bending"], torch.ones(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components["torsion"], torch.zeros(2), atol=1e-6, rtol=1e-6)

    def test_combined_is_sqrt_of_squares(self):
        force = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        torque = torch.tensor([[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
        components = stem_load_components(force, torque, self.axis, **self.limits)
        expected = torch.sqrt(
            torch.tensor([0.5**2, 0.5**2]) + torch.tensor([0.5**2, 0.5**2])
        )
        torch.testing.assert_close(components["combined"], expected, atol=1e-6, rtol=1e-6)


class TestStemDamage(unittest.TestCase):
    def test_break_after_critical_duration(self):
        attached = torch.ones(2, dtype=torch.bool)
        damage = torch.zeros(2)
        steps = 240
        for _ in range(steps):
            damage, broken = update_stem_damage(
                damage,
                torch.ones(2),
                dt_s=1.0 / 120.0,
                critical_duration_s=torch.full((2,), 1.0),
                onset_ratio=0.55,
                recovery_time_s=1.5,
                attached=attached,
            )
        self.assertTrue(bool(broken.all()))
        self.assertTrue(bool((damage >= 1.0).all()))

    def test_no_break_from_single_spike(self):
        attached = torch.ones(2, dtype=torch.bool)
        damage = torch.zeros(2)
        damage, broken = update_stem_damage(
            damage,
            torch.full((2,), 5.0),
            dt_s=1.0 / 120.0,
            critical_duration_s=torch.full((2,), 1.0),
            onset_ratio=0.55,
            recovery_time_s=1.5,
            attached=attached,
        )
        self.assertFalse(bool(broken.any()))

    def test_recovers_below_onset(self):
        attached = torch.ones(2, dtype=torch.bool)
        damage = torch.full((2,), 0.8)
        damage, broken = update_stem_damage(
            damage,
            torch.zeros(2),
            dt_s=0.5,
            critical_duration_s=torch.full((2,), 1.0),
            onset_ratio=0.55,
            recovery_time_s=1.0,
            attached=attached,
        )
        self.assertTrue(bool((damage < 0.8).all()))
        self.assertFalse(bool(broken.any()))

    def test_detached_never_breaks(self):
        attached = torch.zeros(2, dtype=torch.bool)
        damage = torch.zeros(2)
        for _ in range(240):
            damage, broken = update_stem_damage(
                damage,
                torch.ones(2),
                dt_s=1.0 / 120.0,
                critical_duration_s=torch.full((2,), 1.0),
                onset_ratio=0.55,
                recovery_time_s=1.5,
                attached=attached,
            )
        self.assertFalse(bool(broken.any()))
        torch.testing.assert_close(damage, torch.zeros(2))


class TestSuccessVerifier(unittest.TestCase):
    def test_stage1_requires_palm_and_all_five_digits(self):
        kwargs = dict(
            palm_support=torch.tensor([True, True, True]),
            hold_streak=torch.tensor([20, 20, 19]),
            center_distance=torch.tensor([0.02, 0.02, 0.02]),
            relative_speed=torch.tensor([0.01, 0.01, 0.01]),
            dropped=torch.tensor([False, False, False]),
            force_safe=torch.tensor([True, True, True]),
            damage_ok=torch.tensor([True, True, True]),
            maximum_center_distance_m=0.08,
            maximum_relative_speed_m_s=0.35,
            success_hold_steps=20,
        )
        conditions = evaluate_stage1_hold_conditions(
            enclosure_contact=torch.tensor(
                [[True, True, True, True, True],
                 [True, True, True, True, False],
                 [True, True, True, True, True]]
            ),
            **kwargs,
        )
        self.assertTrue(bool(conditions["success"][0]))
        self.assertFalse(bool(conditions["success"][1]))
        self.assertFalse(bool(conditions["success"][2]))

    def _all_true(self) -> dict[str, torch.Tensor]:
        return {
            "secure_before_pick": torch.ones(2, dtype=torch.bool),
            "break_opposition_ok": torch.ones(2, dtype=torch.bool),
            "stem_broken": torch.ones(2, dtype=torch.bool),
            "hold_streak": torch.full((2,), 20, dtype=torch.long),
            "center_distance": torch.zeros(2),
            "relative_speed": torch.zeros(2),
            "dropped": torch.zeros(2, dtype=torch.bool),
            "force_safe": torch.ones(2, dtype=torch.bool),
            "damage_ok": torch.ones(2, dtype=torch.bool),
        }

    def test_full_success(self):
        kwargs = self._all_true()
        conditions = evaluate_success_conditions(
            maximum_center_distance_m=0.08,
            maximum_relative_speed_m_s=0.35,
            success_hold_steps=20,
            **kwargs,
        )
        self.assertTrue(bool(conditions["success"].all()))
        for name, value in conditions.items():
            if name == "success":
                continue
            self.assertTrue(bool(value.all()), msg=f"condition {name} failed")

    def test_each_condition_is_required(self):
        def check(mutator, name):
            kwargs = self._all_true()
            mutator(kwargs)
            conditions = evaluate_success_conditions(
                maximum_center_distance_m=0.08,
                maximum_relative_speed_m_s=0.35,
                success_hold_steps=20,
                **kwargs,
            )
            self.assertFalse(bool(conditions["success"].any()), msg=f"{name} not required")

        check(lambda k: k.__setitem__("secure_before_pick", torch.zeros(2, dtype=torch.bool)), "secure_before_pick")
        check(lambda k: k.__setitem__("break_opposition_ok", torch.zeros(2, dtype=torch.bool)), "break_opposition_ok")
        check(lambda k: k.__setitem__("stem_broken", torch.zeros(2, dtype=torch.bool)), "stem_broken")
        check(lambda k: k.__setitem__("hold_streak", torch.full((2,), 19, dtype=torch.long)), "hold_streak")
        check(lambda k: k.__setitem__("center_distance", torch.full((2,), 0.09)), "center_distance")
        check(lambda k: k.__setitem__("relative_speed", torch.full((2,), 0.40)), "relative_speed")
        check(lambda k: k.__setitem__("dropped", torch.ones(2, dtype=torch.bool)), "dropped")
        check(lambda k: k.__setitem__("force_safe", torch.zeros(2, dtype=torch.bool)), "force_safe")
        check(lambda k: k.__setitem__("damage_ok", torch.zeros(2, dtype=torch.bool)), "damage_ok")

    def test_hold_streak_resets_by_condition(self):
        kwargs = self._all_true()
        kwargs["hold_streak"] = torch.tensor([19, 20], dtype=torch.long)
        conditions = evaluate_success_conditions(
            maximum_center_distance_m=0.08,
            maximum_relative_speed_m_s=0.35,
            success_hold_steps=20,
            **kwargs,
        )
        self.assertTrue(bool(conditions["success"][1]))
        self.assertFalse(bool(conditions["success"][0]))


class TestTomatoCriticContract(unittest.TestCase):
    def test_tomato_critic_pack_roundtrip(self):
        obs = _actor_observation(3)
        privileged = {
            "physical_fingertip_force": torch.zeros((3, 5)),
            "physical_palm_force": torch.zeros((3, 1)),
            "contact_points_in_palm": torch.zeros((3, 15)),
            "palm_contact_point_in_palm": torch.zeros((3, 3)),
            "contact_counts": torch.zeros((3, 6)),
            "object_mass": torch.zeros((3, 1)),
            "static_friction": torch.zeros((3, 1)),
            "effective_stiffness": torch.zeros((3, 1)),
            "peak_indentation": torch.zeros((3, 1)),
            "force_exposure": torch.zeros((3, 1)),
            "impact_exposure": torch.zeros((3, 1)),
            "damage_score": torch.zeros((3, 1)),
            "stem_loads": torch.zeros((3, 4)),
            "stem_damage": torch.zeros((3, 1)),
            "slip_speed": torch.zeros((3, 1)),
            "tactile_latent_gain": torch.zeros((3, 5)),
            "tactile_latent_bias": torch.zeros((3, 5)),
        }
        state = pack_tomato_critic_state(obs, **privileged)
        self.assertEqual(state.shape, (3, TOMATO_CRITIC_STATE_DIM))
        self.assertTrue(torch.equal(state[:, :ACTOR_OBSERVATION_DIM], obs))
        self.assertTrue(torch.equal(block(state, "stem_loads"), privileged["stem_loads"]))
        self.assertTrue(torch.equal(block(state, "slip_speed"), privileged["slip_speed"]))

    def test_tomato_critic_requires_all_blocks(self):
        obs = _actor_observation(1)
        with self.assertRaises(ValueError):
            pack_tomato_critic_state(obs, physical_fingertip_force=torch.zeros((1, 5)))
        bad = {name: torch.zeros((1, int(sl.stop - sl.start))) for name, sl in
               __import__("simulation.ppo_tomato_grasp.hand_ppo_contract", fromlist=["x"]).TOMATO_CRITIC_PRIVILEGED_BLOCKS.items()}
        bad["stem_loads"] = torch.zeros((1, 3))
        with self.assertRaises(ValueError):
            pack_tomato_critic_state(obs, **bad)

    def test_legacy_148d_contract_unchanged(self):
        obs = _actor_observation(2)
        privileged = {name: torch.zeros((2, int(sl.stop - sl.start))) for name, sl in
                      __import__("simulation.ppo_tomato_grasp.hand_ppo_contract", fromlist=["x"]).CRITIC_PRIVILEGED_BLOCKS.items()}
        state = pack_critic_state(obs, **privileged)
        self.assertEqual(state.shape, (2, CRITIC_STATE_DIM))


if __name__ == "__main__":
    unittest.main()
