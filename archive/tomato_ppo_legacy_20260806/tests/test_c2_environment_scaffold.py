"""C2-1 contract environment and isolated training-runtime gates."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from module.skills.tomato_pick import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT_ID,
    TomatoPickAction,
)
from simulation.env_manager import preflight
from simulation.training import build_training_plan, execute_training
from simulation.tomato_pick import ContractTomatoPickEnv, RuleTomatoPickPolicy
from simulation.tomato_pick.config import (
    ASSET_PROFILE_PATH,
    ISAAC_RUNTIME_PROFILE_PATH,
    PPO_PROFILE_PATH,
    RUNTIME_MANIFEST_PATH,
    SIM2REAL_PROFILE_PATH,
    load_asset_profile,
    load_isaac_runtime_profile,
    load_object,
    load_ppo_profile,
    load_sim2real_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class C2EnvironmentScaffoldTests(unittest.TestCase):
    def test_contract_environment_exactly_exposes_frozen_spaces(self) -> None:
        env = ContractTomatoPickEnv()
        try:
            observation, info = env.reset(seed=7)
            self.assertEqual(len(observation.values), OBSERVATION_DIM)
            self.assertEqual(env.observation_space_spec.shape, (OBSERVATION_DIM,))
            self.assertEqual(env.action_space_spec.shape, (ACTION_DIM,))
            self.assertEqual(env.action_space_spec.low, -1.0)
            self.assertEqual(env.action_space_spec.high, 1.0)
            self.assertEqual(info["backend_kind"], "contract_smoke_not_physics")
            self.assertFalse(info["physics_valid"])
            self.assertFalse(info["suitable_for_ppo_training"])
            self.assertFalse(info["actor_contains_privileged_state"])
        finally:
            env.close()

    def test_rule_policy_completes_one_hundred_seed_contract_smoke(self) -> None:
        env = ContractTomatoPickEnv()
        policy = RuleTomatoPickPolicy()
        try:
            for seed in range(100):
                observation, _ = env.reset(seed=seed)
                policy.reset()
                final_info = None
                for _ in range(120):
                    action = policy.act(observation)
                    observation, _, terminated, truncated, info = env.step(action)
                    if info["success"]:
                        self.assertTrue(info["stem_detached"])
                        self.assertTrue(info["retained_without_slip"])
                        self.assertFalse(info["slip_detected"])
                        self.assertFalse(info["dropped"])
                    if terminated or truncated:
                        final_info = info
                        break
                self.assertIsNotNone(final_info, seed)
                self.assertTrue(final_info["success"], (seed, final_info))
                self.assertEqual(final_info["stem_break_event_count"], 1)
        finally:
            env.close()

    def test_seed_reset_is_deterministic(self) -> None:
        def rollout(seed: int) -> list[tuple[str, bool, int]]:
            env = ContractTomatoPickEnv()
            policy = RuleTomatoPickPolicy()
            trace: list[tuple[str, bool, int]] = []
            try:
                observation, _ = env.reset(seed=seed)
                for _ in range(120):
                    observation, _, terminated, truncated, info = env.step(
                        policy.act(observation)
                    )
                    trace.append(
                        (
                            str(info["phase"]),
                            bool(info["stem_break_event"]),
                            int(info["episode_step"]),
                        )
                    )
                    if terminated or truncated:
                        break
            finally:
                env.close()
            return trace

        self.assertEqual(rollout(19), rollout(19))

    def test_opening_after_contact_is_drop_and_never_success(self) -> None:
        env = ContractTomatoPickEnv()
        policy = RuleTomatoPickPolicy()
        try:
            observation, _ = env.reset(seed=0)
            while env.state_summary["phase"] == "contact":
                observation, _, _, _, _ = env.step(policy.act(observation))
            opening = [0.0] * ACTION_DIM
            for index in (0, 1, 2, 3, 4, 11, 12, 13, 14, 15):
                opening[6 + index] = -1.0
            _, reward, terminated, truncated, info = env.step(
                TomatoPickAction(tuple(opening))
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["dropped"])
            self.assertFalse(info["stem_detached"])
            self.assertFalse(info["success"])
            self.assertLess(reward, 0.0)
        finally:
            env.close()

    def test_stem_break_event_is_one_shot(self) -> None:
        env = ContractTomatoPickEnv()
        policy = RuleTomatoPickPolicy()
        events = 0
        try:
            observation, _ = env.reset(seed=3)
            for _ in range(120):
                observation, _, terminated, truncated, info = env.step(
                    policy.act(observation)
                )
                events += int(info["stem_break_event"])
                if terminated or truncated:
                    self.assertTrue(info["success"])
                    self.assertEqual(info["stem_break_event_count"], 1)
                    break
            self.assertEqual(events, 1)
        finally:
            env.close()

    def test_asset_profile_records_training_ready_assets_and_launch_authorization(self) -> None:
        assets = load_asset_profile()
        robot = assets["robot"]
        task_assets = assets["task_assets"]
        self.assertEqual(robot["status"], "selected_by_c2_0")
        self.assertTrue((ROOT / robot["usd"]).is_file())
        self.assertTrue((ROOT / robot["mass_manifest"]).is_file())
        self.assertEqual(task_assets["tomato"]["collision"], "convex_hull_nonuniform_scale")
        self.assertIn("kelvin_voigt", task_assets["stem"]["breakable_constraint"])
        self.assertTrue(assets["ready_for_isaac_physics"])
        self.assertTrue(assets["ready_for_ppo_training"])
        self.assertTrue(assets["training_launch_authorized"])
        self.assertEqual(assets["training_launch_blockers"], [])
        self.assertTrue(assets["physics_smoke"]["passed"])
        self.assertTrue(assets["physics_smoke"]["six_dof_cartesian_ik_exercised"])
        self.assertTrue(assets["physics_smoke"]["sixteen_hand_joint_targets_exercised"])
        self.assertTrue(assets["reward_audit"]["passed"])
        self.assertTrue(assets["ppo_optimizer_probe"]["passed"])
        self.assertFalse(assets["provenance"]["old_fruit_grasp_baseline_reused"])

    def test_ppo_profile_matches_frozen_interface_and_lifecycle(self) -> None:
        profile = load_ppo_profile()
        compatibility = profile["compatibility"]
        policy = profile["policy"]
        gate = profile["activation_gate"]
        self.assertEqual(profile["trainer"]["algorithm"], "PPO")
        self.assertEqual(profile["trainer"]["backend"], "rsl_rl")
        self.assertEqual(policy["actor_observation_dim"], OBSERVATION_DIM)
        self.assertEqual(policy["action_dim"], ACTION_DIM)
        self.assertEqual(compatibility["observation_layout_id"], OBSERVATION_LAYOUT_ID)
        self.assertEqual(compatibility["action_layout_id"], ACTION_LAYOUT_ID)
        self.assertTrue(gate["asset_profile_ready_for_ppo_training"])
        self.assertTrue(gate["isaac_runtime_preflight_passed"])
        self.assertTrue(gate["direct_rl_env_cuda_smoke_passed"])
        self.assertTrue(gate["six_dof_and_sixteen_joint_action_smoke_passed"])
        self.assertTrue(gate["two_update_optimizer_probe_passed"])
        self.assertTrue(gate["sim2real_full_and_nominal_audits_passed"])
        self.assertTrue(gate["ppo_zero_training_preflight_passed"])
        self.assertIn(
            profile["status"],
            {
                "ready_waiting_for_explicit_training_authorization",
                "active_formal_training",
            },
        )
        self.assertEqual(
            gate["explicit_training_authorized"],
            profile["status"] == "active_formal_training",
        )
        self.assertFalse(gate["old_baseline_inputs_reused"])

    def test_managed_isaac_runtime_records_real_gpu_and_robot_smoke(self) -> None:
        profile = load_isaac_runtime_profile()
        primary = profile["primary"]
        verification = profile["verification"]
        self.assertEqual(profile["status"], "installed_and_smoke_tested")
        self.assertEqual(primary["isaac_sim"], "6.0.1.0")
        self.assertEqual(primary["python"], "3.12.3")
        self.assertTrue(Path(primary["launcher"]).is_file())
        self.assertTrue(Path(primary["python_executable"]).is_file())
        self.assertTrue(verification["cuda_physx_steps_passed"])
        self.assertEqual(verification["formal_robot_num_bodies"], 71)
        self.assertEqual(verification["formal_robot_num_independent_joints"], 56)
        self.assertEqual(verification["formal_robot_authored_mimic_relations"], 10)
        self.assertFalse(verification["formal_robot_source_mutated"])

    def test_sim2real_profile_uses_deployable_measurements_and_correlated_mass(self) -> None:
        profile = load_sim2real_profile()
        interface = profile["policy_interface"]
        perception = profile["perception_bridge"]
        tomato = profile["correlated_tomato_sampling"]
        self.assertEqual(interface["observation_layout_id"], OBSERVATION_LAYOUT_ID)
        self.assertFalse(interface["actor_privileged_state_allowed"])
        self.assertFalse(perception["raw_point_cloud_in_actor"])
        self.assertFalse(perception["exact_sim_pose_in_actor"])
        self.assertTrue(tomato["do_not_sample_mass_independently"])
        self.assertEqual(tomato["mass_formula"], "4/3*pi*a*b*c*density")
        randomization = profile["training_randomization_policy"]
        self.assertEqual(
            randomization["mode"], "full_target_distribution_from_first_rollout"
        )
        self.assertFalse(randomization["curriculum_or_later_patch"])

    def test_environment_roles_never_auto_install_or_access_devices(self) -> None:
        manifest = load_object(RUNTIME_MANIFEST_PATH)
        policy = manifest["policy"]
        self.assertFalse(policy["automatic_environment_mutation"])
        self.assertFalse(policy["automatic_package_install"])
        self.assertFalse(policy["automatic_download"])
        self.assertFalse(policy["real_can_access"])
        self.assertFalse(policy["vendor_sdk_import"])
        self.assertFalse(policy["model_api_credentials_visible"])
        self.assertFalse(manifest["roles"]["contract_smoke"]["can_train_deployable_policy"])
        self.assertFalse(manifest["roles"]["isaac_ppo"]["connects_real_can"])

    def test_current_contract_preflight_passes_and_isaac_preflight_blocks(self) -> None:
        contract = preflight("contract_smoke")
        isaac = preflight("isaac_ppo", isaaclab_root=ROOT / "does-not-exist")
        self.assertTrue(contract.ready)
        self.assertFalse(contract.blockers)
        self.assertFalse(isaac.ready)
        failed = {check.check_id for check in isaac.checks if not check.passed}
        self.assertTrue(
            {
                "isaaclab_launcher_present",
                "training_dispatcher_present",
            }.issubset(failed)
        )
        self.assertNotIn("ppo_gate_open", failed)
        self.assertNotIn("task_assets_ready", failed)
        self.assertNotIn("isaac_adapter_implemented", failed)

    def test_training_plan_is_non_executable_while_gate_is_blocked(self) -> None:
        plan = build_training_plan(isaaclab_root=ROOT / "does-not-exist")
        self.assertFalse(plan.ready)
        self.assertEqual(plan.command, ())
        self.assertEqual(plan.task_id, "LinkAgent-TomatoPick-Direct-v0")
        self.assertEqual(plan.trainer_backend, "rsl_rl")
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            execute_training(plan, explicitly_confirmed=True)
        with self.assertRaisesRegex(PermissionError, "explicit confirmation"):
            execute_training(plan, explicitly_confirmed=False)

    def test_importing_agent_core_or_simulation_never_loads_heavy_runtime(self) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys; import module.runtime; import simulation; "
                    "import simulation.tomato_pick; print(json.dumps(sorted(sys.modules)))"
                ),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        loaded = set(json.loads(probe.stdout))
        forbidden = {
            "gymnasium",
            "isaaclab",
            "isaaclab_tasks",
            "isaaclab_rl",
            "rsl_rl",
            "torch",
        }
        self.assertTrue(forbidden.isdisjoint(loaded), forbidden & loaded)

    def test_simulation_configs_contain_no_secret_fields(self) -> None:
        secret_names = {"api_key", "authorization", "password", "secret", "token"}

        def keys(value: object) -> set[str]:
            found: set[str] = set()
            if isinstance(value, dict):
                for key, child in value.items():
                    found.add(str(key).lower())
                    found.update(keys(child))
            elif isinstance(value, list):
                for child in value:
                    found.update(keys(child))
            return found

        for path in (
            ASSET_PROFILE_PATH,
            ISAAC_RUNTIME_PROFILE_PATH,
            PPO_PROFILE_PATH,
            RUNTIME_MANIFEST_PATH,
            SIM2REAL_PROFILE_PATH,
        ):
            self.assertTrue(secret_names.isdisjoint(keys(load_object(path))), path)


if __name__ == "__main__":
    unittest.main()
