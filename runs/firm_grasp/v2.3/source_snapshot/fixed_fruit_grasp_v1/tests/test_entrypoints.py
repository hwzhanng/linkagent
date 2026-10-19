"""固定果实训练和评估入口的纯 CPU 工具测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from research.fixed_fruit_grasp_v1.run_utils import (
    configure_fixed_fruit_agent_cfg,
    agent_config_sha256,
    atomic_write_json,
    completed_episode_delta,
    contract_sha256,
    fixed_fruit_agent_settings,
    hash_run_inputs,
    InspectionWindowStopState,
    normalize_episode_record,
    open_hand_action_provider,
    resolve_checkpoint_pose,
    score_evaluation_batch,
    summarize_episode_rows,
    summarize_inspection_window,
    validate_checkpoint_compatibility,
)


class EntryPointUtilityTests(unittest.TestCase):
    """验证入口共享的身份、回合、检查点和开手基线约定。"""

    def test_atomic_json_maps_non_finite_values_to_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            atomic_write_json(path, {"missing": float("nan"), "value": 1.0})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"missing": None, "value": 1.0})

    def test_contract_identity_ignores_runtime_fields_but_keeps_asset_identity(self) -> None:
        base = {"stage": "contact", "num_envs": 32, "seed": 1, "asset_config": {"mass": 1.0}}
        same_task = {"stage": "hold", "num_envs": 1, "seed": 9, "asset_config": {"mass": 1.0}}
        changed_asset = {"stage": "contact", "asset_config": {"mass": 1.1}}
        self.assertEqual(contract_sha256(base), contract_sha256(same_task))
        self.assertNotEqual(contract_sha256(base), contract_sha256(changed_asset))

    def test_agent_configuration_comparison_allows_device_move_only(self) -> None:
        config = {"device": "cuda:0", "algorithm": {"learning_rate": 0.0003}}
        moved = {"device": "cpu", "algorithm": {"learning_rate": 0.0003}}
        changed = {"device": "cpu", "algorithm": {"learning_rate": 0.0001}}
        self.assertEqual(agent_config_sha256(config), agent_config_sha256(moved))
        self.assertNotEqual(agent_config_sha256(config), agent_config_sha256(changed))

    def test_entrypoint_agent_settings_use_fixed_schedule_and_conservative_init_std(self) -> None:
        agent_cfg = SimpleNamespace(
            algorithm=SimpleNamespace(learning_rate=1.0e-5, schedule="adaptive"),
            actor=SimpleNamespace(
                distribution_cfg=SimpleNamespace(init_std=0.20),
            ),
        )
        configure_fixed_fruit_agent_cfg(agent_cfg)
        self.assertEqual(
            fixed_fruit_agent_settings(agent_cfg, 14),
            {
                "num_actions": 14,
                "learning_rate": 1.0e-4,
                "schedule": "fixed",
                "initial_std": 0.08,
            },
        )

    def test_resume_hashes_ignore_smoke_tests_but_bind_semantic_sources_and_assets(self) -> None:
        semantic_files = (
            "research/fixed_fruit_grasp_v1/train.py",
            "research/fixed_fruit_grasp_v1/evaluate.py",
            "research/fixed_fruit_grasp_v1/run_utils.py",
            "research/fixed_fruit_grasp_v1/env.py",
            "research/fixed_fruit_grasp_v1/contract.py",
            "research/fixed_fruit_grasp_v1/physical_contacts.py",
            "research/fixed_fruit_grasp_v1/physical_hold.py",
            "research/fixed_fruit_grasp_v1/collision_cooking.py",
            "research/small_fruit_pinch_v1/source/agents.py",
        )
        identity = {"asset_config": {"fruit_diameter_m": 0.067}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_dir = root / "research/fixed_fruit_grasp_v1"
            package_dir.mkdir(parents=True)
            for relative_path in semantic_files:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("semantic-v1", encoding="utf-8")
            smoke = package_dir / "smoke.py"
            smoke.write_text("smoke-v1", encoding="utf-8")
            tests_dir = package_dir / "tests"
            tests_dir.mkdir()
            (tests_dir / "unrelated.py").write_text("test-v1", encoding="utf-8")

            original_hashes = hash_run_inputs(package_dir, identity)
            metadata = {
                "contract_id": "fixed_fruit_grasp_v1",
                "contract_sha256": contract_sha256(identity),
                "stage": "contact",
                "source_and_asset_hashes": original_hashes,
            }
            smoke.write_text("smoke-v2", encoding="utf-8")
            (tests_dir / "unrelated.py").write_text("test-v2", encoding="utf-8")
            noise_only_hashes = hash_run_inputs(package_dir, identity)
            self.assertNotEqual(
                original_hashes["package_source_sha256"],
                noise_only_hashes["package_source_sha256"],
            )
            self.assertEqual(
                original_hashes["training_semantics_sha256"],
                noise_only_hashes["training_semantics_sha256"],
            )
            self.assertEqual(
                validate_checkpoint_compatibility(
                    metadata,
                    identity,
                    "contact",
                    source_and_asset_hashes=noise_only_hashes,
                ),
                "full_resume",
            )

            (root / "research/fixed_fruit_grasp_v1/collision_cooking.py").write_text(
                "semantic-v2", encoding="utf-8"
            )
            changed_semantics = hash_run_inputs(package_dir, identity)
            with self.assertRaisesRegex(ValueError, "training_semantics_sha256"):
                validate_checkpoint_compatibility(
                    metadata,
                    identity,
                    "contact",
                    source_and_asset_hashes=changed_semantics,
                )

            changed_asset = dict(original_hashes)
            changed_asset["asset_files_manifest_sha256"] = "different-assets"
            with self.assertRaisesRegex(ValueError, "asset_files_manifest_sha256"):
                validate_checkpoint_compatibility(
                    metadata,
                    identity,
                    "contact",
                    source_and_asset_hashes=changed_asset,
                )

    def test_checkpoint_stage_transition_requires_same_task_identity(self) -> None:
        identity = {
            "stage": {"name": "contact", "required_hold_steps": 120},
            "physics": {
                "support": {
                    "kind": "declared_stem_spring",
                    "release_step": 120,
                    "release_time_s": 1.2,
                    "declared_break_force_n": 100.0,
                },
                "sim_dt_s": 0.0025,
            },
            "initial_pose": {"pregrasp_distance_m": 0.083},
            "asset_config": {"fruit_diameter_m": 0.067},
        }
        hold_identity = {
            **identity,
            "stage": {"name": "hold", "required_hold_steps": 60},
            "physics": {
                **identity["physics"],
                "support": {
                    **identity["physics"]["support"],
                    "release_step": 60,
                    "release_time_s": 0.6,
                },
            },
        }
        metadata = {
            "contract_id": "fixed_fruit_grasp_v1",
            "contract_sha256": contract_sha256(identity),
            "stage": "contact",
        }
        self.assertEqual(
            validate_checkpoint_compatibility(metadata, hold_identity, "hold"),
            "actor_only_stage_transition",
        )
        with self.assertRaises(ValueError):
            changed_physics = {**hold_identity, "physics": {**hold_identity["physics"], "sim_dt_s": 0.005}}
            validate_checkpoint_compatibility(metadata, changed_physics, "hold")

    def test_evaluation_pose_defaults_to_checkpoint_and_rejects_mismatch(self) -> None:
        identity = {
            "initial_pose": {
                "pregrasp_distance_m": 0.083,
                "closed_grasp_fruit_center_m": [0.01, -0.02, 0.03],
            }
        }
        distance, center = resolve_checkpoint_pose(identity, None, None)
        self.assertAlmostEqual(distance, 0.083)
        self.assertEqual(center, (0.01, -0.02, 0.03))
        self.assertEqual(
            resolve_checkpoint_pose(identity, 0.083, (0.01, -0.02, 0.03)),
            (0.083, (0.01, -0.02, 0.03)),
        )
        with self.assertRaises(ValueError):
            resolve_checkpoint_pose(identity, 0.084, None)

    def test_episode_rows_preserve_null_and_missing_reason(self) -> None:
        row = normalize_episode_record({"env_id": 0, "episode_id": 4, "success": None})
        self.assertIsNone(row["success"])
        self.assertIn("success", row["missing_reason"])
        self.assertIsNone(row["contact_fraction"])
        self.assertIn("contact_fraction", row["missing_reason"])

    def test_episode_delta_reads_only_new_rows_and_detects_truncation(self) -> None:
        row = {"env_id": 0, "episode_id": 1, "success": False, "reason": "drop"}

        class Environment:
            completed_episodes = [row]

        rows, cursor = completed_episode_delta(Environment(), 0)
        self.assertEqual(cursor, 1)
        self.assertEqual(len(rows), 1)
        Environment.completed_episodes.clear()
        with self.assertRaises(RuntimeError):
            completed_episode_delta(Environment(), cursor)

    def test_missing_contact_is_not_interpreted_as_zero(self) -> None:
        rows = [
            {"success": False, "contact_fraction": 0.0, "reason": "drop"}
            for _ in range(31)
        ] + [{"success": False, "reason": "missing_contact"}]
        summary = summarize_inspection_window(rows)
        self.assertIsNone(summary["contact_fraction_zero"])
        self.assertEqual(summary["contact_fraction_missing_count"], 1)
        self.assertIsNotNone(summary["contact_fraction_zero_missing_reason"])
        self.assertIsNone(summary["required_contact_fraction_zero"])

    def test_required_contact_stop_uses_contiguous_windows_and_resets_on_progress(self) -> None:
        stopper = InspectionWindowStopState()
        zero_required = summarize_inspection_window(
            [
                {"success": False, "contact_fraction": 0.4, "required_contact_fraction": 0.0}
                for _ in range(32)
            ]
        )
        positive_required = summarize_inspection_window(
            [
                {"success": False, "contact_fraction": 0.4, "required_contact_fraction": 0.1}
                for _ in range(32)
            ]
        )
        missing_required = summarize_inspection_window(
            [{"success": False, "reason": "missing_contact"} for _ in range(32)]
        )

        self.assertIsNone(stopper.observe(zero_required))
        self.assertIsNone(stopper.observe(positive_required))
        self.assertEqual(stopper.consecutive_zero_contact_windows, 0)
        self.assertIsNone(stopper.observe(zero_required))
        self.assertEqual(
            stopper.observe(zero_required),
            "two_consecutive_zero_required_contact_windows",
        )
        self.assertIsNone(missing_required["required_contact_fraction_zero"])
        missing_stopper = InspectionWindowStopState()
        self.assertIsNone(missing_stopper.observe(missing_required))
        self.assertEqual(missing_stopper.consecutive_zero_contact_windows, 0)

    def test_eval_batch_scores_only_remaining_rows_and_keeps_overshoot_unscored(self) -> None:
        batch = [
            {"env_id": 3, "episode_id": 8, "success": False},
            {"env_id": 1, "episode_id": 9, "success": None},
            {"env_id": 0, "episode_id": 7, "success": True},
        ]
        scored, unscored = score_evaluation_batch(batch, scored_count=19, target_count=20)

        self.assertEqual([(row["env_id"], row["score_index"]) for row in scored], [(0, 20)])
        self.assertEqual([row["env_id"] for row in unscored], [1, 3])
        self.assertTrue(scored[0]["scored"])
        self.assertTrue(all(row["scored"] is False for row in unscored))
        self.assertTrue(all(row["score_index"] is None for row in unscored))
        self.assertIsNone(unscored[0]["success"])
        self.assertIn("success", unscored[0]["missing_reason"])

        after_target, still_unscored = score_evaluation_batch(
            [{"env_id": 2, "episode_id": 10, "success": False}],
            scored_count=20,
            target_count=20,
        )
        self.assertEqual(after_target, [])
        self.assertEqual(len(still_unscored), 1)
        self.assertFalse(still_unscored[0]["scored"])
        with self.assertRaisesRegex(ValueError, "at most 20"):
            score_evaluation_batch([], scored_count=0, target_count=21)

    def test_inspection_window_requires_repeated_major_immediate_failure(self) -> None:
        rows = [
            {
                "success": False,
                "contact_fraction": 0.25,
                "reason": "spawn_failure",
                "failure_phase": "immediate",
            }
            for _ in range(16)
        ] + [
            {"success": True, "contact_fraction": 0.5, "reason": None}
            for _ in range(16)
        ]
        first = summarize_inspection_window(rows)
        second = summarize_inspection_window(rows)
        self.assertEqual(first["major_immediate_failure_reason"], "spawn_failure")
        self.assertEqual(
            first["major_immediate_failure_reason"],
            second["major_immediate_failure_reason"],
        )

    def test_complete_episode_denominator_is_not_reduced_for_missing_success(self) -> None:
        summary = summarize_episode_rows(
            [
                {"success": True, "contact_fraction": 0.5},
                {"success": None, "contact_fraction": 0.1},
            ]
        )
        self.assertEqual(summary["episodes_completed"], 2)
        self.assertEqual(summary["success_observed"], 1)
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(summary["success_missing_count"], 1)

    def test_neutral_baseline_setter_precedes_inverse_action_methods(self) -> None:
        class Environment:
            called = False

            def set_open_hand_baseline(self, enabled: bool) -> None:
                self.called = enabled

            def open_hand_action(self) -> list[float]:
                raise AssertionError("the neutral-target override is the selected provider")

        env = Environment()
        name, action = open_hand_action_provider(env)
        self.assertEqual(name, "set_open_hand_baseline")
        self.assertIsNone(action)
        self.assertTrue(env.called)

    def test_explicit_open_action_fallback_is_used_only_without_setter(self) -> None:
        class Environment:
            def default_joint_pos_action(self) -> list[float]:
                return [0.25] * 14

        name, action = open_hand_action_provider(Environment())
        self.assertEqual(name, "default_joint_pos_action")
        self.assertEqual(len(action), 14)


if __name__ == "__main__":
    unittest.main()
