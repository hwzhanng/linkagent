from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT))
from research.thumb_pad_grasp_v1 import record_video  # noqa: E402


def _args(*extra: str) -> list[str]:
    return [
        "--mode", "evaluate", "--policy", "checkpoint",
        "--checkpoint", "/tmp/same-task.pt", "--output", "/tmp/eval",
        *extra,
    ]


def test_cli_keeps_checkpoint_evaluator_and_bounded_camera_args() -> None:
    args, run_args = record_video._parse_args(
        _args("--episodes", "2", "--camera", "side", "--frame-every", "2", "--device", "cpu")
    )
    assert (args.camera, args.frame_every, args.episodes, args.num_envs, args.view_env_index) == (
        "side", 2, 2, 1, 0
    )
    assert "--enable_cameras" in run_args
    assert run_args[run_args.index("--episodes") + 1] == "2"
    assert run_args[run_args.index("--num-envs") + 1] == "1"
    assert run_args[run_args.index("--checkpoint") + 1] == "/tmp/same-task.pt"


def test_cli_selects_environment_three_in_four_env_evaluation() -> None:
    args, run_args = record_video._parse_args(
        _args("--episodes", "8", "--num-envs", "4", "--view-env-index", "3")
    )
    assert (args.episodes, args.num_envs, args.view_env_index) == (8, 4, 3)
    assert run_args[run_args.index("--episodes") + 1] == "8"
    assert run_args[run_args.index("--num-envs") + 1] == "4"


@pytest.mark.parametrize("extra", [
    ("--mode", "train", "--policy", "checkpoint", "--checkpoint", "x", "--output", "o"),
    ("--mode", "evaluate", "--policy", "open", "--checkpoint", "x", "--output", "o"),
    ("--mode", "evaluate", "--policy", "checkpoint", "--checkpoint", "", "--output", "o"),
    ("--episodes", "0"),
    ("--episodes", "9"),
    ("--num-envs", "2"),
    ("--num-envs", "5"),
    ("--num-envs", "4", "--view-env-index", "4"),
    ("--view-env-index", "-1"),
])
def test_cli_rejects_non_checkpoint_or_unbounded_evaluation(extra: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        record_video._parse_args(_args(*extra))


def test_camera_uses_actual_selected_environment_origin_and_binds_receipt(tmp_path: Path) -> None:
    config = SimpleNamespace(eye=None, lookat=None, backend_source="visualizer")
    state = {"camera": "side", "view_env_index": 3, "requested_num_envs": 4}
    scene = SimpleNamespace(
        env_origins=np.asarray([
            (0.0, 0.0, 0.0), (5.0, 0.0, 0.0),
            (0.0, 7.0, 0.0), (11.0, -4.0, 2.5),
        ])
    )

    record_video._configure_recording_camera(config, scene, state)

    eye_local, lookat_local = record_video._CAMERAS["side"]
    origin = (11.0, -4.0, 2.5)
    expected_eye = tuple(origin[i] + eye_local[i] for i in range(3))
    expected_lookat = tuple(origin[i] + lookat_local[i] for i in range(3))
    assert config.eye == expected_eye
    assert config.lookat == expected_lookat
    assert config.backend_source == "renderer"
    assert state["camera_origin_world_m"] == list(origin)
    assert state["actual_num_envs"] == 4

    receipt = record_video._receipt({
        **state,
        "record_video_source_sha256": "source-hash",
        "frame_every": 2,
        "frames_sampled": 0,
        "blank_frames_skipped": 0,
        "frame_limit_reached": False,
        "camera_configured": True,
        "capture_error": None,
        "encoder_error": None,
        "runtime_error": None,
        "runtime_started": True,
        "evaluation_return_code": 0,
        "video_sha256": None,
        "partial_sha256": None,
        "video_path": None,
        "partial_path": None,
        "output": tmp_path,
    })
    assert receipt["schema_version"] == "thumb_pad_video_receipt.v1"
    assert receipt["record_video_source_sha256"] == "source-hash"
    assert receipt["selected_env_index"] == 3
    assert receipt["actual_num_envs"] == 4
    assert receipt["camera"]["origin_type"] == "env"
    assert receipt["camera"]["env_index"] == 3
    assert receipt["camera"]["origin_world_m"] == list(origin)
    assert receipt["camera"]["eye_world_m"] == list(expected_eye)
    assert receipt["camera"]["lookat_world_m"] == list(expected_lookat)


def test_caption_identifies_selected_environment_episode_and_age() -> None:
    class Env:
        episode_length_buf = np.asarray([1, 2, 3, 17])
        _video_episode_number = 5

    state = {"view_env_index": 3}
    record_video._update_contact_overlay(Env(), state)
    assert state["overlay_caption"] == "env 3 | episode 5 | age 17"


def test_episode_number_advances_only_when_selected_environment_finishes() -> None:
    class DoneEnv:
        def __init__(self, _cfg: object, *, render_mode: str | None, **_kwargs: object) -> None:
            self.render_mode = render_mode
            self.num_envs = 4
            self.episode_length_buf = np.zeros(4, dtype=np.int64)

        def step(self, _action: object) -> tuple[None, None, np.ndarray, np.ndarray, dict]:
            self.episode_length_buf += 1
            self.episode_length_buf[3] = 0
            return None, None, np.asarray([True, False, False, True]), np.zeros(4, dtype=bool), {}

        def close(self) -> None:
            pass

    state = {
        "camera": "close", "frame_every": 2, "episodes": 1,
        "capture_enabled": False, "view_env_index": 3, "requested_num_envs": 4,
        "frames_sampled": 0, "blank_frames_skipped": 0, "frame_limit_reached": False,
        "video_sha256": None, "partial_sha256": None, "video_path": None,
        "partial_path": None,
    }
    env = record_video._recording_env(DoneEnv, state)(SimpleNamespace(viewer=SimpleNamespace()))
    env.step(None)
    assert env._video_episode_number == 2
    assert state["actual_num_envs"] == 4


def test_record_video_is_excluded_from_critical_source_manifest() -> None:
    from research.thumb_pad_grasp_v1 import run as evaluator

    dependency = evaluator.SOURCE_ROOT / "research/fixed_fruit_grasp_v1/env.py"
    files, _manifest_sha256 = evaluator._critical_source_manifest({
        "source_dependencies": {
            "test_dependency": {
                "source_path": str(dependency),
                "sha256": evaluator._sha256_file(dependency),
            }
        }
    })
    assert "thumb_pad_grasp_v1/record_video.py" not in files
    assert "thumb_pad_grasp_v1/env.py" in files


def test_wrapper_records_actual_rgb_frames_and_video_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Writer:
        def __init__(self, path: Path) -> None:
            self.stream = path.open("wb")

        def append_data(self, frame: np.ndarray) -> None:
            self.stream.write(frame.tobytes())

        def close(self) -> None:
            self.stream.close()

    monkeypatch.setattr(record_video, "_open_writer", lambda path, _fps: Writer(path))
    state = {
        "camera": "close", "frame_every": 2, "episodes": 1,
        "capture_enabled": True, "camera_configured": False,
        "encoder_error": None, "capture_error": None, "runtime_error": None,
        "runtime_started": True, "evaluation_return_code": 0,
        "frames_sampled": 0, "blank_frames_skipped": 0, "frame_limit_reached": False,
        "video_sha256": None, "partial_sha256": None, "video_path": None,
        "partial_path": None, "receipt_path": tmp_path / "video_receipt.json",
        "output": tmp_path, "inside_run": False, "receipt_before_close": False,
    }

    class DummyEnv:
        def __init__(self, _cfg: object, *, render_mode: str | None, **_kwargs: object) -> None:
            self.render_mode = render_mode
            self.steps = 0

        def render(self) -> np.ndarray:
            return np.full((4, 5, 3), 23, dtype=np.uint8)

        def step(self, _action: object) -> int:
            self.steps += 1
            return self.steps

        def close(self) -> None:
            pass

    cfg = SimpleNamespace(viewer=SimpleNamespace(), video_recorder=None)
    env = record_video._recording_env(DummyEnv, state)(cfg)
    assert env.render_mode == "rgb_array"
    assert cfg.viewer.eye == record_video._CAMERAS["close"][0]
    for _ in range(5):
        env.step(None)
    env.close()

    assert state["frames_sampled"] == 3
    path = state["video_path"]
    assert path.is_file()
    assert state["video_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_receipt_separates_video_capture_from_evaluation_status(tmp_path: Path) -> None:
    state = {
        "camera": "front", "frame_every": 4, "episodes": 2,
        "frames_sampled": 1, "blank_frames_skipped": 0, "frame_limit_reached": False,
        "camera_configured": True, "capture_error": None, "encoder_error": None,
        "runtime_error": None, "runtime_started": True, "evaluation_return_code": 0,
        "video_sha256": "abc", "partial_sha256": None, "video_path": tmp_path / "evaluation.mp4",
        "partial_path": None, "output": tmp_path, "receipt_before_close": False,
    }
    receipt = record_video._receipt(state)
    assert receipt["status"] == "RECORDED"
    assert receipt["evaluation_status"] is None
    assert receipt["camera"]["verified_by_valid_rgb_frames"] is True

