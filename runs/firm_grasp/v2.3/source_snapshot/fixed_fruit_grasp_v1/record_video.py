#!/usr/bin/env python3
"""用真实 env.render() 帧包裹固定果实评估器，录像失败不改变评估结果。"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_RUN_ROOT = Path(__file__).resolve().parents[3]
_EVALUATE_PATH = _SOURCE_ROOT / "research/fixed_fruit_grasp_v1/evaluate.py"
_FROZEN_EVALUATE_SHA256 = "45523659bcdeee5838ef6e3756c73405dc6fcfaba2914f56583d5f722caae63b"
_FRAME_EVERY_STEPS = 4
_VIDEO_FPS = 5
_MAX_EPISODES = 2
_MAX_FRAMES_PER_EPISODE = 300
_MAX_TOTAL_FRAMES = _MAX_EPISODES * _MAX_FRAMES_PER_EPISODE
_CAMERA_EYE = (0.48, -0.62, 0.38)
_CAMERA_LOOKAT = (0.0, 0.0, 0.1125)
_NO_ADAPTER_REASON = "no verified camera capture adapter is configured for this task"
_CACHE_ALIAS_ROOT = Path("/tmp/linkbot-ppo-grasp-learning-20261013/cache")


def _option_value(argv: list[str], name: str) -> str | None:
    value = None
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(name + "="):
            value = token.partition("=")[2]
    return value


def _bounded_eval_args(argv: list[str], *, enable_cameras: bool) -> list[str]:
    """固定单环境，并把请求回合数限制在 1 到 2。"""
    kept: list[str] = []
    episodes = _MAX_EPISODES
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--episodes":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError("--episodes 后需要正整数")
            episodes = min(int(argv[index + 1]), _MAX_EPISODES)
            index += 2
        elif token.startswith("--episodes="):
            episodes = min(int(token.partition("=")[2]), _MAX_EPISODES)
            index += 1
        elif token == "--num-envs":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError("--num-envs 后需要数值；录像适配器固定使用 1 个环境")
            index += 2
        elif token.startswith("--num-envs="):
            index += 1
        else:
            kept.append(token)
            index += 1
    if enable_cameras and "--enable_cameras" not in kept:
        kept.append("--enable_cameras")
    return kept + ["--episodes", str(episodes), "--num-envs", "1"]


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _configure_external_cache(output: Path) -> dict[str, str]:
    """使用已验证的短路径缓存根，避免 Warp/NVRTC 临时路径超长。"""
    cache_root = _CACHE_ALIAS_ROOT / output.stem
    names = {
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
        "CUDA_CACHE_PATH": "cuda",
        "__GL_SHADER_DISK_CACHE_PATH": "gl",
        "OMNI_USER_DIR": "omni-user",
        "IMAGEIO_USERDIR": "imageio",
        "MPLCONFIGDIR": "matplotlib",
        "PYTHONPYCACHEPREFIX": "pycache",
    }
    configured: dict[str, str] = {}
    for variable, name in names.items():
        directory = cache_root / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
        configured[variable] = str(directory)
    sys.dont_write_bytecode = True
    return configured


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt(state: dict[str, Any]) -> None:
    """原子落盘并 fsync；回执故障只写 stderr，不传播到 evaluator。"""
    receipt = state["receipt_path"]
    videos = state["videos"]
    good = [item for item in videos if item.get("status") == "recorded"]
    if good and len(good) == len(videos) and state.get("capture_enabled"):
        status = "RECORDED"
    elif good or any(item.get("path") for item in videos):
        status = "PARTIAL"
    elif not state.get("capture_enabled"):
        status = "UNAVAILABLE"
    else:
        status = "NO_VIDEO"
    if good and len(good) == len(videos):
        missing_reason = None
    else:
        missing_reason = (
            state.get("actual_runtime_error")
            or state.get("capture_error")
            or state.get("encoder_error")
            or next(
                (
                    item.get("reason_detail") or item.get("error")
                    for item in videos
                    if item.get("reason_detail") or item.get("error")
                ),
                None,
            )
            or (
                "runtime_camera_missing: Kit runtime entered, but no valid RGB frames were recorded."
                if state.get("runtime_started")
                else "runtime_not_started: evaluator did not enter its Kit runtime."
            )
        )
    camera_missing_reason = (
        None
        if state["frames_sampled"] > 0
        else state.get("actual_runtime_error")
        or state.get("capture_error")
        or state.get("encoder_error")
        or (
            "runtime_camera_missing: Kit runtime entered, but no valid RGB frames were recorded."
            if state.get("runtime_started")
            else "runtime_not_started: evaluator did not enter its Kit runtime."
        )
    )
    payload = {
        "adapter": "record_video.py / Isaac Lab env.render() RGB frames",
        "status": status,
        "missing_reason": missing_reason,
        "summary_path": str(state["summary_path"]),
        "video_directory": str(state["video_dir"]),
        "evaluate_sha256": state["evaluate_sha256"],
        "evaluate_sha256_matches_frozen": state["evaluate_sha256"] == _FROZEN_EVALUATE_SHA256,
        "evaluator_summary_video_recording": {
            "recorded": False,
            "missing_reason": _NO_ADAPTER_REASON,
            "interpretation": (
                "这是未修改的 evaluator 内置字段；本回执单独描述外置 camera adapter 的录像。"
            ),
        },
        "capture_source": "env.render() RGB array from the active Isaac Lab renderer",
        "implementation_camera_verification": {
            "verified": state["frames_sampled"] > 0,
            "missing_reason": camera_missing_reason,
        },
        "sampling": {
            "policy_steps_per_frame": _FRAME_EVERY_STEPS,
            "policy_rate_hz": 20,
            "nominal_fps": _VIDEO_FPS,
            "max_episodes": _MAX_EPISODES,
            "max_frames_per_episode": _MAX_FRAMES_PER_EPISODE,
            "max_total_frames": _MAX_TOTAL_FRAMES,
        },
        "camera": {"eye_world_m": _CAMERA_EYE, "lookat_world_m": _CAMERA_LOOKAT},
        "cache_directories": state["cache_directories"],
        "encoder_preflight_error": state.get("encoder_error"),
        "capture_error": state.get("capture_error"),
        "actual_runtime_error": state.get("actual_runtime_error"),
        "runtime_started": state.get("runtime_started", False),
        "frames_sampled": state["frames_sampled"],
        "blank_frames_skipped": state["blank_frames_skipped"],
        "evaluation_return_code": state.get("evaluation_return_code"),
        "videos": videos,
        "written_before_simulation_app_close": True,
    }
    temporary_path = None
    try:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=receipt.parent,
            prefix=f".{receipt.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, receipt)
        _sync_directory(receipt.parent)
    except Exception as exc:
        print(f"录像回执写入失败（评估结果保持原样）：{type(exc).__name__}: {exc}", file=sys.stderr)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _probe_encoder() -> tuple[bool, str | None]:
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg

        executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not executable.is_file() or not os.access(executable, os.X_OK):
            return False, f"ffmpeg executable unavailable: {executable}"
        return True, None
    except Exception as exc:
        return False, f"imageio/ffmpeg unavailable: {type(exc).__name__}: {exc}"


def _recording_env_class(base_class: type, state: dict[str, Any]) -> type:
    class RecordingFixedFruitGraspEnv(base_class):
        """只在 evaluator 已启动 Kit 后启用 RGB 渲染和有界写盘。"""

        def __init__(self, cfg: Any, **kwargs: Any) -> None:
            self._record_policy_steps = 0
            self._record_last_sampled_step = -1
            self._record_frame_count = 0
            self._record_writer = None
            self._record_partial_path: Path | None = None
            self._record_failure: str | None = None
            if state["capture_enabled"]:
                cfg.viewer.eye = _CAMERA_EYE
                cfg.viewer.lookat = _CAMERA_LOOKAT
                cfg.viewer.origin_type = "world"
                if cfg.video_recorder is not None:
                    recorder_class = cfg.video_recorder.class_type

                    class OptionalVideoRecorder:
                        """相机后端初始化失败时保留物理评估。"""

                        def __init__(self, recorder_cfg: Any, scene: Any) -> None:
                            try:
                                self._recorder = recorder_class(recorder_cfg, scene)
                            except Exception as exc:
                                self._recorder = None
                                state["capture_enabled"] = False
                                state["capture_error"] = f"camera backend init: {type(exc).__name__}: {exc}"

                        def render_rgb_array(self) -> Any:
                            if self._recorder is None:
                                return None
                            return self._recorder.render_rgb_array()

                    cfg.video_recorder.class_type = OptionalVideoRecorder
                render_mode = "rgb_array"
            else:
                render_mode = None
            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _capture_policy_frame(self) -> None:
            if not state["capture_enabled"] or self._record_writer is False:
                return
            if self._record_frame_count >= _MAX_FRAMES_PER_EPISODE:
                return
            if sum(item["frame_count"] for item in state["videos"]) + self._record_frame_count >= _MAX_TOTAL_FRAMES:
                return
            try:
                import imageio.v2 as imageio
                import numpy as np

                frame = self.render()
                if frame is None:
                    raise RuntimeError(state.get("capture_error") or "env.render() returned None")
                rgb = np.asarray(frame)
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"expected uint8 RGB frame, received shape={rgb.shape}, dtype={rgb.dtype}")
                if not np.any(rgb):
                    # Kit PerspectiveVideo 预热时会返回全零占位帧；跳过，不写合成画面。
                    state["blank_frames_skipped"] += 1
                    return
                if self._record_writer is None:
                    episode_number = len(state["videos"]) + 1
                    self._record_partial_path = state["video_dir"] / f"episode-{episode_number:02d}.partial.mp4"
                    self._record_writer = imageio.get_writer(
                        str(self._record_partial_path),
                        format="FFMPEG",
                        mode="I",
                        fps=_VIDEO_FPS,
                        codec="libx264",
                        macro_block_size=1,
                    )
                self._record_writer.append_data(rgb)
                self._record_frame_count += 1
                state["frames_sampled"] += 1
            except Exception as exc:
                self._record_failure = f"{type(exc).__name__}: {exc}"
                state["capture_error"] = self._record_failure
                state["capture_enabled"] = False
                writer = self._record_writer
                self._record_writer = False
                if writer not in (None, False):
                    try:
                        writer.close()
                    except Exception:
                        pass

        def _finish_video(self, row: dict[str, Any] | None) -> None:
            writer = self._record_writer
            partial_path = self._record_partial_path
            self._record_writer = None
            self._record_partial_path = None
            label = (
                "stage-success" if row is not None and row.get("success") is True
                else "failure" if row is not None and row.get("success") is False
                else "incomplete"
            )
            video_number = len(state["videos"]) + 1
            item: dict[str, Any] = {
                "episode_number": video_number,
                "episode_id": row.get("episode_id") if row else None,
                "stage_success": row.get("success") if row else None,
                "goal_success": row.get("goal_success") if row else None,
                "reason": row.get("reason") if row else "evaluation ended before episode completion",
                "frame_count": self._record_frame_count,
                "fps": _VIDEO_FPS,
                "policy_steps_per_frame": _FRAME_EVERY_STEPS,
                "status": "unavailable",
                "path": None,
                "sha256": None,
            }
            if writer not in (None, False):
                try:
                    writer.close()
                    if partial_path is None or not partial_path.is_file():
                        raise FileNotFoundError("ffmpeg did not produce the temporary MP4")
                    with partial_path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    final_path = state["video_dir"] / f"episode-{video_number:02d}-{label}.mp4"
                    os.replace(partial_path, final_path)
                    _sync_directory(final_path.parent)
                    item.update(
                        status="recorded" if row is not None else "incomplete",
                        path=str(final_path),
                        sha256=_sha256_file(final_path),
                    )
                except Exception as exc:
                    item["status"] = "encoding_failed"
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    if partial_path is not None and partial_path.exists():
                        item["partial_path"] = str(partial_path)
            elif self._record_frame_count == 0:
                item["reason_detail"] = self._record_failure or (
                    state.get("capture_error")
                    or state.get("encoder_error")
                    or "episode ended before the first 4-step sample"
                )
            if writer is False and partial_path is not None and partial_path.is_file():
                item.update(
                    status="encoding_failed",
                    partial_path=str(partial_path),
                    error=self._record_failure or state.get("capture_error"),
                )
            self._record_frame_count = 0
            self._record_failure = None
            state["videos"].append(item)
            _write_receipt(state)

        def _reset_idx(self, env_ids: Any) -> None:
            completed = getattr(self, "completed_episodes", None)
            previous_count = len(completed) if isinstance(completed, list) else 0
            if (
                self._record_policy_steps > 0
                and self._record_policy_steps % _FRAME_EVERY_STEPS == 0
                and self._record_last_sampled_step != self._record_policy_steps
            ):
                self._record_last_sampled_step = self._record_policy_steps
                self._capture_policy_frame()
            super()._reset_idx(env_ids)
            completed = getattr(self, "completed_episodes", None)
            if isinstance(completed, list):
                for row in completed[previous_count:]:
                    self._finish_video(row)

        def step(self, action: Any) -> Any:
            self._record_policy_steps += 1
            result = super().step(action)
            if (
                self._record_policy_steps % _FRAME_EVERY_STEPS == 0
                and self._record_last_sampled_step != self._record_policy_steps
            ):
                self._record_last_sampled_step = self._record_policy_steps
                self._capture_policy_frame()
            return result

        def close(self) -> None:
            if self._record_writer not in (None, False) or self._record_frame_count:
                self._finish_video(None)
            else:
                _write_receipt(state)
            super().close()

    RecordingFixedFruitGraspEnv.__name__ = "RecordingFixedFruitGraspEnv"
    return RecordingFixedFruitGraspEnv


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "-h" in arguments or "--help" in arguments:
        evaluate = importlib.import_module("research.fixed_fruit_grasp_v1.evaluate")
        try:
            return int(evaluate.main(arguments) or 0)
        except SystemExit as exc:
            return int(exc.code or 0)

    try:
        output_argument = _option_value(arguments, "--output")
        if not output_argument:
            raise ValueError("必须提供 evaluator 的 --output 路径")
        output = Path(output_argument).expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        output = output.resolve()
        if not _under_root(output.parent, _RUN_ROOT):
            raise ValueError(f"summary、视频和缓存必须位于外盘任务目录 {_RUN_ROOT}")
        video_dir = output.with_name(f"{output.stem}.videos")
        receipt_path = output.with_name(f"{output.stem}.video_receipt.json")
        reserved_outputs = (
            output,
            output.with_name(f"{output.stem}.episodes.jsonl"),
            output.with_name(f"{output.stem}.failure.json"),
            output.with_name(f"{output.stem}.receipt.json"),
            receipt_path,
        )
        collisions = [str(path) for path in reserved_outputs if path.exists()]
        if collisions or (video_dir.exists() and any(video_dir.iterdir())):
            raise FileExistsError(
                "录像评估输出已存在，请使用新的 --output 名称："
                + ", ".join(collisions + [str(video_dir)] if video_dir.exists() else collisions)
            )
        video_dir.mkdir(parents=True, exist_ok=True)
        cache_directories = _configure_external_cache(output)
        capture_enabled, encoder_error = _probe_encoder()
        bounded_arguments = _bounded_eval_args(arguments, enable_cameras=capture_enabled)
        evaluate_hash = _sha256_file(_EVALUATE_PATH)
        if evaluate_hash != _FROZEN_EVALUATE_SHA256:
            raise RuntimeError(
                "冻结的 evaluate.py SHA-256 不匹配；拒绝启动，避免适配器套在不同 evaluator 上"
            )
    except Exception as exc:
        print(f"录像适配器预检失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    state: dict[str, Any] = {
        "summary_path": output,
        "video_dir": video_dir,
        "receipt_path": receipt_path,
        "evaluate_sha256": evaluate_hash,
        "cache_directories": cache_directories,
        "capture_enabled": capture_enabled,
        "encoder_error": encoder_error,
        "capture_error": None,
        "actual_runtime_error": None,
        "runtime_started": False,
        "evaluation_return_code": None,
        "frames_sampled": 0,
        "blank_frames_skipped": 0,
        "videos": [],
    }
    evaluate = importlib.import_module("research.fixed_fruit_grasp_v1.evaluate")
    original_run = evaluate._run
    original_write_failure = evaluate._write_failure

    def _write_failure_with_video_receipt(*args: Any, **kwargs: Any) -> Any:
        failure_exc = kwargs.get("exc")
        if failure_exc is None and len(args) > 1:
            failure_exc = args[1]
        phase = kwargs.get("phase")
        if phase is None and len(args) > 2:
            phase = args[2]
        if isinstance(failure_exc, BaseException):
            state["actual_runtime_error"] = (
                f"actualruntimeerror[{phase or 'unknown_phase'}]: "
                f"{type(failure_exc).__name__}: {failure_exc}"
            )
        result = original_write_failure(*args, **kwargs)
        _write_receipt(state)
        return result

    evaluate._write_failure = _write_failure_with_video_receipt

    def _run_with_camera_adapter(args: Any, summary_path: Path) -> Any:
        state["runtime_started"] = True
        env_module = importlib.import_module("research.fixed_fruit_grasp_v1.env")
        original_env_class = env_module.FixedFruitGraspEnv
        env_module.FixedFruitGraspEnv = _recording_env_class(original_env_class, state)
        state["summary_path"] = summary_path
        try:
            result = original_run(args, summary_path)
            if isinstance(result, tuple) and result:
                state["evaluation_return_code"] = result[0]
            return result
        except BaseException as exc:
            state["actual_runtime_error"] = (
                f"actualruntimeerror[evaluate._run]: {type(exc).__name__}: {exc}"
            )
            raise
        finally:
            env_module.FixedFruitGraspEnv = original_env_class
            _write_receipt(state)

    # evaluate.main() 先创建 AppLauncher，再调用这个包装函数；此时 Kit 已就绪。
    evaluate._run = _run_with_camera_adapter
    try:
        return int(evaluate.main(bounded_arguments) or 0)
    finally:
        _write_receipt(state)


if __name__ == "__main__":
    raise SystemExit(main())
