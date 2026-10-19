#!/usr/bin/env python3
"""为 thumb_pad_grasp_v1 的原生 checkpoint 评估附加真实 RGB 录像。

本模块是录像辅助入口，明确不属于 run.py 的 critical source manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

_CAMERAS = {
    "close": ((0.34, -0.30, 0.24), (0.14, -0.08, 0.13)),
    "front": ((0.40, -0.38, 0.27), (0.14, -0.08, 0.13)),
    "side": ((0.32, 0.20, 0.24), (0.14, -0.08, 0.13)),
}
_POLICY_HZ = 20
_MAX_FRAMES = 600


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--camera", choices=tuple(_CAMERAS), default="close")
    parser.add_argument("--frame-every", type=int, choices=(2, 4), default=4)
    parser.add_argument("--mode")
    parser.add_argument("--policy")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--view-env-index", type=int, default=0)
    args, rest = parser.parse_known_args(argv)

    if args.mode != "evaluate" or args.policy != "checkpoint":
        raise ValueError("必须显式使用 --mode evaluate --policy checkpoint")
    if not args.checkpoint or not args.output:
        raise ValueError("必须提供同任务 --checkpoint 和 --output")
    args.episodes = 1 if args.episodes is None else args.episodes
    args.num_envs = 1 if args.num_envs is None else args.num_envs
    if not 1 <= args.episodes <= 8:
        raise ValueError("录像入口限定 --episodes 1..8")
    if args.num_envs not in (1, 4):
        raise ValueError("录像入口限定 --num-envs 1 或 4")
    if not 0 <= args.view_env_index < args.num_envs:
        raise ValueError("--view-env-index 必须满足 0 <= index < --num-envs")

    clean = rest + [
        "--mode", "evaluate", "--policy", "checkpoint",
        "--checkpoint", args.checkpoint, "--output", args.output,
        "--episodes", str(args.episodes), "--num-envs", str(args.num_envs),
    ]
    if "--enable_cameras" not in clean:
        clean.append("--enable_cameras")
    return args, clean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_writer(path: Path, fps: int) -> Any:
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(path), format="FFMPEG", mode="I", fps=fps,
        codec="libx264", macro_block_size=1,
    )


def _float_tuple(value: Any) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(float(item) for item in value)


def _configure_recording_camera(config: Any, scene: Any, state: dict[str, Any]) -> None:
    """以 IsaacLab 实际 env_origins 配置 world-space 视频相机。"""
    env_origins = getattr(scene, "env_origins", None)
    if env_origins is None:
        raise RuntimeError("scene.env_origins 不可用，无法确认所选环境的相机世界原点")
    actual_num_envs = len(env_origins)
    env_index = int(state["view_env_index"])
    if not 0 <= env_index < actual_num_envs:
        raise ValueError(
            f"view env index {env_index} 超出 scene.env_origins 的范围 0..{actual_num_envs - 1}"
        )
    requested_num_envs = state.get("requested_num_envs")
    if requested_num_envs is not None and int(requested_num_envs) != actual_num_envs:
        raise RuntimeError(
            f"requested num_envs={requested_num_envs}, but scene.env_origins has {actual_num_envs} rows"
        )

    origin = _float_tuple(env_origins[env_index])
    if len(origin) != 3:
        raise ValueError(f"scene.env_origins[{env_index}] must be a 3D world origin")
    eye_local, lookat_local = _CAMERAS[state["camera"]]
    eye_world = tuple(origin[i] + eye_local[i] for i in range(3))
    lookat_world = tuple(origin[i] + lookat_local[i] for i in range(3))
    config.eye = eye_world
    config.lookat = lookat_world
    # IsaacLab 的录像器会把 Kit capture 相机直接设到 cfg.eye/lookat；
    # 避免其 visualizer 同步逻辑把 env 局部坐标重新覆盖进 world 坐标字段。
    if hasattr(config, "backend_source"):
        config.backend_source = "renderer"
    state.update(
        actual_num_envs=actual_num_envs,
        camera_origin_world_m=list(origin),
        camera_eye_world_m=list(eye_world),
        camera_lookat_world_m=list(lookat_world),
        camera_origin_type="env",
    )


def _receipt(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("video_sha256") and not state.get("capture_error"):
        status = "RECORDED"
    elif state.get("partial_sha256") or state["frames_sampled"]:
        status = "PARTIAL"
    elif state.get("encoder_error") or not state.get("runtime_started"):
        status = "UNAVAILABLE"
    else:
        status = "NO_VIDEO"

    output = state.get("output")
    metadata, summary = {}, {}
    if isinstance(output, Path):
        for name, dest in (("run_metadata.json", metadata), ("summary.json", summary)):
            try:
                value = json.loads((output / name).read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    dest.update(value)
            except (OSError, json.JSONDecodeError):
                pass
    legacy_state = "view_env_index" not in state
    eye_world = state.get("camera_eye_world_m")
    lookat_world = state.get("camera_lookat_world_m")
    if legacy_state:
        eye, lookat = _CAMERAS[state["camera"]]
        eye_world = list(eye)
        lookat_world = list(lookat)
    return {
        "schema_version": "thumb_pad_video_receipt.v1",
        "record_video_source_sha256": state.get("record_video_source_sha256"),
        "selected_env_index": state.get("view_env_index", 0),
        "actual_num_envs": state.get("actual_num_envs"),
        "contact_overlay": {"enabled":state.get('pad_overlay_enabled',False),
            "error":state.get('pad_overlay_error'),"geometry_sha256":state.get('pad_overlay_geometry_sha256'),
            "legend":"green outlines=confirmed PAD boundaries; contact points and normals green PAD / red NON_PAD / yellow UNKNOWN; force>=0.01N display threshold only",
            "physics":"display geometry only; no CollisionAPI or RigidBodyAPI; labels are visualizations of runtime trace, not independent success evidence"},
        "task_id": "thumb_pad_grasp_v1",
        "status": status,
        "evaluation_status": summary.get("status"),
        "evaluation_return_code": state.get("evaluation_return_code"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "raw_pad_trace": str(output / "pad_trace.jsonl") if output else None,
        "video": {
            "path": str(state["video_path"]) if state.get("video_path") else None,
            "partial_path": str(state["partial_path"]) if state.get("partial_sha256") else None,
            "sha256": state.get("video_sha256"),
            "partial_sha256": state.get("partial_sha256"),
            "frame_count": state["frames_sampled"],
            "fps": _POLICY_HZ // state["frame_every"],
        },
        "camera": {
            "selection": state["camera"],
            "origin_type": state.get("camera_origin_type", "world" if legacy_state else None),
            "env_index": state.get("view_env_index", 0),
            "origin_world_m": state.get("camera_origin_world_m"),
            "eye_world_m": eye_world,
            "lookat_world_m": lookat_world,
            "configured": bool(state.get("camera_configured")),
            "verified_by_valid_rgb_frames": state["frames_sampled"] > 0,
        },
        "sampling": {
            "policy_steps_per_frame": state["frame_every"],
            "blank_frames_skipped": state["blank_frames_skipped"],
            "frame_limit_reached": state["frame_limit_reached"],
        },
        "errors": {
            "encoder": state.get("encoder_error"),
            "capture": state.get("capture_error"),
            "runtime": state.get("runtime_error"),
        },
        "receipt_written_before_app_close": bool(state.get("receipt_before_close")),
    }


def _write_receipt(state: dict[str, Any]) -> None:
    path = state.get("receipt_path")
    if not isinstance(path, Path) or not path.parent.is_dir():
        return
    before_close = bool(state.get("inside_run"))
    if before_close:
        state["receipt_before_close"] = True
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(_receipt(state), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        if before_close:
            state["receipt_before_close"] = False
        state["capture_error"] = state.get("capture_error") or f"receipt: {type(exc).__name__}: {exc}"
        print(f"录像回执写入失败：{state['capture_error']}", file=sys.stderr, flush=True)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _update_contact_overlay(env: Any, state: dict[str, Any]) -> None:
    """纯显示USD几何：批准边界与实际接触点/法向，不添加任何物理API。"""
    env_index = int(state.get("view_env_index", 0))
    episode_number = int(getattr(env, "_video_episode_number", 1))
    episode_age = None
    try:
        episode_age = int(env.episode_length_buf[env_index].item())
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    age_label = "?" if episode_age is None else str(episode_age)
    caption_prefix = f"env {env_index} | episode {episode_number} | age {age_label}"
    state["overlay_caption"] = caption_prefix
    if not hasattr(env,'_robot'):
        return
    try:
        import numpy as np
        import omni.usd
        from pxr import UsdGeom, Gf
        from scipy.spatial.transform import Rotation
        stage=omni.usd.get_context().get_stage()
        if stage is None:raise RuntimeError('no active USD stage')
        if 'pad_overlay_geometry' not in state:
            config_path=Path(__file__).parents[1]/'thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json'
            state['pad_overlay_geometry']=json.loads(config_path.read_text())['digits']
            state['pad_overlay_geometry_sha256']=_sha256(config_path)
        poses=env._robot.data.body_pose_w.torch[env_index].cpu().numpy()
        for digit in state['pad_overlay_geometry']:
            name=digit['digit'];body=digit['body_path'].rsplit('/',1)[-1]
            pose=poses[env._robot.body_names.index(body)]
            edges=np.array(digit['boundary_edges_body_local_m']).reshape(-1,3)
            world=edges@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
            curve=UsdGeom.BasisCurves.Define(stage,'/World/PadAudit/'+name)
            curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([2]*(len(world)//2))
            curve.CreatePointsAttr([Gf.Vec3f(*map(float,point)) for point in world])
            curve.CreateWidthsAttr([.00035]);curve.SetWidthsInterpolation('constant')
            curve.CreateDisplayColorAttr([(0.1,1.,0.1)])
        points=[];colors=[];ends=[];normalcolors=[]
        palette={1:(.05,1.,.05),2:(1.,.1,.1),3:(1.,.85,.05)}
        for row in getattr(env.pad_observer,'details',[]):
            for i,(eid,force) in enumerate(zip(row['env_ids'],row['forces_n'])):
                if eid!=env_index or force<.01:continue
                point=np.array(row['points_w'][i]);normal=-np.array(row['raw_normals_w'][i]);color=palette[row['labels'][i]]
                points.append(Gf.Vec3f(*map(float,point)));colors.append(color)
                ends.extend([Gf.Vec3f(*map(float,point)),Gf.Vec3f(*map(float,point+normal*.008))]);normalcolors.extend([color,color])
        marker=UsdGeom.Points.Define(stage,'/World/PadAudit/actual_contact_points')
        marker.CreatePointsAttr(points);marker.CreateWidthsAttr([.0018]*len(points))
        marker.CreateDisplayColorAttr(colors);marker.GetDisplayColorPrimvar().SetInterpolation('vertex')
        normals=UsdGeom.BasisCurves.Define(stage,'/World/PadAudit/actual_contact_normals')
        normals.CreateTypeAttr('linear');normals.CreateCurveVertexCountsAttr([2]*len(points));normals.CreatePointsAttr(ends)
        normals.CreateWidthsAttr([.0004]);normals.SetWidthsInterpolation('constant')
        normals.CreateDisplayColorAttr(normalcolors);normals.GetDisplayColorPrimvar().SetInterpolation('vertex')
        truth=env.pad_observer.latest
        pad=float(truth['pad_force_n'][env_index,0]) if truth is not None else 0.
        side=float(truth['nonpad_force_n'][env_index,0]) if truth is not None else 0.
        streak=int(env.g2_guard.streak[env_index].max());released=bool(env._stem_detached[env_index])
        state['overlay_caption']=f'{caption_prefix} | thumb PAD {pad:.2f}N | nonPAD {side:.2f}N | hold {streak/120:.3f}s | released {released}'
        state['pad_overlay_enabled']=True
    except Exception as exc:
        state['pad_overlay_error']=f'{type(exc).__name__}: {exc}'


def _recording_env(base: type, state: dict[str, Any]) -> type:
    state.setdefault("view_env_index", 0)

    class RecordingThumbPadEnv(base):
        def __init__(self, cfg: Any, **kwargs: Any) -> None:
            self._video_steps = 0
            self._video_frames = 0
            self._video_writer = None
            self._video_error: str | None = None
            self._video_episode_number = 1
            render_mode = None
            if state["capture_enabled"]:
                try:
                    eye, lookat = _CAMERAS[state["camera"]]
                    cfg.viewer.eye, cfg.viewer.lookat = eye, lookat
                    cfg.viewer.origin_type = "env"
                    cfg.viewer.env_index = int(state["view_env_index"])
                    if cfg.video_recorder is not None:
                        recorder = cfg.video_recorder.class_type

                        class OptionalRecorder:
                            def __init__(self, config: Any, scene: Any) -> None:
                                try:
                                    _configure_recording_camera(config, scene, state)
                                    self.inner = recorder(config, scene)
                                    state["camera_configured"] = True
                                except Exception as exc:
                                    self.inner = None
                                    state["capture_error"] = f"camera backend: {type(exc).__name__}: {exc}"
                                    state["capture_enabled"] = False

                            def render_rgb_array(self) -> Any:
                                return self.inner.render_rgb_array() if self.inner is not None else None

                        cfg.video_recorder.class_type = OptionalRecorder
                    render_mode = "rgb_array"
                except Exception as exc:
                    state["capture_enabled"] = False
                    state["capture_error"] = f"camera config: {type(exc).__name__}: {exc}"
            super().__init__(cfg, render_mode=render_mode, **kwargs)
            actual_num_envs = int(getattr(self, "num_envs", state.get("requested_num_envs", 1)))
            state["actual_num_envs"] = actual_num_envs
            if not 0 <= int(state["view_env_index"]) < actual_num_envs:
                state["capture_enabled"] = False
                state["capture_error"] = state.get("capture_error") or (
                    f"view env index {state['view_env_index']} is outside runtime num_envs={actual_num_envs}"
                )

        def _capture(self) -> None:
            if not state["capture_enabled"] or self._video_writer is False:
                return
            if state["frames_sampled"] >= _MAX_FRAMES:
                state["frame_limit_reached"] = True
                return
            try:
                import numpy as np

                _update_contact_overlay(self,state)
                frame = self.render()
                if frame is None:
                    raise RuntimeError("env.render() returned None")
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                rgb = np.asarray(frame)
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"expected uint8 RGB; got {rgb.shape}, {rgb.dtype}")
                if not np.any(rgb):
                    state["blank_frames_skipped"] += 1
                    return
                if state.get('overlay_caption'):
                    from PIL import Image,ImageDraw
                    annotated=Image.fromarray(rgb).copy();draw=ImageDraw.Draw(annotated)
                    draw.rectangle((0,0,rgb.shape[1],19),fill=(0,0,0))
                    draw.text((4,3),state['overlay_caption'],fill=(255,255,255))
                    rgb=np.asarray(annotated)
                if self._video_writer is None:
                    state["partial_path"] = state["output"] / "evaluation.partial.mp4"
                    self._video_writer = _open_writer(
                        state["partial_path"], _POLICY_HZ // state["frame_every"]
                    )
                self._video_writer.append_data(rgb)
                state["frames_sampled"] += 1
            except Exception as exc:
                self._video_error = f"{type(exc).__name__}: {exc}"
                state["capture_error"] = self._video_error
                state["capture_enabled"] = False
                writer, self._video_writer = self._video_writer, False
                if writer not in (None, False):
                    try:
                        writer.close()
                    except Exception:
                        pass

        def _finish(self) -> None:
            writer, self._video_writer = self._video_writer, None
            partial = state.get("partial_path")
            if writer not in (None, False):
                try:
                    writer.close()
                    if partial is None or not partial.is_file():
                        raise FileNotFoundError("ffmpeg did not produce an MP4")
                    final = state["output"] / "evaluation.mp4"
                    os.replace(partial, final)
                    state["video_path"] = final
                    state["video_sha256"] = _sha256(final)
                except Exception as exc:
                    state["capture_error"] = state.get("capture_error") or f"encoding: {type(exc).__name__}: {exc}"
            if partial is not None and partial.is_file() and not state.get("video_sha256"):
                try:
                    state["partial_sha256"] = _sha256(partial)
                except OSError:
                    pass
            _write_receipt(state)

        def step(self, action: Any) -> Any:
            # 在动作执行前采样当前真实场景，首帧包含 reset 后的初始手果位置。
            if self._video_steps % state["frame_every"] == 0:
                self._capture()
            self._video_steps += 1
            result = super().step(action)
            if isinstance(result, (tuple, list)) and len(result) >= 4:
                try:
                    import numpy as np

                    terminated, truncated = result[2], result[3]
                    if hasattr(terminated, "detach"):
                        terminated = terminated.detach().cpu().numpy()
                    if hasattr(truncated, "detach"):
                        truncated = truncated.detach().cpu().numpy()
                    done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
                    done = done.reshape(-1)
                    index = int(state["view_env_index"])
                    if index < done.size and bool(done[index]):
                        self._video_episode_number += 1
                except (TypeError, ValueError):
                    pass
            return result

        def close(self) -> None:
            try:
                self._finish()
            finally:
                super().close()

    return RecordingThumbPadEnv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        help_parser = argparse.ArgumentParser(
            description="为原生 thumb_pad_grasp_v1 checkpoint 评估附加 RGB 录像。",
            epilog="其余 run.py 评估参数（例如 --device、--seed、--max-env-steps）会原样透传。",
        )
        help_parser.add_argument("--mode", choices=("evaluate",), required=True)
        help_parser.add_argument("--policy", choices=("checkpoint",), required=True)
        help_parser.add_argument("--checkpoint", required=True)
        help_parser.add_argument("--output", required=True)
        help_parser.add_argument("--episodes", type=int, choices=range(1, 9), default=1)
        help_parser.add_argument("--num-envs", type=int, choices=(1, 4), default=1)
        help_parser.add_argument("--view-env-index", type=int, default=0)
        help_parser.add_argument("--camera", choices=tuple(_CAMERAS), default="close")
        help_parser.add_argument("--frame-every", type=int, choices=(2, 4), default=4)
        help_parser.print_help()
        return 0
    try:
        args, run_args = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        print(f"录像参数错误：{exc}", file=sys.stderr)
        return 2

    run = importlib.import_module("research.firm_grasp_v1.run")
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        enabled = ffmpeg.is_file() and os.access(ffmpeg, os.X_OK)
        encoder_error = None if enabled else f"ffmpeg executable unavailable: {ffmpeg}"
    except Exception as exc:
        enabled, encoder_error = False, f"imageio/ffmpeg unavailable: {type(exc).__name__}: {exc}"

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    state: dict[str, Any] = {
        "camera": args.camera, "frame_every": args.frame_every, "episodes": args.episodes,
        "requested_num_envs": args.num_envs, "view_env_index": args.view_env_index,
        "record_video_source_sha256": _sha256(Path(__file__).resolve()),
        "capture_enabled": enabled, "camera_configured": False, "encoder_error": encoder_error,
        "capture_error": None, "runtime_error": None, "runtime_started": False,
        "evaluation_return_code": None, "frames_sampled": 0, "blank_frames_skipped": 0,
        "frame_limit_reached": False, "video_sha256": None, "partial_sha256": None,
        "video_path": None, "partial_path": None, "videos": [], "output": output,
        "receipt_path": output / "video_receipt.json",
        "inside_run": False, "receipt_before_close": False, "output_created": False,
    }
    original_run, original_prepare = run._run, run._prepare_output

    def prepare(path: Path) -> Path:
        created = original_prepare(path)
        state.update(output=created, receipt_path=created / "video_receipt.json", output_created=True)
        return created

    def run_with_video(namespace: Any, destination: Path, stop: Any, phase: Any) -> int:
        state.update(runtime_started=True, inside_run=True, output=destination,
                     receipt_path=destination / "video_receipt.json")
        env_module = importlib.import_module("research.firm_grasp_v1.env")
        original_env = env_module.FirmGraspEnv
        env_module.FirmGraspEnv = _recording_env(original_env, state)
        try:
            result = original_run(namespace, destination, stop, phase)
            state["evaluation_return_code"] = int(result or 0)
            return result
        except BaseException as exc:
            state["runtime_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            env_module.FirmGraspEnv = original_env
            _write_receipt(state)
            state["inside_run"] = False

    run._run, run._prepare_output = run_with_video, prepare
    try:
        try:
            result = int(run.main(run_args) or 0)
        except SystemExit as exc:
            result = int(exc.code or 0)
        except BaseException as exc:
            state["runtime_error"] = f"{type(exc).__name__}: {exc}"
            result = 1
        state["evaluation_return_code"] = result
        if state["output_created"]:
            try:
                failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
                state["runtime_error"] = state.get("runtime_error") or (
                    f"{failure.get('phase')}: {failure.get('exception_type')}: {failure.get('exception')}"
                )
            except (OSError, json.JSONDecodeError):
                pass
            _write_receipt(state)
        return result
    finally:
        run._run, run._prepare_output = original_run, original_prepare


if __name__ == "__main__":
    raise SystemExit(main())
