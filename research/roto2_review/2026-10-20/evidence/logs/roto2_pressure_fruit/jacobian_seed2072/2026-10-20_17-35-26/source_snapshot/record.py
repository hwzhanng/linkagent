"""Film an actual PhysX first episode, with visual-only orientation markers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


@torch.no_grad()
def record_checkpoint(trainer, wrapped, checkpoint: Path, output: Path,
                      seed: int, env_index: int = 0, max_steps: int = 480):
    if not 0 <= env_index < wrapped.num_envs:
        raise ValueError("Video environment index is out of range")
    output.parent.mkdir(parents=True, exist_ok=True)
    agent = trainer.agent
    modules = torch.load(checkpoint, map_location=wrapped.device, weights_only=True)
    for key in ("encoder", "policy"):
        getattr(agent, key).load_state_dict(modules[key])
        getattr(agent, key).eval()
    if agent.encoder.state_preprocessor is not None:
        agent.encoder.state_preprocessor.load_state_dict(modules["state_preprocessor"])
    states, _ = wrapped.reset(hard=True)
    raw = wrapped._unwrapped
    direction = float(raw._goal[env_index].sign())
    dt = float(raw.cfg.sim.dt * raw.cfg.decimation)
    fps = round(1 / dt)

    # Explicitly follow measured poses: adding children after Fabric initialization
    # can leave their inherited transforms stale. These have no physics APIs.
    from pxr import UsdGeom, Gf
    from isaaclab.sim.utils.stage import get_current_stage
    stage = get_current_stage()
    from .core import quat_rotate
    marker_ops = []
    marker_offsets = []
    for i, (axis, color) in enumerate([((1, 0, 0), (1, .95, .75)),
                                      ((0, 1, 0), (.2, .8, .95)),
                                      ((0, 0, 1), (.95, .85, .1))]):
        marker = UsdGeom.Sphere.Define(stage, f"/World/VideoOrientationMarker_{i}")
        marker.CreateRadiusAttr(.0027)
        marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        marker_ops.append(UsdGeom.Xformable(marker).AddTranslateOp())
        marker_offsets.append(axis)
    offsets = torch.tensor(marker_offsets, device=raw.device) * raw._axes_m[env_index] * .98

    def update_markers():
        pose = raw._fruit.data.root_pose_w.torch[env_index]
        quat = raw._wxyz(pose[3:7])
        positions = pose[:3] + quat_rotate(quat.expand(3, -1), offsets)
        for op, pos in zip(marker_ops, positions.cpu().tolist()):
            op.Set(Gf.Vec3d(*pos))
        return pose.detach().cpu().tolist()

    from .reward_geometry import pad_frames
    centers, ux, vy, _ = pad_frames(raw.device)
    corners = centers[:,None] + torch.tensor([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]],device=raw.device)[None,:,0:1]*.0048*ux[:,None] + torch.tensor([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]],device=raw.device)[None,:,1:2]*.0072*vy[:,None]
    pad_ops = []
    for j in range(5):
        group = []
        for k in range(4):
            dot = UsdGeom.Sphere.Define(stage, f'/World/VideoPad_{j}_{k}')
            dot.CreateRadiusAttr(.0012)
            color = dot.CreateDisplayColorAttr([Gf.Vec3f(.9,.6,.1)])
            group.append((UsdGeom.Xformable(dot).AddTranslateOp(), color))
        pad_ops.append(group)
    def update_pads():
        ids = raw._contact_body_ids[:5]
        q = raw._wxyz(raw._robot.data.body_quat_w.torch[env_index,ids])
        pos = raw._robot.data.body_pos_w.torch[env_index,ids]
        points = pos[:,None] + quat_rotate(q[:,None].expand(-1,4,-1),corners)
        for j in range(5):
            active = bool(raw._pad_forces[env_index,j] > .15)
            color = Gf.Vec3f(.1,.95,.3) if active else Gf.Vec3f(.9,.6,.1)
            for k,(op,attr) in enumerate(pad_ops[j]):
                op.Set(Gf.Vec3d(*points[j,k].cpu().tolist()));attr.Set([color])
    initial_pose = update_markers()
    update_pads()

    origin = raw.scene.env_origins[env_index].cpu().numpy()
    capture = raw.video_recorder._capture
    capture.cfg.eye = tuple(origin + np.array((.34, -.34, 1.09)))
    capture.cfg.lookat = tuple(origin + np.array((.065, 0, .88)))
    if raw.cfg.hand_posture == "palm_up":
        capture.cfg.eye = tuple(origin + np.array((.18, -.36, 1.32)))
        capture.cfg.lookat = tuple(origin + np.array((-.12, 0, .96)))
    capture.cfg.window_width = 1280
    capture.cfg.window_height = 720
    for _ in range(35):
        frame = raw.render()
    if frame is None or frame.shape != (720, 1280, 3) or np.std(frame) < 3:
        raise RuntimeError("Renderer did not produce a nonblank 1280x720 RGB image")

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    stage_label = {"contact":"接触课程", "rotation":"转动课程", "regrasp":"小角度换指课程", "spin":"双向转动课程", "continuous":"连续转动课程"}[raw.cfg.reward_stage]
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    trace = []
    turns = 0.0
    support_steps = 0
    peak = 0.0
    ended = False
    failed = False
    last_frame = None
    try:
        for step in range(min(max_steps, int(raw.max_episode_length + 2))):
            latent = agent.encoder(states)
            action, _, _ = agent.policy.act(latent, deterministic=True)
            states, _, terminated, truncated, info = wrapped.step(action)
            event = info["step"]
            turns += float(event["rotation_rate"][env_index]) * direction * dt / (2 * math.pi)
            pressure = float(event["peak_pressure_pa"][env_index])
            support = bool(event["secure_support"][env_index])
            support_steps += support
            peak = max(peak, pressure)
            failed = bool(terminated.reshape(-1)[env_index])
            ended = bool((terminated | truncated).reshape(-1)[env_index])
            # Auto-reset happens inside step(). Do not show the newly reset pose
            # as if it were the failed/finished terminal state.
            if ended and last_frame is not None:
                frame = last_frame
                pose = None
            else:
                pose = update_markers()
                update_pads()
                frame = raw.render().copy()
                last_frame = frame
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 1280, 103), fill=(17, 24, 35))
            draw.text((24, 13), f"PPO {stage_label}  |  真实 PhysX 仿真  |  种子 {seed} / 环境 {env_index}", font=font, fill="white")
            draw.text((24, 50), f"时间 {(step+1)*dt:5.2f} 秒    实际姿态转角 {turns:6.3f} 圈    压力 {pressure/1000:5.1f} kPa", font=font, fill=(255, 196, 136))
            draw.rectangle((0, 678, 1280, 720), fill=(17, 24, 35))
            draw.text((24, 690), "原速 | 果实彩点显示姿态 | 指尖四角：橙色无有效触摸 / 绿色有效触摸 | 压力为代理量", font=small, fill=(200, 210, 222))
            if ended:
                label = f"回合结束：实际净转角 {turns:.3f} 圈；有效区域承托 {support_steps/(step+1):.1%}"
                draw.rectangle((24, 121, 1000, 188), fill=(125, 40, 35))
                draw.text((39, 132), label, font=font, fill="white")
                draw.text((39, 162), "回合结束：画面停留在自动复位前一帧，指标为终止步数值", font=small, fill="white")
            encoder.stdin.write(np.asarray(im).tobytes())
            if step in (0, 100, 250) or ended:
                im.save(output.with_name(f"{output.stem}_frame{step:03d}.jpg"))
            trace.append({'step':step+1,'time_s':(step+1)*dt,'directed_turns':turns,
                          'secure_support':support,'pressure_pa':pressure,
                          'failed':failed,'ended':ended,'fruit_pose_native':event['fruit_pose_native'][env_index].cpu().tolist(),
                          'pad_contact':event['pad_contact'][env_index].cpu().tolist()})
            if ended:
                # A clearly marked two-second still lets the viewer read the result.
                for _ in range(2*fps):
                    encoder.stdin.write(np.asarray(im).tobytes())
                break
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    from scipy.spatial.transform import Rotation
    all_poses = np.array([initial_pose] + [t['fruit_pose_native'] for t in trace])
    q = all_poses[:,3:7]
    if not raw._native_xyzw: q=q[:,[1,2,3,0]]
    matrices=Rotation.from_quat(q).as_matrix()
    headings=np.arctan2(matrices[:,1,0],matrices[:,0,0])
    delta=np.arctan2(np.sin(np.diff(headings)),np.cos(np.diff(headings)))
    offline_turns=float(delta.sum()*direction/(2*math.pi))
    if abs(offline_turns-turns)>1e-4: raise RuntimeError('Video pose/metric mismatch')
    result = {'checkpoint' :str(checkpoint), 'seed':seed, 'env_index':env_index,
              'parallel_envs':wrapped.num_envs, 'physics_steps_recorded':len(trace),
              'domain_profile':raw.cfg.domain_profile, 'reward_stage':raw.cfg.reward_stage,
              'hand_posture':raw.cfg.hand_posture,
              'training_goal_direction':raw.cfg.goal_direction,
              'task_acceptance_claimed':False,
              'fps':fps,'real_time_playback':True,'episode_completed':ended,
              'failure_termination':failed,'directed_turns':turns,
              'secure_support_fraction':support_steps/max(1,len(trace)),
              'peak_pressure_proxy_pa':peak,
              'independent_pose_turns':offline_turns,
              'rotation_metric_status':'VERIFIED against independent scipy quaternion heading reconstruction',
              'pad_geometry':'vendor sensor-cap mesh fitted prior; not hardware calibrated',
              'goal_direction':direction,
              'terminal_frame_note':'Last pre-reset pose is held at terminal step; 2s labeled end card follows.',
              'visual_only_orientation_markers':True,'trace':trace,'video':str(output)}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2),flush=True)
