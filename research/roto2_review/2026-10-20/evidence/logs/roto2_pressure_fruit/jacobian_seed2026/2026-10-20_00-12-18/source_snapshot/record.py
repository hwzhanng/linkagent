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

    # Children of the actual fruit follow its real pose; no collision or rigid-body APIs.
    from pxr import UsdGeom, Gf
    from isaaclab.sim.utils.stage import get_current_stage
    stage = get_current_stage()
    parent = f"/World/envs/env_{env_index}/Fruit"
    for i, (axis, color) in enumerate([((1, 0, 0), (1, .95, .75)),
                                      ((0, 1, 0), (.2, .8, .95)),
                                      ((0, 0, 1), (.95, .85, .1))]):
        marker = UsdGeom.Sphere.Define(stage, f"{parent}/visual_orientation_{i}")
        marker.CreateRadiusAttr(.085)
        marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.Xformable(marker).AddTranslateOp().Set(Gf.Vec3d(*(a * .97 for a in axis)))

    origin = raw.scene.env_origins[env_index].cpu().numpy()
    capture = raw.video_recorder._capture
    capture.cfg.eye = tuple(origin + np.array((.34, -.34, 1.09)))
    capture.cfg.lookat = tuple(origin + np.array((.065, 0, .88)))
    capture.cfg.window_width = 1280
    capture.cfg.window_height = 720
    for _ in range(35):
        frame = raw.render()
    if frame is None or frame.shape != (720, 1280, 3) or np.std(frame) < 3:
        raise RuntimeError("Renderer did not produce a nonblank 1280x720 RGB image")

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
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
            else:
                frame = raw.render().copy()
                last_frame = frame
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 1280, 103), fill=(17, 24, 35))
            draw.text((24, 13), f"FINAL PPO | Actual PhysX simulation | Seed {seed} / Env {env_index}", font=font, fill="white")
            draw.text((24, 50), f"t={(step+1)*dt:5.2f}s   rotation={turns:5.2f} turns   grip={'SECURE' if support else 'UNSTABLE'}   pressure={pressure/1000:5.1f} kPa", font=font, fill=(205, 223, 242))
            draw.rectangle((0, 678, 1280, 720), fill=(17, 24, 35))
            draw.text((24, 690), "1x playback | Colored dots: visual-only orientation markers | Pressure is a simulation proxy", font=small, fill=(200, 210, 222))
            if ended:
                success = turns >= 1 and support_steps/(step+1) >= .65 and not failed and peak <= raw.cfg.maximum_pressure_proxy_pa
                label = "SUCCESS" if success else "FAILED / CRITERIA NOT MET"
                draw.rectangle((24, 121, 780, 188), fill=(20, 90, 55) if success else (125, 40, 35))
                draw.text((39, 132), label, font=font, fill="white")
                draw.text((39, 162), "Episode ended; terminal metrics shown on last pre-reset frame", font=small, fill="white")
            encoder.stdin.write(np.asarray(im).tobytes())
            if step in (0, 100, 250) or ended:
                im.save(output.with_name(f"{output.stem}_frame{step:03d}.jpg"))
            trace.append({'step':step+1,'time_s':(step+1)*dt,'directed_turns':turns,
                          'secure_support':support,'pressure_pa':pressure,
                          'failed':failed,'ended':ended})
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
    result = {'checkpoint':str(checkpoint), 'seed':seed, 'env_index':env_index,
              'parallel_envs':wrapped.num_envs, 'physics_steps_recorded':len(trace),
              'fps':fps,'real_time_playback':True,'episode_completed':ended,
              'failure_termination':failed,'directed_turns':turns,
              'secure_support_fraction':support_steps/max(1,len(trace)),
              'peak_pressure_proxy_pa':peak,
              'success':bool(ended and not failed and turns>=1 and support_steps/len(trace)>=.65 and peak<=raw.cfg.maximum_pressure_proxy_pa),
              'terminal_frame_note':'Last pre-reset pose is held at terminal step; 2s labeled end card follows.',
              'visual_only_orientation_markers':True,'trace':trace,'video':str(output)}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2),flush=True)
