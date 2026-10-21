"""Actual PhysX N-1 video, explicitly labelled as a probe, not a learned policy."""
from pathlib import Path
import json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .core import FINGERS


class CalibrationVideo:
    def __init__(self, raw, path, index, mode):
        if not 0 <= index < raw.num_envs: raise ValueError('Video index outside environment batch')
        self.raw,self.path,self.index,self.mode=raw,Path(path),index,mode
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.trace=[];self.condition='hold' if index%6==0 else FINGERS[index%6-1]
        capture=raw.video_recorder._capture;origin=raw.scene.env_origins[index].cpu().numpy()
        capture.cfg.eye=tuple(origin+np.array((.18,-.36,1.32)))
        capture.cfg.lookat=tuple(origin+np.array((-.12,0,.96)))
        capture.cfg.window_width=1280;capture.cfg.window_height=720
        for _ in range(35):frame=raw.render()
        if frame is None or frame.shape!=(720,1280,3) or np.std(frame)<3:
            raise RuntimeError('Missing/nonphysical calibration render')
        self.font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
        self.encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','-',
            '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)

    def append(self, step, data):
        i=self.index;raw=self.raw
        phase='保持对照' if i%6==0 else ('初始保持' if step<40 else '卸载手指' if step<80 else '释放保持' if step<90 else '恢复抓姿')
        im=Image.fromarray(raw.render().copy());draw=ImageDraw.Draw(im)
        pads=data['pad'][i].cpu().tolist();forces=data['whole_finger_force'][i].cpu().tolist()
        draw.rectangle((0,0,1280,115),fill=(17,24,35))
        draw.text((20,10),f'N−1 物理测试（非PPO策略） | {self.condition} | {phase} | {(step+1)*raw.step_dt:.2f}s',font=self.font,fill='white')
        draw.text((20,43),f'位移 {float(data["displacement"][i])*1000:.1f}mm | 速度 {float(data["speed"][i]):.3f}m/s | 稳定 {bool(data["stable"][i])}',font=self.font,fill=(255,196,136))
        draw.text((20,76),'有效pad '+str([int(v) for v in pads])+' | 整指力N '+str([round(v,2) for v in forces]),font=self.font,fill='white')
        self.encoder.stdin.write(np.asarray(im).tobytes())
        if step in (0,39,79,89,149):im.save(self.path.with_name(f'{self.path.stem}_frame{step:03d}.jpg'))
        self.trace.append(dict(step=step+1,phase=phase,pose=data['pose'][i].cpu().tolist(),pad=pads,
            whole_finger_force_n=forces,stable=bool(data['stable'][i]),failed=bool(data['failed'][i])))

    def close(self):
        self.encoder.stdin.close()
        if self.encoder.wait()!=0:raise RuntimeError('Calibration video encoding failed')
        subprocess.run(['ffmpeg','-v','error','-i',str(self.path),'-f','null','-'],check=True)
        self.path.with_suffix('.json').write_text(json.dumps(dict(kind='N-1 physical probe; no learned controller',
            mode=self.mode,env=self.index,condition=self.condition,frames=len(self.trace),fps=20,
            decode_verified=True,trace=self.trace),indent=2)+'\n')
