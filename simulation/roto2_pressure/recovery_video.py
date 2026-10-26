"""Labelled video of a physical recovery probe, explicitly not a PPO rollout."""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RecoveryVideo:
    def __init__(self, raw, path, index):
        assert 0 <= index < raw.num_envs
        self.raw,self.path,self.index=raw,Path(path),index
        self.path.parent.mkdir(parents=True,exist_ok=True)
        assert not self.path.exists()
        self.trace=[]
        origin=raw.scene.env_origins[index].cpu().numpy()
        cfg=raw.video_recorder._capture.cfg
        cfg.eye=tuple(origin+np.array((.18,-.36,1.32)))
        cfg.lookat=tuple(origin+np.array((-.12,0,.96)))
        cfg.window_width=1280;cfg.window_height=720
        for _ in range(35):frame=raw.render()
        assert frame is not None and frame.shape==(720,1280,3) and np.std(frame)>3
        self.font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
        self.encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','-',
            '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)

    def append(self,step,d):
        i=self.index;phase=(step-40)%200
        label='初始保持' if step<40 else ('松开末端' if phase<40 else '回收关节' if phase<80 else '重新接触与保持' if phase<160 else '返回初始目标')
        cycle=0 if step<40 else 1+(step-40)//200
        im=Image.fromarray(self.raw.render().copy());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,1280,115),fill=(17,24,35))
        pads=d['pad'][i].cpu().tolist();forces=d['whole_finger_force'][i].cpu().tolist()
        draw.text((20,10),f'物理路径复验（非PPO策略） | 循环{cycle} | {label} | {(step+1)*self.raw.step_dt:.2f}s',font=self.font,fill='white')
        draw.text((20,43),f'位移 {float(d["displacement"][i])*1000:.1f}mm | 稳定 {bool(d["stable"][i])} | 峰压 {float(d["peak_pressure"][i]):.0f}Pa',font=self.font,fill=(255,196,136))
        draw.text((20,76),'有效pad '+str([int(v) for v in pads])+' | 整指力N '+str([round(v,2) for v in forces]),font=self.font,fill='white')
        self.encoder.stdin.write(np.asarray(im).tobytes())
        if step in (39,79,119,199,399,599):im.save(self.path.with_name(f'{self.path.stem}_frame{step:03d}.jpg'))
        self.trace.append(dict(step=step,cycle=cycle,phase=label,pose=d['pose'][i].cpu().tolist(),pad=pads,
                              whole_finger_force_n=forces,stable=bool(d['stable'][i])))

    def close(self):
        self.encoder.stdin.close()
        assert self.encoder.wait()==0
        subprocess.run(['ffmpeg','-v','error','-i',str(self.path),'-f','null','-'],check=True)
        self.path.with_suffix('.json').write_text(json.dumps(dict(kind='Physical recovery replay, NOT PPO policy',
            frames=len(self.trace),fps=20,index=self.index,decode_verified=True,trace=self.trace),indent=2)+'\n')
