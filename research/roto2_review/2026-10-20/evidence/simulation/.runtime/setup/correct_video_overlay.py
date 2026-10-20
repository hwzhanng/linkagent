import json,math,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
root=Path('/home/descfly/linkagent-ppo');folder=root/'artifacts/simulation';trace=json.loads((folder/'diagnostics/raw_pose_trace.json').read_text());rows=trace['trace'];poses=np.array([r['fruit_pose_native'][3:] for r in rows if r.get('fruit_pose_native')]);rot=Rotation.from_quat(poses);yaw=np.unwrap(rot.as_euler('xyz')[:,2]);yaw_change=(yaw-yaw[0])*180/np.pi;delta=(rot[1:]*rot[:-1].inv()).as_rotvec();total_z=float(delta[:,2].sum()/(2*np.pi));source=folder/'diagnostics/legacy_velocity_overlay.mp4';output=folder/'final_ppo_simulation_verified.mp4'
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(source)]));stream=probe['streams'][0];n=int(stream['nb_frames']);w,h=stream['width'],stream['height'];fps=20
reader=subprocess.Popen(['ffmpeg','-v','error','-i',str(source),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
for frame_idx in range(n):
 data=reader.stdout.read(w*h*3)
 if len(data)!=w*h*3: raise RuntimeError('Incomplete decoded frame')
 i=min(frame_idx,len(rows)-1);r=rows[i];actual=float(yaw_change[min(i,len(yaw_change)-1)]);im=Image.fromarray(np.frombuffer(data,dtype=np.uint8).reshape(h,w,3));draw=ImageDraw.Draw(im)
 draw.rectangle((0,0,w,103),fill=(17,24,35));draw.text((24,10),'最终模型真实仿真回放  |  发现旋转指标失真  |  种子 2026 / 环境 0',font=font,fill='white')
 draw.text((24,48),f"时间 {r['time_s']:5.2f} 秒    实际偏航变化 {actual:+5.1f}°    旧指标 {r['directed_turns']:5.2f} 圈（无效）",font=font,fill=(255,196,136))
 draw.rectangle((0,678,w,720),fill=(17,24,35));draw.text((24,688),f"原速播放  |  姿态以首帧为零点  |  支撑：{'稳定' if r['secure_support'] else '不稳定'}  |  压力代理 {r['pressure_pa']/1000:.1f} kPa  |  彩色点仅用于观察",font=small,fill=(205,218,231))
 if frame_idx>=len(rows)-1:
  draw.rectangle((24,121,1000,188),fill=(125,40,35));draw.text((39,127),'未完成连续旋转：旧成功判定无效',font=font,fill='white');draw.text((39,160),'本回合主要是小幅摆动；画面停留在复位前一帧，末尾停留 2 秒',font=small,fill='white')
 writer.stdin.write(np.asarray(im).tobytes())
 if frame_idx in (100,250,len(rows)-1): im.save(folder/f'verified_frame{frame_idx:03d}.jpg')
writer.stdin.close();assert writer.wait()==0;assert reader.wait()==0
summary={'video':str(output),'checkpoint':trace['checkpoint'],'seed':trace['seed'],'env_index':trace['env_index'],'actual_physx_render':True,'duration_s':n/fps,'fps':fps,'resolution':[w,h],'frame_count':n,'legacy_reported_turns':trace['directed_turns'],'measured_net_yaw_change_degrees':float(yaw_change[-1]),'measured_yaw_range_degrees':float(np.ptp(yaw)*180/np.pi),'quaternion_increment_world_z_net_turns':total_z,'maximum_frame_relative_rotation_degrees':float(np.max(np.linalg.norm(delta,axis=1))*180/np.pi),'rotation_success_verified':False,'previous_success_metric_invalid':True,'note':'Real rendered frames retained. Overlay corrected using actual recorded quaternion poses. Last terminal image holds pre-reset pose, plus 2-second labeled end card.'}
(folder/'video_verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
