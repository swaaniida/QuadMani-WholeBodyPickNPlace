"""Correct initialization label, verify video and record evidence."""
from pathlib import Path
import json,hashlib
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;root=O.parents[1];video=O/'b1z1_stand_ee_20s.mp4';fixed=O/'video_corrected.mp4'
meta=json.loads((O/'metrics.json').read_text());reader=imageio.get_reader(str(video));vm=reader.get_meta_data()
fontpath=root/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
count=0
with imageio.get_writer(str(fixed),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
 for i,frame in enumerate(reader):
  if i==0:
   im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,43,1279,85),fill='white')
   draw.text((16,48),'t 0.02s  |  vx=vy=yaw 명령 0  |  EE 목표: 초기 정지 자세  |  위치 오차 2.0cm',font=ImageFont.truetype(str(fontpath),22),fill='#19334b')
   frame=np.asarray(im);im.save(O/'frame_0000.png')
  writer.append_data(frame);count+=1
reader.close();assert count==500
check=imageio.get_reader(str(fixed));cm=check.get_meta_data();assert abs(cm['duration']-20)<.05 and cm['fps']==25;check.close();fixed.replace(video)
for name,digest in meta['protected_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
(O/'video_verification.json').write_text(json.dumps(dict(frames=count,fps=25,seconds=cm['duration'],size=cm['size'],protected_files_unchanged=True,first_frame_note='First sample is initialization, followed by six prescribed EE segments.'),indent=2))
# Keep review sheet in agreement with the corrected first frame.
shots=[O/('frame_%04d.png'%i) for i in [0,250,500,750,998]]
sheet=Image.new('RGB',(1280*3,670*2),'white')
for k,p in enumerate(shots):sheet.paste(Image.open(p),((k%3)*1280,(k//3)*670))
sheet.save(O/'contact_sheet.jpg',quality=90)
print('Verified',count,'frames',cm['duration'],'seconds',cm['size'])
