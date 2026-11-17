"""Render saved PhysX EE-range stress-test states with target/base-height HUD."""
import argparse,sys
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from render_ee_tracking import load,render

def main():
 p=argparse.ArgumentParser();p.add_argument('--rollout',required=True);p.add_argument('--output',required=True);p.add_argument('--env',type=int,default=3);p.add_argument('--stride',type=int,default=2);a=p.parse_args()
 folder=Path(a.rollout);trace=np.load(folder/'trajectories.npz');e=a.env;state=load(folder,e);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 n=len(trace['step']);low=int(np.argmin(trace['target'][:,e,2]))
 with imageio.get_writer(str(out),fps=25/a.stride,codec='libx264',quality=8,macro_block_size=1) as writer:
  for i in range(0,n,a.stride):
   elapsed=float(trace['step'][i])*.02
   scene=render(state,i,'DYNAMIC EE RANGE • 45k POLICY',e,1,elapsed)
   canvas=Image.new('RGB',(512,324),(248,248,248));canvas.paste(Image.fromarray(scene),(0,0));draw=ImageDraw.Draw(canvas)
   target_z=float(trace['target'][i,e,2]);actual_z=float(trace['ee'][i,e,2]);base_z=float(trace['root'][i,e,2]);err=float(trace['position_error'][i,e])
   draw.text((7,292),'EE target height %.3f m  |  measured TCP %.3f m  |  base height %.3f m'%(target_z,actual_z,base_z),fill=(15,15,15))
   draw.text((7,307),'EE error %.3f m  |  floor-adjacent goals sampled; no scripted crouch'%(err),fill=(15,15,15))
   writer.append_data(np.asarray(canvas))
   if i==low:canvas.save(out.with_name(out.stem+'_floor_target.jpg'),quality=94)
   if i%(a.stride*100)==0:print('rendered %.1fs'%(elapsed),flush=True)
 state['renderer'].close()
 print('Saved %d frames; lowest EE target at %.3fm.'%(len(range(0,n,a.stride)),float(trace['target'][low,e,2])))
if __name__=='__main__':main()
