"""Run upstream demo.py; automate ONLY GUI click/display on headless astrago.

The positive click is chosen inside the official DAVIS initial annotation.
No annotation is injected as a model output or as an AOT reference mask.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT/'upstream/Tracking_SAM'
OUT = ROOT/'outputs/tracking/official_demo'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(UP))
from tracking_SAM import main_tracker
from tracking_SAM.plt_clicker import Annotator


if __name__ == '__main__':
    os.chdir(UP)
    torch.set_num_threads(4)
    annotation = cv2.imread('sample_data/DAVIS_bear/init_mask/00000.png', cv2.IMREAD_GRAYSCALE)>0
    distance = cv2.distanceTransform(annotation.astype(np.uint8), cv2.DIST_L2, 5)
    y,x = np.unravel_index(np.argmax(distance),distance.shape)
    rows=[]
    initial_mask=[]
    def click_without_gui(self):
        self.fig=plt.figure()
        self.ax1=self.fig.add_subplot(1,1,1)
        self._Annotator__on_button_press(SimpleNamespace(xdata=int(x),ydata=int(y),button=1))
        initial_mask.append(self.get_mask().copy())
        plt.close(self.fig)
    Annotator.main=click_without_gui
    original_annotate=main_tracker.annotate_init_frame
    original_propagate=main_tracker.propagate_one_frame
    def annotate(self,*args,**kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        value=original_annotate(self,*args,**kwargs)
        torch.cuda.synchronize()
        rows.append(dict(stage='sam_init_and_aot_reference',index=0,
                         elapsed_ms=(time.perf_counter()-start)*1000,
                         mask_pixels=int(initial_mask[-1].sum()),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        cv2.imwrite(str(OUT/'initial_sam_mask.png'),initial_mask[-1]*255)
        return value
    def propagate(self,image):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        mask=original_propagate(self,image)
        torch.cuda.synchronize()
        assert mask.shape==image.shape[:2] and mask.any()
        idx=len(rows)
        rows.append(dict(stage='aott_vos',index=idx,elapsed_ms=(time.perf_counter()-start)*1000,
                         mask_pixels=int((mask>0).sum()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        if idx%10==0 or idx==1:
            cv2.imwrite(str(OUT/f'mask_{idx:05d}.png'),(mask>0).astype(np.uint8)*255)
        return mask
    main_tracker.annotate_init_frame=annotate
    main_tracker.propagate_one_frame=propagate
    calls=[0]
    def key(_):
        calls[0]+=1
        return ord('a') if calls[0]==1 else 255
    def display(name,image):
        if name=='Tracked':
            cv2.imwrite(str(OUT/'last_overlay.png'),image)
    cv2.waitKey=key
    cv2.imshow=display
    cv2.destroyAllWindows=lambda:None
    spec=importlib.util.spec_from_file_location('official_tracking_demo',UP/'demo.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights=UP/'pretrained_weights'
    start=time.perf_counter()
    module.main(str(weights/'sam_vit_h_4b8939.pth'),str(weights/'AOTT_PRE_YTB_DAV.pth'),
                str(weights/'groundingdino_swint_ogc.pth'),1)
    with (OUT/'frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    times=[r['elapsed_ms'] for r in rows if r['stage']=='aott_vos']
    warm=times[1:]
    iou=np.logical_and(initial_mask[0]>0,annotation).sum()/np.logical_or(initial_mask[0]>0,annotation).sum()
    result=dict(upstream_demo='demo.py main() executed with GUI-only headless harness',
                click_xy=[int(x),int(y)],initial_sam_iou=float(iou),
                frames=len(rows),vos_frames=len(times),sam_init_ms=rows[0]['elapsed_ms'],
                warm_vos_mean_ms=float(np.mean(warm)),warm_vos_p50_ms=float(np.percentile(warm,50)),
                warm_vos_p95_ms=float(np.percentile(warm,95)),warm_vos_compute_fps=1000/float(np.mean(warm)),
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
                elapsed_s=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,
                windows_camera_tested=False)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
