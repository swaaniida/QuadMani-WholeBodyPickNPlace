"""Static scientific 3D snapshots of each analyzed capture (CPU, Agg backend)."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from obb_branch_analysis import BRANCHES,COLORS,ground_basis


def main(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    target=path/'snapshots';target.mkdir(exist_ok=True)
    mesh=trimesh.load(ROOT/'local_client/assets/franka_umi_vis_mesh.obj',force='mesh',process=False)
    for entry in manifest['frames']:
        if entry['status']!='ok':continue
        with np.load(path/entry['result_file'],allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        pc=d['point_cloud'];g=d['grasps'];tags=d['branch_tags'];F=ground_basis(d['ground_plane'])
        center=pc.mean(0);scale=max(.16,float(np.linalg.norm(np.ptp(pc,axis=0)))*.8)
        fig=plt.figure(figsize=(15,5));fig.suptitle(entry['frame_id'])
        for col,tag in enumerate(BRANCHES):
            ax=fig.add_subplot(1,3,col+1,projection='3d');color=np.array(COLORS[tag])/255
            ax.scatter(*pc[::max(1,len(pc)//2000)].T,s=1,c=d['colors'][::max(1,len(pc)//2000)]/255)
            bg=d['background_points'];bg=bg[::max(1,len(bg)//1500)]
            ax.scatter(*bg.T,s=.3,c='gray',alpha=.15)
            corners=F[:3,3]+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])@F[:3,:2].T*max(.3,scale)
            ax.add_collection3d(Poly3DCollection([corners],facecolor='orange',alpha=.18))
            ids=np.flatnonzero(tags==tag)
            for i in ids:
                p=g[i,:3,3];end=p+g[i,:3,2]*.06
                ax.plot(*np.stack((p,end)).T,color=color,alpha=.35,linewidth=.8)
            if len(ids):
                i=int(ids[np.argmax(d['confidences'][ids])]);v=mesh.vertices@g[i,:3,:3].T+g[i,:3,3]
                ax.add_collection3d(Poly3DCollection(v[mesh.faces],facecolor=color,alpha=.22,linewidth=0))
                ax.set_title(f'{tag}: {len(ids)} poses\nTop C={d["confidences"][i]:.3f}; clearance={d["clearance_m"][i]*1000:.1f} mm')
            for k,axis in enumerate(('x','y','z')):getattr(ax,'set_'+axis+'lim')(center[k]-scale,center[k]+scale)
            ax.set(xlabel='camera X / m',ylabel='camera Y / m',zlabel='camera Z / m');ax.set_box_aspect((1,1,1))
        fig.tight_layout();fig.savefig(target/(Path(entry['result_file']).stem+'.png'),dpi=130);plt.close(fig)


if __name__=='__main__':main(sys.argv[1])
