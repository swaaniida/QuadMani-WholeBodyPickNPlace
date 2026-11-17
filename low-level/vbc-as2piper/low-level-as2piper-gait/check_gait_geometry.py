"""Measure learned foot lift and slip using actual mesh and box geometry."""
import argparse,json
from pathlib import Path
import numpy as np
from gait_geometry import distances,box_distances

NAMES=['stand','forward','turn','forward_turn']

def measure(folder,stride=4):
    folder=Path(folder);meta=json.loads((folder/'policy_evaluation.json').read_text())
    extra=json.loads((folder/'metadata.json').read_text())
    x=np.load(folder/'policy_evaluation.npz')['samples']
    if x.shape[-1]<82:raise ValueError('Full world foot positions and dynamic box poses required')
    terrain=np.load(meta['actual_terrain_cache']);vertices=terrain['vertices'].copy()
    vertices[:,:2]-=float(terrain['border_size']);indices=terrain['triangles']
    sample_indices=np.arange(100,len(x),stride);v=x[sample_indices];feet=v[:,:,49:61].reshape(*v.shape[:2],4,3)
    gap=np.empty(feet.shape[:-1],dtype=np.float32)
    tangent_slip=np.full_like(gap,np.nan)
    for env in range(v.shape[1]):
        points=feet[:,env].reshape(-1,3)
        lo=points[:,:2].min(0)-.06;hi=points[:,:2].max(0)+.06;local=[]
        for offset in range(0,len(indices),200000):
            tri=vertices[indices[offset:offset+200000]]
            keep=((tri[:,:,:2].min(1)<=hi)&(tri[:,:,:2].max(1)>=lo)).all(-1)
            if keep.any():local.append(tri[keep])
        if not local:raise ValueError('No actual triangles near recorded foot trajectory')
        triangle_distance,triangle_closest=distances(points,np.concatenate(local),return_closest=True)
        box_pose=np.repeat(v[:,env,75:82],4,axis=0)
        box_distance,box_closest=box_distances(points,box_pose,extra['box_size_m'],return_closest=True)
        use_box=box_distance<triangle_distance
        distance=np.minimum(triangle_distance,box_distance)
        gap[:,env]=distance.reshape(len(v),4)-.028
        if v.shape[-1]>=120:
            nearest=np.where(use_box[:,None],box_closest,triangle_closest)
            direction=(nearest-points)/np.maximum(distance[:,None],1e-9)
            velocity=v[:,env,90:114].reshape(-1,6)
            foot_velocity=velocity[:,:3]+np.cross(velocity[:,3:6],.028*direction)
            box_velocity=np.repeat(v[:,env,114:120],4,axis=0)
            surface_velocity=box_velocity[:,:3]+np.cross(box_velocity[:,3:6],nearest-box_pose[:,:3])
            relative=foot_velocity-np.where(use_box[:,None],surface_velocity,0.)
            tangent=relative-(relative*direction).sum(-1)[:,None]*direction
            tangent_slip[:,env]=np.linalg.norm(tangent,axis=-1).reshape(len(v),4)
        if env%16==0:print('Checked environment',env,flush=True)
    valid=(v[:,:,2]<.5)&(np.abs(v[:,:,40:42]).max(-1)<.35)
    reference=gap<.002;proxy=v[:,:,45:49]>.5;group=meta['envs_per_command']
    result={'reference':'Shortest distance to actual terrain triangles or dynamic oriented box minus sphere radius 0.028m. Geometric near-contact gap<2mm, not GRF. Excludes first 2s, done frames and abs roll/pitch>=0.35rad. Unsigned geometry is not a penetration-depth test.',
            'slip_note':'Uses recorded v_COM + omega cross vertical sphere-bottom offset; approximate contact-point slip on rough terrain.',
            'sampling_dt_s':stride*.02,'completed_updates':meta['completed_updates'],'conditions':meta['conditions']}
    for i,name in enumerate(NAMES):
        sl=slice(i*group,(i+1)*group);ok=valid[:,sl];g=gap[:,sl];r=reference[:,sl];p=proxy[:,sl]
        loaded=ok[...,None]&r
        slip=v[:,sl,35:39]
        freq=np.fft.rfftfreq(len(g),stride*.02);spectrum=abs(np.fft.rfft(r-r.mean(0),axis=0))**2
        band=(freq>.5)&(freq<5);psd=spectrum.mean((1,2))
        result[name]={'valid_fraction':float(ok.mean()),'gap_p95_per_foot_m':[float(np.quantile(g[:,:,k][ok],.95)) for k in range(4)],
            'contact_fraction_per_foot':[(float(r[:,:,k][ok].mean())) for k in range(4)],
            'contact_proxy_agreement':float((p[ok]==r[ok]).mean()),
            'contact_proxy_precision':float((p[ok]&r[ok]).sum()/max(1,p[ok].sum())),
            'contact_proxy_recall':float((p[ok]&r[ok]).sum()/max(1,r[ok].sum())),
            'geometric_contact_estimated_bottom_point_slip_m_s':float(slip[loaded].mean()) if loaded.any() else None,
            'contact_dominant_Hz':float(freq[band][psd[band].argmax()]),
            'geometric_two_or_more_contact_fraction':float((r.sum(-1)>=2)[ok].mean()),
            'geometric_no_contact_fraction':float((r.sum(-1)==0)[ok].mean()),
            'geometric_phase_contact_agreement':float((r==(v[:,sl,17:21]>.5))[ok].mean()),
            'missing_geometric_stance_per_foot':[float((~r[:,:,k])[(v[:,sl,17+k]>.9)&ok].mean()) for k in range(4)],
            'missing_proxy_stance_same_samples':[float((~p[:,:,k])[(v[:,sl,17+k]>.9)&ok].mean()) for k in range(4)],
            'closest_surface_tangent_slip_m_s':float(tangent_slip[:,sl][loaded].mean()) if v.shape[-1]>=120 and loaded.any() else None,
            'grid_minus_contact_ground_abs_p95_m':float(np.quantile(np.abs(v[:,sl,61:65]-v[:,sl,65:69])[loaded],.95)) if loaded.any() else None,
            'counterfactual_grid_height_reward':float(v[:,sl,73][ok].mean()),
            'counterfactual_last_support_height_reward':float(v[:,sl,74][ok].mean())}
    np.savez_compressed(folder/'geometric_contact.npz',gap=gap,valid=valid,sample_indices=sample_indices,tangent_slip=tangent_slip)
    (folder/'geometric_gait.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation');p.add_argument('--stride',type=int,default=4);a=p.parse_args()
    print(json.dumps(measure(a.evaluation,a.stride),indent=2))
