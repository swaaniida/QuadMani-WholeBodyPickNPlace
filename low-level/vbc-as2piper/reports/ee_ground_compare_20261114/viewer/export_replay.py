"""Export exact visual geometry and recorded-state FK poses. No physics rollout."""
import json
import os
import argparse
from pathlib import Path
import numpy as np
import mujoco

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data_final'


def main():
    global HERE, DATA
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=str(DATA))
    parser.add_argument('--output',default=str(HERE))
    args=parser.parse_args()
    DATA=Path(args.data).resolve()
    HERE=Path(args.output).resolve()
    HERE.mkdir(parents=True,exist_ok=True)
    os.umask(0o002)
    meta=json.loads((DATA/'metadata.json').read_text(encoding='utf-8'))
    z=np.load(DATA/'states.npz')
    model=mujoco.MjModel.from_xml_path(str(DATA/'render_0.xml'))
    data=mujoco.MjData(model)
    qids=[int(model.joint(n).qposadr[0]) for n in meta['dof_names']]
    descriptors=[]
    pos=[]
    idx=[]
    pos_offset=index_offset=0
    mesh_cache={}
    meshes=[]
    kinds={int(mujoco.mjtGeom.mjGEOM_BOX):'box',int(mujoco.mjtGeom.mjGEOM_SPHERE):'sphere',int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder',int(mujoco.mjtGeom.mjGEOM_CAPSULE):'capsule'}
    for gid in range(model.ngeom):
        if model.geom_group[gid]!=2:
            continue
        geom_type=int(model.geom_type[gid])
        item={'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid),
              'body':int(model.geom_bodyid[gid]),'position':model.geom_pos[gid].tolist(),
              'quaternion':model.geom_quat[gid][[1,2,3,0]].tolist(),
              'rgba':model.geom_rgba[gid].tolist(),'size':model.geom_size[gid].tolist()}
        if geom_type==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(model.geom_dataid[gid])
            if mid not in mesh_cache:
                va,vn=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
                fa,fn=int(model.mesh_faceadr[mid]),int(model.mesh_facenum[mid])
                positions=np.asarray(model.mesh_vert[va:va+vn],dtype='<f4')
                faces=np.asarray(model.mesh_face[fa:fa+fn],dtype='<u4').reshape(-1)
                mesh_cache[mid]=len(meshes)
                meshes.append({'positionOffset':pos_offset,'vertexCount':vn,'indexOffset':index_offset,'indexCount':len(faces)})
                pos.append(positions.tobytes());idx.append(faces.tobytes())
                pos_offset+=positions.nbytes;index_offset+=faces.nbytes
            item.update(kind='mesh',mesh=mesh_cache[mid])
        else:
            assert geom_type in kinds,geom_type
            item['kind']=kinds[geom_type]
        descriptors.append(item)
    frames=len(z['time'])
    poses=np.zeros((frames,4,model.nbody,7),dtype='<f4')
    max_error=0.
    reset_discontinuities=[]
    for i in range(frames):
        for j in range(4):
            root=z['root'][i,j]
            data.qpos[:3]=root[:3]
            data.qpos[3:7]=root[[6,3,4,5]]
            data.qpos[qids]=z['q'][i,j]
            mujoco.mj_forward(model,data)
            poses[i,j,:,:3]=data.xpos
            poses[i,j,:,3:]=data.xquat[:,[1,2,3,0]]
            error=float(np.linalg.norm(data.body('piper_umi_nominal_tcp').xpos-z['ee'][i,j]))
            near_reset=any(e['env']==meta['video_envs'][j] and abs(z['time'][i]-e['time'])<.061 for e in meta['reset_events'])
            assert error<.001 or near_reset,(i,j,error)
            if error>=.001:reset_discontinuities.append([i,j,error])
            else:max_error=max(max_error,error)
    signals=np.concatenate([z['root'][:,:,:3],z['target'],z['ee'],z['target_quat'],z['ee_quat'],
        z['velocity'][:,:,None],z['yaw'][:,:,None],z['position_error'][:,:,None],
        z['orientation_error_deg'][:,:,None],z['done'][:,:,None].astype(float),
        np.repeat(z['phase'][:,None,None],4,axis=1),z['finger_force'][:,:,None]],axis=-1).astype('<f4')
    assert signals.shape==(frames,4,24)
    offsets={'root':0,'target':3,'ee':6,'targetQuat':9,'eeQuat':13,'vx':17,'yaw':18,'positionError':19,'orientationError':20,'done':21,'phase':22,'fingerForce':23}
    if 'commands' in z:
        signals=np.concatenate([signals,z['commands'],z['velocity_xy'][:,:,1:2]],axis=-1).astype('<f4')
        offsets.update(commands=24,vy=27)
    manifest={'iterations':meta['iterations'],'frames':frames,'fps':25,'duration':meta['duration_s'],'bodies':model.nbody,
              'bodyNames':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)],
              'meshes':meshes,'geoms':descriptors,'signalStride':signals.shape[-1],'knots':meta['knots'],
              'signalOffsets':offsets,'movingCommands':'commands' in z,
              'fixedVyZero':bool('commands' not in z or np.count_nonzero(z['commands'][:,:,1])==0),
              'targetFrame':meta.get('position_frame'),
              'velocitySegments':meta.get('velocity_segments'),
              'resetEvents':meta['reset_events'],'videoEnvs':meta['video_envs'],'metrics':meta['metrics'],
              'positionBytes':pos_offset,'indexBytes':index_offset,'poseBytes':poses.nbytes,'signalBytes':signals.nbytes,
              'coordinateFrame':'Initial AS2 base xy / ground z, metres, z up; quaternion xyzw',
              'physics':'Recorded PhysX states; FK export only; no re-simulation',
              'maxNonresetTcpFkError':max_error,'resetDiscontinuities':reset_discontinuities,
              'massKg':float(model.body_mass.sum())}
    (HERE/'positions.bin').write_bytes(b''.join(pos))
    (HERE/'indices.bin').write_bytes(b''.join(idx))
    poses.tofile(HERE/'poses.bin');signals.tofile(HERE/'signals.bin')
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Exported:',len(descriptors),'visual geometries,',model.nbody,'bodies,',frames,'frames, max FK error',max_error,flush=True)


if __name__=='__main__':main()
