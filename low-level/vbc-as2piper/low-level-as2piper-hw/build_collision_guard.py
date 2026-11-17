"""Export conservative collision OBBs from the actual URDF/MuJoCo asset."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mujoco_mount_compare'))
from mount_workspace_study import StudyConverter,setup
from verify_mounts import stl_triangles
from prepare_asset import prepare


def main():
    asset=prepare(ROOT/'resources/robots/as2piper/urdf/as2piper_umi_v7_mesh_seated_plate_only.urdf')
    directory=ROOT/'resources/commands';directory.mkdir(exist_ok=True)
    xml=directory/'plate_forward.xml';StudyConverter(asset,xml).build()
    m=mujoco.MjModel.from_xml_path(str(xml));d,names,_,_=setup(m,'plate_only_forward')
    centers=[];halves=[];indices=[]
    groups=[(i-1,'piper_link'+str(i),['piper_link'+str(i)]) for i in range(2,7)]
    groups.append((5,'piper_link6',['piper_flange_link','piper_gripper_base','piper_gripper_link1','piper_gripper_link2']))
    for index,frame,links in groups:
        bid=m.body(frame).id;points=[]
        for g in range(m.ngeom):
            if not m.geom_contype[g] or m.body(int(m.geom_bodyid[g])).name not in links:continue
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];v=m.mesh_vert[start:start+m.mesh_vertnum[mesh]]
            elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                v=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
            else:raise ValueError('Unhandled arm collision geometry')
            world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            points.extend(((world-d.xpos[bid])@d.xmat[bid].reshape(3,3)).tolist())
        points=np.array(points);low=points.min(0);high=points.max(0)
        centers.append(((low+high)/2).tolist());halves.append(((high-low)/2+.002).tolist());indices.append(index)
    body=stl_triangles(ROOT/'resources/robots/as2piper/meshes/as2/base_link.STL').reshape(-1,3)
    low=body.min(0);high=body.max(0)
    output=dict(asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),link_index=indices,center=centers,half=halves,
        obstacle_center=[((low+high)/2).tolist()],obstacle_half=[((high-low)/2+.005).tolist()],
        joint_low=(m.jnt_range[[m.joint(n).id for n in names],0]+.01).tolist(),
        joint_high=(m.jnt_range[[m.joint(n).id for n in names],1]-.01).tolist(),
        description='Conservative OBBs: links2-6 + flange/gripper/open fingers versus full AS2 body/head mesh AABB. Arm pad2mm/body pad5mm. Fixed mount and link1 excluded as intended connection.',
        limitations='Conservative boxes can reject mesh-clear poses. Not a continuous collision proof; no legs/objects/cables or arm self-collision pairs.')
    (directory/'arm_collision.json').write_text(json.dumps(output,indent=2))
    print(output['description'])


if __name__=='__main__':main()
