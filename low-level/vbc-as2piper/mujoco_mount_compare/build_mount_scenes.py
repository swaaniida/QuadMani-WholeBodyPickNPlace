#!/usr/bin/env python3
"""Build two physically dynamic MuJoCo MJCF robots from the existing AS2+PiPER+UMI V7 URDF.

Pure Python 3.8 stdlib; MuJoCo is NOT required for generation.
Do not modify the source robot URDF or the legacy VBC experiment.
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / 'low-level/resources/robots/as2piper/urdf/as2piper_umi_actuated_v7.urdf'
MODELS = HERE / 'models'


def nums(s, default):
    return [float(x) for x in (s if s is not None else default).split()]


def vec(a):
    return ' '.join('{:.10g}'.format(float(v)) for v in a)


def qmul(a, b):
    w,x,y,z=a; W,X,Y,Z=b
    return [w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
            w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W]


def quat_from_rpy(rpy):
    r,p,y=rpy
    qx=[math.cos(r/2),math.sin(r/2),0,0]
    qy=[math.cos(p/2),0,math.sin(p/2),0]
    qz=[math.cos(y/2),0,0,math.sin(y/2)]
    q=qmul(qmul(qz,qy),qx)
    return vec(q)


def origin_attrs(elem):
    if elem is None: return {'pos':'0 0 0','quat':'1 0 0 0'}
    return {'pos':vec(nums(elem.get('xyz'),'0 0 0')),
            'quat':quat_from_rpy(nums(elem.get('rpy'),'0 0 0'))}


def box_inertia(m, xyz):
    a,b,c=xyz
    return [m*(b*b+c*c)/12, m*(a*a+c*c)/12, m*(a*a+b*b)/12]


def box_body(parent, name, pos, dims, mass, color, solid=True):
    b=ET.SubElement(parent,'body',name=name,pos=vec(pos))
    ET.SubElement(b,'inertial',pos='0 0 0',mass='{:.9g}'.format(mass),
                  diaginertia=vec(box_inertia(mass,dims)))
    ET.SubElement(b,'geom',name='hardware_'+name,type='box',
                  size=vec([x/2 for x in dims]),rgba=color,
                  contype='1' if solid else '0', conaffinity='1' if solid else '0',
                  friction='0.9 0.005 0.0001')
    return b


class Converter:
    def __init__(self, source, output_path, config, variant, strict=True):
        self.source=source; self.output_path=output_path
        self.cfg=config; self.variant=variant; self.strict=strict
        self.robot=ET.parse(source).getroot()
        self.links={x.get('name'):x for x in self.robot.findall('link')}
        self.children={}
        self.joints=self.robot.findall('joint')
        for joint in self.joints:
            pname=joint.find('parent').get('link')
            self.children.setdefault(pname, []).append(joint)
        active=[j for j in self.joints if j.get('type') not in ('fixed','floating')]
        if len(active)!=20:
            raise ValueError('Expected V7 20 movable joints (found {}); wrong source?'.format(len(active)))
        for required in ('base_link','piper_base_link'):
            if required not in self.links:
                raise ValueError('Missing {}'.format(required))
        self.asset_cache={}
        self.missing=[]; self.skipped=[]
        self.root=ET.Element('mujoco',model='as2piper_'+variant)
        ET.SubElement(self.root,'compiler',angle='radian',autolimits='true')
        ET.SubElement(self.root,'option',timestep=vec([config['time_step_s']]),
                      gravity='0 0 {}'.format(config['gravity_m_s2']),integrator='implicitfast')
        ET.SubElement(self.root,'size',njmax='5000',nconmax='1000')
        visual=ET.SubElement(self.root,'visual')
        ET.SubElement(visual,'map',znear='0.01')
        self.assets=ET.SubElement(self.root,'asset')
        self.world=ET.SubElement(self.root,'worldbody')
        ET.SubElement(self.world,'light',name='sun',pos='1 -2 3',dir='-0.3 0.4 -1')
        ET.SubElement(self.world,'geom',name='ground',type='plane',pos='0 0 0',
                      size='5 5 0.1',material='groundmat',friction='1.2 0.01 0.001')
        ET.SubElement(self.assets,'material',name='groundmat',rgba='0.42 0.48 0.48 1')
        self.act=ET.SubElement(self.root,'actuator')
        self.ns=[]

    def _find_mesh(self, mesh_elem):
        raw=mesh_elem.get('filename','')
        if raw.startswith('package://'):
            raise ValueError('Unexpected package:// URI in {}'.format(raw))
        p=(self.source.parent/raw).resolve()
        if p.suffix.lower()=='.dae':
            # MuJoCo does not parse COLLADA visual meshes. Use corresponding official STL.
            p=(p.parent.parent/(p.stem+'.stl')).resolve()
        if not p.exists():
            self.missing.append(raw)
            return None
        if p.suffix.lower() not in ('.stl','.obj','.msh'):
            self.skipped.append(str(p))
            return None
        return p

    def add_geom(self, b, g, visual, link_name, i):
        geometry=g.find('geometry')
        if geometry is None or len(geometry)==0: return
        shape=geometry[0]
        tag=shape.tag
        attrs={'name': ('v_' if visual else 'c_')+link_name+'_'+str(i),
               'group': '2' if visual else '3',
               'contype':'0' if visual else '1',
               'conaffinity':'0' if visual else '1'}
        attrs.update(origin_attrs(g.find('origin')))
        if tag=='box':
            attrs['type']='box'; attrs['size']=vec([v*0.5 for v in nums(shape.get('size'),'1 1 1')])
        elif tag=='sphere':
            attrs['type']='sphere'; attrs['size']=shape.get('radius','0.01')
        elif tag=='cylinder':
            attrs['type']='cylinder'; attrs['size']=vec([float(shape.get('radius')),float(shape.get('length'))/2])
        elif tag=='mesh':
            path=self._find_mesh(shape)
            if path is None:
                if self.strict: raise FileNotFoundError('Missing/unsupported mesh {} link {}'.format(shape.get('filename'),link_name))
                return
            scale=vec(nums(shape.get('scale'),'1 1 1'))
            key=(str(path),scale)
            if key not in self.asset_cache:
                name='mesh_{}'.format(len(self.asset_cache))
                # mesh file is resolved relative to output MJCF, not current working directory
                import os
                rel=os.path.relpath(str(path),str(self.output_path.parent))
                ET.SubElement(self.assets,'mesh',name=name,file=Path(rel).as_posix(),scale=scale)
                self.asset_cache[key]=name
            attrs['type']='mesh'; attrs['mesh']=self.asset_cache[key]
        else:
            self.skipped.append('{} {}'.format(link_name,tag))
            return
        if not visual:
            attrs['rgba']='0.7 0.7 0.7 0.0' # collision geoms remain active, hidden by default
            attrs['friction']='1.0 0.005 0.0001' if 'foot' in link_name else '0.7 0.005 0.0001'
        else:
            material=g.find('material/color')
            attrs['rgba']= material.get('rgba','0.7 0.7 0.7 1') if material is not None else (
                '0.20 0.22 0.25 1' if 'piper' in link_name else '0.80 0.85 0.91 1')
        ET.SubElement(b,'geom',attrs)

    def add_inertial(self,b,link):
        inertial=link.find('inertial')
        if inertial is None: return
        massnode=inertial.find('mass')
        tensor=inertial.find('inertia')
        if massnode is None or tensor is None: return
        m=float(massnode.get('value','0'))
        if m<=0: return
        i=[float(tensor.get(k,'0')) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
        attrs=origin_attrs(inertial.find('origin'))
        # Rotate URDF inertia tensor into the body frame.
        # MuJoCo fullinertia must not be combined with quat.
        w,x,y,z = (float(v) for v in attrs.pop('quat').split())
        R = [
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ]
        xx,yy,zz,xy,xz,yz = i
        I = [[xx,xy,xz], [xy,yy,yz], [xz,yz,zz]]
        B = [[sum(R[a][k]*I[k][l]*R[b][l]
                  for k in range(3) for l in range(3))
              for b in range(3)] for a in range(3)]
        attrs['mass']=vec([m])
        attrs['fullinertia']=vec([
            B[0][0], B[1][1], B[2][2],
            B[0][1], B[0][2], B[1][2]
        ])
        ET.SubElement(b,'inertial',attrs)

    def joint_motor(self,joint):
        name=joint.get('name'); t=joint.get('type');
        if t=='fixed': return
        if t not in ('revolute','continuous','prismatic'):
            raise ValueError('Unsupported joint {} type {}'.format(name,t))
        attrs={'name':name,'type':'slide' if t=='prismatic' else 'hinge',
               'axis':vec(nums(joint.find('axis').get('xyz'),'0 0 1')) if joint.find('axis') is not None else '0 0 1'}
        limit=joint.find('limit')
        if limit is not None and limit.get('lower') is not None and limit.get('upper') is not None:
            lo=float(limit.get('lower')); hi=float(limit.get('upper'))
            if hi>lo:
                attrs['range']=vec([lo,hi]); attrs['limited']='true'
        attrs['damping']='0.7' if t=='prismatic' else ('0.7' if name.startswith('piper_') else '0.9')
        attrs['armature']='0.0001' if t=='prismatic' else '0.004'
        if t=='prismatic': cap=35
        elif name.startswith('piper_'): cap=9
        elif ('hip_joint' in name): cap=30
        else: cap=45
        ET.SubElement(self.act,'motor',name='u_'+name,joint=name,gear='1',
                      ctrllimited='true',ctrlrange=vec([-cap,cap]))
        return attrs

    def make_link(self,link_name,parent,pose,via=None):
        b=ET.SubElement(parent,'body',{'name':link_name,**pose})
        if via is not None and via.get('type')!='fixed':
            ET.SubElement(b,'joint',self.joint_motor(via))
        link=self.links[link_name]
        self.add_inertial(b,link)
        for i,x in enumerate(link.findall('visual')): self.add_geom(b,x,True,link_name,i)
        for i,x in enumerate(link.findall('collision')): self.add_geom(b,x,False,link_name,i)
        for joint in self.children.get(link_name,[]):
            if joint.find('child').get('link')=='piper_base_link': continue
            self.make_link(joint.find('child').get('link'),b,origin_attrs(joint.find('origin')),joint)
        return b

    def build(self):
        cfg=self.cfg
        base=self.make_link('base_link',self.world,
                            {'pos':'0 0 {}'.format(cfg['initial_body_height_m']),'quat':'1 0 0 0'})
        ET.SubElement(base,'freejoint',name='base_free')
        mx=float(cfg['robot_mount_x_m']); my=float(cfg['robot_mount_y_m'])
        deck=float(cfg['estimated_as2_mount_surface_z_m'])
        pt=float(cfg['plate_thickness_m'])
        railh=float(cfg['profile_height_m']) if self.variant=='profiles_plate' else 0.0
        if railh:
            length=float(cfg['profile_length_m']); width=float(cfg['profile_width_m'])
            weight=float(cfg['profile_mass_per_meter_kg'])*length
            for side,sgn in (('left',1),('right',-1)):
                box_body(base,'al_profile_'+side,
                         [mx,my+sgn*float(cfg['profile_center_y_m']),deck+railh/2],
                         [length,width,railh],weight,'0.48 0.50 0.53 1')
        plate_w=float(cfg['plate_width_y_m']);plate_l=float(cfg['plate_length_x_m'])
        plate_m=plate_l*plate_w*pt*float(cfg['aluminum_density_kg_m3'])
        plate=box_body(base,'mount_plate',[mx,my,deck+railh+pt/2],
                       [plate_l,plate_w,pt],plate_m,'0.75 0.62 0.25 1')
        # PiPER base_link is attached rigidly to top of the mounting plate.
        self.make_link('piper_base_link',plate,{'pos':vec([0,0,pt/2]),'quat':'1 0 0 0'})
        # Foot markers for camera and EE frame reference; nominal TCP is from URDF if it exists.
        for name in ('FL_foot','FR_foot','RL_foot','RR_foot'):
            pass
        for name in ('base_link','piper_link6'):
            pass
        MODELS.mkdir(parents=True,exist_ok=True)
        ET.ElementTree(self.root).write(str(self.output_path),encoding='utf-8',xml_declaration=True)
        return {'variant':self.variant,'mjcf':str(self.output_path),
                'mount_x_m':mx,'mount_y_m':my,'piper_mount_height_relative_to_as2_base_m':deck+railh+pt,
                'plate_mass_kg':plate_m,
                'profile_total_mass_kg':2*float(cfg['profile_mass_per_meter_kg'])*float(cfg['profile_length_m']) if railh else 0,
                'mesh_assets':len(self.asset_cache), 'missing_meshes':self.missing,
                'skipped_meshes':self.skipped, 'actuator_count':len(self.act)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--config',type=Path,default=HERE/'mount_config.json')
    p.add_argument('--allow-missing-meshes',action='store_true',help='ONLY for offline XML structural checks; not for physics')
    args=p.parse_args()
    if not args.source.is_file():
        raise SystemExit('Missing source V7 URDF: {}\nCopy this package into vbc-as2piper/mujoco_mount_compare/'.format(args.source))
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    results=[]
    for kind in ('profiles_plate','plate_only'):
        out=MODELS/(kind+'.xml')
        generator=Converter(args.source,out,cfg,kind,strict=not args.allow_missing_meshes)
        result=generator.build()
        results.append(result)
        print('{}: model={}, PiPER z={:.4f}m, mount mass={:.4f}kg, motors={}, mesh={}'.format(
            kind,out,result['piper_mount_height_relative_to_as2_base_m'],
            result['plate_mass_kg']+result['profile_total_mass_kg'],result['actuator_count'],result['mesh_assets']))
    (MODELS/'manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Generated two MJCF robot scenes. These do not alter any V7 URDF.')

if __name__=='__main__':main()
