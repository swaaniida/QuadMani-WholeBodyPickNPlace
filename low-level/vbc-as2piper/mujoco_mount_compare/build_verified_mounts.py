#!/usr/bin/env python3
"""Mesh-seated design candidates; AS2 structural fastening remains unverified."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from build_mount_scenes import Converter, SOURCE, HERE, origin_attrs, box_inertia, vec

URDFS = SOURCE.parent
CFG = json.loads((HERE/'mount_config.json').read_text())
CFG.update(robot_mount_x_m=-0.02, robot_mount_y_m=0.,
           estimated_as2_mount_surface_z_m=0.07602847821788371,
           plate_length_x_m=0.14, plate_width_y_m=0.12,
           profile_length_m=0.14, profile_center_y_m=0.035,
           description='Mesh-seated candidate on upper shell; structural interface NOT verified')

def fixed(robot, name, parent, child, xyz):
    j=ET.SubElement(robot,'joint',name=name,type='fixed')
    ET.SubElement(j,'parent',link=parent); ET.SubElement(j,'child',link=child)
    ET.SubElement(j,'origin',xyz=vec(xyz),rpy='0 0 0')

def box(robot,name,dims,mass,color):
    link=ET.SubElement(robot,'link',name=name)
    inert=ET.SubElement(link,'inertial'); ET.SubElement(inert,'origin',xyz='0 0 0',rpy='0 0 0')
    ET.SubElement(inert,'mass',value=str(mass))
    i=box_inertia(mass,dims)
    ET.SubElement(inert,'inertia',ixx=str(i[0]),iyy=str(i[1]),izz=str(i[2]),ixy='0',ixz='0',iyz='0')
    for tag in ['visual','collision']:
        e=ET.SubElement(link,tag);ET.SubElement(e,'origin',xyz='0 0 0',rpy='0 0 0')
        ET.SubElement(ET.SubElement(e,'geometry'),'box',size=vec(dims))
        if tag=='visual':ET.SubElement(ET.SubElement(e,'material',name=name+'_color'),'color',rgba=color)

class DirectConverter(Converter):
    """Convert the explicit URDF tree without injecting a second mounting transform."""
    def make_link(self,link_name,parent,pose,via=None):
        b=ET.SubElement(parent,'body',dict(name=link_name,**pose))
        if via is not None and via.get('type')!='fixed':ET.SubElement(b,'joint',self.joint_motor(via))
        link=self.links[link_name];self.add_inertial(b,link)
        for i,x in enumerate(link.findall('visual')):self.add_geom(b,x,True,link_name,i)
        for i,x in enumerate(link.findall('collision')):self.add_geom(b,x,False,link_name,i)
        for j in self.children.get(link_name,[]):
            self.make_link(j.find('child').get('link'),b,origin_attrs(j.find('origin')),j)
        return b
    def build(self):
        base=self.make_link('base_link',self.world,dict(pos='0 0 0.35',quat='1 0 0 0'))
        ET.SubElement(base,'freejoint',name='base_free')
        ET.SubElement(self.root.find('visual'),'global',offwidth='1280',offheight='720')
        ET.indent(self.root) if hasattr(ET,'indent') else None
        ET.ElementTree(self.root).write(str(self.output_path),encoding='utf-8',xml_declaration=True)


def main():
    # Exact triangle clipping over the plate footprint, never global max(Z).
    from verify_mounts import stl_triangles, clipped_max
    x=CFG['robot_mount_x_m'];L=CFG['plate_length_x_m'];W=CFG['plate_width_y_m']
    CFG['estimated_as2_mount_surface_z_m']=clipped_max(
        stl_triangles(SOURCE.parent/'../meshes/as2/base_link.STL'),[x-L/2,x+L/2,-W/2,W/2])
    out=HERE/'models/mesh_seated';out.mkdir(parents=True,exist_ok=True)
    (out/'mount_config.json').write_text(json.dumps(CFG,indent=2))
    for variant in ['profiles_plate','plate_only']:
        r=ET.parse(SOURCE).getroot();r.set('name','as2piper_umi_v7_mesh_seated_'+variant)
        r.remove(r.find("joint[@name='as2_to_piper']"))
        ET.SubElement(r,'link',name='as2_assumed_mount_interface')
        fixed(r,'as2_assumed_interface_fixed','base_link','as2_assumed_mount_interface',
              [CFG['robot_mount_x_m'],0,CFG['estimated_as2_mount_surface_z_m']])
        h=CFG['profile_height_m'] if variant=='profiles_plate' else 0
        if h:
            for side,sgn in [('left',1),('right',-1)]:
                name='al_profile_'+side
                box(r,name,[CFG['profile_length_m'],CFG['profile_width_m'],h],
                    CFG['profile_length_m']*CFG['profile_mass_per_meter_kg'],'0.48 0.50 0.53 1')
                fixed(r,name+'_fixed','as2_assumed_mount_interface',name,[0,sgn*CFG['profile_center_y_m'],h/2])
        dims=[CFG[k] for k in ['plate_length_x_m','plate_width_y_m','plate_thickness_m']]
        mass=dims[0]*dims[1]*dims[2]*CFG['aluminum_density_kg_m3']
        box(r,'mount_plate',dims,mass,'0.75 0.62 0.25 1')
        fixed(r,'mount_plate_fixed','as2_assumed_mount_interface','mount_plate',[0,0,h+dims[2]/2])
        fixed(r,'plate_to_piper_fixed','mount_plate','piper_base_link',[0,0,dims[2]/2])
        path=URDFS/('as2piper_umi_v7_mesh_seated_'+variant+'.urdf')
        ET.ElementTree(r).write(str(path),encoding='utf-8',xml_declaration=True)
        dest=out/(variant+'.xml');c=DirectConverter(path,dest,CFG,variant);c.build()
        print(variant,path,'PiPER z',CFG['estimated_as2_mount_surface_z_m']+h+dims[2], 'motors',len(c.act))

if __name__=='__main__':main()
