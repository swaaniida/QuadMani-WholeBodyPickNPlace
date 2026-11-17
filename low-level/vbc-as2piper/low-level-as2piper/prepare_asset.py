"""Create an Isaac Gym import copy, retaining the UMI TCP fixed joint."""
from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET


def prepare(source):
    tree=ET.parse(source)
    joint=tree.getroot().find("joint[@name='piper_umi_tcp_joint']")
    assert joint is not None
    joint.set('dont_collapse','true')
    for fixed in tree.getroot().findall('joint'):
        if fixed.find('child').get('link') in ('FL_foot','FR_foot','RL_foot','RR_foot'):
            fixed.set('dont_collapse','true')
    output=source.with_name('as2piper_plate_training.urdf')
    content=ET.tostring(tree.getroot(),encoding='utf-8',xml_declaration=True)
    if output.exists() and output.read_bytes()==content:
        return output
    # A group collaborator can replace a file in this shared directory, but
    # cannot chmod a file owned by another user. Set permissions only on our
    # newly created inode, then atomically publish the complete asset.
    fd,name=tempfile.mkstemp(prefix=output.name+'.',suffix='.tmp',dir=str(output.parent))
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fchmod(handle.fileno(),0o664)
        os.replace(name,str(output))
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return output


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    print(prepare(root/'resources/robots/as2piper/urdf/as2piper_umi_v7_mesh_seated_plate_only.urdf'))
