#!/usr/bin/env python3
"""Export the audited Git index, never the whole working directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from audit_publish import ROOT


def main():
    subprocess.run(['python3',str(ROOT/'scripts/audit_publish.py')],check=True,cwd=ROOT)
    names=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0')
    rows=[];output=ROOT/'graspgenx-realsense/dist/grasp-pose-github-source.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for entry in names:
            if not entry:continue
            meta,name=entry.split(b'\t',1);mode,oid,stage=meta.split()
            if stage!=b'0' or mode not in (b'100644',b'100755'):raise ValueError('Nonregular/unmerged index entry')
            path=name.decode();data=subprocess.check_output(['git','cat-file','blob',oid.decode()],cwd=ROOT)
            info=zipfile.ZipInfo('grasp-pose/'+path);info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(int(mode,8)<<16)
            z.writestr(info,data);rows.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('grasp-pose/SOURCE_MANIFEST.json',json.dumps(rows,indent=2)+'\n')
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    print(output)
    print('ZIP SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':main()
