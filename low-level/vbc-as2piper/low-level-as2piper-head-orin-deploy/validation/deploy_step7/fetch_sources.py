from pathlib import Path
import urllib.request,json,hashlib
OUT=Path(__file__).parent/'official'
items={
'piper_sdk':['piper_sdk/interface/piper_interface.py','README.MD','CHANGELOG.MD','piper_sdk/piper_msgs/msg_v2/transmit/arm_joint_mit_ctrl.py'],
'pyAgxArm':['docs/piper/firmware_reference.md','docs/piper/piper_api.md','pyAgxArm/protocols/can_protocol/drivers/piper/default/driver.py']}
manifest=[]
for repo,paths in items.items():
 req=urllib.request.Request('https://api.github.com/repos/agilexrobotics/'+repo+'/commits/master',headers={'User-Agent':'AS2-local-audit'})
 commit=json.load(urllib.request.urlopen(req,timeout=30))['sha']
 for path in paths:
  url='https://raw.githubusercontent.com/agilexrobotics/'+repo+'/'+commit+'/'+path
  try:data=urllib.request.urlopen(url,timeout=30).read()
  except Exception as e:print(path,type(e).__name__);continue
  dest=OUT/repo/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  manifest.append(dict(repo=repo,commit=commit,path=path,url=url,sha256=hashlib.sha256(data).hexdigest()))
 # Save tree metadata to identify official model config without full SDK install.
 req=urllib.request.Request('https://api.github.com/repos/agilexrobotics/'+repo+'/git/trees/'+commit+'?recursive=1',headers={'User-Agent':'AS2-local-audit'})
 tree=json.load(urllib.request.urlopen(req,timeout=30))
 (OUT/(repo+'_tree.json')).write_text(json.dumps(tree))
(OUT/'sources.json').write_text(json.dumps(manifest,indent=2));print('saved',len(manifest),'official files')
