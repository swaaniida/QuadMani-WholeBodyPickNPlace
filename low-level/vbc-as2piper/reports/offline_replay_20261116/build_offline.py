"""Package existing 8774 replay into a self-contained file; source stays intact."""
from pathlib import Path
import re,json,gzip,base64,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'reports/ee_ground_compare_20261114/viewer'
OUT=SRC/'downloads';OUT.mkdir(exist_ok=True)
HERE=Path(__file__).resolve().parent
hashes={};blobs={};scenarios={}
def read(p):
 b=p.read_bytes();hashes[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b
def asset(p):
 b=read(p);key=hashlib.sha256(b).hexdigest()
 if key not in blobs:blobs[key]=base64.b64encode(gzip.compress(b,compresslevel=9,mtime=0)).decode()
 return key
for key,folder in [('motion36_vy0',SRC/'datasets/motion36_vy0'),('ground64',SRC)]:
 m=json.loads(read(folder/'manifest.json'))
 scenarios[key]={'manifest':m,'files':{n:asset(folder/n) for n in ['positions.bin','indices.bin','poses.bin','signals.bin']}}
three=read(SRC/'vendor/three/three.module.js').decode()
match=re.search(r'^export \{ (.*?) \};\s*$',three,re.M|re.S);assert match and not re.search(r'^import ',three,re.M)
exports=[]
for v in match.group(1).split(','):
 v=v.strip();pair=v.split(' as ');exports.append(pair[-1]+':'+pair[0])
three=three[:match.start()]+'\nreturn {'+','.join(exports)+'};\n'+three[match.end():]
controls=read(SRC/'vendor/three/OrbitControls.js').decode();match=re.match(r"import \{(.*?)\} from 'three';",controls,re.S);assert match
controls='const {'+match.group(1)+'}=THREE;'+controls[match.end():]
controls=controls.replace('export { OrbitControls };','return OrbitControls;')
app=read(SRC/'viewer_vy0.js').decode()
app=re.sub(r"^import .*?;\n",'',app,flags=re.M)
app=app.replace("const scenarioKey=requestedScenario==='ground64'?'ground64':requestedScenario==='motion36_lateral'?'motion36_lateral':'motion36_vy0';","const scenarioKey=requestedScenario==='ground64'?'ground64':'motion36_vy0';")
app=re.sub(r'^const manifestUrl=.*?;\n','',app,flags=re.M)
app=re.sub(r'^async function fetchFile\(.*?\n','',app,flags=re.M)
app=app.replace('manifest=await fetchFile(manifestUrl.href);','manifest=OFFLINE.scenarios[scenarioKey].manifest;')
old="const buffers=await Promise.all(['positions.bin','indices.bin','poses.bin','signals.bin'].map(p=>fetchFile(new URL(p,manifestUrl).href,true)));"
new="const buffers=await Promise.all(['positions.bin','indices.bin','poses.bin','signals.bin'].map(p=>readEmbedded(OFFLINE.scenarios[scenarioKey].files[p])));"
assert old in app;app=app.replace(old,new)
app=app.replace('serve.sh로 실행한 HTTP 주소에서 열어 주세요.','압축을 풀고 HTML 파일을 Chrome 또는 Edge에서 열어 주세요.')
assert not re.search(r'\b(fetch|import)\s*\(',app)
bootstrap='''
const OFFLINE=JSON.parse(document.getElementById('offline-assets').textContent);
const decodedAssets=new Map();
async function readEmbedded(key){
 if(!decodedAssets.has(key))decodedAssets.set(key,(async()=>{
  if(!globalThis.DecompressionStream)throw new Error('이 브라우저는 압축 해제를 지원하지 않습니다. 최신 Chrome 또는 Edge로 열어 주세요.');
  const text=atob(OFFLINE.blobs[key]);const bytes=new Uint8Array(text.length);
  for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 })());
 return decodedAssets.get(key);
}
'''
html=read(SRC/'index.html').decode();html=re.sub(r'<script type="importmap">.*?</script>','',html,flags=re.S)
html=html.replace('<option value="motion36_lateral">이전 좌우 명령 시험 · 보관</option>','')
font=base64.b64encode(read(SRC/'vendor/fonts/NotoSansKR-Medium.ttf')).decode()
html=html.replace('./vendor/fonts/NotoSansKR-Medium.ttf','data:font/ttf;base64,'+font)
script="(()=>{'use strict';\nconst THREE=(()=>{\n"+three+'\n})();\nconst OrbitControls=(()=>{\n'+controls+'\n})();\n'+bootstrap+app+'\n})();'
# HTML parsing must not terminate a script on a JS string literal.
script=re.sub(r'</script',r'<\/script',script,flags=re.I)
pack='<script type="application/json" id="offline-assets">'+json.dumps({'scenarios':scenarios,'blobs':blobs},separators=(',',':'))+'</script>'
assert '<script type="module" src="./viewer_vy0.js"></script>' in html
html=html.replace('<script type="module" src="./viewer_vy0.js"></script>',pack+'\n<script>'+script+'</script>')
assert 'motion36_lateral' not in html.split('<script type="application/json"')[0]
name='as2piper_3d_offline.html';out=OUT/name;out.write_text(html)
readme='''AS2 + PiPER 3D 기록 뷰어 — 오프라인\n\n1. ZIP을 완전히 압축 해제합니다.\n2. as2piper_3d_offline.html을 Chrome 또는 Edge에서 엽니다.\n3. 서버·Python·인터넷 연결·추가 데이터 파일은 필요 없습니다.\n\n포함: vy=0 EE+보행36초, 정지+바닥EE64초. 이전 좌우 명령 시험 제외.\n체크포인트 선택,4분할 비교,시점 회전/확대/이동,카메라 연결,시간/속도/반복,팔 투명도,EE 경로·방향축·속도화살표 유지.\n현재8774의 기존 기록 재생이며 새 학습모델/시뮬레이션은 추가하지 않았습니다.\n\nThree.js/OrbitControls의 MIT license는 THREE_LICENSE.txt에 포함합니다.\n'''
license=read(SRC/'vendor/three/LICENSE').decode()
with zipfile.ZipFile(OUT/'as2piper_3d_offline.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.write(out,name);z.writestr('사용방법.txt',readme);z.writestr('THREE_LICENSE.txt',license)
(HERE/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
(HERE/'build.json').write_text(json.dumps({'html':str(out.relative_to(ROOT)),'html_bytes':out.stat().st_size,'zip_bytes':(OUT/'as2piper_3d_offline.zip').stat().st_size,'scenarios':list(scenarios),'unique_binary_assets':len(blobs),'network_dependencies':0},indent=2))
print((HERE/'build.json').read_text())
