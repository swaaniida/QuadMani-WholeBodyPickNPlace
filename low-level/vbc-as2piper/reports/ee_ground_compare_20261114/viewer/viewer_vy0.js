import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $=s=>document.querySelector(s), main=$('#main'), canvas=$('#view'), panels=$('#panels');
let renderer, manifest, poses, signals, worlds=[], selected=3, compare=false, playing=false, time=0, last=performance.now(), ready=false, syncing=false, dirty=true;
const requestedScenario=new URLSearchParams(location.search).get('scenario');
const scenarioKey=requestedScenario==='ground64'?'ground64':requestedScenario==='motion36_lateral'?'motion36_lateral':'motion36_vy0';
const manifestUrl=new URL({ground64:'./manifest.json',motion36_lateral:'./datasets/motion36/manifest.json',motion36_vy0:'./datasets/motion36_vy0/manifest.json'}[scenarioKey],location.href);
const progress=s=>$('#progress').textContent=s;
function failure(e){console.error(e);$('#loading').classList.remove('off');$('#loading').classList.add('error');$('#loading').innerHTML='<div><strong>3D 뷰어를 불러오지 못했습니다</strong><div class="error-text"></div><p>serve.sh로 실행한 HTTP 주소에서 열어 주세요.</p></div>';$('#loading .error-text').textContent=e.message||String(e);}
window.addEventListener('error',e=>failure(e.error||e.message));
window.addEventListener('unhandledrejection',e=>failure(e.reason));
async function fetchFile(name,binary=false){const response=await fetch(name);if(!response.ok)throw new Error(`${name}: HTTP ${response.status}`);return binary?response.arrayBuffer():response.json();}
function signal(frame,j,field,n=1){if(!(field in manifest.signalOffsets))return n===1?0:new Float32Array(n);const at=(frame*4+j)*manifest.signalStride+manifest.signalOffsets[field];return n===1?signals[at]:signals.subarray(at,at+n);}
function frameIndex(){return Math.max(0,Math.min(manifest.frames-1,Math.round(time*manifest.fps)-1));}
function phaseAt(f,j){return manifest.knots[Math.round(signal(f,j,'phase'))][3];}
function displayMode(){dirty=true;panels.classList.toggle('single',!compare);$('#mode').textContent=compare?'단일 모델 보기':'4개 동시 비교';worlds.forEach((w,j)=>{w.panel.classList.toggle('active',j===selected);w.controls.enabled=compare||j===selected;});resize();}
function setSelected(j){selected=j;$('#checkpoint').value=String(j);displayMode();}
function syncCamera(source){dirty=true;if(syncing||!$('#linked').checked)return;syncing=true;for(const w of worlds){if(w===source)continue;const shift=new THREE.Vector3();if(ready&&$('#follow').checked){const a=signal(frameIndex(),source.index,'root',3),b=signal(frameIndex(),w.index,'root',3);shift.set(b[0]-a[0],b[1]-a[1],0);}w.camera.position.copy(source.camera.position).add(shift);w.camera.quaternion.copy(source.camera.quaternion);w.controls.target.copy(source.controls.target).add(shift);w.controls.update();}syncing=false;}
function cameraPreset(kind){const w=worlds[selected];const anchor=ready&&$('#follow').checked?signal(frameIndex(),selected,'root',3):[0,0,0];w.controls.target.set(anchor[0]+.25,anchor[1],.34);const delta={home:[1.15,-1.75,1.0],side:[0,-1.9,.45],front:[1.9,0,.45],top:[.001,0,2.1]}[kind];w.camera.position.copy(w.controls.target).add(new THREE.Vector3(...delta));w.controls.update();syncCamera(w);}
function resize(){dirty=true;if(!renderer)return;renderer.setSize(main.clientWidth,main.clientHeight,false);for(const w of worlds){const r=w.panel.getBoundingClientRect();if(r.width&&r.height){w.camera.aspect=r.width/r.height;w.camera.updateProjectionMatrix();}}}
function setPlaying(value){playing=value;last=performance.now();$('#play').textContent=playing?'일시정지':'재생';}
function seek(value){dirty=true;time=Math.max(0,Math.min(manifest.duration,Number(value)));setPlaying(false);updatePose();}

function makeWorld(j,geometries){
 const panel=document.createElement('div');panel.className='panel';panel.dataset.checkpoint=j;panel.innerHTML='<div class="tag"></div>';panels.append(panel);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#e9eef2');
 const camera=new THREE.PerspectiveCamera(43,1,.005,40);camera.up.set(0,0,1);camera.position.set(1.4,-1.75,1.34);
 const controls=new OrbitControls(camera,panel);controls.target.set(.25,0,.34);controls.enableDamping=false;controls.minDistance=.16;controls.maxDistance=12;controls.minPolarAngle=.015;controls.maxPolarAngle=Math.PI-.015;controls.update();
 const world={index:j,scene,camera,controls,panel,bodies:[],armMaterials:[]};controls.addEventListener('change',()=>syncCamera(world));panel.addEventListener('pointerdown',()=>{if(selected!==j)setSelected(j);});
 scene.add(new THREE.HemisphereLight(0xffffff,0x667785,1.65));const light=new THREE.DirectionalLight(0xffffff,1.65);light.position.set(1,-2,3);scene.add(light);const fill=new THREE.DirectionalLight(0xc9dceb,.55);fill.position.set(-1,2,1.5);scene.add(fill);
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(24,24),new THREE.MeshStandardMaterial({color:0xd3dde5,roughness:1}));floor.position.z=-.001;scene.add(floor);const grid=new THREE.GridHelper(24,120,0x8093a1,0xb9c8d4);grid.rotation.x=Math.PI/2;grid.position.z=.0003;scene.add(grid);scene.add(new THREE.AxesHelper(.15));
 for(let b=0;b<manifest.bodies;b++){const group=new THREE.Group();scene.add(group);world.bodies.push(group);}
 for(const item of manifest.geoms){
  let geometry;
  if(item.kind==='mesh')geometry=geometries[item.mesh];
  else if(item.kind==='box')geometry=new THREE.BoxGeometry(...item.size.map(x=>2*x));
  else if(item.kind==='sphere')geometry=new THREE.SphereGeometry(item.size[0],20,12);
  else if(item.kind==='cylinder'){geometry=new THREE.CylinderGeometry(item.size[0],item.size[0],2*item.size[1],24);geometry.rotateX(Math.PI/2);}
  else if(item.kind==='capsule')geometry=new THREE.CapsuleGeometry(item.size[0],2*item.size[1],6,12);
  else throw new Error('지원하지 않는 형상 '+item.kind);
  const color=new THREE.Color().setRGB(...item.rgba.slice(0,3));const material=new THREE.MeshStandardMaterial({color,roughness:.74,metalness:.08,side:THREE.DoubleSide});const mesh=new THREE.Mesh(geometry,material);mesh.position.fromArray(item.position);mesh.quaternion.fromArray(item.quaternion);world.bodies[item.body].add(mesh);
  if(item.name.includes('piper'))world.armMaterials.push(material);
 }
 world.lastRoot=new THREE.Vector3(...signal(0,j,'root',3));world.velocityArrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),.3,0x287bd8,.06,.035);scene.add(world.velocityArrow);
 world.target=new THREE.Mesh(new THREE.SphereGeometry(.013,16,10),new THREE.MeshBasicMaterial({color:0xf34b39}));world.actual=new THREE.Mesh(new THREE.SphereGeometry(.009,16,10),new THREE.MeshBasicMaterial({color:0x00a7b5}));scene.add(world.target,world.actual);
 world.targetAxes=new THREE.AxesHelper(.09);world.actualAxes=new THREE.AxesHelper(.065);scene.add(world.targetAxes,world.actualAxes);
 const pts=[];for(let f=0;f<manifest.frames;f+=4)pts.push(new THREE.Vector3(...signal(f,j,'target',3)));
 world.targetPath=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xd85345,transparent:true,opacity:.35}));scene.add(world.targetPath);if(manifest.movingCommands){world.targetPath.geometry.dispose();world.targetTraceArray=new Float32Array(76*3);world.targetPath.geometry=new THREE.BufferGeometry();world.targetPath.geometry.setAttribute('position',new THREE.BufferAttribute(world.targetTraceArray,3));world.targetPath.frustumCulled=false;}
 const traceGeometry=new THREE.BufferGeometry();world.traceArray=new Float32Array(76*3);traceGeometry.setAttribute('position',new THREE.BufferAttribute(world.traceArray,3));world.trace=new THREE.Line(traceGeometry,new THREE.LineBasicMaterial({color:0x009ca9}));world.trace.frustumCulled=false;scene.add(world.trace);
 return world;
}

function updatePose(){
 if(!ready)return;const f=frameIndex();
 for(let j=0;j<4;j++){
  const w=worlds[j];for(let b=0;b<manifest.bodies;b++){const at=((f*4+j)*manifest.bodies+b)*7;w.bodies[b].position.fromArray(poses,at);w.bodies[b].quaternion.fromArray(poses,at+3);}
  w.target.position.fromArray(signal(f,j,'target',3));w.actual.position.fromArray(signal(f,j,'ee',3));w.targetAxes.position.copy(w.target.position);w.actualAxes.position.copy(w.actual.position);w.targetAxes.quaternion.fromArray(signal(f,j,'targetQuat',4));w.actualAxes.quaternion.fromArray(signal(f,j,'eeQuat',4));w.targetAxes.visible=w.actualAxes.visible=$('#axes').checked;
  w.targetPath.visible=w.trace.visible=$('#targetPath').checked;let count=0;for(let k=Math.max(0,f-75);k<=f;k++){w.traceArray.set(signal(k,j,'ee',3),count*3);if(w.targetTraceArray)w.targetTraceArray.set(signal(k,j,'target',3),count*3);count++;}w.trace.geometry.attributes.position.needsUpdate=true;w.trace.geometry.setDrawRange(0,count);if(w.targetTraceArray){w.targetPath.geometry.attributes.position.needsUpdate=true;w.targetPath.geometry.setDrawRange(0,count);}
  const cmd=signal(f,j,'commands',3);const root=signal(f,j,'root',3);const baseIndex=manifest.bodyNames.indexOf('base_link');const q=w.bodies[baseIndex].quaternion;const yaw=Math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));const speed=Math.hypot(cmd[0],cmd[1]);w.velocityArrow.visible=$('#velocityArrows').checked&&speed>.001;w.velocityArrow.position.set(root[0],root[1],root[2]+.20);if(speed>.001){w.velocityArrow.setDirection(new THREE.Vector3(cmd[0]*Math.cos(yaw)-cmd[1]*Math.sin(yaw),cmd[0]*Math.sin(yaw)+cmd[1]*Math.cos(yaw),0).normalize());w.velocityArrow.setLength(speed*1.2,.06,.035);}
  const reset=manifest.resetEvents.some(e=>e.env===manifest.videoEnvs[j]&&time>=e.time&&time-e.time<.8);
  w.panel.querySelector('.tag').innerHTML=`<b>${manifest.iterations[j].toLocaleString()} iteration</b><br>위치 ${(signal(f,j,'positionError')*100).toFixed(1)}cm · 방향 ${signal(f,j,'orientationError').toFixed(1)}°`+(manifest.movingCommands?`<br>명령 vx ${cmd[0].toFixed(2)} / vy ${cmd[1].toFixed(2)} / yaw ${cmd[2].toFixed(2)}<br>실제 ${signal(f,j,'vx').toFixed(2)} / ${signal(f,j,'vy').toFixed(2)} / ${signal(f,j,'yaw').toFixed(2)}`:'')+(reset?'<br><span class="reset">실패 RESET</span>':'');
 }
 const roots=worlds.map((w,j)=>new THREE.Vector3(...signal(f,j,'root',3)));if($('#follow').checked){syncing=true;const shared=roots[selected].clone().sub(worlds[selected].lastRoot);shared.z=0;worlds.forEach((w,j)=>{const delta=roots[j].clone().sub(w.lastRoot);delta.z=0;if(delta.lengthSq()>1e-16){w.camera.position.add(delta);w.controls.target.add(delta);w.controls.update();dirty=true;}});syncing=false;}worlds.forEach((w,j)=>w.lastRoot.copy(roots[j]));
 const target=signal(f,selected,'target',3),actual=signal(f,selected,'ee',3),cmd=signal(f,selected,'commands',3);
 $('#time').textContent=time.toFixed(2)+' s';$('#timeline').value=time;$('#phase').textContent=phaseAt(f,selected);
 $('#readout').textContent=`목표 z: ${(target[2]*100).toFixed(1)} cm\n실제 z: ${(actual[2]*100).toFixed(1)} cm\n위치 오차: ${(signal(f,selected,'positionError')*100).toFixed(2)} cm\n방향 오차: ${signal(f,selected,'orientationError').toFixed(2)}°\n명령 vx/vy: ${cmd[0].toFixed(2)} / ${cmd[1].toFixed(2)} m/s\n명령 yaw: ${cmd[2].toFixed(2)} rad/s\n실제 vx/vy: ${signal(f,selected,'vx').toFixed(3)} / ${('vy' in manifest.signalOffsets)?signal(f,selected,'vy').toFixed(3):'미기록'} m/s\n실제 yaw: ${signal(f,selected,'yaw').toFixed(3)} rad/s\n목표 XYZ: ${Array.from(target,x=>x.toFixed(3)).join(', ')}\n실제 XYZ: ${Array.from(actual,x=>x.toFixed(3)).join(', ')}`;
 window.__replayState={scenario:scenarioKey,duration:manifest.duration,root:Array.from(signal(f,selected,'root',3)),command:Array.from(cmd),actualVelocity:[signal(f,selected,'vx'),('vy' in manifest.signalOffsets)?signal(f,selected,'vy'):null,signal(f,selected,'yaw')],ready:true,frame:f,time,selected,compare,playing,meshCount:manifest.geoms.length,camera:worlds[selected].camera.position.toArray(),target:Array.from(target),actual:Array.from(actual)};
}

async function load(){
 manifest=await fetchFile(manifestUrl.href);$('#scenario').value=scenarioKey;$('#timeline').max=manifest.duration;$('#duration').textContent=manifest.duration.toFixed(2)+' s';$('#clipBadge').textContent=manifest.duration+'초 · '+(manifest.movingCommands&&manifest.fixedVyZero?'vy 명령 0 고정':'기록된 PhysX 동작');$('#follow').checked=!!manifest.movingCommands;$('#conditions').textContent=manifest.movingCommands?(manifest.fixedVyZero?'vx·yaw 변화 / vy 명령 0 고정 · DR/push OFF. EE xy는 몸통·방향을 따르고 z는 지면 기준입니다.':'이전 vx·vy·yaw 명령 시험 · DR/push OFF. 비영 vy는 미학습 시험입니다.'):'선속도·각속도 명령 0 · 명목 물리값 · DR/push OFF. 지면에 고정된 EE 목표입니다.';progress(`형상 ${manifest.geoms.length}개 · 동작 ${manifest.frames}프레임`);
 const buffers=await Promise.all(['positions.bin','indices.bin','poses.bin','signals.bin'].map(p=>fetchFile(new URL(p,manifestUrl).href,true)));
 const [pb,ib,poseBuffer,signalBuffer]=buffers;
 if(pb.byteLength!==manifest.positionBytes||ib.byteLength!==manifest.indexBytes||poseBuffer.byteLength!==manifest.poseBytes||signalBuffer.byteLength!==manifest.signalBytes)throw new Error('기록 파일 크기가 manifest와 일치하지 않습니다.');
 poses=new Float32Array(poseBuffer);signals=new Float32Array(signalBuffer);
 const geometries=manifest.meshes.map(m=>{const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(pb,m.positionOffset,m.vertexCount*3),3));g.setIndex(new THREE.BufferAttribute(new Uint32Array(ib,m.indexOffset,m.indexCount),1));g.computeVertexNormals();g.computeBoundingSphere();return g;});
 progress('3D 로봇 형상 생성 중…');renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();failure(new Error('WebGL 연결이 끊겼습니다. 페이지를 새로고침하세요.'));});
 worlds=manifest.iterations.map((_,j)=>makeWorld(j,geometries));if(!manifest.geoms.length)throw new Error('로봇 형상이 없습니다.');ready=true;displayMode();updatePose();if($('#follow').checked)cameraPreset('home');$('#loading').classList.add('off');$('#status').textContent=`로봇 형상 ${manifest.geoms.length}개 로드 완료 · ${manifest.massKg.toFixed(3)}kg · 기록 재생 전용`;
 window.__replay={getCameras:()=>worlds.map(w=>w.camera.position.toArray()),stats:()=>renderer.info.render,seek,setPlaying,setSelected,setCompare:v=>{compare=v;displayMode();},manifest};
}

$('#scenario').onchange=e=>{const url=new URL(location.href);url.searchParams.set('scenario',e.target.value);location.href=url.href;};
$('#play').onclick=()=>ready&&setPlaying(!playing);$('#restart').onclick=()=>ready&&seek(0);$('#timeline').oninput=e=>ready&&seek(e.target.value);$('#checkpoint').onchange=e=>ready&&setSelected(Number(e.target.value));$('#mode').onclick=()=>{if(ready){compare=!compare;displayMode();}};
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>ready&&cameraPreset(b.dataset.view));$('#focus').onclick=()=>{if(!ready)return;const w=worlds[selected],point=new THREE.Vector3(...signal(frameIndex(),selected,'root',3)).add(new THREE.Vector3(.15,0,.15));w.camera.position.add(point.clone().sub(w.controls.target));w.controls.target.copy(point);w.controls.update();syncCamera(w);};
$('#opacity').oninput=e=>{for(const w of worlds)for(const material of w.armMaterials){material.opacity=Number(e.target.value);material.transparent=material.opacity<1;material.depthWrite=material.opacity===1;}};
document.addEventListener('input',()=>dirty=true);window.addEventListener('resize',resize);window.addEventListener('keydown',e=>{if(!ready||['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();setPlaying(!playing);}if(e.key==='ArrowRight')seek(time+.2);if(e.key==='ArrowLeft')seek(time-.2);if(e.key.toLowerCase()==='r')cameraPreset('home');});
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);});
function animate(now){requestAnimationFrame(animate);const dt=(now-last)/1000;last=now;if(!ready)return;if(playing){time+=dt*Number($('#speed').value);if(time>manifest.duration){if($('#loop').checked)time%=manifest.duration;else{time=manifest.duration;setPlaying(false);}}}updatePose();for(const w of worlds)if(w.controls.enabled)w.controls.update();if(!dirty&&!playing)return;dirty=false;renderer.setScissorTest(true);const box=canvas.getBoundingClientRect();for(let j=0;j<4;j++){if(!compare&&j!==selected)continue;const w=worlds[j],b=w.panel.getBoundingClientRect();renderer.setViewport(b.left-box.left,box.bottom-b.bottom,b.width,b.height);renderer.setScissor(b.left-box.left,box.bottom-b.bottom,b.width,b.height);renderer.render(w.scene,w.camera);}}
load().catch(failure);requestAnimationFrame(animate);
