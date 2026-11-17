# 누적 설계 지식·실패 사례·판단

## 모델/초기자세

- AS2 다리12+PiPER6+양finger2=20DOF.18policyaction중팔6은원본마스킹;팔은DLS6DIK/position drive. finger±0.049m열림유지.
- 최신카메라/거치대포함URDF명목질량23.4303404741327kg. baseline45k실제훈련asset23.32806408133kg. hardware.json의질량은이전gain/DR설계참고값이며actual simulator mass는URDF에서결정.
- arm base/base_link=(0.22435,0,0.1319)m;Orin origin=(-0.03,0,0.111875)m,질량0.8725kg. 실물체결/커버지지/볼트강도검증과시각결합은구분.
- legdefault4다리모두[0,0.8,-1.6]rad;arm=[0,1.1,-0.55,0,0,0]. standingbase0.3244069024m에서28mm반경foot가네발모두기구학적으로접지. reset0.3264069024m은2mm상공. termination0.1m와혼동금지.
- 원본B1leg hip좌+0.2/우−0.2,thigh0.8,calf−1.5.40mmfoot collision로default접지base0.516164m,baseheightreward0.55m와33.836mm차이. 이것은기본각고정계산이며학습된B1이피할수없는penalty는아님. 원본B1visual sphere30mm≠collision40mm.
- UMI V7 형상·방향보존.142.5mm그리퍼베이스폭은EE축길이가아님. 단순링크6아닌실제TCP(piper_umi_nominal_tcp)로도달성판정.

## EE 좌표/생성

- 실제armbase관측·IK는실제body회전반영. 목표중심은standingarmbase의가상yaw frame으로height/roll/pitch-invariant: x0.22435,y0,z0.4563069024m+body아래지형높이.
- XY/yaw는locomotion을따르지만crouch/tilt가목표z를같이끌어내리지않음.
- 현재독립구면 r[.35,.80],고도각[−pi/2.5,pi/3](−72~60°),좌우각±1.2rad. 지면높이필터[.02,1.14912722545]m. orientationdelta각±.08rad;rolloffset0;pitch=2.455216137744035−고도각+delta. 바닥전용pitch연동제거.
- baseline45k: r[.40,.48],고도각[.10,.50],좌우±.25rad,실제armbase중심이body높이/회전을따름. 현재와범위를혼동하지않음.
- 원본B1:r[.4,.95],고도각−72~60°,좌우±1.2,center[.3,0,.7],pitch=.38−고도각,rollpi/2,delta각±.5.
- sampling10회실패시이전위치뿐아니라orientationdelta도복원하는오류수정. 후보bank/IKvalidity는현재없음. bbox 전방20mm연장.
- nominalfeet+bodypose/6DIK/충돌조사에서211목표중205해발견,6개미해결. 완전한연속범위의도달성인증아님. far-forwardground일부는bodypitch약40°필요.
- currentasset의CoMstatic조사에서최악전방margin78.2mm/rear하중여유18.5%수준,nominal약35.1%. 동적안정성·마찰·payload를인증하는수치는아님. 기록된RR무접촉이단순전방CoM전복때문이라는증거는없음.

## Observation/PPO/ROA

- current67+privileged18+history67×10=755.배포67+670=737. contact현재4+history40은완전제거;zero padding없음. gait5는활성. orientation59:62의기존zero는별개보존.
- current67:IMU0:5,q5:23,dq23:41,이전legcommand41:53,velocity53:56,EEgoal56:59,orientationzero59:62,gaitindex62:63,clock63:67.
- priv18:addedbasemass/COMxyz/grippermass/friction/legmotor12. 학습priv encoder→latent20,historyencoder670→latent20;actor입력87,critic85.90차원이라는설명은잘못.
- iteration%20==0 history rollout/DAgger update,나머지privileged rollout/PPO update. 학습때history도같이학습되지만실물추론시온라인학습안함.
- arm action마스킹/Leg·Arm reward/advantage·critichead·mixing은원본경로유지. reward branch를단순합쳐기여도계산하지않음.

## 보상과 접촉

- standstill=exp(−.05×다리12관절L1default편차),정지에서만+1scale. 접지/하중균등보상이아님. RR이떠도기준각근처면높은점수가능.
- 원본tracking_lin_vel_max는목표초과속도에서포화. 높은+2근처보상과과속이공존. 현재초과속도전용penalty없음.
- feet_contact_forces=reset후2초부터Σmax(norm(netF)−threshold,0),scale−.001. 현재B70N,기존B200N. hardforcecap아님. sensor-onlyfoot_contacts_z는0,PhysX netforce/contact_filt기반reward는유지.
- contactshaping force/vel은positive mismatch×−2. 정지walkingmask로0,이동정상. 실제footworldvelocity/netforce참조. force/vel이0인것을AS2에센서가없어서라고해석하지않음.
- feet_height_target.18m은footcenter높이관련보상의목표. 발바닥18cm/반드시18cm스윙강제아님. 기존B1.30에기하비약.60적용;scale+1유지.
- reward수집은weighted→episode_sums,실제leg/armrew_buf는각/100. Episode_rew는누적/Tmax,정상50Hz에선per-step weighted평균의약50배. Episode_metric/baseheight는높이값이고penalty가아님. 같은W&B이름·단위로비교.
- B1heightreward는worldZ,AS2는localterrain-relative. Rough에서직접동일수치비교주의.3개node최소높이는정확한발직하지형면아님.
- 반경.028sphere의정확한접촉도verticalheight−radius는r(secθ−1)오차(10°0.43/20°1.80/30°4.33mm). 5cmgrid최솟값의위치/양자화오차가더해지면30°예에서28mm가짜들림. 이는기하예이며새실험실측아님.
- exactPhysX mesh로upward support triangle nearestpoint거리−r가유력한개선안. rawheightfield와Delatin단순화mesh가다를수있음. sidewall/edge/penetration분리,공간index/GPU국소탐색필요. 아직구현안함.
- FootLift제안−.2×stop×tanh(max(hmax−.005,0)/.04):기존45k평지RR27.66mm/P9545.20mm자료에서평균−.09824(가상계산). 실제학습적용/효과검증아님. 정지외란회복step도비용을받고max는높은발이하의다른발개선을구분하지못함.
- Rough net-contact-force와ground contact의불일치약56%조사기록존재. netF만으로실제지지율/발높이/안정성인증금지. AS2 EDU실기footforce센서없음을전제로44observation제거;torque/Jacobian estimator는추후하드웨어과제.

## 제어·DR·외란

- policy/IK50Hz,PhysX200Hz(decimation4). 4번에걸친보간이아니라동일목표hold. 논문Z1실기800Hz와학습루프를혼동하지않음. PiPER800Hz독립실기bridge없음.
- arm/gripperPhysXpositiondrive400/40유지. 과거armstrength샘플.7~1.3은실제positiondrive에적용되지않던경로;현재[1,1]명시. legDR유지. MIT SDK gain숫자를PhysX에기계적으로복사하지않음.
- arm target slew3rad/s(.06rad/50Hz step),joint limitclamp. 실제측정관절속도보장아님. effort100Nm모델값유지,정격미확정. arm delay0/accelNone;임의로실기값을확정하지않음.
- legactiondelay는240000policytick전0,이후1tick(20ms)원본실효schedule유지. config3→1은실제지연을3→1줄인것과다름.
- push는8초주기로baseXY속도를대입,연속외력이아님. moving±.3963886m/s,정지정확0명령배수2.5→±.9909716. 실제deltaV는직전속도와차이. N힘과구분.
- termination height.1m/rollpitch.8rad,bodycontact기존목록유지. reset초기높이와termination기준높이구분. arbitrarythreshold완화로넘어짐가리지않음.

## 재현/운영 실패 사례

- terminal/VSCode종료로학습소실가능: nohup+setsid,stdin/dev/null,로그경로명시. py -u/PYTHONUNBUFFERED=1 권장.실시간지표는ablation_metrics.jsonl.
- wandb0.15.12는Forge서버settings=null로실패;project-local0.22.3에서해결. endpoint는api.forge.coreweave.com,로그인은각자. 로그인정보공유금지.
- wandb-core/gpu_stats실행비트누락664로PermissionError:775복구. binary를일괄일반파일chmod하지않음.
- 기존checkpoint가있어도mainoptimizer외historyoptimizer/counter/RNG/환경history누락으로bitwise동일resume불가. 새train은seed초기가중치로시작하며resumeid로착각하지않음.
- A100MIG EGL렌더실패;OSMesaCPU렌더성공.초기MuJoCo3.1.6,후기재생3.5.0/Python3.11사용. framebuffer크기와offwidth/offheight일치필요.
- PD/scripteddemo/기구학한계시각화/학습policyrollout/실물인증을구분. meshmaxZ가실제체결면이라는가정으로상판이뜬과거오류재발금지.
- offline HTML은js/font/data를내장해야file://에서동작. 현재다운로드는vy0/ground64만,좌우미학습시험제외. 원본서버자료와현재70N model은다른실험.

## 원본 B1과 AS2에서 특히 혼동하면 안 되는 점

- B1 학습 checkpoint는 gait clock을 CLI에서 활성화한799차원(proprio71+priv18+history710)이다. 원본 config의 gait=False만 보고 실제 학습도 비활성으로 단정하지 않는다.
- 과거 AS2 schema:contactON/gaitOFF=744,contactOFF/gaitOFF=700,contactOFF/gaitON=755. 현재 배포737. B1 contact44를0으로 둔 ablation은799차원을 그대로 유지한 분석이며, AS2의 실제44제거 구현과 다르다.
- 원본 B1 feet_contact_forces는force_sensor_tensor의6축 전체 norm(N과Nm혼합)에40threshold를 적용했다. 실제 net contact force의40N과 동등하지 않다. B1 평지정지 sensor6평균9.94/P9920.25 대 netforce평균142.04N/P99232.82N. 원본페널티가 거의0이어도 정상하중이40N미만이라는 뜻이 아니다.
- 원본 B1 shaping은 음의mismatch에음의scale을곱해양의보상이되는부호문제,vel은갱신되지않는cache를읽는문제가있었다. AS2는positive mismatch×negative scale,livevelocity참조로수정했다. 문서의과거gaitOFF시점과현재ON시점을구분한다.
- 정지 mask는|vx_cmd|>.2 또는|vy_cmd|>.2 또는|yaw_cmd|>.5의보행mask반대다. Push2.5배의정지는commands.abs().max()<1e-6로별도정의되어있다.
- Rough force norm 통계가낮다고 실제지지하중/충격이감소했다고해석하지않는다. B1/AS2 모두trimesh force집계한계를분리해측정해야한다.
