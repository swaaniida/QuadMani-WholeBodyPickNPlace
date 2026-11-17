# 현재 상태

## 코드와 실험

- 현재70N: `low-level-as2piper-head-orin-deploy`의 condition B=70N. A40N/Cscale0은보존,실행대상아님. force scale−0.001/feetheight0.18 scale+1 유지.
- W&B project `as2piper-low`,70N 원실행ID `l8onmp0b`,원실행명 `as2piper_headpiper_orin_70N_20261115T133137Z`.
- 200N 원실행은nocontact의 `as2piper_headpiper_orin_200N_0.18m`,ID `gdr8lou4`,45000완료.
- 70N은아직진행중인실험의스냅샷. 저장범위는checkpoint_index.json과PROJECT_SNAPSHOT.json이기준. 최신저장≠성능검증완료.

## 확정 검증:70N10k/15k/20k

평지/rough각32환경,4명령×8환경,40초,초기2초제외,DR/pushoff,history actor. 중앙EE작은궤적이며확장된바닥작업영역전체를시험한것은아님.

|항목|10k|15k|20k|
|---|---:|---:|---:|
|평지전진평균(m/s),명령0.4|0.668|0.560|0.594|
|평지전진속도RMSE(m/s)|0.291|0.174|0.205|
|Rough전진평균(m/s)|0.678|0.558|0.596|
|평지정지RR contact(%),norm>2N|0.046|5.217|0.329|
|평지전진EE평균(cm)|1.910|1.695|1.707|
|평지전진몸통 높이 오차(cm)|0.729|2.035|1.140|
|평지전진forceP99(N)|294.7|171.4|169.6|
|평지복합forceP99(N)|289.2|169.2|216.8|

세체크포인트모두해당평가의실패reset0.20k는15k대비과속/정지접지에서후퇴하며높이추종일부개선. 속도·충격기준15k가더균형적이지만실기배포완료판정은불가. 보고서/원시데이터는reports/deploy70_iter*/와각experiments/.../evaluations에포함.

## 아직 적용하지 않은 것

- Foot Lift Penalty(max+tanh), Ground Proximity, Valid Contact 보상은검토/제안이며현재실행에추가하지않았다.
- Rough정확한sphere/triangle 지지면 거리 보상helper도미구현. 기존지면최솟값계산유지.
- 전신IK feasible bank/online IK validity rejection은사용자요청으로제거. 원본구면+기본box/ground/path검사만유지.
- 실제PiPER effort정격/latency/readback/전류환산/실기통신800Hz bridge 검증미완료. SDK0.6.1/FW S-V1.5-8 참고,명목400/40/effort100은실측인증값이아님.
- 실물안전한토크제한·estop·SDK 명령변환·센서관측 재현을완료했다고주장하지않는다.
