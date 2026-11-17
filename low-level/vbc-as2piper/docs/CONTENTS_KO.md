# 포함 범위와 보존 정책

- 원본폴더구조를유지하여기존상대import/resource경로를살렸다. 상위저장소는low-level외프로젝트를담을수있다.
- deploy70 / nocontact200의완성checkpoint전체,seed초기가중치,manifest,평가samples/result,ablation_metrics압축본포함. snapshot이후생성모델은목록에없음.
- 과거AS2/forward/hw/gait의소스·자산·검증문서·각run의latest및주요round checkpoint선별포함. 이계열중일부는VBC원복/분리보상실험용이다. 최신70N기본으로실행하지않음.
- source_copy_map.json은출처,checkpoint_index.json은모델목록. 최종실제파일hash는MANIFEST_SHA256.json. baseline200URDF는추가camera이전의학습당시파일로export에서만복원했다.
- reports/는원시NPZ·평가표·workspace/CoM/충돌조사·뷰어와스크립트. 이전보고서설정은그시점기록이다. 서술이현재와충돌하면CURRENT_STATE/manifest/실제코드를우선한다.
- tools/ + mujoco_mount_compare/ + robot-sources/ + wrist_camera_integration/source_bundle은URDF/UMI/장착판/카메라형상재구성자료. 처음중앙장착부터head구조까지기록을보존하며모두인증된기계설계라고주장하지않음.
- 원본B1소스/45000checkpoint와rsl_rl을reference/vbc-legacy에복사했다. 서버원본은수정하지않았다.
- 별도Unitree공식mjlab학습및DeepWBC전체저장소는VBC low-level과다른프로젝트여서포함하지않는다. Orin원본URDF는참고자료로만포함.
- SDK/venv/cache/빌드산출물/원시W&B run기록/인증정보는포함하지않음. Loss/Perf/Train/Episode history는numeric JSONL압축으로추출보존한다.
- 과거restart_push25 helper는.reference.txt로보관;삭제/kill포함restart shell은실행진입점에서제외. 일반train/evaluate스크립트만사용한다.
- historical코드/manifest의서버절대경로는출처기록이다. 필요시scripts/relocate_historical.py로한정치환한다. coretrain/evaluate는상대root계산이며새launcher캐시는프로젝트내.cache를사용한다.

## 추가 보존한 과거 근거

초기AS2 audit,forward port audit,하드웨어 reset/IK/보상 audit,gait reward 분리실험의코드·설정·원시NPZ·그림·영상도함께보존했다.중복mesh/배열은Git/LFS의동일SHA객체로중복제거될수있지만작업폴더의논리용량은더크다. prototype의모든중간weight가아니라checkpoint_index에명시된latest/주요iteration을선별했다.
