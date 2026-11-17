# 실제 수행한 검증

- 현재70N/이전200N의13개 native source hash가각학습manifest와전부일치. baseline200 URDF는실제훈련당시파일로export에서만복원.
- 복사된새경로의legged_gym/rsl_rl을import하는것을assert.70N10k/15k/20k및200N45kstrictload,CPU history추론,737입력JIT와동일출력확인. optimizer0/새장시간학습0.
- 주요URDF3개(최신AS2/45k당시AS2/B1)의외부mesh참조누락0. DOF20/20/19,질량23.430340/23.328064/57.986970kg.
- 민감파일명/실제키패턴/미다운로드LFS pointer/원본소스hash/핵심Python구문검사통과.
- MANIFEST_SHA256.json은최종파일별무결성목록. verify_snapshot.py로clone후다시검증.
- 신규OS에PyPI/SDK전체를처음부터설치하거나5천개역사스크립트를전부재실행한검증은아니다. 별도설치된현서버의SDK/의존성과cache를사용한경로독립성/추론검사다.
- EXPORT_PORTABILITY.patch는export에만적용한launcher cache/stdout/기구생성기경로변경. PPO/보상/관측/원래코드와기존학습을변경하지않음.
- 최신저장31800까지의weight는보존했지만성능검증은20k까지. LFS실제객체push와새clone검사는실제원격push후별도진행해야한다.
