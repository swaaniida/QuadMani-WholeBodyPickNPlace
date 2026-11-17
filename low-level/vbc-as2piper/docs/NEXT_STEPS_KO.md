# 다음 담당자가 먼저 확인할 사항

1. 새로clone한폴더에서GitLFS객체를모두받고verify_snapshot.py를통과시킨다.외부SDK를확보한다.원래서버권한/개인홈/linked.git을복사하지않는다.
2. 최신70N 스냅샷은31800까지저장됐지만성능검증은20k까지다.그이후모델을좋은모델로가정하지말고같은평가조건으로비교한다.
3. 과속/정지RR접지부족을해결하기위한보상변경은사용자승인전적용하지않는다. standstill기본각과높이목표는현재기구학적으로일치한다.절댓값높이오차만보고body가낮은지높은지단정하지말고signed높이/foot clearance/positiveFz/관절각을함께계측한다.
4. Rough지형height/force집계오차를독립접촉기하와대조한다. FootLift등미적용제안과실제학습설정을구분한다.
5. Ground-picking의전신동작/EEorientation/충돌/CoM/토크/latency는중앙EE평가만으로보증되지않는다.작업범위변경은기존표본/한계pose검토와별도실험으로다룬다.
6. hardware최종실측:PiPER torque한계/전류환산/실gain/지연/모터명령주기,AS2 state/SDK입력변환,손목0~.15kg/등0~.30kg추가DR의올바른링크/관성갱신/고정조인트병합/중복질량여부.숫자가미확정이면임의로확정하지않는다.
7. checkpoint불완전resume형식을실제상태완전복원으로설명하지않는다.학습중단상태를정확히보존하려면historyoptimizer/counter/RNG/환경state까지새포맷설계가필요하지만기존PPO알고리즘을멋대로바꾸지않는다.
