#!/usr/bin/env python3
"""Render baseline and three surviving low-inertia alternatives side by side."""
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/compact_pose_search'


def main():
    raw=json.loads((OUT/'results.json').read_text())
    labels={'baseline':'Current','candidate_1':'A: lowest roll/pitch','candidate_10':'B: moderate fold','candidate_13':'C: balanced axes'}
    selected=[]
    for variant in ('plate_only','profiles_plate'):
        m=mujoco.MjModel.from_xml_path(str(HERE/'models/locomani'/(variant+'.xml')))
        r=mujoco.Renderer(m,height=240,width=400)
        sheet=Image.new('RGB',(1200,4*266),'white');draw=ImageDraw.Draw(sheet)
        base=next(x for x in raw['evaluations'] if x['variant']==variant and x['name']=='baseline')
        for row,name in enumerate(labels):
            x=next(x for x in raw['evaluations'] if x['variant']==variant and x['name']==name)
            assert x['max_self_contacts']==0 and x['object_center_visible'] and x['warnings']==0
            x['inertia_change_percent']=(100*(np.array(x['inertia_diag_kg_m2'])/base['inertia_diag_kg_m2']-1)).tolist()
            d=mujoco.MjData(m);d.qpos[:]=x['qpos'];mujoco.mj_forward(m,d)
            for col,view in enumerate(('side','front','wrist')):
                if view=='wrist':camera='wrist_camera'
                else:
                    camera=mujoco.MjvCamera();camera.lookat[:]=[.1,0,.42];camera.distance=1.3
                    camera.azimuth=90 if view=='side' else 0;camera.elevation=-10
                r.update_scene(d,camera=camera);pixels=r.render()
                Image.fromarray(pixels).save(OUT/('%s_%s_%s.png'%(variant,name,view)))
                sheet.paste(Image.fromarray(pixels),(col*400,row*266+26))
                draw.text((col*400+5,row*266+5),labels[name]+' / '+view,fill='black')
                if view=='wrist':
                    x['visible_red_object_pixels']=int(((pixels[:,:,0]>120)&(pixels[:,:,0]>1.7*pixels[:,:,1])&(pixels[:,:,0]>1.7*pixels[:,:,2])).sum())
                    assert x['visible_red_object_pixels']>3
            selected.append(x)
        sheet.save(OUT/(variant+'_comparison.png'));r.close()
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2))
    lines=['# 낮은 관성 팔 자세 후보','',
           '현재 상부 손목 카메라 위치와 장착 모델은 고정했다. 기본 자세 설정은 변경하지 않았다.',
           '', '| 구성 / 후보 | q2, q3, q5 (rad) | 팔 CoM z (mm) | 전신 CoM z (mm) | roll/pitch/yaw 관성 변화 |',
           '|---|---|---:|---:|---|']
    for x in selected:
        q=np.array(x['q_target_rad'])[[1,2,4]]
        lines.append('| %s / %s | %s | %.1f | %.1f | %s %% |'%(x['variant'],labels[x['name']],', '.join('%.3f'%v for v in q),
            x['arm_com_relative_base_m'][2]*1000,x['com_relative_base_m'][2]*1000,
            ' / '.join('%+.1f'%v for v in x['inertia_change_percent'])))
    lines += ['', '[plate_only 비교 이미지](plate_only_comparison.png) · [profiles_plate 비교 이미지](profiles_plate_comparison.png)', '',
        'q1/q4/q6은 모두 0이다. A는 roll/pitch 감소와 높이를 우선하지만 yaw 관성이 증가한다. B는 세 축을 모두 낮추는 완만한 후보, C는 B보다 yaw 감소에 유리한 후보다. 카메라 이미지에서도 C는 전방 물체 주변의 여유가 남아 다음 학습 초기 자세의 우선 평가 후보로 권고한다. B는 roll/pitch 감소 우선 대안이다. A/B/C는 4초 중력 PD 유지와 카메라 중심 광선 검사를 두 장착 구성에서 통과했다. 그러나 A의 실제 손목 영상은 AS2 머리가 중앙 시야를 크게 차지하므로 물체 중심이 보인다는 검사만으로 시야 유지에 적합하다고 판단하면 안 된다. A는 낮은 관성의 비교 기준으로만 남기고 인식용 우선 후보에서 제외한다.', '',
        '관성은 로봇의 각 링크 자체 관성과 평행축 항을 합친 전신 CoM 기준 locked inertia다. 축은 AS2 base 좌표의 roll(X), pitch(Y), yaw(Z)다. 물체와 질량 미정 카메라는 제외했다. 관절 모터 반사 관성이나 실제 보행 동적 안정성 지표는 아니다. 높이는 AS2 base 기준이며 지면 높이가 아니다.', '',
        '탐색: seed=31, 좌우 대칭 3관절 공간 35,000회 표본, 관절 한계 여유 및 8Nm 정적 중력 하중 필터, 중심 물체 FOV/가림 광선 검사. 상위 8개와 중간 관절각 12개를 각 구성에서 중력 PD로 4초 검증했다. 한계에 가까운 첫 탐색 결과(first_pass.json)는 별도 보존했다.', '',
        '시야는 전방 고정 sphere 중심 x=0.61m에서 검사했다. 비교 이미지에서 양 손가락을 확인해야 하며 다른 거리/크기의 물체, 접근 경로 전체의 시야, 발별 동적 지지, ground reach 전환, 충돌 여유 거리와 보행 외란은 미검증이다. 정적 충돌 0만으로 실제 CAD 간섭 여유를 보장하지 않는다. 현재 PD 실제 관절각과 명령각은 다르며 selected.json에 모두 기록했다.', '',
        '재실행: `PYTHONPATH=$PWD/mujoco_mount_compare/.deps:$PWD/mujoco_mount_compare OPENBLAS_NUM_THREADS=1 python mujoco_mount_compare/search_compact_poses.py` 후 OSMesa 환경에서 `report_compact_poses.py`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:15]))


if __name__=='__main__':main()
