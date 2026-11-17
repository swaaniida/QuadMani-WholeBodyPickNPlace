from pathlib import Path
import json,csv
import numpy as np
O=Path(__file__).resolve().parent;a=json.loads((O/'scan.json').read_text());records=a['records'];sols=[s for x in records for s in x['solutions']]
worst=min(((x,s) for x in records for s in x['solutions']),key=lambda pair:pair[1]['front_edge_x_margin_m'])
summary=dict(goals=len(records),goals_with_pose=sum(bool(x['solutions']) for x in records),poses=len(sols),static_support_infeasible=sum(not s['static_support_feasible'] for s in sols),CoM_outside_support=sum(s['com_margin_m']<0 for s in sols),front_edge_margin_min_mm=min(s['front_edge_x_margin_m'] for s in sols)*1000,support_polygon_margin_min_mm=min(s['com_margin_m'] for s in sols)*1000,rear_total_min_N=min(s.get('rear_total_min_N',float('inf')) for s in sols),max_possible_min_foot_N_min=min(s.get('max_possible_min_foot_N',0) for s in sols),leg_effort_support_infeasible=sum(not s.get('URDF_leg_effort_support_feasible',False) for s in sols),leg_effort_maxmin_foot_min_N=min(s.get('URDF_effort_max_possible_min_foot_N',0) or 0 for s in sols),mass_kg=a['mass_kg'],baseline=a['baseline'],worst_goal={k:v for k,v in worst[0].items() if k not in ['solutions','best_attempt']},worst_pose=worst[1],radius_cap_comparison=[])
for cap in [.65,.75,.8]:
 sub=[x for x in records if x['r']<=cap+1e-8];ss=[s for x in sub for s in x['solutions']]
 summary['radius_cap_comparison'].append(dict(cap=cap,goals=len(sub),goals_with_pose=sum(bool(x['solutions']) for x in sub),poses=len(ss),front_margin_min_mm=min(s['front_edge_x_margin_m'] for s in ss)*1000,rear_total_min_N=min(s.get('rear_total_min_N',float('inf')) for s in ss),CoM_outside=sum(s['com_margin_m']<0 for s in ss)))
(O/'summary.json').write_text(json.dumps(summary,indent=2))
with (O/'target_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['r','phi_deg','yaw_deg','x','y','z','pose_count','min_front_margin_mm','min_rear_N'])
 for x in records:
  ss=x['solutions'];w.writerow([x['r'],np.rad2deg(x['phi']),np.rad2deg(x['yaw']),*x['target'],len(ss),min((s['front_edge_x_margin_m']*1000 for s in ss),default=''),min((s.get('rear_total_min_N',float('inf')) for s in ss),default='')])
print(json.dumps({k:v for k,v in summary.items() if k not in ['baseline','worst_pose','worst_goal']},indent=2))
