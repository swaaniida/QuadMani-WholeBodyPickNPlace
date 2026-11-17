#!/usr/bin/env python3
"""Simple metrics from headless MuJoCo rollouts (not proof of robust locomotion)."""
import argparse
import csv
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def metrics(file):
    with file.open(encoding='utf-8') as f:
        d=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    if not d:raise ValueError('Empty log: '+str(file))
    def rms(x):return math.sqrt(sum(v*v for v in x)/len(x))
    return {'file':file.name,'base_final_z_m':d[-1]['base_z_m'],
            'base_min_z_m':min(r['base_z_m'] for r in d),
            'roll_rms_deg':rms([r['roll_deg'] for r in d]),
            'pitch_rms_deg':rms([r['pitch_deg'] for r in d]),
            'com_z_mean_m':sum(r['com_z_m'] for r in d)/len(d),
            'mean_sum_abs_joint_torque_Nm':sum(r['sum_abs_joint_torque_Nm'] for r in d)/len(d),
            'mean_sum_abs_gripper_force_N':sum(r['sum_abs_gripper_force_N'] for r in d)/len(d),
            'contact_samples':len([r for r in d if r['contacts']>0]),
            'samples':len(d)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['stand','crouch','arm','step'],default='stand')
    a=p.parse_args()
    for x in ['profiles_plate','plate_only']:
        file=HERE/'results'/'{}_{}.csv'.format(x,a.mode)
        if not file.exists():
            print('Missing:',file);continue
        print('\n{}:'.format(x))
        for k,v in metrics(file).items(): print('  {}: {}'.format(k,round(v,5) if isinstance(v,float) else v))

if __name__=='__main__':main()
