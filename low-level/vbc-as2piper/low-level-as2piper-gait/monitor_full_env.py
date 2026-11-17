"""Read-only milestone evaluations of matched full-environment PPO trials."""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from summarize_gait import summarize
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[200,400,600]);p.add_argument('--group',type=int,default=16);p.add_argument('--export-all',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for step in a.checkpoints:
        ckpt=ROOT/'logs/as2piper-low'/a.run/('model_%d.pt'%step)
        start=time.monotonic()
        while not ckpt.exists():
            finished=ckpt.parent/'study_finished.json'
            if finished.exists():
                print('Training completed before remaining milestone; nothing to evaluate:',ckpt,flush=True)
                return
            if time.monotonic()-start>21600:raise TimeoutError(str(ckpt))
            time.sleep(10)
        target=out/('eval_%d'%step)
        if not (target/'gait_summary.json').exists():
            command=[sys.executable,'-u','evaluate_hw.py',str(ckpt),'--vbc-port',
                '--gait-preset',a.preset,'--arm-goals','--terrain','trimesh','--domain-rand',
                '--vx','.25','--yaw','.6','--seconds','20','--envs-per-command',str(a.group),
                '--seed','20261102','--output',str(target)]
            if a.export_all or step==a.checkpoints[-1]:command+=['--export-terrain']
            print('Evaluating',ckpt,flush=True)
            with (ROOT/'experiments/gait_full_env_v1/evaluation.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                with (out/('eval_%d.log'%step)).open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            d=summarize(target)
        else:d=json.loads((target/'gait_summary.json').read_text())
        record={k:{m:d[k][m] for m in ['mean_vx_m_s','mean_yaw_rad_s','vx_abs_error_m_s',
             'yaw_abs_error_rad_s','fall_resets_total','ee_position_mean_m']} for k in ['stand','forward','turn','forward_turn']}
        record['checkpoint']=str(ckpt);record['completed_updates']=d['completed_updates']
        record['local_acceptance_pass']=d['local_acceptance_pass']
        record['full_environment_baseline_pass']=d.get('full_environment_baseline_pass',False)
        record['passed']=record['full_environment_baseline_pass']
        (out/('result_%d.json'%step)).write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
