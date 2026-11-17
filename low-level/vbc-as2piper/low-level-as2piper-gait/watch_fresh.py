"""Evaluate fresh-training milestones and request a checkpointed local stop."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from summarize_gait import summarize

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--preset',default='stable');p.add_argument('--output',required=True)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[200,400,600])
    p.add_argument('--rough-dr',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'logs/as2piper-low'/a.run;start=time.monotonic()
    for it in a.checkpoints:
        checkpoint=folder/('model_%d.pt'%it)
        while not checkpoint.exists():
            if time.monotonic()-start>10800:raise TimeoutError('Fresh training checkpoint did not finish')
            time.sleep(5)
        target=out/('eval_%d'%it)
        print('Evaluating fresh checkpoint',checkpoint,flush=True)
        with (out/('eval_%d.log'%it)).open('w') as log:
            command=[sys.executable,'-u','evaluate_hw.py',str(checkpoint),'--vbc-port',
                '--gait-preset',a.preset,'--arm-goals','--vx','.25','--yaw','.6','--output',str(target)]
            if a.rough_dr:command+=['--terrain','trimesh','--domain-rand']
            subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        d=summarize(target);summary=dict(checkpoint=str(checkpoint),passed=d['local_acceptance_pass'],
            criteria=d['local_acceptance_criteria'],error_score=d['selection_error_score'])
        print(json.dumps(summary),flush=True);(out/'progress.json').write_text(json.dumps(summary,indent=2))
        if d['local_acceptance_pass']:
            (out/'selected_checkpoint.txt').write_text(str(checkpoint)+'\n')
            (out/'stop_after_update').write_text('Fresh nominal command-following criteria passed; preserve final completed update.\n')
            return


if __name__=='__main__':main()
