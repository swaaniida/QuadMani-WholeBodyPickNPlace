"""Evaluate only this diagnostic run's checkpoints while its training continues."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'audit_hw/vbc_walk_verify_20261030'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extended',action='store_true');args=parser.parse_args()
    run=(OUT/('extended_run_name.txt' if args.extended else 'run_name.txt')).read_text().strip();folder=ROOT/'logs/as2piper-low'/run
    target_iteration=1000 if args.extended else 300
    checkpoints=[400,600,800,1000] if args.extended else [200,300]
    jobs={};completed={};start=time.monotonic();last_print=0
    while True:
        text=(OUT/('training_extended_retry.log' if args.extended else 'training.log')).read_text(errors='replace')
        iterations=re.findall(r'Learning iteration (\d+)/'+str(target_iteration),text)
        current=int(iterations[-1]) if iterations else -1
        for it in checkpoints:
            target=OUT/('eval_'+str(it))
            if it not in jobs and it not in completed and (folder/('model_%d.pt'%it)).exists():
                if (target/'policy_evaluation.json').exists():
                    completed[it]=0;continue
                log=(OUT/('eval_%d.log'%it)).open('w')
                cmd=[sys.executable,str(ROOT/'evaluate_hw.py'),str(folder/('model_%d.pt'%it)),
                     '--vbc-port','--arm-goals','--vx','.25','--yaw','.6','--output',str(target)]
                jobs[it]=(subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT),log)
                print('Started policy evaluation',it,flush=True)
        for it,(process,log) in list(jobs.items()):
            code=process.poll()
            if code is not None:
                log.close();completed[it]=code;del jobs[it]
                print('Policy evaluation finished',it,'exit',code,flush=True)
        progress=dict(run=run,iteration=current,evaluations=completed,running_evaluations=list(jobs),elapsed_s=time.monotonic()-start)
        (OUT/('progress_extended.json' if args.extended else 'progress.json')).write_text(json.dumps(progress,indent=2))
        if time.monotonic()-last_print>45:
            print(json.dumps(progress),flush=True);last_print=time.monotonic()
        if target_iteration in completed or ((OUT/'stop_after_update').exists() and not jobs):
            if any(completed.values()):raise RuntimeError('An evaluation failed; inspect its log')
            return
        if time.monotonic()-start>7200:raise TimeoutError('Diagnostic checkpoints did not finish within 2h')
        time.sleep(5)


if __name__=='__main__':main()
