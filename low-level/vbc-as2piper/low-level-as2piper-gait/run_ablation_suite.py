"""Sequential, restartable paired PPO trials; never manages other workloads."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/gait_ablation_v1'
PRESETS=['original','velocity','jerk','stop','data_stop','speed','stable']


def summarize():
    results={}
    for preset in PRESETS:
        p=OUT/(preset+'_eval')/'policy_evaluation.json'
        if p.exists():
            d=json.loads(p.read_text())
            results[preset]={k:{name:v[name] for name in ('mean_vx_m_s','mean_yaw_rad_s','vx_abs_error_m_s',
                'yaw_abs_error_rad_s','fall_resets','upright_fraction','phase_contact_agreement','stance_slip_mean_m_s')}
                for k,v in d.items() if isinstance(v,dict) and 'mean_vx_m_s' in v}
    (OUT/'comparison.json').write_text(json.dumps(results,indent=2))
    return results


def main():
    os.environ.update(WANDB_MODE='disabled',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',PYTHONUNBUFFERED='1')
    # The first matched control was started explicitly; wait only for its file.
    first=ROOT/'logs/as2piper-low/gait_ablation_v1_original/model_433.pt'
    start=time.monotonic()
    while not first.exists():
        if time.monotonic()-start>1800:raise TimeoutError('Matched control did not finish')
        time.sleep(5)
    for preset in PRESETS:
        run='gait_ablation_v1_'+preset
        checkpoint=ROOT/'logs/as2piper-low'/run/'model_433.pt'
        if not checkpoint.exists():
            cmd=[sys.executable,'-u','train_gait.py','--preset',preset,'--controlled','--num-envs','1024',
                 '--iterations','120','--checkpoint',str(OUT/'shared_start_313.pt'),'--run',run,'--seed','7']
            print('Training',preset,flush=True)
            with (OUT/(preset+'_train.log')).open('w') as log:
                subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        evaluation=OUT/(preset+'_eval')
        if not (evaluation/'policy_evaluation.json').exists():
            print('Evaluating',preset,flush=True)
            cmd=[sys.executable,'-u','evaluate_hw.py',str(checkpoint),'--vbc-port','--gait-preset',preset,
                 '--arm-goals','--vx','.25','--yaw','.6','--output',str(evaluation)]
            with (OUT/(preset+'_eval.log')).open('w') as log:
                subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        results=summarize();print(json.dumps({preset:results[preset]}),flush=True)
    print('Paired study complete',flush=True)


if __name__=='__main__':main()
