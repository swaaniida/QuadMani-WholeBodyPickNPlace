"""Sample visible MIG memory while the actual replay networking worker runs."""
import csv
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    out=ROOT/'outputs/tracking'
    samples=[]
    with (out/'worker-gpu.log').open('w') as log:
        process=subprocess.Popen([str(ROOT/'.venv-client/bin/python'),str(ROOT/'scripts/tracking_worker_smoke.py')],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while process.poll() is None:
            raw=subprocess.check_output(['nvidia-smi'],text=True)
            match=re.search(r'(\d+)MiB\s*/\s*40192MiB',raw)
            if match:samples.append(dict(elapsed_s=time.monotonic()-start,mig_used_mib=int(match.group(1))))
            time.sleep(.2)
        assert process.returncode==0,process.returncode
    assert samples
    with (out/'gpu_samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
    result=dict(peak_observed_mig_mib=max(r['mig_used_mib'] for r in samples),
                total_mig_mib=40192,sampling_interval_s=.2,samples=len(samples),
                workload='Both real models resident; actual static RGB-D replay tracking and grasp RPC')
    (out/'gpu_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
