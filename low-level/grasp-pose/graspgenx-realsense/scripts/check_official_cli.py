"""Launch the exact upstream CLI on a separate loopback port, verify, stop it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'upstream/GraspGenX'
sys.path.insert(0,str(ROOT/'local_client'))
from grasp_client import GraspClient

if __name__=='__main__':
    out=ROOT/'outputs/benchmarks'
    command=[sys.executable,str(UP/'client-server/graspgenx_server.py'),
             '--config',str(UP/'ext/graspgenx_checkpoints/release'),
             '--assets_dir',str(UP/'ext/gripper_descriptions/gripper_descriptions/assets'),
             '--default_gripper','franka_umi','--host','127.0.0.1','--port','5557']
    with (out/'official_cli.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        client=GraspClient(port=5557,timeout_ms=500)
        try:
            deadline=time.monotonic()+120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Official CLI exited {process.returncode}; see official_cli.log')
                try:
                    health=client.health()
                    break
                except TimeoutError:
                    if time.monotonic()>deadline:
                        raise TimeoutError('Official server did not start within 120s')
                    time.sleep(.2)
            binding=subprocess.check_output(['ss','-ltnp','( sport = :5557 )'],text=True)
            assert '127.0.0.1:5557' in binding
            client.close()
            client=GraspClient(port=5557)
            pc_file=sorted((UP/'assets/sample_data/object_pc').glob('*.json'))[0]
            pc=np.asarray(json.loads(pc_file.read_text())['pc'],np.float32)
            result=client.infer(pc,api='name',planner='diffusion')
            assert len(result['grasps'])==100
            report=dict(command=command,health=health,metadata=client.metadata(),binding=binding,
                        returned_grasps=len(result['grasps']),server_timing=result['server_timing'],
                        network_rtt_ms=result['network_rtt_ms'])
            (out/'official_cli.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
