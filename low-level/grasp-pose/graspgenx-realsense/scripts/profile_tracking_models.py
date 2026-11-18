"""Isolated measurement only: no production files, sessions or settings modified."""
import argparse,csv,json,os,re,subprocess,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/model_profile/manual'
OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path[:0]=[str(ROOT/'tracking_service'),str(ROOT),str(ROOT/'local_client'),str(ROOT/'upstream/Tracking_SAM')]


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)))


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def resized_source():
    source=ROOT/'upstream/GraspGenX/assets/sample_data/real_world/00'
    rgb=cv2.cvtColor(cv2.imread(str(source/'rgb.png')),cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    rgb=cv2.resize(rgb,(640,480))
    depth=cv2.resize(np.load(source/'depth.npy').astype(np.float32),(640,480),interpolation=cv2.INTER_NEAREST)
    K=np.array(json.loads((source/'meta_data.json').read_text())['intrinsics'],np.float32)
    K[0]*=640/w;K[1]*=480/h
    return rgb,depth,K


def client_main(args):
    from tracking_client import TrackingWorker
    from pointcloud import FilterConfig
    rgb,depth,K=resized_source()
    intr=dict(width=640,height=480,fx=float(K[0,0]),fy=float(K[1,1]),ppx=float(K[0,2]),ppy=float(K[1,2]),model='distortion.none',coeffs=[0.]*5)
    epoch=time.perf_counter();bgr=rgb[:,:,::-1].copy()
    class Capture:
        def latest(self):
            i=int((time.perf_counter()-epoch)*30)
            return SimpleNamespace(frame_id=f'profile-replay-{i}',bgr=bgr,depth_raw=depth,depth_m=depth,K=K,
                depth_scale=1.,intrinsics=intr,color_intrinsics=intr,unit_rays=None,camera_timestamp_ms=None,
                host_timestamp_s=None,ready_monotonic=epoch+i/30,acquisition_ms=None,alignment_ms=None,conversion_ms=None)
    capture=Capture();rows=[];grasps=[];initialized=False;first_measured=None;last_frame=None
    worker=TrackingWorker(capture,FilterConfig(foreground_band_m=0),port=args.port,fps=args.cap,
                          output=OUT/f'client_{args.label}')
    def observer(s):
        nonlocal first_measured
        if not initialized or not s['response']['grasp_allowed']:return
        now=time.perf_counter()
        row=dict(frame_id=s['frame'].frame_id,arrival=now,tracking_rtt_ms=s['tracking_rtt_ms'],
                 frame_encode_ms=s['frame_encode_ms'],service_ms=s['response']['service_ms'],
                 sam_init_ms=s['response']['metrics']['sam_init_ms'],vos_ms=s['response']['metrics']['vos_ms'])
        rows.append(row)
    worker.frame_observer=observer;worker.start()
    deadline=time.perf_counter()+180
    try:
        while time.perf_counter()<deadline:
            latest,status,busy=worker.state()
            if not initialized and latest is not None:
                worker.select(capture.latest(),point=(310,300));initialized=True
            done=worker.take_result()
            if done is not None:grasps.append(dict(infer_ms=done['result']['server_timing']['infer_ms'],**done['result'].get('gpu',{})))
            if initialized and len(rows)>=111:break
            if args.concurrent and latest is not None and latest['response']['grasp_allowed'] and not busy and latest['frame'].frame_id!=last_frame:
                try:worker.request_grasp();last_frame=latest['frame'].frame_id
                except ValueError:pass
            if 'error' in status.lower():print('client status',status,flush=True)
            time.sleep(.005)
        else:raise TimeoutError('Isolated replay worker timed out')
    finally:worker.close()
    measured=rows[10:111]
    intervals=np.diff([r['arrival'] for r in measured])
    result=dict(source='static real recorded RGB-D resized 640x480; localhost; no USB/SSH/renderer',
        frames=100,observations=len(measured),warmup_frames=10,upload_cap=args.cap,
        fps=100/(measured[-1]['arrival']-measured[0]['arrival']),interval_ms=stats(intervals*1000),
        timing={k:stats([r[k] for r in measured if r[k] is not None]) for k in ('tracking_rtt_ms','frame_encode_ms','service_ms','vos_ms')},
        grasp_results=grasps,torch_imported='torch' in sys.modules)
    write_csv(OUT/f'stream_{args.label}.csv',measured)
    (OUT/f'stream_{args.label}.json').write_text(json.dumps(result,indent=2))
    print('CLIENT DONE',args.label,result['fps'],flush=True)


def model_main(args):
    import torch
    import torch.nn.functional as F
    import zmq,msgpack,msgpack_numpy
    from tracking_service.serve import TrackingService
    torch.set_num_threads(4)
    # Only this measurement process is capped; existing services are untouched.
    torch.cuda.set_per_process_memory_fraction(.35)
    report=dict(environment=dict(torch=torch.__version__,cuda_runtime=torch.version.cuda,python=sys.version,
        gpu=torch.cuda.get_device_name(),total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
        sm=torch.cuda.get_device_properties(0).multi_processor_count,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark),warmup=10,measured_frames=100,input_hw=[480,640])
    phase=['load'];samples=[];stop=threading.Event()
    def mem():
        return dict(allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    def poll():
        while not stop.is_set():
            raw=subprocess.check_output(['nvidia-smi'],text=True)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB',raw)
            if m:samples.append(dict(time=time.time(),phase=phase[0],used_mib=int(m[1]),total_mib=int(m[2])))
            stop.wait(.2)
    thread=threading.Thread(target=poll,daemon=True);thread.start()
    def checkpoint():
        (OUT/'profile.json').write_text(json.dumps(report,indent=2))
        write_csv(OUT/'gpu_samples.csv',samples)
    def cuda_stage(fn):
        torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();begin.record();value=fn();end.record();end.synchronize()
        return value,begin.elapsed_time(end),(time.perf_counter()-start)*1000
    def wall_stage(fn):
        start=time.perf_counter();value=fn();return value,(time.perf_counter()-start)*1000
    def gpu(row,name,fn):
        value,event,wall=cuda_stage(fn);row[name+'_gpu_ms']=event;row[name+'_wall_ms']=wall;return value
    try:
        print('Loading isolated official TrackingService replica',flush=True)
        service=TrackingService()
        service.out=OUT/'isolated_service_logs';service.out.mkdir(exist_ok=True)
        tracker=service.backend.tracker;sam=tracker.sam;vos=tracker.vos_tracker
        torch.cuda.synchronize();report['loaded']=mem()
        models={'SAM':sam,'AOTT':vos.model,'GroundingDINO':tracker.dino_model}
        report['models']={}
        checkpoints={'SAM':'sam_vit_h_4b8939.pth','AOTT':'AOTT_PRE_YTB_DAV.pth','GroundingDINO':'groundingdino_swint_ogc.pth'}
        for name,model in models.items():
            n=sum(p.numel() for p in model.parameters())
            file=ROOT/'upstream/Tracking_SAM/pretrained_weights'/checkpoints[name]
            dtype={str(t.dtype) for t in model.parameters()}
            report['models'][name]=dict(parameters=n,parameters_m=n/1e6,checkpoint=file.name,checkpoint_bytes=file.stat().st_size,
                checkpoint_MB=file.stat().st_size/1e6,fp32_MiB=n*4/2**20,fp16_MiB=n*2/2**20,dtypes=sorted(dtype),
                buffer_MiB=sum(b.numel()*b.element_size() for b in model.buffers())/2**20)
        report['model_config']={k:v for k,v in vars(vos.cfg).items() if k.startswith('MODEL_') or k.startswith('TEST_')}
        report['sam_config']=dict(backbone='ViT-H',depth=len(sam.image_encoder.blocks),embedding=1280,heads=16,
            image_size=sam.image_encoder.img_size,patch_size=16,window_size=14,global_attn=[7,15,23,31],prompt_dim=256,
            multimask_output=False,precision='FP32; no autocast')
        checkpoint()
        rgb,depth,K=resized_source();predictor=tracker.sam_predictor
        def sam_one(image,point=(310,300)):
            torch.cuda.synchronize();row={};start=time.perf_counter()
            resized,row['resize_cpu_ms']=wall_stage(lambda:predictor.transform.apply_image(image))
            tensor=gpu(row,'h2d',lambda:torch.as_tensor(resized,device='cuda').permute(2,0,1).contiguous()[None])
            pre=gpu(row,'normalize_pad',lambda:sam.preprocess(tensor))
            features=gpu(row,'encoder',lambda:sam.image_encoder(pre))
            coords,row['prompt_cpu_ms']=wall_stage(lambda:predictor.transform.apply_coords(np.array([point],np.float32),image.shape[:2]))
            coord,label=gpu(row,'prompt_h2d',lambda:(torch.as_tensor(coords,dtype=torch.float,device='cuda')[None],
                                                                   torch.ones((1,1),dtype=torch.int,device='cuda')))
            sparse,dense=gpu(row,'prompt_encoder',lambda:sam.prompt_encoder(points=(coord,label),boxes=None,masks=None))
            low,iou=gpu(row,'decoder',lambda:sam.mask_decoder(image_embeddings=features,image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False))
            masks=gpu(row,'postprocess',lambda:sam.postprocess_masks(low,tuple(resized.shape[:2]),image.shape[:2])>sam.mask_threshold)
            torch.cuda.synchronize()
            arrays,row['d2h_cpu_ms']=wall_stage(lambda:(masks[0].cpu().numpy(),iou[0].cpu().numpy(),low[0].cpu().numpy()))
            row['total_ms']=(time.perf_counter()-start)*1000
            row['model_gpu_ms']=sum(row[k+'_gpu_ms'] for k in ('encoder','prompt_encoder','decoder'))
            return arrays[0][0].astype(np.uint8),row
        def aot_one(image):
            torch.cuda.synchronize();row={};start=time.perf_counter()
            sample=dict(current_img=image,meta=dict(obj_num=vos.obj_num,height=image.shape[0],width=image.shape[1],flip=False))
            data,row['preprocess_cpu_ms']=wall_stage(lambda:vos.transform(sample)[0])
            tensor=gpu(row,'h2d_cast',lambda:data['current_img'].cuda(vos.gpu_id,non_blocking=True).float().unsqueeze(0))
            gpu(row,'match',lambda:vos.engine.match_propogate_one_frame(tensor))
            logits=gpu(row,'decode',lambda:vos.engine.decode_current_logits(image.shape[:2]))
            def post():
                label=torch.argmax(torch.softmax(logits,dim=1),dim=1,keepdim=True).float()
                return label,F.interpolate(label,size=vos.engine.input_size_2d,mode='nearest')
            label,small=gpu(row,'postprocess',post)
            gpu(row,'memory_update',lambda:vos.engine.update_memory(small))
            torch.cuda.synchronize()
            mask,row['d2h_cpu_ms']=wall_stage(lambda:label.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8))
            vos.frame_cnt+=1
            row['total_ms']=(time.perf_counter()-start)*1000
            row['model_gpu_ms']=sum(row[k+'_gpu_ms'] for k in ('match','decode','memory_update'))
            row['internal_h']=tensor.shape[-2];row['internal_w']=tensor.shape[-1]
            return mask,row
        def summarize(rows):
            return {k:stats([r[k] for r in rows]) for k in rows[0] if k.endswith('_ms')}
        with torch.inference_mode():
            print('SAM warmup + 100 initial segmentations',flush=True);phase[0]='SAM'
            samrows=[]
            for i in range(110):
                if i==10:torch.cuda.reset_peak_memory_stats()
                mask,row=sam_one(rgb)
                assert mask.any()
                torch.cuda.synchronize();start=time.perf_counter()
                vos.reset_engine();vos.add_reference_frame(rgb,mask);torch.cuda.synchronize()
                row['aot_reference_ms']=(time.perf_counter()-start)*1000
                row['sam_plus_reference_ms']=row['total_ms']+row['aot_reference_ms']
                if i>=10:samrows.append(row)
                if i%20==0:print('SAM',i,'ms',row['total_ms'],flush=True)
            report['sam_latency']=summarize(samrows);report['sam_memory']=mem();write_csv(OUT/'sam.csv',samrows);checkpoint()
            # Genuine moving DAVIS frames, forward then reverse; no abrupt end-to-start jump.
            files=sorted((ROOT/'upstream/Tracking_SAM/sample_data/DAVIS_bear/images').glob('*.jpg'))
            frames=[cv2.cvtColor(cv2.resize(cv2.imread(str(p)),(640,480)),cv2.COLOR_BGR2RGB) for p in files]
            annotation=cv2.imread(str(ROOT/'upstream/Tracking_SAM/sample_data/DAVIS_bear/init_mask/00000.png'),0)>0
            distance=cv2.distanceTransform(annotation.astype(np.uint8),cv2.DIST_L2,5)
            y,x=np.unravel_index(distance.argmax(),distance.shape)
            click=(x*640/annotation.shape[1],y*480/annotation.shape[0])
            ref,_=sam_one(frames[0],click)
            vos.reset_engine();vos.add_reference_frame(frames[0],ref)
            order=list(range(len(frames)))+list(range(len(frames)-2,0,-1))
            print('AOTT warmup + 100 moving-frame samples',flush=True);phase[0]='AOTT';aotrows=[]
            torch.cuda.empty_cache()
            for i in range(1,111):
                if i==11:torch.cuda.reset_peak_memory_stats()
                mask,row=aot_one(frames[order[i%len(order)]])
                if i>=11:aotrows.append(dict(row,source_frame=files[order[i%len(order)]].name))
            report['aot_latency']=summarize(aotrows);report['aot_memory']=mem()
            report['aot_input']=[aotrows[0]['internal_h'],aotrows[0]['internal_w']]
            report['aot_data']='82 genuine DAVIS bear frames forward/reverse; 10 warmup + 100 measured; no GT mask injection'
            write_csv(OUT/'aot.csv',aotrows);checkpoint()
            print('FLOPs operator profiler (one representative frame, not latency samples)',flush=True);phase[0]='FLOPs'
            for name,fn in [('SAM',lambda:sam_one(rgb)),('AOTT',lambda:aot_one(frames[0]))]:
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True,with_flops=True) as prof:
                        fn()
                    ops=[dict(op=e.key,count=e.count,flops=e.flops or 0) for e in prof.key_averages()]
                    flops=sum(r['flops'] for r in ops)
                    dense=sum(r['flops'] for r in ops if r['op'] in ('aten::conv2d','aten::mm','aten::bmm','aten::addmm'))
                    report.setdefault('flops',{})[name]=dict(counted_GFLOPs=flops/1e9,dense_GMACs=dense/2e9,
                        note='Supported operator estimate; custom correlation, softmax, interpolation and other uncounted operators excluded. Not complete FLOPs.')
                    write_csv(OUT/f'flops_{name}.csv',ops)
                except Exception as exc:report.setdefault('flops',{})[name]=dict(error=str(exc))
            checkpoint()
        # Exact official TrackingWorker protocol, isolated localhost service and session.
        socket=zmq.Context.instance().socket(zmq.REP);socket.setsockopt(zmq.LINGER,0)
        socket.bind('tcp://127.0.0.1:*');port=int(socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(':',1)[1])
        print('Isolated replay socket',port,flush=True)
        for label,cap,concurrent in [('cap10',10,False),('uncapped30',30,False),('concurrent',30,True)]:
            phase[0]='stream_'+label;torch.cuda.reset_peak_memory_stats();peak_a=peak_r=0
            argv=[str(ROOT/'.venv-client/bin/python'),str(Path(__file__).resolve()),'--client','--port',str(port),'--cap',str(cap),'--label',label]
            if concurrent:argv.append('--concurrent')
            with (OUT/f'client_{label}.log').open('w') as log:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+180
            while child.poll() is None and time.monotonic()<deadline:
                if not socket.poll(200):continue
                packet=msgpack.unpackb(socket.recv(),raw=False,object_hook=msgpack_numpy.decode)
                start=time.perf_counter()
                try:result=service.dispatch(packet);result['service_ms']=(time.perf_counter()-start)*1000
                except Exception as exc:result=dict(error=f'{type(exc).__name__}: {exc}')
                socket.send(msgpack.packb(result,default=msgpack_numpy.encode,use_bin_type=True))
                peak_a=max(peak_a,torch.cuda.max_memory_allocated()/2**20)
                peak_r=max(peak_r,torch.cuda.max_memory_reserved()/2**20)
            if child.poll() is None:child.terminate();child.wait(timeout=10);raise TimeoutError('Replay child timed out')
            if child.returncode:raise RuntimeError(f'Child failed {label}; see log')
            report.setdefault('stream',{})[label]=json.loads((OUT/f'stream_{label}.json').read_text())
            report['stream'][label]['process_memory']=dict(mem(),observed_peak_allocated_mib=peak_a,observed_peak_reserved_mib=peak_r)
            print('STREAM',label,report['stream'][label]['fps'],flush=True);checkpoint()
        socket.close();service.pool.shutdown(wait=True)
    finally:
        stop.set();thread.join(timeout=3)
        for p in sorted({r['phase'] for r in samples}):
            vals=[r['used_mib'] for r in samples if r['phase']==p]
            report.setdefault('nvidia_smi',{})[p]=dict(min_mib=min(vals),peak_observed_mib=max(vals),samples=len(vals),nominal_sampling_interval_s=.2)
        checkpoint()
        print('REPORT',OUT/'profile.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--client',action='store_true');p.add_argument('--port',type=int)
    p.add_argument('--cap',type=float,default=30);p.add_argument('--label',default='test');p.add_argument('--concurrent',action='store_true')
    args=p.parse_args()
    if args.client:client_main(args)
    else:model_main(args)
