# Tracking SAM model and performance profile

2026-11-12, astrago. Existing service PIDs 1229001 (GraspGenX) / 1326898 (Tracking SAM) remained alive and unchanged. Source SHA256 checks passed. No installation, service restart, checkpoint modification or production setting changes.

## Measurement conditions

- Actual hardware: NVIDIA A100-SXM4-80GB, MIG **3g.40gb**, **40,192 MiB**, **42 SM**; this is not 3g.20gb.
- Driver 535.161.08; driver-reported CUDA support 12.2; actual Tracking PyTorch **1.13.1+cu117**, CUDA runtime **11.7**, Python 3.10.22. Existing GraspGenX environment: PyTorch 2.5.1+cu121 / CUDA runtime 12.1 (read from its installed version.py).
- Batch 1, FP32 weights, eval/inference mode, no autocast; matmul TF32 disabled. Existing model and inference source untouched; isolated measurement process uses official components. The measurement process alone has a 35% allocator limit; measured peaks stayed below that limit.
- Input RGB **640x480**. SAM: long-edge 1024, padded encoder input **1024x1024**. AOTT official align_corners/stride transform produces **481x641**.
- SAM: 10 warmups + 100 full image-encoder/prompt/decoder runs, recomputing the embedding on an actual recorded real_world/00 RGB-D frame. No cached-embedding shortcut. Repeated image, not 100 unique captures.
- AOTT: 10 warmups + 100 tracking steps on genuine DAVIS bear frames (82-frame source, forward then reverse). The initial click is selected inside the dataset annotation; only the actual SAM output initializes AOTT. Ground truth is not used as a predicted/reference mask.
- GPU stages use CUDA events and synchronization. CPU preprocessing, H2D/casts, GPU postprocessing and D2H are separate. Segmented wrapper totals include instrumentation/launch/synchronization overhead.
- Streaming: actual CPU-only TrackingWorker / JPEG90 + lossless depth + msgpack/ZMQ / official TrackingService, isolated ephemeral localhost port. Each condition uses 10 warmup observations then 100 frame intervals (101 observations). Synthetic replay timestamps/IDs label repeats of real data; no fake masks or grasp poses.
- The original Tracking service is not used/reset for the experiment. Co-run GraspGenX requests use its existing localhost server. All production services remain resident; the extra profiling replica affects whole-MIG memory accounting.

## Models, parameters and theoretical weight memory

Checkpoint MB = bytes / 1,000,000; GPU MiB = bytes / 1,048,576. Parameter counts exclude buffers and do not double-count modules shared with the predictor/engine. FP16 values are theoretical, not an FP16 runtime experiment.

| Model | Parameters M | Checkpoint | File MB | FP32 weights MiB | FP16 weights MiB |
|---|---:|---|---:|---:|---:|
| SAM | 641.091 | `sam_vit_h_4b8939.pth` | 2564.55 | 2445.57 | 1222.78 |
| AOTT | 5.734 | `AOTT_PRE_YTB_DAV.pth` | 23.33 | 21.87 | 10.94 |
| GroundingDINO | 172.840 | `groundingdino_swint_ogc.pth` | 694.00 | 659.33 | 329.67 |
| SAM+AOTT | 646.825 | total | 2587.88 | 2467.44 | 1233.72 |
| Full resident model incl. DINO | 819.664 | total | 3281.88 | 3126.77 | 1563.39 |

**SAM:** ViT-H, 32 blocks, embedding 1280, 16 heads, patch16, window14, global attention blocks 7/15/23/31, 256-d prompt embedding, two-way mask transformer depth2, single positive click, multimask_output=False.
**AOTT:** AOT-Tiny, MobileNetV2 [24,32,96,1280] feature channels, 256-d embedding, 1 LSTT layer, 8 self/8 attention heads, FPN decoder, max10 objects (tested1), align_corners=True, TEST_MULTISCALE=[1], no flip, max long side1040, long-term memory gap9999, short-term skip1. Pretraining stage pre_ytb_dav.
**GroundingDINO:** Swin-T + BERT-base-uncased. Loaded by the existing main_tracker even for click mode; no DINO inference is timed here. Counts include its frozen text encoder.

## Latency, 100 measured samples after warmup

| Stage | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| SAM GPU model only | 1442.52 | 1421.42 | 1527.26 |
| SAM including pre/post/transfers | 1461.63 | 1442.97 | 1551.59 |
| AOTT reference initialization | 31.34 | 28.67 | 34.60 |
| SAM + AOTT reference | 1492.97 | 1471.31 | 1579.17 |
| AOTT GPU model only | 16.70 | 16.26 | 21.73 |
| AOTT including pre/post/transfers | 28.96 | 30.21 | 34.85 |

SAM GPU model = image encoder + prompt encoder + mask decoder. AOTT GPU model = matching/encoder/LSTT + decoder + learned memory update; excludes CPU normalization, H2D/cast, external label postprocessing, D2H. Decoder includes its official interpolation.

| Additional stage | SAM mean ms | AOTT mean ms |
|---|---:|---:|
| CPU preprocessing | 8.84 | 6.03 |
| Image H2D/cast wall | 2.27 | 1.49 |
| GPU postprocess event | 0.08 | 0.14 |
| D2H and CPU output conversion | 0.24 | 0.39 |

SAM prompt H2D is separately reported in raw JSON (not part of the image H2D row). Stage P95 values should not be added together.

## Tracking throughput

| Condition | FPS |
|---|---:|
| AOTT GPU model only, reciprocal mean event time | 59.86 |
| AOTT full model wrapper, reciprocal mean time | 34.54 |
| RGB-D replay, existing default upload cap10 | 9.33 |
| RGB-D replay, upload cap30 | 19.26 |
| RGB-D replay cap30 + actual GraspGenX concurrent inference | 17.90 |

Concurrent GraspGenX: 7 completed real inferences, mean 643.47 ms. This short co-run does not claim 100 grasp requests; the 100-frame requirement is met for tracking. Current Windows USB/SSH/renderer throughput is **not measurable remotely** and is not represented by these localhost replay numbers.

## GPU memory

PyTorch numbers below are for the isolated Tracking replica with SAM+AOTT+DINO resident, not a per-model standalone GPU session. Allocated is live allocator bytes; reserved includes the PyTorch cache. Inference peak values use reset_peak_memory_stats after warmup; streaming peak includes initial SAM segmentation.

| Phase | Allocated MiB | Reserved MiB | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|
| After model loading | 3194.37 | 3288.00 | 3194.37 | 3288.00 |
| SAM benchmark | 3225.88 | 6972.00 | 6446.63 | 6972.00 |
| AOTT benchmark | 3230.45 | 3612.00 | 3488.44 | 3612.00 |
| Tracking replica co-run (incl. SAM init) | 3202.62 | 6936.00 | 6419.37 | 6936.00 |

Existing GraspGenX allocator co-run peaks: allocated 784.29 MiB, reserved 970.00 MiB (reported by the actual server). Separate per-process maxima are not necessarily simultaneous and should not be summed as an exact joint peak.

- nvidia-smi whole-MIG baseline before the measurement: **18,899 MiB**. Whole-MIG sampled maximum during this experiment: **26,494 MiB**. During the concurrency phase: **26,458 MiB**. Post-measurement snapshot after the profiling process exited: **19,019 MiB**; runtime caches may differ from the initial baseline. Includes both original services, their caches/context, and the extra Tracking profiling replica. These are not clean two-model-only memory requirements.
- nvidia-smi is sampled (nominal .2s plus query time), so it is an observed peak; PyTorch maxima are allocator high-water marks. AOTT nvidia phase may include the SAM→AOTT cache-release transition. Use PyTorch AOTT allocated/reserved above for that phase.
- Exact clean original-two-service-only peak or 3g.20GB measurements would require changing the active workload/allocation, so they were not performed. Existing process-specific Tracking allocator state cannot be read externally without instrumenting its process; its replica was measured instead.

## FLOPs / MACs

| Model | Supported-operator GFLOPs | Conv/matmul GMACs |
|---|---:|---:|
| SAM | 5967.46 | 2981.95 |
| AOTT | 27.77 | 13.88 |

One representative frame at the same external 640x480 resolution. SAM internally runs1024², AOTT481x641. PyTorch profiler with_flops covers supported convolution/mm/bmm/addmm and some elementwise operations. AOTT custom correlation, attention aggregation reductions, softmax, interpolation and other unsupported operations are omitted. Thus these are **partial operator counts / lower-bound estimates, not complete FLOPs**. MACs are derived only from conv/matrix multiply FLOPs/2, not all elementwise FLOPs.

## Artifacts

`profile.json`, `sam.csv`, `aot.csv`, `stream_*.csv/json`, `gpu_samples.csv`, `flops_*.csv`, `profile.log`, `source_before.sha256`, `gpu_before.txt`, `gpu_after.txt`. The measurement script is `profile.py`; production code is unchanged.

Official installed source: Tracking_SAM d5e5e81a361dce4ddbe3264a2b8027b2b43ea3fd, AOT submodule ada8a3cbf0ba6dde563a49e78e56dbbcde01d143. Model construction/configuration is read from tracking_SAM/tracking_SAM.py, tracking_SAM/aott.py and configs/models/default.py; SAM backbone definition is the installed official segment_anything/build_sam.py.
