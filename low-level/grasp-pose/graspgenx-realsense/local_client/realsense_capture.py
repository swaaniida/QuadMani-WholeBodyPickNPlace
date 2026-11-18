"""Latest-frame-only acquisition thread; reconnects without growing a queue."""
from dataclasses import dataclass
import threading
import time
import uuid
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    frame_id: str
    camera_timestamp_ms: float
    host_timestamp_s: float
    ready_monotonic: float
    bgr: np.ndarray
    depth_raw: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    intrinsics: dict
    color_intrinsics: dict
    depth_scale: float
    unit_rays: np.ndarray
    acquisition_ms: float
    alignment_ms: float
    conversion_ms: float
    serial: str


def intrinsics_dict(intr):
    return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                ppx=intr.ppx, ppy=intr.ppy, model=str(intr.model), coeffs=list(intr.coeffs))


def make_unit_rays(rs, intr):
    """Cache calibrated rays at z=1; use SDK deprojection for nonzero distortion."""
    v, u = np.indices((intr.height, intr.width), dtype=np.float32)
    if intr.model == rs.distortion.none or not np.any(intr.coeffs):
        return np.stack(((u-intr.ppx)/intr.fx, (v-intr.ppy)/intr.fy, np.ones_like(u)), -1)
    if intr.model == rs.distortion.modified_brown_conrady:
        raise RuntimeError('SDK deprojection does not support modified_brown_conrady; use a rectified color stream')
    rays = np.empty((intr.height, intr.width, 3), dtype=np.float32)
    for y in range(intr.height):
        for x in range(intr.width):
            rays[y, x] = rs.rs2_deproject_pixel_to_point(intr, [x, y], 1.0)
    return rays


class RealSenseCapture:
    def __init__(self, serial=None):
        self.serial = serial
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self.status = 'Looking for RealSense...'
        self.thread = threading.Thread(target=self._run, name='realsense', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._latest

    def close(self):
        self._stop.set()
        self.thread.join(timeout=8)

    def _open(self, rs):
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError('No USB RealSense device detected')
        for device in devices:
            print('RealSense:', device.get_info(rs.camera_info.name),
                  'serial=', device.get_info(rs.camera_info.serial_number), flush=True)
        matching = [d for d in devices if self.serial is None or
                    d.get_info(rs.camera_info.serial_number) == self.serial]
        if not matching:
            raise RuntimeError(f'RealSense serial {self.serial} not found')
        device = matching[0]
        serial = device.get_info(rs.camera_info.serial_number)
        modes = {rs.stream.depth: set(), rs.stream.color: set()}
        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                if profile.is_video_stream_profile() and profile.stream_type() in modes:
                    v = profile.as_video_stream_profile()
                    if profile.stream_type() == rs.stream.depth and profile.format() != rs.format.z16:
                        continue
                    modes[profile.stream_type()].add((v.width(), v.height(), v.fps()))
        rank = lambda m: (m != (640, 480, 30), abs(m[2]-30), abs(m[0]*m[1]-640*480))
        depths = sorted(modes[rs.stream.depth], key=rank)
        colors = sorted(modes[rs.stream.color], key=rank)
        print('Supported depth modes:', depths, '\nSupported color modes:', colors, flush=True)
        if not depths or not colors:
            raise RuntimeError('Device must provide both depth and RGB streams')
        # Prefer matching modes, then a bounded set of available combinations.
        shared = sorted(set(depths) & set(colors), key=rank)
        pairs = [((640, 480, 30), (640, 480, 30))]
        pairs += [(m, m) for m in shared]
        pairs += [(d, c) for d in depths[:5] for c in colors[:5]]
        seen, errors = set(), []
        for d, c in pairs:
            if (d, c) in seen:
                continue
            seen.add((d, c))
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, d[0], d[1], rs.format.z16, d[2])
            config.enable_stream(rs.stream.color, c[0], c[1], rs.format.bgr8, c[2])
            try:
                profile = pipeline.start(config)
                scale = profile.get_device().first_depth_sensor().get_depth_scale()
                print(f'Active depth={d}, color={c}, depth_scale={scale} m/unit', flush=True)
                return pipeline, rs.align(rs.stream.color), scale, serial
            except RuntimeError as exc:
                errors.append(str(exc))
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
        raise RuntimeError('No supported simultaneous RGB-D mode: ' + '; '.join(errors[-3:]))

    def _run(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            self.status = f'RealSense SDK import failed: {exc}'
            return
        while not self._stop.is_set():
            pipeline = None
            try:
                pipeline, align, scale, serial = self._open(rs)
                session = uuid.uuid4().hex[:10]
                ray_key, rays = None, None
                while not self._stop.is_set():
                    t0 = time.perf_counter()
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    t1 = time.perf_counter()
                    aligned = align.process(frames)
                    t2 = time.perf_counter()
                    depth, color = aligned.get_depth_frame(), aligned.get_color_frame()
                    if not depth or not color:
                        continue
                    intr = depth.profile.as_video_stream_profile().intrinsics
                    ci = color.profile.as_video_stream_profile().intrinsics
                    key = repr(intrinsics_dict(intr))
                    if key != ray_key:
                        self.status = 'Preparing calibrated pixel rays...'
                        rays = make_unit_rays(rs, intr)
                        ray_key = key
                        print('Aligned depth intrinsics:', intrinsics_dict(intr), flush=True)
                        print('Color intrinsics:', intrinsics_dict(ci), flush=True)
                    raw = np.asanyarray(depth.get_data()).copy()
                    bgr = np.asanyarray(color.get_data()).copy()
                    if raw.shape != bgr.shape[:2]:
                        raise RuntimeError('Aligned depth/color dimensions differ')
                    depth_m = raw.astype(np.float32) * scale
                    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float32)
                    ready = time.perf_counter()
                    frame = RGBDFrame(
                        f'{serial}-{session}-{color.get_frame_number()}', color.get_timestamp(),
                        time.time(), ready, bgr, raw, depth_m, K, intrinsics_dict(intr),
                        intrinsics_dict(ci), scale, rays, (t1-t0)*1000, (t2-t1)*1000,
                        (ready-t2)*1000, serial)
                    with self._lock:
                        self._latest = frame
                    self.status = f'Connected {serial}'
            except Exception as exc:
                self.status = f'Camera disconnected/error: {exc}; retrying...'
                print(self.status, flush=True)
            finally:
                with self._lock:
                    self._latest = None
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except RuntimeError:
                        pass
            self._stop.wait(2)
