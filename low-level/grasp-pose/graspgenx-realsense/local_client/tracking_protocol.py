"""CPU-only frame codec shared by Windows and Tracking SAM service."""
from dataclasses import asdict
import time
import zlib
import cv2
import numpy as np

MAX_PIXELS = 4096*2160


def png_mask(mask):
    ok, data = cv2.imencode('.png', (mask > 0).astype(np.uint8))
    if not ok:
        raise ValueError('Mask encoding failed')
    return data.tobytes()


def read_mask(data, shape):
    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != tuple(shape):
        raise ValueError('Mask/image shape mismatch')
    return (mask > 0).astype(np.uint8)


def frame_packet(frame, session, seq, config, point=None, text=None):
    start = time.perf_counter()
    ok, rgb = cv2.imencode('.jpg', frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError('RGB JPEG encoding failed')
    if frame.depth_raw.dtype == np.uint16:
        depth = np.ascontiguousarray(frame.depth_raw, dtype='<u2')
        encoding, scale = 'u16_zlib', frame.depth_scale
    else:
        depth = np.ascontiguousarray(frame.depth_m, dtype='<f4')
        encoding, scale = 'f32_m_zlib', 1.0
    compressed = zlib.compress(depth.tobytes(), level=1)
    packet = dict(action='frame', session=session, seq=int(seq), frame_id=frame.frame_id,
                  stream_id=frame.frame_id.rsplit('-',1)[0],
                  camera_timestamp_ms=frame.camera_timestamp_ms,
                  host_timestamp_s=frame.host_timestamp_s,
                  rgb_jpeg=rgb.tobytes(), depth_zlib=compressed, depth_encoding=encoding,
                  shape=list(depth.shape), depth_scale=float(scale), K=frame.K,
                  intrinsics=frame.intrinsics, color_intrinsics=frame.color_intrinsics,
                  filters=asdict(config))
    if point is not None:
        packet['point'] = list(point)
    if text:
        packet['text'] = text
    encode_ms=(time.perf_counter()-start)*1000
    return packet, dict(frame_encode_ms=encode_ms, upload_payload_bytes=len(compressed)+len(rgb))


def decode_frame(packet):
    shape=tuple(int(v) for v in packet['shape'])
    if len(shape)!=2 or min(shape)<=0 or np.prod(shape)>MAX_PIXELS:
        raise ValueError('Invalid image dimensions')
    encoding=packet['depth_encoding']
    if encoding not in ('u16_zlib','f32_m_zlib'):
        raise ValueError('Unsupported depth encoding')
    dtype=np.dtype('<u2' if encoding=='u16_zlib' else '<f4')
    size=int(np.prod(shape))*dtype.itemsize
    decoder=zlib.decompressobj()
    raw=decoder.decompress(packet['depth_zlib'],size+1)
    if len(raw)!=size or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError('Invalid lossless depth payload length')
    rgb=cv2.imdecode(np.frombuffer(packet['rgb_jpeg'],np.uint8),cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2]!=shape:
        raise ValueError('RGB/depth pixel dimensions differ')
    scale=float(packet['depth_scale'])
    if not np.isfinite(scale) or not 0<scale<=1:
        raise ValueError('Invalid depth scale')
    if encoding=='f32_m_zlib' and scale!=1:
        raise ValueError('Float meter encoding must have scale 1')
    K=np.asarray(packet['K'],dtype=np.float32)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Invalid calibrated intrinsics')
    if not np.allclose(K[2],[0,0,1]):
        raise ValueError('Invalid intrinsics last row')
    depth=np.frombuffer(raw,dtype=dtype).reshape(shape).astype(np.float32)*scale
    depth[~np.isfinite(depth) | (depth<=0)]=0
    return rgb,depth,K


def filter_tracked_depth(depth_m, tracking_mask, cfg):
    """Apply controls only inside the tracker mask; preserve disconnected parts."""
    if depth_m.shape!=tracking_mask.shape:
        raise ValueError('Mask/depth shape mismatch')
    if not 0<cfg.near_m<cfg.far_m or cfg.foreground_band_m<0:
        raise ValueError('Require 0 < near < far and band >= 0')
    if not 0<=cfg.foreground_percentile<=100:
        raise ValueError('Foreground percentile must be in [0,100]')
    mask=(tracking_mask>0) & np.isfinite(depth_m) & (depth_m>=cfg.near_m) & (depth_m<=cfg.far_m)
    if mask.any() and cfg.foreground_band_m>0:
        baseline=np.percentile(depth_m[mask],cfg.foreground_percentile)
        mask &= depth_m<=baseline+cfg.foreground_band_m
    return mask.astype(np.uint8)


class MaskGuard:
    """Conservative geometric lost-state latch; not a semantic identity score."""
    def __init__(self,min_pixels=100,min_iou=.03,max_area_ratio=4.):
        self.min_pixels=min_pixels
        self.min_iou=min_iou
        self.max_area_ratio=max_area_ratio
        self.reset()

    def reset(self):
        self.previous=None
        self.lost=False
        self.reason='uninitialized'

    def check(self,mask,initial=False):
        current=mask>0
        if initial:
            self.reset()
        if self.lost:
            return False,self.reason
        area=int(current.sum())
        reason=None
        if area<self.min_pixels:
            reason='empty_or_too_small'
        elif area/current.size>.85:
            reason='implausibly_large_mask'
        elif self.previous is not None:
            ratio=area/max(int(self.previous.sum()),1)
            iou=np.logical_and(current,self.previous).sum()/max(np.logical_or(current,self.previous).sum(),1)
            if not 1/self.max_area_ratio<=ratio<=self.max_area_ratio:
                reason='mask_area_jump'
            elif iou<self.min_iou:
                reason='temporal_overlap_lost'
        if reason:
            self.lost=True
            self.reason=reason
            return False,reason
        self.previous=current.copy()
        self.reason='tracking'
        return True,'tracking'
