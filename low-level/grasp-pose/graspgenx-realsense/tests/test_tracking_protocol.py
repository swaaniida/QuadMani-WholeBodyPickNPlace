from dataclasses import replace
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
import numpy as np
import pytest
from pointcloud import FilterConfig
from tracking_protocol import frame_packet,decode_frame,read_mask,png_mask,filter_tracked_depth,MaskGuard


def source_frame():
    raw=np.arange(20*30,dtype=np.uint16).reshape(20,30)+500
    return SimpleNamespace(bgr=np.full((20,30,3),180,np.uint8),depth_raw=raw,
        depth_m=raw.astype(np.float32)*.001,depth_scale=.001,
        K=np.array([[100,0,15],[0,100,10],[0,0,1]],np.float32),frame_id='test-stream-1',
        camera_timestamp_ms=12,host_timestamp_s=30,intrinsics={},color_intrinsics={})


def test_lossless_depth_scale_and_same_pixel_geometry():
    f=source_frame()
    p,_=frame_packet(f,'test_session',1,FilterConfig())
    bgr,depth,K=decode_frame(p)
    assert bgr.shape==f.bgr.shape
    np.testing.assert_array_equal(depth,f.depth_m)
    np.testing.assert_array_equal(K,f.K)
    assert p['stream_id']=='test-stream'


def test_float_dataset_depth_is_lossless_without_fake_sensor_scale():
    f=source_frame();f.depth_raw=f.depth_m.copy()
    p,_=frame_packet(f,'test',2,FilterConfig())
    assert p['depth_encoding']=='f32_m_zlib' and p['depth_scale']==1
    np.testing.assert_array_equal(decode_frame(p)[1],f.depth_m)


def test_reject_mismatched_payload_shape_and_bad_K():
    p,_=frame_packet(source_frame(),'test',1,FilterConfig())
    p['shape']=[10,10]
    with pytest.raises(ValueError):decode_frame(p)
    p,_=frame_packet(source_frame(),'test',1,FilterConfig())
    p['K'][0,0]=0
    with pytest.raises(ValueError):decode_frame(p)


def test_mask_png_round_trip_and_shape_guard():
    m=np.zeros((20,30),np.uint8);m[2:12,4:15]=1
    np.testing.assert_array_equal(read_mask(png_mask(m),m.shape),m)
    with pytest.raises(ValueError):read_mask(png_mask(m),(30,20))


def test_depth_filter_never_includes_background_or_discards_disconnected_mask():
    d=np.ones((20,30),np.float32)
    mask=np.zeros_like(d,np.uint8);mask[1:5,1:5]=1;mask[12:17,20:25]=1
    cfg=FilterConfig(foreground_band_m=0)
    np.testing.assert_array_equal(filter_tracked_depth(d,mask,cfg),mask)
    d[2,2]=0
    filtered=filter_tracked_depth(d,mask,cfg)
    assert filtered.sum()==mask.sum()-1


def test_guard_latches_loss_until_explicit_reinitialization():
    guard=MaskGuard(min_pixels=10)
    mask=np.zeros((50,60),np.uint8);mask[5:15,5:15]=1
    assert guard.check(mask,initial=True)[0]
    assert not guard.check(np.zeros_like(mask))[0]
    assert not guard.check(mask)[0]
    assert guard.check(mask,initial=True)[0]
    shifted=np.roll(mask,30,axis=1)
    assert not guard.check(shifted)[0]
