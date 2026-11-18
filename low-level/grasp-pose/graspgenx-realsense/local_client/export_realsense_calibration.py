"""Read actual D435i intrinsics/depth-to-color extrinsics on the Windows camera."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from realsense_capture import intrinsics_dict


def extrinsic_matrix(rs,e):
    # librealsense stores extrinsics.rotation in column-major order.
    T=np.eye(4);T[:3,:3]=np.asarray(e.rotation).reshape(3,3,order='F');T[:3,3]=e.translation
    for p in ([0,0,0],[1,2,3],[.1,-.2,.7]):
        np.testing.assert_allclose(T[:3,:3]@p+T[:3,3],rs.rs2_transform_point_to_point(e,p),atol=1e-6)
    return T


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--serial')
    p.add_argument('--output',type=Path,default=Path('realsense_calibration.json'))
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480);p.add_argument('--fps',type=int,default=30)
    a=p.parse_args()
    import pyrealsense2 as rs
    pipeline=rs.pipeline();config=rs.config()
    if a.serial:config.enable_device(a.serial)
    config.enable_stream(rs.stream.depth,a.width,a.height,rs.format.z16,a.fps)
    config.enable_stream(rs.stream.color,a.width,a.height,rs.format.bgr8,a.fps)
    profile=pipeline.start(config)
    try:
        depth=profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color=profile.get_stream(rs.stream.color).as_video_stream_profile()
        device=profile.get_device();align=rs.align(rs.stream.color)
        for _ in range(10):frames=pipeline.wait_for_frames(5000)
        aligned=align.process(frames);intr=aligned.get_depth_frame().profile.as_video_stream_profile().intrinsics
        T=extrinsic_matrix(rs,depth.get_extrinsics_to(color))
        ci=color.intrinsics;K=np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1.]])
        data=dict(captured_utc=datetime.now(timezone.utc).isoformat(),serial=device.get_info(rs.camera_info.serial_number),
            camera_model=device.get_info(rs.camera_info.name),width=a.width,height=a.height,fps=a.fps,
            color_intrinsics=intrinsics_dict(ci),depth_native_intrinsics=intrinsics_dict(depth.intrinsics),
            aligned_depth_intrinsics=intrinsics_dict(intr),K_color=K.tolist(),
            depth_scale=device.first_depth_sensor().get_depth_scale(),
            T_color_optical_depth_optical=T.tolist(),T_depth_optical_color_optical=np.linalg.inv(T).tolist(),
            note='SDK device calibration; does not measure hand-eye or physical TCP offset')
        a.output.write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
        from mounted_wrist_rig import MountedWristRig,calibrated_urdf,DEFAULT_RIG
        rig=MountedWristRig(calibration=a.output)
        transforms=dict(T_color_optical_TCP=rig.T_camera_tcp.tolist(),T_TCP_color_optical=np.linalg.inv(rig.T_camera_tcp).tolist(),
            status='SDK depth/color extrinsics + nominal CAD housing/TCP; not hand-eye calibrated',camera_calibration=str(a.output))
        a.output.with_name('camera_tcp_transform.json').write_text(json.dumps(transforms,indent=2))
        print('URDF with device depth/color extrinsics:',calibrated_urdf(DEFAULT_RIG,data))
        print('Camera -> nominal UMI TCP (m):',rig.T_camera_tcp[:3,3])
    finally:pipeline.stop()


if __name__=='__main__':main()
