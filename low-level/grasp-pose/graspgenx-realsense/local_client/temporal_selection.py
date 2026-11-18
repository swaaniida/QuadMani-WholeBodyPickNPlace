"""Model-independent metric grasp selection. No poses are generated or modified."""
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import numpy as np
from grasp_ranking import rank_grasps


@dataclass(frozen=True)
class TemporalConfig:
    position_sigma_m: float = .03  # Legacy configuration; position is not scored.
    rotation_sigma_deg: float = 20.
    collision_margin_m: float = 0.  # Legacy config compatibility; no additional margin is applied.
    plane_distance_m: float = .008
    max_plane_points: int = 5000
    plane_min_inliers: int = 250
    plane_min_fraction: float = .12
    plane_max_rmse_m: float = .006
    plane_max_object_height_m: float = .4
    # Optional calibrated up normal expressed in camera optical coordinates.
    ground_up_camera: object = None
    ground_max_tilt_deg: float = 25.
    voxel_m: float = .02
    max_background_points: int = 12000
    mask_dilation_pixels: int = 11
    min_icp_points: int = 300
    icp_correspondence_m: float = .04
    icp_min_fitness: float = .55
    icp_max_rmse_m: float = .018
    icp_max_step_m: float = .25
    icp_max_step_deg: float = 25.
    min_normal_eigenvalue: float = .003
    keyframe_interval: int = 10
    keyframe_realign_every: int = 5
    max_keyframes: int = 6
    max_keyframe_correction_m: float = .05
    max_keyframe_correction_deg: float = 10.
    max_temporal_gap_s: float = 1.  # Legacy setting; no temporal age limit is applied.
    icp_wait_s: float = .15
    result_cache_size: int = 1024

    def __post_init__(self):
        positive=('position_sigma_m','rotation_sigma_deg','plane_distance_m','plane_max_rmse_m',
                  'plane_max_object_height_m','voxel_m','icp_correspondence_m','icp_max_rmse_m',
                  'icp_max_step_m','icp_max_step_deg','max_keyframe_correction_m','max_keyframe_correction_deg',
                  'max_temporal_gap_s')
        if any(not np.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in positive):raise ValueError('Positive finite metric/time parameters required')
        if self.collision_margin_m<0 or self.icp_wait_s<0:raise ValueError('Margins/wait cannot be negative')
        for k in ('max_plane_points','plane_min_inliers','max_background_points','mask_dilation_pixels','min_icp_points','keyframe_interval','keyframe_realign_every','max_keyframes','result_cache_size'):
            if not isinstance(getattr(self,k),int) or getattr(self,k)<1:raise ValueError(f'{k} must be a positive integer')
        if not 0<self.icp_min_fitness<=1 or not 0<self.plane_min_fraction<=1:raise ValueError('Fitness/fraction must be (0,1]')
        if self.ground_up_camera is not None:
            n=np.asarray(self.ground_up_camera,float)
            if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:raise ValueError('Invalid ground up normal')

    @classmethod
    def load(cls,path=None):return cls(**json.loads(Path(path).read_text())) if path else cls()


def capture_time(frame):
    for name,scale in (('camera_timestamp_ms',.001),('host_timestamp_s',1.),('ready_monotonic',1.)):
        value=getattr(frame,name,None)
        if value is not None and np.isfinite(value):return name,float(value)*scale
    raise ValueError('No finite capture timestamp')


def same_capture_sequence(clock,generation,stamp,previous_clock,previous_generation,previous_time):
    # Long inference/capture gaps do not invalidate a trusted prior selection.
    return (clock==previous_clock and generation==previous_generation and previous_time is not None
            and stamp>previous_time)


def rotation_degrees(a,b):
    return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))


def pose_change(a,b):
    if a is None or b is None:return None,None
    return float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),float(rotation_degrees(a[:3,:3],b[:3,:3]))


def collision_clearance(grasps,vertices,plane):
    """Minimum signed CAD-vertex/solid-ground distance, exact for planar triangles.

    Plane normal points into free space. Open CAD mesh is the supplied official
    franka_umi geometry, NOT calibrated PiPER TCP or an arm collision model.
    """
    g=np.asarray(grasps,float);v=np.asarray(vertices,float);plane=np.asarray(plane,float)
    n=plane[:3];norm=np.linalg.norm(n)
    if norm<1e-8 or not np.isfinite(plane).all():raise ValueError('Invalid plane')
    n=n/norm;offset=plane[3]/norm
    # Equivalent to transforming every vertex, with much smaller allocations.
    local_normals=np.einsum('nji,j->ni',g[:,:3,:3],n)
    return np.min(local_normals@v.T,axis=1)+g[:,:3,3]@n+offset


def select_candidates(grasps,confidences,points,vertices,plane,config,previous_world=None,
                      T_world_camera=None,icp_reliable=False,ground_uncertainty_m=0.):
    g=np.asarray(grasps);c=np.asarray(confidences,float)
    baseline=rank_grasps(g,c,points)  # also validates SE(3) and C in [0,1].
    # Pin the requested baseline even if a laptop has a different legacy weight setting.
    baseline['scores']=baseline['components']@np.array([.6,.1,.25,.05])
    baseline['order']=np.argsort(-baseline['scores'],kind='stable')
    n=len(c);temporal=np.full(n,np.nan);normalized_c=baseline['components'][:,0];scores=normalized_c.copy()
    # Establish ground eligibility before final scoring. Baseline remains unfiltered
    # for comparison; no confidence top-k is allowed to discard input candidates.
    clearance=collision_clearance(g,vertices,plane) if plane is not None and n else np.full(n,np.nan)
    if not np.isfinite(ground_uncertainty_m) or ground_uncertainty_m<0:raise ValueError('Plane RMSE must be finite and nonnegative')
    valid=np.isfinite(clearance)&(clearance>ground_uncertainty_m)
    usable=bool(icp_reliable and T_world_camera is not None and previous_world is not None)
    if usable:
        world=np.asarray(T_world_camera)@g
        angles=np.array([rotation_degrees(previous_world[:3,:3],R) for R in world[:,:3,:3]])
        temporal=np.exp(-.5*(angles/config.rotation_sigma_deg)**2)
        scores=.6*normalized_c+.4*temporal
    order=np.flatnonzero(valid);order=order[np.argsort(-scores[order],kind='stable')]
    return dict(selected=int(order[0]) if len(order) else None,
                baseline_selected=int(baseline['order'][0]) if n else None,
                scores=scores,temporal=temporal,temporal_used=usable,baseline=baseline,
                clearance_m=clearance,valid=valid,order=order,
                status='ok' if len(order) else ('no_ground_plane' if plane is None else 'no_valid_grasp'))
