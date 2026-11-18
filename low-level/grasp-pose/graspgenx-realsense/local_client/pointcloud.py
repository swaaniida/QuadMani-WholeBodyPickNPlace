"""Metric optical-frame geometry. No normalization or frame rotation."""
from dataclasses import dataclass
import time
import cv2
import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    near_m: float = 0.1
    far_m: float = 2.0
    foreground_band_m: float = 0.08
    foreground_percentile: float = 10.0
    voxel_m: float = 0.003
    max_points: int = 12000
    min_component_pixels: int = 30


def segment(depth_m, roi, cfg):
    """Depth band starts near a robust foreground percentile; keep largest island."""
    if not (0 < cfg.near_m < cfg.far_m) or cfg.foreground_band_m < 0:
        raise ValueError('Require 0 < near < far and foreground band >= 0')
    if not 0 <= cfg.foreground_percentile <= 100:
        raise ValueError('Foreground percentile must be in [0,100]')
    x, y, w, h = map(int, roi)
    height, width = depth_m.shape
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > width or y+h > height:
        raise ValueError('ROI outside captured image')
    crop = depth_m[y:y+h, x:x+w]
    valid = np.isfinite(crop) & (crop >= cfg.near_m) & (crop <= cfg.far_m)
    if not valid.any():
        raise ValueError('No valid depth in ROI; adjust near/far or select again')
    foreground = float(np.percentile(crop[valid], cfg.foreground_percentile))
    if cfg.foreground_band_m > 0:
        valid &= crop <= foreground + cfg.foreground_band_m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise ValueError('No foreground component')
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[label, cv2.CC_STAT_AREA] < cfg.min_component_pixels:
        raise ValueError('Foreground component too small')
    mask = np.zeros(depth_m.shape, dtype=bool)
    mask[y:y+h, x:x+w] = labels == label
    return mask


def deproject(depth_m, K, mask, unit_rays=None):
    """SDK unit rays account for aligned-frame lens distortion when provided."""
    v, u = np.where(mask & np.isfinite(depth_m) & (depth_m > 0))
    z = depth_m[v, u]
    if unit_rays is not None:
        points = unit_rays[v, u] * z[:, None]
    else:
        points = np.column_stack(((u-K[0, 2])*z/K[0, 0],
                                  (v-K[1, 2])*z/K[1, 1], z))
    return np.ascontiguousarray(points, dtype=np.float32), v, u


def downsample(points, colors, voxel_m=0.003, max_points=12000):
    if voxel_m < 0 or max_points < 100:
        raise ValueError('voxel >= 0 and max_points >= 100 required')
    indices = np.arange(len(points))
    if voxel_m > 0 and len(points):
        _, indices = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0,
                               return_index=True)
    if len(indices) > max_points:
        indices = indices[np.random.default_rng(42).choice(len(indices), max_points, replace=False)]
    return np.ascontiguousarray(points[indices], dtype=np.float32), colors[indices]


def prepare(frame, roi, cfg):
    start = time.perf_counter()
    mask = segment(frame.depth_m, roi, cfg)
    segmented = time.perf_counter()
    points, v, u = deproject(frame.depth_m, frame.K, mask, frame.unit_rays)
    colors = frame.bgr[v, u, ::-1].copy()
    generated = time.perf_counter()
    points, colors = downsample(points, colors, cfg.voxel_m, cfg.max_points)
    end = time.perf_counter()
    if len(points) < 100:
        raise ValueError(f'Only {len(points)} points; need >=100, adjust ROI/filter/voxel')
    return points, colors, mask, {
        'segmentation_ms': (segmented-start)*1000,
        'pointcloud_ms': (generated-segmented)*1000,
        'downsampling_ms': (end-generated)*1000,
    }
