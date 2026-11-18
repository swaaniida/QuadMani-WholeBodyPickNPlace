# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Extracted unchanged functions from NVlabs/GraspGenX commit 44eb1d77d6d27bc97ee361e3afb717a2115e30a9.
# Vendored to keep Windows free from graspgenx package initialization and torch.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import trimesh

import trimesh.transformations as tra

import viser

import viser.transforms as vtf

def get_color_from_score(labels, use_255_scale=False):
    """Convert score labels to RGB colors (red=low, green=high)."""
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        score = scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )
        return score.astype(np.int32)

def matrix_to_wxyz_position(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 homogeneous transformation matrix to wxyz quaternion and position.

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (wxyz quaternion, position)
    """
    rotation_matrix = T[:3, :3]
    so3 = vtf.SO3.from_matrix(rotation_matrix)
    wxyz = so3.wxyz
    position = T[:3, 3]
    return wxyz, position

def create_gripper_control_points_for_viz(
    width: float, depth: float, height: float = 0.0
) -> List[np.ndarray]:
    """
    Create control points for gripper visualization based on width, depth, and height.
    This follows the x-grasp convention for parallel jaw grippers.

    Args:
        width: gripper opening width (X extent)
        depth: gripper depth/fingertip z position (Z extent)
        height: gripper height (Y extent) - used for mid_point positioning

    Returns:
        List of control point arrays for visualization (each array is shape [N, 4])
    """
    hw = width / 2  # half width
    hh = height / 2  # half height

    # Create canonical control points
    # Format: left fingertip, right fingertip, mid-finger points, base
    left_tip = np.array([-hw, 0, depth, 1])
    right_tip = np.array([hw, 0, depth, 1])
    left_mid = np.array([-hw, 0, depth * 0.6, 1])
    right_mid = np.array([hw, 0, depth * 0.6, 1])
    base = np.array([0, 0, 0, 1])

    # Mid point at fingertip depth, offset by half height in Y
    mid_point = np.array([0, -hh, depth, 1])

    # Create visualization line path:
    # left_mid -> left_tip -> mid_point -> base -> mid_point -> right_tip -> right_mid
    vis_points = np.array(
        [left_mid, left_tip, mid_point, base, mid_point, right_tip, right_mid]
    )

    return [vis_points]

def generate_control_points_from_sweep_volume(sweep_volume: Dict) -> List[np.ndarray]:
    """Generate control points from sweep volume."""
    sv_extents = sweep_volume["extents"]
    f = sweep_volume["offset"][2]

    w = sv_extents[0]
    d = sv_extents[1]
    h = sv_extents[2]

    control_points = np.array(
        [
            [w / 2, 0, h / 2 + f, 1],
            [w / 2, 0, -h / 2 + f, 1],
            [0, 0, -h / 2 + f, 1],
            [0, 0, 0, 1],
            [0, 0, -h / 2 + f, 1],
            [-w / 2, 0, -h / 2 + f, 1],
            [-w / 2, 0, h / 2 + f, 1],
        ]
    )
    control_points = [
        control_points,
    ]  # Not sure why this is needed, but it is.
    return control_points

def visualize_x_grasp(
    vis: viser.ViserServer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_info=None,
    width: float = None,
    depth: float = None,
    height: float = 0.0,
    sweep_volume: Dict = None,
    linewidth: float = 2.0,
    **kwargs,
):
    """
    Visualize a grasp using XGripperInfo control points or width/depth/height parameters.

    Args:
        vis: viser server object
        name: str, name for this grasp visualization
        transform: 4x4 homogeneous transform for the grasp pose
        color: RGB color list
        gripper_info: XGripperInfo object with control_points_visualization (optional)
        width: gripper width (used if gripper_info not provided)
        depth: gripper depth (used if gripper_info not provided)
        height: gripper height (used if gripper_info not provided)
        linewidth: width of the line segments
    """
    if vis is None:
        return []

    # Get control points — prefer sweep_volume for accurate extremities
    if (
        gripper_info is not None
        and hasattr(gripper_info, "sweep_volume")
        and gripper_info.sweep_volume is not None
    ):
        # gripper_info.sweep_volume is [extents(3), offset(3)]
        sv = gripper_info.sweep_volume
        sv_dict = {"extents": sv[:3].tolist(), "offset": sv[3:].tolist()}
        grasp_vertices = generate_control_points_from_sweep_volume(sv_dict)
    elif sweep_volume is not None:
        grasp_vertices = generate_control_points_from_sweep_volume(sweep_volume)
    elif width is not None and depth is not None:
        grasp_vertices = create_gripper_control_points_for_viz(width, depth, height)
    else:
        # Default fallback
        grasp_vertices = create_gripper_control_points_for_viz(0.08, 0.05, 0.0)

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz, position = matrix_to_wxyz_position(transform.astype(float))

    handles = []
    for i, ctrl_pts in enumerate(grasp_vertices):
        # Convert to numpy array
        ctrl_pts = np.array(ctrl_pts, dtype=np.float32)

        # Handle different input shapes
        if ctrl_pts.ndim == 1:
            # Single point, skip
            continue

        # Ensure shape is [N, 4] (N points with homogeneous coords)
        if ctrl_pts.shape[0] == 4 and ctrl_pts.shape[1] != 4:
            # Shape is [4, N], transpose it
            ctrl_pts = ctrl_pts.T

        points_3d = ctrl_pts[:, :3]  # Shape: [N, 3]

        num_points = points_3d.shape[0]
        if num_points < 2:
            continue

        # Create line segments
        segments = np.zeros((num_points - 1, 2, 3), dtype=np.float32)
        for j in range(num_points - 1):
            segments[j, 0, :] = points_3d[j]
            segments[j, 1, :] = points_3d[j + 1]

        handle = vis.scene.add_line_segments(
            f"{name}/{i}",
            points=segments,
            colors=color_tuple,
            line_width=linewidth,
            wxyz=wxyz,
            position=position,
        )
        handles.append(handle)

    return handles

