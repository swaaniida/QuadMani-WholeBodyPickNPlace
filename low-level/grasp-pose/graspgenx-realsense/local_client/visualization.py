"""CPU-only browser 3D viewer and capture-matched RGB projection."""
from pathlib import Path
import time
import cv2
import numpy as np
import trimesh
import viser
from official_viser_helpers import matrix_to_wxyz_position, visualize_x_grasp
from grasp_client import load_sweep_params
from grasp_ranking import rank_grasps, display_order, comparison_label


def project_points(points, K, color_intrinsics=None):
    points = np.asarray(points, dtype=np.float64)
    if np.any(points[:, 2] <= 1e-4):
        return None
    if color_intrinsics and np.any(color_intrinsics.get('coeffs', [])):
        import pyrealsense2 as rs
        i = rs.intrinsics()
        for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs'):
            setattr(i, key, color_intrinsics[key])
        model = color_intrinsics['model'].split('.')[-1]
        i.model = getattr(rs.distortion, model)
        uv = np.array([rs.rs2_project_point_to_pixel(i, p.tolist()) for p in points])
    else:
        uvw = points @ np.asarray(K).T
        uv = uvw[:, :2] / uvw[:, 2:3]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e6:
        return None
    return np.rint(uv).astype(int)


def overlay(bgr, K, grasps, scores, frame_id, color_intrinsics=None, topk=10, point_cloud=None, original_indices=None):
    image = bgr.copy()
    if color_intrinsics:
        i = color_intrinsics
        K = np.array([[i['fx'], 0, i['ppx']], [0, i['fy'], i['ppy']], [0, 0, 1]])
    ranking = rank_grasps(grasps, scores, point_cloud, original_indices) if point_cloud is not None else None
    order = display_order(ranking, topk) if ranking is not None else np.argsort(-scores)[:topk]
    draw_order = list(order)
    if ranking is not None and len(order):
        leaders = list(dict.fromkeys((ranking['confidence_best'], int(ranking['order'][0]))))
        draw_order = [idx for idx in order if idx not in leaders] + leaders
    for rank, idx in enumerate(draw_order):
        is_selected = idx == order[0]
        label = comparison_label(idx, ranking) if ranking is not None else ('C#1' if rank == 0 else '')
        highlight = (0, 255, 255) if is_selected else ((255, 255, 0) if label == 'C#1' else (180, 180, 180))
        s_rank = int(np.flatnonzero(ranking['order'] == idx)[0])+1 if ranking is not None else rank+1
        text = f'#{s_rank} C={scores[idx]:.3f}'
        if ranking is not None:
            text += f" S={ranking['scores'][idx]:.3f} {label}"
        T = grasps[idx]
        local = np.vstack((np.zeros(3), np.eye(3)*0.04))
        camera_points = local @ T[:3, :3].T + T[:3, 3]
        uv = project_points(camera_points, K, color_intrinsics)
        if uv is None:
            continue
        origin = tuple(uv[0])
        for j, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            visible, a, b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), origin, tuple(uv[j+1]))
            if visible:
                cv2.line(image, a, b, color, 3 if is_selected else 1, cv2.LINE_AA)
        cv2.circle(image, origin, 8 if label else 3, highlight, 3 if label else 1)
        cv2.putText(image, text, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, highlight, 1, cv2.LINE_AA)
    cv2.putText(image, f'CAPTURE {frame_id}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 255, 255), 1, cv2.LINE_AA)
    if ranking is not None and len(grasps):
        for row, idx in enumerate(dict.fromkeys((int(ranking['order'][0]), ranking['confidence_best']))):
            c, r, p, y = ranking['components'][idx]
            line = f"{comparison_label(idx, ranking)} S={ranking['scores'][idx]:.4f} C={c:.4f} R={r:.3f} P={p:.3f} Y={y:.3f}"
            cv2.rectangle(image, (0, 29+row*23), (image.shape[1], 52+row*23), (20,20,20), -1)
            cv2.putText(image, line, (8, 46+row*23), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (0,255,255) if row == 0 else (255,255,0), 1, cv2.LINE_AA)
    return image


class GraspViewer:
    def __init__(self, port=8080):
        self.server = viser.ViserServer(host='127.0.0.1', port=port, label='RealSense GraspGenX PoC')
        self.mesh = None
        mesh_path = Path(__file__).parent / 'assets/franka_umi_vis_mesh.obj'
        if mesh_path.exists():
            try:
                self.mesh = trimesh.load(mesh_path, force='mesh', process=False)
            except Exception as exc:
                print('CAD load failed; using labeled official sweep wireframe:', exc)
        self.params = load_sweep_params()
        self.info = self.server.gui.add_markdown('Select an object with S. Browser view is local only.')
        self.details = self.server.gui.add_markdown('No inference yet.')
        self.mesh_checkbox = self.server.gui.add_checkbox('Show official franka_umi CAD', initial_value=True)
        self.mesh_handles = []

        @self.mesh_checkbox.on_update
        def toggle(_):
            for handle in self.mesh_handles:
                handle.visible = self.mesh_checkbox.value

    def show(self, points, colors, grasps=None, scores=None, frame_id='', frame_label='camera_optical', original_indices=None):
        start = time.perf_counter()
        vis = self.server
        vis.scene.reset()
        self.mesh_handles = []
        vis.scene.set_up_direction('-y')  # Viewer navigation only; data are never rotated.
        vis.scene.add_frame('/camera', axes_length=0.12, axes_radius=0.002)
        vis.scene.add_label('/camera/label', f'{frame_label}: X right, Y down, Z forward; meters')
        vis.scene.add_point_cloud('/object', points=points, colors=colors, point_size=0.0025)
        grasps = np.zeros((0, 4, 4)) if grasps is None else grasps
        scores = np.zeros(0) if scores is None else scores
        ranking = rank_grasps(grasps, scores, points, original_indices) if frame_label == 'camera_optical' else None
        order = display_order(ranking) if ranking is not None else np.argsort(-scores)[:10]
        detail = []
        sweep = dict(extents=self.params['extents_open'], offset=self.params['offset_open'])
        for rank, idx in enumerate(order):
            s_rank = int(np.flatnonzero(ranking['order'] == idx)[0])+1 if ranking is not None else rank+1
            label = comparison_label(idx, ranking) if ranking is not None else ('C#1' if rank == 0 else '')
            score_text = f'C={scores[idx]:.4f}'
            if ranking is not None:
                score_text += f" S={ranking['scores'][idx]:.4f}"
            T = grasps[idx]
            name = f'/grasp_{rank+1:02d}'
            wxyz, position = matrix_to_wxyz_position(T)
            color = (255, 210, 0) if rank == 0 else ((0, 255, 255) if label == 'C#1' else (int(255*(1-scores[idx])), int(255*scores[idx]), 100))
            visualize_x_grasp(vis, name+'/wire', T, color=color, sweep_volume=sweep,
                              linewidth=5 if label else 2)
            vis.scene.add_frame(name+'/axes', wxyz=wxyz, position=position,
                                axes_length=0.06 if rank == 0 else 0.035,
                                axes_radius=0.002 if rank == 0 else 0.0008)
            vis.scene.add_label(name+'/score', f'#{s_rank} {score_text} {label}', position=position)
            if self.mesh is not None:
                h = vis.scene.add_mesh_simple(name+'/CAD', vertices=self.mesh.vertices,
                                             faces=self.mesh.faces, color=color,
                                             wxyz=wxyz, position=position,
                                             opacity=0.5 if label else 0.15,
                                             visible=self.mesh_checkbox.value)
                self.mesh_handles.append(h)
            preference = '' if ranking is None else f"R/P/Y: `{ranking['components'][idx, 1:].round(5).tolist()}`\n\n"
            detail.append(f'**#{s_rank} {score_text} {label}**\n\n' + preference +
                          f'position m: `{position.round(5).tolist()}`\n\n'
                          f'quaternion wxyz (viewer): `{wxyz.round(5).tolist()}`\n\n'
                          f'```\n{np.array2string(T, precision=5)}\n```')
        self.info.content = (f'Capture `{frame_id}`; frame `{frame_label}`; {len(points)} points; '
                             f'{len(grasps)} grasps; showing top {len(order)}.\n\n'
                             + ('Official franka_umi visual mesh, static open configuration. '
                                if self.mesh is not None else 'No CAD: official sweep-volume wireframe only. ')
                             + 'Pose = gripper base, not robot TCP. '
                             + ('Gold = S#1; cyan = original C#1 (both gold if identical). '
                                'Top 10 by S plus C#1 if outside top 10. C remains raw model confidence. '
                                'S = 0.6C + 0.1R + 0.25P + 0.05Y. '
                                if ranking is not None else 'Non-camera frame: confidence ranking only. ')
                             +
                             'Mouse drag: orbit; scroll: zoom.')
        self.details.content = '\n\n'.join(detail) if detail else 'Point cloud preview; no grasp inferred.'
        center = np.mean(points, axis=0)

        def aim(client):
            client.camera.position = center + np.array([0.3, -0.25, -0.5])
            client.camera.look_at = center
            client.camera.up_direction = (0, -1, 0)

        for client in vis.get_clients().values():
            aim(client)
        # Replace the callback through on_client_connect's handle to avoid
        # accumulating a new callback on every result.
        self._aim = aim
        if not getattr(self, '_connected_callback', False):
            @vis.on_client_connect
            def connected(client):
                self._aim(client)
            self._connected_callback = True
        return (time.perf_counter()-start)*1000

    def close(self):
        self.server.stop()
