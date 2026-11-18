"""Camera-frame pose preferences. Pure NumPy postprocessing; no model/RPC changes.

Official GraspGenX base frame: +Z approach, +X finger closing.
R: closing-axis projection in camera XY, relative to camera +X (about Z).
Y: approach-axis projection in camera XZ, relative to camera +Z (about Y).
P: signed approach alignment to the camera-to-object centroid ray.
Each angular preference is (1 + cos(angle))/2. Undefined projections are
neutral (0.5). These are separate preferences, NOT conventional XYZ Euler angles.
"""
import numpy as np

WEIGHTS = np.array([0.6, 0.1, 0.25, 0.05])


def rank_grasps(grasps, confidences, points, original_indices=None):
    g = np.asarray(grasps, dtype=np.float64)
    c = np.asarray(confidences, dtype=np.float64)
    if g.shape != (len(c), 4, 4) or c.ndim != 1:
        raise ValueError('Expected grasps (N,4,4), confidences (N,)')
    if not np.isfinite(g).all() or not np.isfinite(c).all():
        raise ValueError('Non-finite grasp/confidence')
    if np.any(c < -1e-6) or np.any(c > 1+1e-6):
        raise ValueError('Confidence must already be a probability in [0,1]')
    if len(g) and (not np.allclose(g[:, :3, :3].transpose(0, 2, 1) @ g[:, :3, :3], np.eye(3), atol=1e-3)
                   or not np.allclose(np.linalg.det(g[:, :3, :3]), 1, atol=1e-3)
                   or not np.allclose(g[:, 3], [0, 0, 0, 1], atol=1e-5)):
        raise ValueError('Expected SE(3) gripper-base poses')
    pc = np.asarray(points, dtype=np.float64)
    if pc.ndim != 2 or pc.shape[1] != 3 or not len(pc) or not np.isfinite(pc).all():
        raise ValueError('Expected a nonempty finite camera-frame object cloud (N,3)')
    center = pc.mean(axis=0)
    norm = np.linalg.norm(center)
    if norm < 1e-8 or center[2] <= 0:
        raise ValueError('Camera-to-object direction requires an object in front of the camera')
    direction = center / norm
    gx, gz = g[:, :3, 0], g[:, :3, 2]
    def projected_cos(a, b):
        length = np.hypot(a, b)
        return np.divide(a, length, out=np.zeros_like(a), where=length > 1e-8)
    r = (1 + projected_cos(gx[:, 0], gx[:, 1])) / 2
    p = (1 + np.clip(gz @ direction, -1, 1)) / 2
    y = (1 + projected_cos(gz[:, 2], gz[:, 0])) / 2
    components = np.column_stack((np.clip(c, 0, 1), r, p, y))
    scores = components @ WEIGHTS
    order = np.argsort(-scores, kind='stable')
    best_c = int(np.argmax(c)) if len(c) else None
    if best_c is not None and original_indices is not None:
        ids = np.asarray(original_indices)
        if ids.shape != c.shape or len(np.unique(ids)) != len(c):
            raise ValueError('Candidate provenance indices must be unique and match candidates')
        tied = np.flatnonzero(c == c.max())
        best_c = int(tied[np.argmin(ids[tied])])
    return dict(scores=scores, components=components, order=order,
                confidence_best=best_c,
                object_center_m=center, approach_reference=direction)


def rerank_result(result, points):
    """Return a new result; preserve raw confidence values and candidate provenance."""
    ranking = rank_grasps(result['grasps'], result['confidences'], points)
    order = ranking['order']
    out = dict(result)
    for key in ('grasps', 'confidences', 'branch_tags'):
        if key in result:
            if len(result[key]) != len(order):
                raise ValueError(f'{key} candidate count mismatch')
            values = np.asarray(result[key])[order]
            out[key] = values.tolist() if isinstance(result[key], list) else values
    out['selection_scores'] = ranking['scores'][order]
    out['orientation_scores'] = ranking['components'][order, 1:]
    out['original_candidate_indices'] = order
    out['selection_metadata'] = dict(
        formula='S=0.6C+0.1R+0.25P+0.05Y', component_order=['R', 'P', 'Y'],
        normalization='(1+cos(angle))/2; degenerate projection=0.5; C unchanged',
        roll='closing +X projected onto camera XY relative to camera +X',
        pitch='approach +Z aligned to camera-to-object centroid ray',
        yaw='approach +Z projected onto camera XZ relative to camera +Z',
        frame='camera_optical: X right, Y down, Z forward',
        gravity_reference='unavailable; top-down means camera viewing direction, not calibrated gravity',
        symmetry='signed axes; no automatic 180-degree finger-symmetry equivalence',
        object_center_m=ranking['object_center_m'].tolist(),
        approach_reference=ranking['approach_reference'].tolist(),
        confidence_best_original_index=ranking['confidence_best'],
        selected_original_index=int(order[0]) if len(order) else None)
    return out


def display_order(ranking, topk=10):
    order = ranking['order'][:topk].tolist()
    best_c = ranking['confidence_best']
    if best_c is not None and best_c not in order:
        order.append(best_c)
    return order


def comparison_label(idx, ranking):
    best_s = int(ranking['order'][0])
    best_c = ranking['confidence_best']
    if idx == best_s and idx == best_c:
        return 'S#1 = C#1'
    if idx == best_s:
        return 'S#1'
    if idx == best_c:
        return 'C#1'
    return ''
