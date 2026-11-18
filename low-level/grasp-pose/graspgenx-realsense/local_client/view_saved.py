"""Replay real saved results (no inference). Works without a camera."""
import argparse
from pathlib import Path
import time
import webbrowser
import numpy as np
from visualization import GraspViewer

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', type=Path, help='outputs/grasps/*.npz')
    p.add_argument('--port', type=int, default=8080)
    args = p.parse_args()
    viewer = GraspViewer(args.port)
    data = np.load(args.result, allow_pickle=False)
    colors = data['colors'] if 'colors' in data else np.full_like(data['point_cloud'], 180, dtype=np.uint8)
    if colors.dtype.kind == 'f' and colors.max(initial=0) <= 1:
        colors = np.clip(colors*255, 0, 255).astype(np.uint8)
    label = str(data['frame']) if 'frame' in data else 'saved_input_frame'
    viewer.show(data['point_cloud'], colors, data['grasps'], data['confidences'],
                args.result.stem, frame_label=label,
                original_indices=data['original_candidate_indices'] if 'original_candidate_indices' in data else None)
    webbrowser.open(f'http://127.0.0.1:{args.port}')
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        viewer.close()
