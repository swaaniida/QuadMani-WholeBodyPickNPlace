"""Adapter for official Tracking_SAM.main_tracker (SAM ViT-H + AOTT VOS).

Only the GUI click interaction is replaced with point coordinates from the
Windows client. Temporal inference remains the official AOT engine.
"""
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'upstream/Tracking_SAM'
sys.path.insert(0, str(UP))


class TrackingBackend:
    def __init__(self):
        from tracking_SAM import main_tracker
        weights = UP / 'pretrained_weights'
        torch.set_num_threads(4)
        self.tracker = main_tracker(
            str(weights/'sam_vit_h_4b8939.pth'),
            str(weights/'AOTT_PRE_YTB_DAV.pth'),
            str(weights/'groundingdino_swint_ogc.pth'))
        self.tracker.sam.eval()
        self.sam_score = None
        self.reset()

    def reset(self):
        self.tracker.reset_engine()
        self.previous_mask = None

    @torch.inference_mode()
    def process(self, rgb, point=None, text=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        initial = point is not None or text is not None
        if initial:
            self.reset()
            if text is not None:
                # Official language initialization includes DINO box -> SAM
                # refinement -> AOT reference initialization.
                reference = self.tracker.vos_tracker.add_reference_frame
                captured = []
                def capture_reference(image, label):
                    captured.append(label.copy())
                    return reference(image, label)
                self.tracker.vos_tracker.add_reference_frame = capture_reference
                try:
                    self.tracker.annotate_init_frame(rgb, method='dino', category_name=text)
                finally:
                    self.tracker.vos_tracker.add_reference_frame = reference
                mask = captured[0]
                self.sam_score = None
            else:
                x, y = map(float, point)
                if not (0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]):
                    raise ValueError('Click outside input RGB image')
                predictor = self.tracker.sam_predictor
                predictor.set_image(rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[x, y]], np.float32),
                    point_labels=np.ones(1, np.int32), multimask_output=False)
                mask = masks[0].astype(np.uint8)
                if not mask.any():
                    raise ValueError('SAM returned an empty initial mask; click again')
                self.sam_score = float(scores[0])
                self.tracker.vos_tracker.add_reference_frame(rgb, mask)
                self.tracker.tracking = True
        else:
            if not self.tracker.is_tracking():
                raise ValueError('Tracker is uninitialized; click a target first')
            mask = self.tracker.propagate_one_frame(rgb)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        mask = (mask > 0).astype(np.uint8)
        self.previous_mask = mask
        return mask, dict(sam_init_ms=elapsed if initial else None,
                          vos_ms=elapsed if not initial else None,
                          sam_score=self.sam_score,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          allocated_mib=torch.cuda.memory_allocated()/2**20,
                          reserved_mib=torch.cuda.memory_reserved()/2**20)
