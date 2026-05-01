"""Optional SAM segmenter for refining damage area within YOLO boxes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from segment_anything import SamPredictor, sam_model_registry
except Exception:  # pragma: no cover - optional dependency.
    SamPredictor = None
    sam_model_registry = None


@dataclass
class SamConfig:
    """Configuration for SAM loading."""

    checkpoint_path: str = os.getenv("SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")
    model_type: str = os.getenv("SAM_MODEL_TYPE", "vit_b")


class SamSegmenter:
    """Run segmentation on detection crops using SAM bounding-box prompts."""

    def __init__(self, config: SamConfig | None = None, project_root: Path | None = None) -> None:
        """Initialize optional SAM predictor."""
        self.config = config or SamConfig()
        self.project_root = project_root
        self.available = False
        self.initialization_error = ""
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if SamPredictor is None or sam_model_registry is None:
                raise ImportError("segment-anything is not installed.")
            ckpt = Path(self.config.checkpoint_path)
            if not ckpt.is_absolute() and self.project_root is not None:
                ckpt = (self.project_root / ckpt).resolve()
            if not ckpt.exists():
                raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

            sam = sam_model_registry[self.config.model_type](checkpoint=str(ckpt))
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.available = True
        except Exception as exc:
            self.initialization_error = str(exc)

    def segment_bbox(self, image_rgb: np.ndarray, bbox: Tuple[float, float, float, float]) -> Dict[str, object]:
        """Predict mask from bbox and return area stats."""
        default = {"mask": None, "mask_area_ratio": 0.0, "sam_available": self.available}
        if not self.available or self.predictor is None:
            return default
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return default
        self.predictor.set_image(image_rgb)
        box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, _, _ = self.predictor.predict(box=box_np[None, :], multimask_output=False)
        if masks is None or len(masks) == 0:
            return default
        mask = masks[0].astype(bool)
        ratio = float(mask.sum() / max(1, h * w))
        return {"mask": mask, "mask_area_ratio": ratio, "sam_available": True}
