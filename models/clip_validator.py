"""CLIP-based validation layer for YOLO detection crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from consistency import clip


@dataclass
class ClipValidationConfig:
    """Configuration for CLIP crop validation."""

    score_threshold: float = 0.25
    model_name: str = "ViT-B/32"


class ClipValidator:
    """Validate cropped detections against textual damage prompts."""

    PROMPTS = [
        "a dent on a car",
        "a scratch on a car",
        "a cracked car surface",
        "a broken car part",
    ]
    PROMPT_TO_LABEL = {
        "a dent on a car": "dent",
        "a scratch on a car": "scratch",
        "a cracked car surface": "crack",
        "a broken car part": "structural_damage",
    }

    def __init__(self, config: ClipValidationConfig | None = None, device: str | None = None) -> None:
        """Initialize CLIP model or keep soft-fail state when unavailable."""
        self.config = config or ClipValidationConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.available = False
        self.initialization_error = ""

        try:
            if clip is None:
                raise ImportError("CLIP package is not installed.")
            self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
            self.available = True
        except Exception as exc:
            self.initialization_error = str(exc)

    @staticmethod
    def _safe_crop(image_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray | None:
        """Crop bbox region from image with boundary clamping."""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return image_bgr[y1:y2, x1:x2]

    def validate_crop(self, image_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> Dict[str, object]:
        """Run CLIP prompt matching for one detection crop."""
        default = {
            "clip_label": "",
            "clip_score": 0.0,
            "clip_scores": {},
            "clip_keep": True,
            "clip_available": self.available,
        }
        if not self.available or self.model is None or self.preprocess is None:
            return default

        crop = self._safe_crop(image_bgr, bbox)
        if crop is None:
            return default

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(crop_rgb)
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(self.PROMPTS).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
            probs = torch.softmax(sims, dim=0).detach().cpu().numpy()

        best_idx = int(np.argmax(probs))
        best_prompt = self.PROMPTS[best_idx]
        best_score = float(probs[best_idx])
        scores: Dict[str, float] = {
            self.PROMPT_TO_LABEL[p]: float(probs[i]) for i, p in enumerate(self.PROMPTS)
        }
        return {
            "clip_label": self.PROMPT_TO_LABEL[best_prompt],
            "clip_score": best_score,
            "clip_scores": scores,
            "clip_keep": best_score >= self.config.score_threshold,
            "clip_available": True,
        }
