"""ViT-based crop classifier for label refinement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency.
    pipeline = None


@dataclass
class VitClassifierConfig:
    """Configuration for HF ViT classifier."""

    model_id: str = os.getenv(
        "VIT_DAMAGE_MODEL",
        "google/vit-base-patch16-224",
    )


class ViTDamageClassifier:
    """Classify detection crops using a pretrained image classification model."""

    LABEL_ALIASES = {
        "dent": "dent",
        "scratch": "scratch",
        "crack": "crack",
        "cracked": "crack",
        "broken": "structural_damage",
        "broken part": "structural_damage",
        "glass": "glass_damage",
        "glass damage": "glass_damage",
        "tire": "tire_damage",
        "light": "light_damage",
        "missing": "missing_part",
        "structural": "structural_damage",
    }

    def __init__(self, config: VitClassifierConfig | None = None) -> None:
        """Initialize HF pipeline with soft-fail behavior when unavailable."""
        self.config = config or VitClassifierConfig()
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = None
        self.available = False
        self.initialization_error = ""
        try:
            if pipeline is None:
                raise ImportError("transformers is not installed.")
            self.classifier = pipeline(
                "image-classification",
                model=self.config.model_id,
                device=self.device,
            )
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

    def _normalize_label(self, raw_label: str) -> str:
        """Map model output label text to project class names."""
        cleaned = raw_label.strip().lower().replace("_", " ")
        for key, mapped in self.LABEL_ALIASES.items():
            if key in cleaned:
                return mapped
        return cleaned.replace(" ", "_")

    def classify_crop(self, image_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> Dict[str, object]:
        """Run ViT classification for one crop and return top class."""
        default = {
            "vit_label": "",
            "vit_score": 0.0,
            "vit_raw_label": "",
            "vit_available": self.available,
        }
        if not self.available or self.classifier is None:
            return default

        crop = self._safe_crop(image_bgr, bbox)
        if crop is None:
            return default
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(crop_rgb)

        preds = self.classifier(image_pil, top_k=1)
        if not preds:
            return default
        top = preds[0]
        raw_label = str(top.get("label", ""))
        score = float(top.get("score", 0.0))
        return {
            "vit_label": self._normalize_label(raw_label),
            "vit_score": score,
            "vit_raw_label": raw_label,
            "vit_available": True,
        }
