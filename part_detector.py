"""Stage 2 car-part localization using robust geometric zone mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Format: (x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio)
CAR_PART_ZONES: Dict[str, Tuple[float, float, float, float]] = {
    "hood": (0.2, 0.0, 0.8, 0.28),
    "roof": (0.2, 0.0, 0.8, 0.2),
    "front_bumper": (0.1, 0.62, 0.9, 0.8),
    "rear_bumper": (0.1, 0.8, 0.9, 1.0),
    "windshield": (0.15, 0.12, 0.85, 0.35),
    "rear_windshield": (0.15, 0.35, 0.85, 0.55),
    "door_front_left": (0.0, 0.22, 0.22, 0.8),
    "door_front_right": (0.78, 0.22, 1.0, 0.8),
    "door_rear_left": (0.22, 0.22, 0.42, 0.82),
    "door_rear_right": (0.58, 0.22, 0.78, 0.82),
    "fender_left": (0.0, 0.1, 0.25, 0.6),
    "fender_right": (0.75, 0.1, 1.0, 0.6),
    "headlight_left": (0.0, 0.3, 0.2, 0.6),
    "headlight_right": (0.8, 0.3, 1.0, 0.6),
    "trunk": (0.15, 0.3, 0.85, 0.8),
}

VIEW_ALIASES = {"auto", "front", "rear", "left_side", "right_side", "side"}

VIEW_ZONE_FILTERS: Dict[str, List[str]] = {
    "front": [
        "front_bumper",
        "hood",
        "windshield",
        "headlight_left",
        "headlight_right",
        "fender_left",
        "fender_right",
        "door_front_left",
        "door_front_right",
        "roof",
    ],
    "rear": [
        "rear_bumper",
        "trunk",
        "rear_windshield",
        "door_rear_left",
        "door_rear_right",
        "fender_left",
        "fender_right",
        "roof",
    ],
    "left_side": [
        "door_front_left",
        "door_rear_left",
        "fender_left",
        "front_bumper",
        "rear_bumper",
        "roof",
    ],
    "right_side": [
        "door_front_right",
        "door_rear_right",
        "fender_right",
        "front_bumper",
        "rear_bumper",
        "roof",
    ],
}


@dataclass
class PartDetection:
    """Representation of one inferred car part region."""

    part_name: str
    bbox: List[float]
    confidence: float
    method: str


class CarPartDetector:
    """Detect/infer car parts from image geometry, with optional model hook."""

    def __init__(self, use_model: bool = False) -> None:
        """Initialize detector. Geometric mode works without any API or downloads."""
        self.use_model = use_model

    def detect_parts(self, image: np.ndarray, scene_view: str = "auto") -> List[dict]:
        """Return inferred part regions for the given image."""
        image_h, image_w = image.shape[:2]
        view = self._normalize_view(scene_view)
        allowed_parts = set(VIEW_ZONE_FILTERS.get(view, CAR_PART_ZONES.keys()))
        parts: List[dict] = []
        for part_name, zone in CAR_PART_ZONES.items():
            if part_name not in allowed_parts:
                continue
            bbox = self._zone_to_bbox(zone, image_w, image_h)
            parts.append(
                PartDetection(
                    part_name=part_name,
                    bbox=[float(v) for v in bbox],
                    confidence=0.75,
                    method="geometric_zone_mapping",
                ).__dict__
            )
        return parts

    def get_part_for_damage(
        self,
        damage_bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, int] | Tuple[int, int, int],
        scene_view: str = "auto",
    ) -> str:
        """Return most likely car part for a damage bounding box."""
        view = self._normalize_view(scene_view)
        allowed_parts = set(VIEW_ZONE_FILTERS.get(view, CAR_PART_ZONES.keys()))
        best_part = ""
        best_overlap = 0.0
        candidates: List[Tuple[str, float]] = []
        for part_name, zone in CAR_PART_ZONES.items():
            if part_name not in allowed_parts:
                continue
            overlap = self._compute_zone_overlap(damage_bbox, zone, image_shape)
            candidates.append((part_name, overlap))
            if overlap > best_overlap:
                best_overlap = overlap
                best_part = part_name
        if best_overlap <= 0.0:
            return self._quadrant_fallback(damage_bbox, image_shape, view)
        return self._resolve_ambiguous_part(best_part, damage_bbox, image_shape, candidates)

    def _compute_zone_overlap(
        self,
        bbox: Tuple[float, float, float, float],
        zone: Tuple[float, float, float, float],
        image_shape: Tuple[int, int] | Tuple[int, int, int],
    ) -> float:
        """Compute fraction of damage bbox area overlapping a zone."""
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        zx1, zy1, zx2, zy2 = self._zone_to_bbox(zone, image_w, image_h)
        x1, y1, x2, y2 = bbox

        ix1 = max(float(x1), float(zx1))
        iy1 = max(float(y1), float(zy1))
        ix2 = min(float(x2), float(zx2))
        iy2 = min(float(y2), float(zy2))

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        bbox_area = max(1e-6, (float(x2) - float(x1)) * (float(y2) - float(y1)))
        return float(inter_area / bbox_area)

    def _quadrant_fallback(
        self,
        bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, int] | Tuple[int, int, int],
        scene_view: str = "auto",
    ) -> str:
        """Fallback part assignment when no zone overlap is found."""
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        if scene_view == "rear":
            if cy > image_h * 0.72:
                return "rear_bumper"
            if image_w * 0.33 <= cx <= image_w * 0.67:
                return "trunk"
        if scene_view == "front":
            if cy > image_h * 0.62:
                return "front_bumper"
            if image_w * 0.3 <= cx <= image_w * 0.7:
                return "hood"
        if scene_view == "left_side":
            return "door_front_left" if cy < image_h * 0.55 else "door_rear_left"
        if scene_view == "right_side":
            return "door_front_right" if cy < image_h * 0.55 else "door_rear_right"

        if cy < image_h * 0.35 and image_w * 0.33 <= cx <= image_w * 0.67:
            return "hood"
        if cy > image_h * 0.8 and image_w * 0.33 <= cx <= image_w * 0.67:
            return "rear_bumper"
        if cy > image_h * 0.65 and image_w * 0.33 <= cx <= image_w * 0.67:
            return "front_bumper"
        if cx < image_w * 0.5:
            if cy < image_h * 0.45:
                return "door_front_left"
            if cy < image_h * 0.7:
                return "door_rear_left"
            return "fender_left"
        if cy < image_h * 0.45:
            return "door_front_right"
        if cy < image_h * 0.7:
            return "door_rear_right"
        return "fender_right"

    @staticmethod
    def _normalize_view(scene_view: str) -> str:
        """Normalize caller-provided scene view into supported aliases."""
        normalized = scene_view.strip().lower().replace("-", "_")
        if normalized == "side":
            return "left_side"
        if normalized not in VIEW_ALIASES:
            return "auto"
        return normalized

    def _resolve_ambiguous_part(
        self,
        best_part: str,
        bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, int] | Tuple[int, int, int],
        candidates: List[Tuple[str, float]],
    ) -> str:
        """Resolve ties for similar zones using box center heuristics."""
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0

        if best_part in {"front_bumper", "rear_bumper"}:
            return "rear_bumper" if cy > image_h * 0.8 else "front_bumper"
        if best_part in {"windshield", "rear_windshield"}:
            return "rear_windshield" if cy > image_h * 0.36 else "windshield"
        if best_part in {"door_front_left", "door_rear_left"}:
            return "door_front_left" if cx < image_w * 0.2 else "door_rear_left"
        if best_part in {"door_front_right", "door_rear_right"}:
            return "door_front_right" if cx > image_w * 0.82 else "door_rear_right"

        # Tie-break with second-best if overlaps are almost identical.
        sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        if len(sorted_candidates) > 1 and (sorted_candidates[0][1] - sorted_candidates[1][1]) < 0.02:
            return sorted_candidates[0][0]
        return best_part

    @staticmethod
    def _zone_to_bbox(
        zone: Tuple[float, float, float, float], image_w: int, image_h: int
    ) -> Tuple[float, float, float, float]:
        """Convert normalized zone coordinates to absolute bbox."""
        x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio = zone
        return (
            x_min_ratio * image_w,
            y_min_ratio * image_h,
            x_max_ratio * image_w,
            y_max_ratio * image_h,
        )
