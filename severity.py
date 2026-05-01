"""Severity estimation module for vehicle damage detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SeverityConfig:
    """Configuration for severity level computation."""

    low_threshold: float = 0.03
    high_threshold: float = 0.15


class SeverityEstimator:
    """Estimate severity, action, and score for a damage detection."""

    LEVELS = ["LOW", "MEDIUM", "HIGH"]
    ACTIONS = {"LOW": "Monitor", "MEDIUM": "Repair", "HIGH": "Replace"}

    BASE_LEVELS = {
        "scratch": "LOW",
        "dent": "MEDIUM",
        "crack": "MEDIUM",
        "glass_breakage": "HIGH",
        "glass_damage": "HIGH",
        "lamp_breakage": "HIGH",
        "light_damage": "HIGH",
        "tire_flat": "HIGH",
        "tire_damage": "HIGH",
        "structural_damage": "HIGH",
        "missing_part": "HIGH",
    }

    def __init__(self, config: SeverityConfig | None = None) -> None:
        """Initialize the estimator with configurable thresholds."""
        self.config = config or SeverityConfig()

    def estimate(
        self,
        label: str,
        bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, int] | Tuple[int, int, int],
        confidence: float,
    ) -> Dict[str, object]:
        """Estimate severity metadata for one detection.

        Args:
            label: Damage class label.
            bbox: Bounding box as (x1, y1, x2, y2).
            image_shape: Image shape from OpenCV/PIL array.
            confidence: Detector confidence between 0 and 1.

        Returns:
            Dictionary containing severity, action and confidence-adjusted score.
        """
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = bbox
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        box_area = box_w * box_h
        image_area = float(max(1, image_w * image_h))
        area_ratio = box_area / image_area

        normalized_label = label.strip().lower().replace(" ", "_")
        base_level = self.BASE_LEVELS.get(normalized_label, "MEDIUM")
        severity_idx = self.LEVELS.index(base_level)

        if area_ratio > self.config.high_threshold:
            severity_idx = min(severity_idx + 1, len(self.LEVELS) - 1)
        elif area_ratio < self.config.low_threshold:
            severity_idx = max(severity_idx - 1, 0)

        severity = self.LEVELS[severity_idx]
        action = self.ACTIONS[severity]

        # Convert level + confidence into a 0-10 score for easy demo interpretation.
        level_weight = {"LOW": 3.0, "MEDIUM": 6.0, "HIGH": 8.0}[severity]
        area_bonus = min(1.5, max(0.0, area_ratio * 10))
        confidence_bonus = max(0.0, min(2.0, confidence * 2.0))
        score = min(10.0, round(level_weight + area_bonus + confidence_bonus, 2))

        return {
            "severity": severity,
            "action": action,
            "confidence_adjusted_score": score,
            "area_ratio": round(area_ratio, 4),
        }
