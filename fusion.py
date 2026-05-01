"""Two-stage fusion engine combining damage detection and part localization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from part_detector import CarPartDetector
from severity import SeverityEstimator


class DamageFusionEngine:
    """Run two-stage analysis and fuse damage detections with car-part context."""

    def __init__(self, stage1_model_path: str | Path, iou_threshold: float = 0.15) -> None:
        """Initialize stage-1 detector, stage-2 part detector, and severity estimator."""
        model_path = Path(stage1_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Stage 1 model not found: {model_path}")
        self.stage1_model = YOLO(str(model_path))
        self.part_detector = CarPartDetector(use_model=False)
        self.severity_estimator = SeverityEstimator()
        self.iou_threshold = iou_threshold
        self.damage_label_map = {
            "glass shatter": "glass_damage",
            "glass_shatter": "glass_damage",
            "glass breakage": "glass_damage",
            "glass_breakage": "glass_damage",
            "lamp broken": "light_damage",
            "lamp_broken": "light_damage",
            "lamp breakage": "light_damage",
            "lamp_breakage": "light_damage",
            "tire flat": "tire_damage",
            "tire_flat": "tire_damage",
            "broken_glass": "glass_damage",
            "broken_lights": "light_damage",
            "dents": "dent",
            "lost_parts": "missing_part",
            "torn": "structural_damage",
            "punctured": "structural_damage",
        }

    def analyze(
        self,
        image_path: str,
        confidence_threshold: float = 0.35,
        scene_view: str = "auto",
    ) -> Dict[str, object]:
        """Run full two-stage analysis and return structured results."""
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path_obj}")

        image = cv2.imread(str(image_path_obj))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path_obj}")

        results = self.stage1_model.predict(
            source=str(image_path_obj),
            conf=confidence_threshold,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )
        stage1 = results[0] if results else None
        part_regions = self.part_detector.detect_parts(image, scene_view=scene_view)
        detections: List[Dict[str, object]] = []

        if stage1 is not None and stage1.boxes is not None:
            names = stage1.names
            for idx, box in enumerate(stage1.boxes, start=1):
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_damage_type = str(names[cls_id])
                damage_type = self._normalize_damage_label(raw_damage_type)

                best_part_name, best_part_bbox, best_iou = self._match_part((x1, y1, x2, y2), part_regions, image.shape)
                if best_iou <= self.iou_threshold:
                    best_part_name = self.part_detector.get_part_for_damage(
                        (x1, y1, x2, y2),
                        image.shape,
                        scene_view=scene_view,
                    )
                    best_part_bbox = self._part_bbox_from_name(best_part_name, part_regions)

                severity_info = self.severity_estimator.estimate(
                    label=damage_type,
                    bbox=(x1, y1, x2, y2),
                    image_shape=image.shape,
                    confidence=confidence,
                )
                color = self._severity_color(str(severity_info["severity"]))
                detections.append(
                    {
                        "id": idx,
                        "damage_type": damage_type,
                        "part": best_part_name,
                        "severity": severity_info["severity"],
                        "action": severity_info["action"],
                        "score": float(severity_info["confidence_adjusted_score"]),
                        "confidence": confidence,
                        "damage_bbox": [x1, y1, x2, y2],
                        "part_bbox": best_part_bbox,
                        "color": color,
                    }
                )

        overall_severity = self._overall_severity(detections)
        overall_score = round(float(np.mean([d["score"] for d in detections])) if detections else 0.0, 2)
        summary = self._build_summary(detections, overall_severity)
        return {
            "image_path": str(image_path_obj),
            "total_detections": len(detections),
            "detections": detections,
            "overall_severity": overall_severity,
            "overall_score": overall_score,
            "summary": summary,
        }

    def draw_results(self, image: np.ndarray, analysis_result: Dict[str, object]) -> np.ndarray:
        """Draw fused detections with severity colors and global badges."""
        output = image.copy()
        detections = analysis_result.get("detections", [])
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["damage_bbox"]]
            color = tuple(int(v) for v in det["color"])

            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(overlay, 0.15, output, 0.85, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"{det['damage_type']} | {det['part']} | {det['severity']} | {det['score']}"
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        # Detection count badge (top-left)
        count_text = f"Detections: {analysis_result.get('total_detections', 0)}"
        cv2.rectangle(output, (12, 12), (280, 52), (30, 30, 30), -1)
        cv2.putText(output, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        # Overall severity badge (top-right)
        sev = str(analysis_result.get("overall_severity", "LOW"))
        sev_color = tuple(int(v) for v in self._severity_color(sev))
        h, w = output.shape[:2]
        badge_text = f"Overall: {sev}"
        cv2.rectangle(output, (w - 300, 12), (w - 12, 52), sev_color, -1)
        cv2.putText(output, badge_text, (w - 286, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return output

    def _match_part(
        self,
        damage_bbox: Tuple[float, float, float, float],
        part_regions: List[Dict[str, object]],
        image_shape: Tuple[int, int, int],
    ) -> Tuple[str, List[float], float]:
        """Find best-overlap part for a damage box using IoU."""
        best_iou = 0.0
        best_part_name = ""
        best_part_bbox = []
        for part in part_regions:
            part_bbox = tuple(float(v) for v in part["bbox"])
            iou = self._compute_iou(damage_bbox, part_bbox)
            if iou > best_iou:
                best_iou = iou
                best_part_name = str(part["part_name"])
                best_part_bbox = [float(v) for v in part_bbox]
        if best_part_name:
            return best_part_name, best_part_bbox, best_iou
        fallback_part = self.part_detector.get_part_for_damage(damage_bbox, image_shape)
        return fallback_part, self._part_bbox_from_name(fallback_part, part_regions), 0.0

    @staticmethod
    def _part_bbox_from_name(part_name: str, part_regions: List[Dict[str, object]]) -> List[float]:
        """Return part bbox for a part name from inferred regions."""
        for part in part_regions:
            if part.get("part_name") == part_name:
                return [float(v) for v in part.get("bbox", [])]
        return []

    @staticmethod
    def _compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two XYXY boxes."""
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        ix1 = max(x11, x21)
        iy1 = max(y11, y21)
        ix2 = min(x12, x22)
        iy2 = min(y12, y22)
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        area1 = max(0.0, (x12 - x11)) * max(0.0, (y12 - y11))
        area2 = max(0.0, (x22 - x21)) * max(0.0, (y22 - y21))
        union = max(1e-6, area1 + area2 - inter_area)
        return float(inter_area / union)

    @staticmethod
    def _severity_color(severity: str) -> List[int]:
        """Map severity to RGB color."""
        mapping = {"LOW": [0, 200, 0], "MEDIUM": [255, 165, 0], "HIGH": [255, 0, 0]}
        return mapping.get(severity, [128, 128, 128])

    @staticmethod
    def _overall_severity(detections: List[Dict[str, object]]) -> str:
        """Return worst severity over all detections."""
        if not detections:
            return "LOW"
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        worst = max(detections, key=lambda d: rank.get(str(d["severity"]), -1))
        return str(worst["severity"])

    @staticmethod
    def _build_summary(detections: List[Dict[str, object]], overall_severity: str) -> str:
        """Build human-readable summary string."""
        if not detections:
            return "No damage detected. Vehicle appears visually normal in the analyzed view."
        top = max(detections, key=lambda d: float(d["score"]))
        return (
            f"{len(detections)} damage zones detected. Most severe: "
            f"{top['damage_type']} on {top['part']} ({overall_severity}). "
            f"Recommended action: {top['action']}."
        )

    def _normalize_damage_label(self, label: str) -> str:
        """Normalize model label names into unified project class names."""
        cleaned = label.strip().lower().replace("-", "_")
        mapped = self.damage_label_map.get(cleaned, cleaned)
        return mapped
