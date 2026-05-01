"""Post-processing and multi-model fusion utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from severity import SeverityEstimator


LABEL_NORMALIZATION = {
    "glass shatter": "glass_damage",
    "glass_breakage": "glass_damage",
    "tire_flat": "tire_damage",
    "lamp_broken": "light_damage",
    "broken": "structural_damage",
}


def normalize_label(label: str) -> str:
    """Map labels to canonical class names."""
    cleaned = label.strip().lower().replace("-", "_")
    return LABEL_NORMALIZATION.get(cleaned, cleaned)


def iou_xyxy(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU for two XYXY boxes."""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area1 = max(0.0, (x12 - x11)) * max(0.0, (y12 - y11))
    area2 = max(0.0, (x22 - x21)) * max(0.0, (y22 - y21))
    union = max(1e-6, area1 + area2 - inter)
    return float(inter / union)


def merge_overlapping_boxes(detections: List[Dict[str, object]], iou_threshold: float = 0.5) -> List[Dict[str, object]]:
    """Merge highly overlapping detections by confidence-weighted averaging."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    merged: List[Dict[str, object]] = []
    used = set()
    for i, det in enumerate(sorted_dets):
        if i in used:
            continue
        group = [det]
        box_i = tuple(float(v) for v in det["damage_bbox"])
        used.add(i)
        for j in range(i + 1, len(sorted_dets)):
            if j in used:
                continue
            candidate = sorted_dets[j]
            box_j = tuple(float(v) for v in candidate["damage_bbox"])
            if iou_xyxy(box_i, box_j) > iou_threshold:
                group.append(candidate)
                used.add(j)

        if len(group) == 1:
            merged.append(group[0])
            continue

        weights = np.array([max(1e-6, float(g.get("confidence", 0.0))) for g in group], dtype=np.float32)
        coords = np.array([g["damage_bbox"] for g in group], dtype=np.float32)
        weighted_coords = (coords * weights[:, None]).sum(axis=0) / weights.sum()
        best = max(group, key=lambda g: float(g.get("confidence", 0.0)))
        best["damage_bbox"] = [float(v) for v in weighted_coords.tolist()]
        best["confidence"] = float(max(float(g.get("confidence", 0.0)) for g in group))
        merged.append(best)
    return merged


def _majority_vote(labels: List[str]) -> str:
    """Return majority label over non-empty votes."""
    votes = [normalize_label(v) for v in labels if v]
    if not votes:
        return ""
    counts: Dict[str, int] = {}
    for label in votes:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _weighted_label_choice(yolo_label: str, clip_label: str, vit_label: str) -> str:
    """Select label by weighted score when majority vote ties."""
    weights = {"yolo": 0.5, "clip": 0.3, "vit": 0.2}
    score: Dict[str, float] = {}
    for source, label in [("yolo", yolo_label), ("clip", clip_label), ("vit", vit_label)]:
        if not label:
            continue
        canonical = normalize_label(label)
        score[canonical] = score.get(canonical, 0.0) + weights[source]
    if not score:
        return normalize_label(yolo_label)
    return max(score.items(), key=lambda kv: kv[1])[0]


def enrich_and_filter_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, object]],
    clip_validator: object,
    vit_classifier: object,
    sam_segmenter: object,
    clip_threshold: float = 0.25,
    min_area_ratio: float = 0.0008,
    min_confidence: float = 0.30,
    merge_iou_threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """Apply CLIP/ViT/SAM enhancement, filtering, and confidence fusion."""
    h, w = image_bgr.shape[:2]
    img_area = float(max(1, h * w))
    severity_estimator = SeverityEstimator()
    enriched: List[Dict[str, object]] = []

    for det in detections:
        bbox = tuple(float(v) for v in det["damage_bbox"])
        x1, y1, x2, y2 = bbox
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / img_area)
        yolo_label = normalize_label(str(det.get("damage_type", "")))
        yolo_conf = float(det.get("confidence", 0.0))

        clip_out = clip_validator.validate_crop(image_bgr, bbox) if clip_validator is not None else {}
        clip_label = normalize_label(str(clip_out.get("clip_label", ""))) if clip_out else ""
        clip_score = float(clip_out.get("clip_score", 0.0)) if clip_out else 0.0
        if clip_out and clip_out.get("clip_available", False) and clip_score < clip_threshold:
            continue

        vit_out = vit_classifier.classify_crop(image_bgr, bbox) if vit_classifier is not None else {}
        vit_label = normalize_label(str(vit_out.get("vit_label", ""))) if vit_out else ""
        vit_conf = float(vit_out.get("vit_score", 0.0)) if vit_out else 0.0

        vote_label = _majority_vote([yolo_label, clip_label, vit_label])
        weighted_label = _weighted_label_choice(yolo_label, clip_label, vit_label)
        final_label = vote_label or weighted_label or yolo_label

        final_conf = float(0.5 * yolo_conf + 0.3 * clip_score + 0.2 * vit_conf)
        if box_area_ratio < min_area_ratio and final_conf < min_confidence:
            continue

        sam_out = sam_segmenter.segment_bbox(image_bgr[:, :, ::-1], bbox) if sam_segmenter is not None else {}
        mask_ratio = float(sam_out.get("mask_area_ratio", 0.0)) if sam_out else 0.0
        severity_info = severity_estimator.estimate(
            label=final_label,
            bbox=bbox,
            image_shape=image_bgr.shape,
            confidence=final_conf,
        )
        severity_info["confidence_adjusted_score"] = min(
            10.0,
            round(float(severity_info["confidence_adjusted_score"]) + min(2.0, mask_ratio * 15.0), 2),
        )

        det_copy = dict(det)
        det_copy["damage_type"] = final_label
        det_copy["fusion_label"] = final_label
        det_copy["model_votes"] = {"yolo": yolo_label, "clip": clip_label, "vit": vit_label}
        det_copy["model_agreement"] = len({v for v in [yolo_label, clip_label, vit_label] if v})
        det_copy["confidence"] = final_conf
        det_copy["yolo_confidence"] = yolo_conf
        det_copy["clip_score"] = clip_score
        det_copy["vit_confidence"] = vit_conf
        det_copy["clip_available"] = bool(clip_out.get("clip_available", False)) if clip_out else False
        det_copy["vit_available"] = bool(vit_out.get("vit_available", False)) if vit_out else False
        det_copy["sam_available"] = bool(sam_out.get("sam_available", False)) if sam_out else False
        det_copy["mask"] = sam_out.get("mask") if sam_out else None
        det_copy["mask_area_ratio"] = mask_ratio
        det_copy["severity"] = severity_info["severity"]
        det_copy["action"] = severity_info["action"]
        det_copy["score"] = severity_info["confidence_adjusted_score"]
        enriched.append(det_copy)

    merged = merge_overlapping_boxes(enriched, iou_threshold=merge_iou_threshold)
    for idx, det in enumerate(merged, start=1):
        det["id"] = idx
    return merged
