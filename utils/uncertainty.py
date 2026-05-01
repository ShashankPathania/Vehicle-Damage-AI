"""Decision confidence and uncertainty assessment utilities."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def evaluate_uncertainty(detections: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute final decision confidence and uncertainty factors."""
    if not detections:
        return {
            "decision_confidence": 0.0,
            "avg_confidence": 0.0,
            "variance_confidence": 0.0,
            "uncertain": True,
            "uncertainty_factors": ["sparse detections"],
        }

    confidences = np.array([float(d.get("confidence", 0.0)) for d in detections], dtype=np.float32)
    avg_conf = float(np.mean(confidences))
    var_conf = float(np.var(confidences))
    decision_conf = float(max(0.0, min(1.0, avg_conf * (1.0 - var_conf))))

    severe_disagreement_count = 0
    moderate_disagreement_count = 0
    for det in detections:
        votes = det.get("model_votes", {})
        vote_values = [str(v) for v in votes.values() if v]
        unique_vote_count = len(set(vote_values))
        det_conf = float(det.get("confidence", 0.0))
        if unique_vote_count >= 3:
            severe_disagreement_count += 1
        elif unique_vote_count == 2 and det_conf < 0.45:
            moderate_disagreement_count += 1

    disagreement_ratio = (
        severe_disagreement_count + 0.5 * moderate_disagreement_count
    ) / max(1, len(detections))

    reasons: List[str] = []
    if avg_conf < 0.35:
        reasons.append("low average confidence")
    if disagreement_ratio > 0.65:
        reasons.append("low agreement between YOLO/CLIP/ViT")
    if len(detections) == 1 and avg_conf < 0.45:
        reasons.append("sparse detections")

    # Mark uncertain only when there is enough signal of instability.
    uncertain_flag = (len(reasons) >= 2) or (decision_conf < 0.30)

    return {
        "decision_confidence": round(decision_conf, 3),
        "avg_confidence": round(avg_conf, 3),
        "variance_confidence": round(var_conf, 3),
        "uncertain": uncertain_flag,
        "uncertainty_factors": reasons,
    }
