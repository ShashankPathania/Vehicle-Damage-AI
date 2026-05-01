"""Claim consistency checker using CLIP for text-image alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import clip
except Exception:  # pragma: no cover - fallback for missing dependency.
    clip = None


@dataclass
class ClaimConsistencyConfig:
    """Configuration values for consistency decision thresholds."""

    clip_threshold: float = 0.22


class ClaimConsistencyChecker:
    """Validate whether a textual claim matches visual vehicle damage evidence."""

    LOCATION_KEYWORDS = {
        "front": {"front", "bumper", "hood"},
        "rear": {"rear", "back", "trunk"},
        "left": {"left", "left side", "left door", "door"},
        "right": {"right", "right side", "right door", "door"},
        "top": {"roof", "windshield", "hood"},
        "bottom": {"bumper", "tire", "flat"},
    }

    DAMAGE_KEYWORDS = {
        "scratch": {"scratch", "scratched"},
        "dent": {"dent", "dented", "bent"},
        "crack": {"crack", "cracked"},
        "glass_breakage": {"broken glass", "shattered", "broken windshield", "glass"},
        "lamp_breakage": {"broken lamp", "headlight", "taillight", "lamp"},
        "tire_flat": {"flat tire", "tire flat", "flat"},
    }
    PART_TO_ZONES = {
        "front_bumper": ["front", "bottom"],
        "rear_bumper": ["rear", "bottom"],
        "hood": ["front", "top"],
        "trunk": ["rear"],
        "windshield": ["front", "top"],
        "rear_windshield": ["rear", "top"],
        "roof": ["top"],
        "door_front_left": ["front", "left"],
        "door_front_right": ["front", "right"],
        "door_rear_left": ["rear", "left"],
        "door_rear_right": ["rear", "right"],
        "fender_left": ["left"],
        "fender_right": ["right"],
        "headlight_left": ["front", "left"],
        "headlight_right": ["front", "right"],
    }

    def __init__(self, device: str | None = None, config: ClaimConsistencyConfig | None = None) -> None:
        """Load CLIP model and prepare threshold config."""
        self.config = config or ClaimConsistencyConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.initialization_error = ""
        try:
            if clip is None:
                raise ImportError("CLIP package is not installed.")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as exc:
            self.initialization_error = str(exc)

    def _extract_claim_tokens(self, claim_text: str) -> Dict[str, List[str]]:
        """Extract location and damage-type mentions from claim text."""
        text = claim_text.lower()
        found_locations: List[str] = []
        found_damage_types: List[str] = []

        for canonical, keys in self.LOCATION_KEYWORDS.items():
            if any(keyword in text for keyword in keys):
                found_locations.append(canonical)

        for canonical, keys in self.DAMAGE_KEYWORDS.items():
            if any(keyword in text for keyword in keys):
                found_damage_types.append(canonical)

        return {"locations": sorted(set(found_locations)), "damage_types": sorted(set(found_damage_types))}

    def _bbox_to_zone(self, bbox: Tuple[float, float, float, float], image_shape: Tuple[int, int, int]) -> str:
        """Map a detection bounding box to a coarse vehicle zone."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        horizontal = "left" if cx < (w / 2.0) else "right"
        vertical = "front" if cy < (h / 2.0) else "rear"

        if cy < h * 0.25:
            return "top"
        if cy > h * 0.75:
            return "bottom"
        if abs(cx - (w / 2.0)) < w * 0.12:
            return vertical
        return horizontal

    def _build_probes(self, detections: Sequence[Dict[str, object]]) -> List[str]:
        """Generate text probes from detected zones and labels."""
        probes = [
            "a car with damage on the front bumper",
            "a car with damage on the rear",
            "a car with damage on the left side door",
            "a car with damage on the right side",
            "a car with a scratched surface",
            "a car with a dented body",
            "a car with broken glass",
            "a car with a cracked panel",
            "a car with broken lamp",
            "a car with a flat tire",
        ]
        for det in detections:
            label = str(det.get("label", "damage")).replace("_", " ")
            location = str(det.get("location", "body"))
            probes.append(f"a car with {label} on the {location}")
        return list(dict.fromkeys(probes))

    def _clip_similarity(self, image_path: Path, probes: Sequence[str]) -> float:
        """Compute maximum CLIP cosine similarity between image and probe set."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(list(probes)).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        return float(np.max(similarities))

    def check(self, image_path: str, claim_text: str, detections: Sequence[Dict[str, object]]) -> Dict[str, object]:
        """Check claim consistency against detections and CLIP semantics."""
        try:
            if self.model is None or self.preprocess is None:
                raise RuntimeError(self.initialization_error or "CLIP model is unavailable.")

            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image not found: {image_path_obj}")

            image = cv2.imread(str(image_path_obj))
            if image is None:
                raise ValueError(f"Unable to read image: {image_path_obj}")

            claim_entities = self._extract_claim_tokens(claim_text)
            detected_zones = []
            for det in detections:
                location = str(det.get("location", "")).strip().lower()
                if location:
                    mapped = self.PART_TO_ZONES.get(location, [])
                    if mapped:
                        detected_zones.extend(mapped)
                        continue

                bbox = det.get("bbox")
                if bbox is not None:
                    zone = self._bbox_to_zone(tuple(bbox), image.shape)
                    detected_zones.append(zone)

            detected_zones = sorted(set(detected_zones))
            claimed_locations = claim_entities["locations"]

            location_mismatch = False
            mismatch_reasons: List[str] = []
            if claimed_locations and detected_zones:
                overlaps = set(claimed_locations).intersection(set(detected_zones))
                location_mismatch = len(overlaps) == 0
                if location_mismatch:
                    mismatch_reasons.append(
                        f"Claimed location(s) {claimed_locations} do not match detected zone(s) {detected_zones}."
                    )

            probes = self._build_probes(detections)
            claim_probes = [f"a car with {claim_text.strip().lower()}"]
            similarity = self._clip_similarity(image_path_obj, probes + claim_probes)
            clip_mismatch = similarity < self.config.clip_threshold
            if clip_mismatch:
                mismatch_reasons.append(
                    f"CLIP similarity {similarity:.2f} is below threshold {self.config.clip_threshold:.2f}."
                )

            if location_mismatch and clip_mismatch:
                fraud_risk = "HIGH"
                consistency_score = 0.2
                verdict = "Claim appears inconsistent with visual evidence. Manual inspection strongly recommended."
            elif location_mismatch or clip_mismatch:
                fraud_risk = "MEDIUM"
                consistency_score = 0.5
                verdict = "Claim is partially consistent but contains suspicious mismatch signals."
            else:
                fraud_risk = "LOW"
                consistency_score = 0.85
                verdict = "Claim is consistent with detected visual damage patterns."

            return {
                "consistency_score": float(consistency_score),
                "fraud_risk": fraud_risk,
                "mismatch_reasons": mismatch_reasons,
                "verdict": verdict,
                "clip_similarity_score": float(similarity),
                "detected_zones": detected_zones,
                "claimed_zones": claimed_locations,
            }
        except Exception:
            return {
                "fraud_risk": "UNKNOWN",
                "verdict": "Consistency check unavailable",
                "consistency_score": -1,
                "mismatch_reasons": [],
                "clip_similarity_score": -1.0,
                "detected_zones": [],
                "claimed_zones": [],
            }
