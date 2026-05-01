"""Standalone CLI for two-stage damage analysis and claim consistency checks."""

from __future__ import annotations

import argparse
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from dotenv import load_dotenv

from consistency import ClaimConsistencyChecker
from fusion import DamageFusionEngine
from models.clip_validator import ClipValidationConfig, ClipValidator
from models.sam_segmenter import SamSegmenter
from models.vit_classifier import ViTDamageClassifier
from utils.fusion import enrich_and_filter_detections
from utils.llm_reasoner import LLMReasoner
from utils.uncertainty import evaluate_uncertainty

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""
    parser = argparse.ArgumentParser(description="Run vehicle damage assessment inference.")
    parser.add_argument("--image", type=str, default="", help="Path to one input image.")
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=[],
        help="Multiple input image paths (space-separated).",
    )
    parser.add_argument("--claim", type=str, required=True, help="Claim text to validate.")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--clip-threshold", type=float, default=0.25, help="CLIP validation threshold.")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for result image.")
    parser.add_argument("--use-sam", action="store_true", help="Enable SAM segmentation if checkpoint is available.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for pipeline stages.",
    )
    return parser.parse_args()


def _consistency_fallback() -> Dict[str, object]:
    """Return default consistency payload when CLIP or checker fails."""
    return {
        "fraud_risk": "UNKNOWN",
        "verdict": "Consistency check unavailable",
        "consistency_score": -1,
        "mismatch_reasons": [],
        "clip_similarity_score": -1.0,
        "detected_zones": [],
    }


def _consistency_no_detection() -> Dict[str, object]:
    """Return conservative consistency payload when no damage is detected."""
    return {
        "fraud_risk": "UNKNOWN",
        "verdict": "No visual damage detected; manual review required.",
        "consistency_score": 0.0,
        "mismatch_reasons": ["No damage detections available for claim validation."],
        "clip_similarity_score": -1.0,
        "detected_zones": [],
    }


def _format_detected_zone(detections: List[Dict[str, object]]) -> str:
    """Format unique part names as report zone text."""
    parts = sorted({str(d.get("part", "")) for d in detections if d.get("part")})
    return ", ".join(parts) if parts else "N/A"


def _pretty_label(label: str) -> str:
    """Convert snake_case labels into display-friendly text."""
    return label.replace("_", " ")


def _overlay_masks(image: np.ndarray, detections: List[Dict[str, object]]) -> np.ndarray:
    """Draw optional SAM masks on top of annotated image."""
    output = image.copy()
    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        color = np.array(det.get("color", [0, 200, 255]), dtype=np.uint8)
        output[mask] = (0.7 * output[mask] + 0.3 * color).astype(np.uint8)
    return output


def _infer_scene_view(claim_text: str) -> str:
    """Infer likely camera/view orientation from claim text hints."""
    text = claim_text.lower()
    if "rear" in text or "back" in text or "trunk" in text:
        return "rear"
    if "front" in text or "hood" in text or "bonnet" in text:
        return "front"
    if "left" in text:
        return "left_side"
    if "right" in text:
        return "right_side"
    return "auto"


def _resolve_input_images(args: argparse.Namespace, root: Path) -> List[Path]:
    """Resolve input image list from --image/--images args."""
    raw_inputs: List[str] = []
    if args.image.strip():
        raw_inputs.append(args.image.strip())
    raw_inputs.extend([p for p in args.images if p.strip()])
    if not raw_inputs:
        raise ValueError("Provide at least one input via --image or --images.")

    resolved: List[Path] = []
    for item in raw_inputs:
        p = Path(item)
        abs_p = p.resolve() if p.is_absolute() else (root / p).resolve()
        if not abs_p.exists():
            raise FileNotFoundError(f"Image file not found: {abs_p}")
        resolved.append(abs_p)
    return resolved


def _risk_rank(level: str) -> int:
    """Map risk label to sortable rank."""
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": -1}.get(level, -1)


def _aggregate_case_summary(results: List[Dict[str, object]], claim_text: str, llm_reasoner: LLMReasoner) -> Dict[str, object]:
    """Aggregate per-image outputs into a single case verdict."""
    if not results:
        return {
            "final_case_risk": "UNKNOWN",
            "avg_decision_confidence": 0.0,
            "uncertain_image_rate": 0.0,
            "reasoning": {},
        }

    risk_levels = [str(r["consistency"].get("fraud_risk", "UNKNOWN")) for r in results]
    final_case_risk = max(risk_levels, key=_risk_rank)
    decision_confs = [float(r["analysis"].get("uncertainty", {}).get("decision_confidence", 0.0)) for r in results]
    uncertain_flags = [bool(r["analysis"].get("uncertainty", {}).get("uncertain", False)) for r in results]
    all_uncertainty_factors: List[str] = []
    all_mismatch_reasons: List[str] = []
    for r in results:
        all_uncertainty_factors.extend(r["analysis"].get("uncertainty", {}).get("uncertainty_factors", []))
        all_mismatch_reasons.extend(r["consistency"].get("mismatch_reasons", []))

    avg_conf = float(np.mean(decision_confs)) if decision_confs else 0.0
    uncertain_rate = float(np.mean(uncertain_flags)) if uncertain_flags else 0.0
    case_payload = {
        "scope": "multi_image_case",
        "claim_text": claim_text,
        "image_count": len(results),
        "risk_levels_per_image": risk_levels,
        "final_case_risk": final_case_risk,
        "avg_decision_confidence": round(avg_conf, 3),
        "uncertain_image_rate": round(uncertain_rate, 3),
        "mismatch_reasons": list(dict.fromkeys(all_mismatch_reasons))[:8],
        "uncertainty_factors": list(dict.fromkeys(all_uncertainty_factors))[:8],
    }
    case_reasoning = llm_reasoner.reason(case_payload)
    return {
        "final_case_risk": final_case_risk,
        "avg_decision_confidence": round(avg_conf, 3),
        "uncertain_image_rate": round(uncertain_rate, 3),
        "reasoning": case_reasoning,
    }


def main() -> None:
    """Execute end-to-end inference, severity estimation, and claim consistency checking."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.log_level.upper() != "DEBUG":
        # Reduce external HTTP/Hub noise while keeping app pipeline logs visible.
        for noisy_logger in ["httpx", "httpcore", "huggingface_hub", "transformers", "transformers.utils.import_utils"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger = logging.getLogger("vehicle_damage_inference")
    weights_path = Path(args.weights).resolve() if Path(args.weights).is_absolute() else (root / args.weights).resolve()
    output_dir = Path(args.output).resolve() if Path(args.output).is_absolute() else (root / args.output).resolve()
    image_paths = _resolve_input_images(args, root)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}. Train model first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Starting multi-model inference pipeline")
        scene_view = _infer_scene_view(args.claim)
        logger.info("Scene view inferred: %s", scene_view)
        fusion_engine = DamageFusionEngine(stage1_model_path=weights_path)
        logger.info("YOLO fusion engine initialized with weights: %s", weights_path)
        clip_validator = ClipValidator(config=ClipValidationConfig(score_threshold=args.clip_threshold))
        logger.info("CLIP validator available: %s", clip_validator.available)
        if not clip_validator.available and clip_validator.initialization_error:
            logger.warning("CLIP fallback active: %s", clip_validator.initialization_error)
        vit_classifier = ViTDamageClassifier()
        logger.info("ViT classifier available: %s", vit_classifier.available)
        if not vit_classifier.available and vit_classifier.initialization_error:
            logger.warning("ViT fallback active: %s", vit_classifier.initialization_error)
        sam_segmenter = SamSegmenter(project_root=root) if args.use_sam else None
        if args.use_sam:
            sam_available = bool(sam_segmenter and sam_segmenter.available)
            logger.info("SAM segmenter available: %s", sam_available)
            if sam_segmenter and (not sam_segmenter.available) and sam_segmenter.initialization_error:
                logger.warning("SAM fallback active: %s", sam_segmenter.initialization_error)
        else:
            logger.info("SAM segmenter disabled by CLI flag")
        checker = ClaimConsistencyChecker()
        llm_reasoner = LLMReasoner()
        all_results: List[Dict[str, object]] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Skipping unreadable image: %s", image_path)
                continue

            analysis_result = fusion_engine.analyze(
                str(image_path),
                confidence_threshold=args.conf,
                scene_view=scene_view,
            )
            raw_detections = analysis_result.get("detections", [])
            logger.info("[%d/%d] YOLO raw detections=%d", idx, len(image_paths), len(raw_detections))
            enhanced_detections = enrich_and_filter_detections(
                image_bgr=image,
                detections=raw_detections,
                clip_validator=clip_validator,
                vit_classifier=vit_classifier,
                sam_segmenter=sam_segmenter,
                clip_threshold=args.clip_threshold,
            )
            analysis_result["detections"] = enhanced_detections
            analysis_result["total_detections"] = len(enhanced_detections)
            analysis_result["overall_score"] = round(
                float(np.mean([d["score"] for d in enhanced_detections])) if enhanced_detections else 0.0,
                2,
            )
            analysis_result["overall_severity"] = (
                max(enhanced_detections, key=lambda d: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(d["severity"]), 0))[
                    "severity"
                ]
                if enhanced_detections
                else "LOW"
            )
            uncertainty = evaluate_uncertainty(enhanced_detections)
            analysis_result["uncertainty"] = uncertainty
            annotated = fusion_engine.draw_results(image, analysis_result)
            annotated = _overlay_masks(annotated, enhanced_detections)

            clip_input_detections = [
                {"label": d["damage_type"], "bbox": tuple(d["damage_bbox"]), "location": d.get("part", "")}
                for d in enhanced_detections
            ]
            consistency = _consistency_fallback()
            if len(enhanced_detections) == 0:
                consistency = _consistency_no_detection()
            else:
                try:
                    consistency = checker.check(str(image_path), args.claim, clip_input_detections)
                except Exception:
                    consistency = _consistency_fallback()
                    logger.exception("Claim consistency failed for %s; using fallback", image_path.name)

            decision_payload = {
                "image_name": image_path.name,
                "claim_text": args.claim,
                "fraud_risk": consistency.get("fraud_risk", "UNKNOWN"),
                "verdict": consistency.get("verdict", ""),
                "mismatch_reasons": consistency.get("mismatch_reasons", []),
                "clip_similarity_score": consistency.get("clip_similarity_score", -1.0),
                "total_detections": len(enhanced_detections),
                "detected_zone": _format_detected_zone(enhanced_detections),
                "overall_severity": analysis_result.get("overall_severity", "LOW"),
                "overall_score": analysis_result.get("overall_score", 0.0),
                "decision_confidence": uncertainty.get("decision_confidence", 0.0),
                "uncertainty_factors": uncertainty.get("uncertainty_factors", []),
                "top_detections": [
                    {
                        "damage_type": d.get("damage_type", ""),
                        "part": d.get("part", ""),
                        "severity": d.get("severity", ""),
                        "confidence": round(float(d.get("confidence", 0.0)), 3),
                    }
                    for d in enhanced_detections[:5]
                ],
            }
            llm_reasoning = llm_reasoner.reason(decision_payload)
            logger.info(
                "[%d/%d] LLM reasoning provider=%s model=%s",
                idx,
                len(image_paths),
                llm_reasoning.get("provider", "unknown"),
                llm_reasoning.get("model", "unknown"),
            )

            out_img = output_dir / f"result_{timestamp}_{idx:03d}.jpg"
            cv2.imwrite(str(out_img), annotated)
            logger.info("[%d/%d] Annotated output saved: %s", idx, len(image_paths), out_img)

            all_results.append(
                {
                    "image_path": str(image_path),
                    "output_image": str(out_img),
                    "analysis": analysis_result,
                    "consistency": consistency,
                    "llm_reasoning": llm_reasoning,
                }
            )

        if not all_results:
            raise RuntimeError("No images were processed successfully.")

        print("========================================")
        print("     VEHICLE DAMAGE ASSESSMENT REPORT")
        print("     Multi-Model AI Analysis System")
        print("========================================")
        print(f"Claim: {args.claim}")
        print(f"Images processed: {len(all_results)}\n")

        for i, item in enumerate(all_results, start=1):
            analysis_result = item["analysis"]
            consistency = item["consistency"]
            llm_reasoning = item["llm_reasoning"]
            detections = analysis_result.get("detections", [])
            uncertainty = analysis_result.get("uncertainty", {})
            print(f"[{i}] Image: {Path(item['image_path']).name}")
            print(f"  Detections: {analysis_result.get('total_detections', 0)} | Severity: {analysis_result.get('overall_severity', 'LOW')} | Score: {analysis_result.get('overall_score', 0.0):.1f}/10")
            print(f"  Fraud Risk: {consistency.get('fraud_risk', 'UNKNOWN')} | Decision Confidence: {float(uncertainty.get('decision_confidence', 0.0)):.2f}")
            print(f"  LLM Verdict: {llm_reasoning.get('final_verdict', 'N/A')}")
            print(f"  LLM Summary: {llm_reasoning.get('summary', 'N/A')}")
            print(f"  Output: {item['output_image']}\n")

        aggregate = {
            "claim_text": args.claim,
            "image_count": len(all_results),
            "avg_decision_confidence": round(
                float(np.mean([float(r['analysis'].get('uncertainty', {}).get('decision_confidence', 0.0)) for r in all_results])),
                3,
            ),
            "fraud_risk_distribution": {},
            "llm_provider_distribution": {},
            "items": all_results,
        }
        for r in all_results:
            risk = str(r["consistency"].get("fraud_risk", "UNKNOWN"))
            aggregate["fraud_risk_distribution"][risk] = aggregate["fraud_risk_distribution"].get(risk, 0) + 1
            provider = str(r["llm_reasoning"].get("provider", "unknown"))
            aggregate["llm_provider_distribution"][provider] = aggregate["llm_provider_distribution"].get(provider, 0) + 1

        case_summary = _aggregate_case_summary(all_results, args.claim, llm_reasoner)
        aggregate["case_summary"] = case_summary
        print("CASE-LEVEL VERDICT:")
        print(f"  Final Case Risk: {case_summary.get('final_case_risk', 'UNKNOWN')}")
        print(f"  Avg Decision Confidence: {float(case_summary.get('avg_decision_confidence', 0.0)):.2f}")
        print(f"  Uncertain Image Rate: {float(case_summary.get('uncertain_image_rate', 0.0)):.2f}")
        case_reasoning = case_summary.get("reasoning", {})
        print(f"  LLM Provider: {case_reasoning.get('provider', 'unknown')} ({case_reasoning.get('model', 'unknown')})")
        print(f"  LLM Case Verdict: {case_reasoning.get('final_verdict', 'N/A')}")
        print(f"  LLM Case Decision: {case_reasoning.get('decision', 'REVIEW')}")

        report_path = output_dir / f"report_{timestamp}.json"
        report_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print("========================================")
        print(f"Structured report saved: {report_path}")
        print(
            "PIPELINE_OK | "
            f"images={len(all_results)} | "
            f"case_risk={case_summary.get('final_case_risk', 'UNKNOWN')} | "
            f"llm={case_reasoning.get('provider', 'unknown')}:{case_reasoning.get('model', 'unknown')}"
        )
        print("========================================")

    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    main()
