"""Streamlit UI for two-stage damage assessment and claim validation."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
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

LOGGER = logging.getLogger("vehicle_damage_app")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for noisy_logger in ["httpx", "httpcore", "huggingface_hub", "transformers", "transformers.utils.import_utils"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def inject_modern_theme() -> None:
    """Inject a modern visual theme for Streamlit dashboard."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(125, 90, 255, 0.15), transparent 35%),
                radial-gradient(circle at 90% 20%, rgba(0, 180, 255, 0.12), transparent 35%),
                linear-gradient(180deg, #0b1020 0%, #10172a 40%, #121a2f 100%);
            color: #e8ebf5;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e152b 0%, #111a33 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .hero {
            padding: 20px 24px;
            margin-bottom: 16px;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(46, 77, 255, 0.45), rgba(89, 153, 255, 0.25));
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
        }

        .hero h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            color: #f5f7ff;
        }

        .hero p {
            margin: 6px 0 0 0;
            color: #cdd6ef;
            font-size: 0.97rem;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(4px);
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 12px;
            min-height: 84px;
        }

        div[data-testid="stMetricLabel"] {
            color: #b8c5eb;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #f3f6ff;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            background: linear-gradient(135deg, rgba(77, 123, 255, 0.95), rgba(67, 182, 255, 0.85)) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        .small-note {
            color: #b8c5eb;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render top hero section."""
    st.markdown(
        """
        <div class="hero">
            <h1>VehicleScan AI</h1>
            <p>Multi-model vehicle damage analysis with fusion confidence and uncertainty insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_component_logs(
    clip_validator: ClipValidator,
    vit_classifier: ViTDamageClassifier,
    sam_segmenter: SamSegmenter | None,
    use_sam: bool,
    raw_count: int,
    final_count: int,
    uncertainty: Dict[str, object],
    consistency: Dict[str, object],
    llm_reasoning: Dict[str, object],
) -> List[str]:
    """Generate user-friendly component execution logs."""
    llm_provider = str(llm_reasoning.get("provider", "unknown"))
    llm_model = str(llm_reasoning.get("model", "unknown"))
    lines = [
        f"[INIT] CLIP validator: {'READY' if clip_validator.available else 'UNAVAILABLE'}",
        f"[INIT] ViT classifier: {'READY' if vit_classifier.available else 'UNAVAILABLE'}",
        (
            "[INIT] SAM segmenter: DISABLED (toggle off)"
            if not use_sam
            else f"[INIT] SAM segmenter: {'READY' if bool(sam_segmenter and sam_segmenter.available) else 'UNAVAILABLE'}"
        ),
        f"[YOLO] Raw detections found: {raw_count}",
        f"[FUSION] Detections after validation/fusion: {final_count}",
        (
            "[UNCERTAINTY] "
            f"Decision confidence={float(uncertainty.get('decision_confidence', 0.0)):.3f}, "
            f"uncertain={'YES' if bool(uncertainty.get('uncertain', False)) else 'NO'}"
        ),
        f"[CONSISTENCY] Fraud risk from claim checker: {consistency.get('fraud_risk', 'UNKNOWN')}",
        f"[LLM] Provider={llm_provider} | Model={llm_model}",
    ]
    if not clip_validator.available and clip_validator.initialization_error:
        lines.append(f"[WARN] CLIP unavailable, fallback active: {clip_validator.initialization_error}")
    if not vit_classifier.available and vit_classifier.initialization_error:
        lines.append(f"[WARN] ViT unavailable, fallback active: {vit_classifier.initialization_error}")
    if use_sam and sam_segmenter is not None and (not sam_segmenter.available) and sam_segmenter.initialization_error:
        lines.append(f"[WARN] SAM unavailable: {sam_segmenter.initialization_error}")
        lines.append("[HINT] Install `segment-anything` and add checkpoint at `models/sam_vit_b_01ec64.pth`.")
    provider_errors = llm_reasoning.get("provider_errors", [])
    if provider_errors:
        for err in provider_errors:
            lines.append(f"[WARN] LLM provider fallback reason: {err}")
    return lines


def pretty_label(label: str) -> str:
    """Convert snake_case labels into display-friendly text."""
    return label.replace("_", " ")


def infer_scene_view(claim_text: str) -> str:
    """Infer likely scene orientation from user claim text."""
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


def _consistency_no_detection() -> Dict[str, object]:
    """Conservative consistency output when no detections are present."""
    return {
        "fraud_risk": "UNKNOWN",
        "verdict": "No visual damage detected; manual review required.",
        "consistency_score": 0.0,
        "mismatch_reasons": ["No damage detections available for claim validation."],
        "clip_similarity_score": -1.0,
    }


def load_fusion_engine(weights_path: Path) -> DamageFusionEngine:
    """Load fusion engine and fail clearly if weights are missing."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return DamageFusionEngine(stage1_model_path=weights_path)


@st.cache_resource(show_spinner=False)
def load_clip_validator(threshold: float) -> ClipValidator:
    """Load reusable CLIP validator."""
    return ClipValidator(config=ClipValidationConfig(score_threshold=threshold))


@st.cache_resource(show_spinner=False)
def load_vit_classifier() -> ViTDamageClassifier:
    """Load reusable ViT classifier."""
    return ViTDamageClassifier()


@st.cache_resource(show_spinner=False)
def load_sam_segmenter(root_path: str, enabled: bool) -> SamSegmenter | None:
    """Load optional SAM segmenter."""
    if not enabled:
        return None
    return SamSegmenter(project_root=Path(root_path))


def badge_html(level: str) -> str:
    """Return HTML for fraud-risk badge."""
    colors = {
        "LOW": ("#0c8f45", "white", ""),
        "MEDIUM": ("#cc7a00", "white", ""),
        "HIGH": ("#b00020", "white", "⚠️ "),
    }
    bg, fg, prefix = colors.get(level, ("#333333", "white", ""))
    return (
        f"<span style='background:{bg};color:{fg};padding:6px 10px;border-radius:8px;"
        f"font-weight:600'>{prefix}{level}</span>"
    )


def build_report_text(claim_text: str, results: List[Dict[str, object]], case_summary: Dict[str, object]) -> str:
    """Build downloadable text report for multi-image analysis."""
    avg_conf = (
        float(np.mean([float(r["uncertainty"].get("decision_confidence", 0.0)) for r in results]))
        if results
        else 0.0
    )
    lines = [
        "========================================",
        "     VEHICLE DAMAGE ASSESSMENT REPORT",
        "     Multi-Model AI Analysis System",
        "========================================",
        f'Claim Submitted: "{claim_text}"',
        f"Images Processed: {len(results)}",
        f"Average Decision Confidence: {avg_conf:.2f}",
        "",
    ]
    for idx, item in enumerate(results, start=1):
        detections = item["detections"]
        consistency = item["consistency"]
        uncertainty = item["uncertainty"]
        llm_reasoning = item["llm_reasoning"]
        overall_score = round(float(np.mean([d["score"] for d in detections])) if detections else 0.0, 2)
        overall_severity = "LOW"
        if detections:
            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            overall_severity = max(detections, key=lambda d: rank.get(str(d["severity"]), 0))["severity"]
        lines.append(
            f"[{idx}] Image: {item['image_name']} | Detections: {len(detections)} | Fraud Risk: {consistency['fraud_risk']}"
        )
        lines.append(f"  Overall Vehicle Condition: {overall_severity}")
        lines.append(f"  Overall Damage Score: {overall_score} / 10")
        lines.append(
            f"  CLIP Similarity Score: {consistency.get('clip_similarity_score', -1):.2f}"
            if consistency.get("clip_similarity_score", -1) >= 0
            else "  CLIP Similarity Score: unavailable"
        )
        lines.append(f"  Mismatch Reasons: {', '.join(consistency['mismatch_reasons']) or 'None'}")
        lines.append(f"  Verdict: {consistency['verdict']}")
        lines.append(f"  Decision Confidence: {uncertainty.get('decision_confidence', 0.0):.2f}")
        lines.append(f"  Uncertainty Factors: {', '.join(uncertainty.get('uncertainty_factors', [])) or 'None'}")
        lines.append(f"  LLM Provider: {llm_reasoning.get('provider', 'unknown')}")
        lines.append(f"  LLM Final Verdict: {llm_reasoning.get('final_verdict', 'N/A')}")
        lines.append(f"  LLM Summary: {llm_reasoning.get('summary', 'N/A')}")
        lines.append("")

    lines.append("CASE-LEVEL VERDICT:")
    lines.append(f"  Final Case Risk: {case_summary.get('final_case_risk', 'UNKNOWN')}")
    lines.append(f"  Avg Decision Confidence: {float(case_summary.get('avg_decision_confidence', 0.0)):.2f}")
    lines.append(f"  Uncertain Image Rate: {float(case_summary.get('uncertain_image_rate', 0.0)):.2f}")
    reasoning = case_summary.get("reasoning", {})
    lines.append(f"  LLM Provider: {reasoning.get('provider', 'unknown')} ({reasoning.get('model', 'unknown')})")
    lines.append(f"  LLM Case Verdict: {reasoning.get('final_verdict', 'N/A')}")
    lines.append(f"  LLM Case Summary: {reasoning.get('summary', 'N/A')}")
    lines.append("========================================")
    return "\n".join(lines)


def _risk_rank(level: str) -> int:
    """Map risk label to sortable rank."""
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": -1}.get(level, -1)


def build_case_summary(results: List[Dict[str, object]], claim_text: str, llm_reasoner: LLMReasoner) -> Dict[str, object]:
    """Build aggregated verdict across all uploaded images."""
    if not results:
        return {
            "final_case_risk": "UNKNOWN",
            "avg_decision_confidence": 0.0,
            "uncertain_image_rate": 0.0,
            "reasoning": {},
        }
    risk_levels = [str(r["consistency"].get("fraud_risk", "UNKNOWN")) for r in results]
    final_case_risk = max(risk_levels, key=_risk_rank)
    decision_confs = [float(r["uncertainty"].get("decision_confidence", 0.0)) for r in results]
    uncertain_flags = [bool(r["uncertainty"].get("uncertain", False)) for r in results]
    all_uncertainty_factors: List[str] = []
    all_mismatch_reasons: List[str] = []
    for r in results:
        all_uncertainty_factors.extend(r["uncertainty"].get("uncertainty_factors", []))
        all_mismatch_reasons.extend(r["consistency"].get("mismatch_reasons", []))

    payload = {
        "scope": "multi_image_case",
        "claim_text": claim_text,
        "image_count": len(results),
        "risk_levels_per_image": risk_levels,
        "final_case_risk": final_case_risk,
        "avg_decision_confidence": round(float(np.mean(decision_confs)) if decision_confs else 0.0, 3),
        "uncertain_image_rate": round(float(np.mean(uncertain_flags)) if uncertain_flags else 0.0, 3),
        "mismatch_reasons": list(dict.fromkeys(all_mismatch_reasons))[:8],
        "uncertainty_factors": list(dict.fromkeys(all_uncertainty_factors))[:8],
    }
    reasoning = llm_reasoner.reason(payload)
    return {
        "final_case_risk": final_case_risk,
        "avg_decision_confidence": payload["avg_decision_confidence"],
        "uncertain_image_rate": payload["uncertain_image_rate"],
        "reasoning": reasoning,
    }


def main() -> None:
    """Render and run the Streamlit analysis workflow."""
    st.set_page_config(page_title="VehicleScan AI", layout="wide")
    inject_modern_theme()
    render_hero()
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    weights_path = root / "models" / "best.pt"

    st.sidebar.title("VehicleScan AI")
    st.sidebar.caption("Modern Multi-Model Damage Assessment")
    uploaded_files = st.sidebar.file_uploader(
        "Upload vehicle image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    claim_text = st.sidebar.text_area(
        "Describe the damage in your claim",
        placeholder="Example: rear bumper scratched in parking lot incident",
        height=120,
    )
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    clip_threshold = st.sidebar.slider("CLIP validation threshold", 0.1, 0.7, 0.25, 0.05)
    use_sam = st.sidebar.checkbox("Enable SAM segmentation (optional)", value=False)
    st.sidebar.markdown('<p class="small-note">Tip: keep confidence between 0.30-0.45 for balanced precision/recall.</p>', unsafe_allow_html=True)
    run_button = st.sidebar.button("Analyze", type="primary")

    if run_button:
        if not uploaded_files:
            st.error("Please upload at least one image before running analysis.")
            return
        if not claim_text.strip():
            st.error("Please enter a claim description.")
            return

        try:
            fusion_engine = load_fusion_engine(weights_path)
            clip_validator = load_clip_validator(clip_threshold)
            vit_classifier = load_vit_classifier()
            sam_segmenter = load_sam_segmenter(str(root), use_sam)
            llm_reasoner = LLMReasoner()
        except Exception as exc:
            st.error(f"Model initialization failed: {exc}")
            return

        try:
            checker = ClaimConsistencyChecker()
            scene_view = infer_scene_view(claim_text)
            results_bundle: List[Dict[str, object]] = []
            all_table_rows: List[Dict[str, object]] = []
            tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_files))])

            for i, uploaded in enumerate(uploaded_files):
                file_bytes = uploaded.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(file_bytes)
                    temp_image_path = Path(tmp_file.name)

                original = Image.open(temp_image_path).convert("RGB")
                image_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
                analysis_result = fusion_engine.analyze(
                    str(temp_image_path),
                    confidence_threshold=conf_threshold,
                    scene_view=scene_view,
                )
                raw_detection_count = len(analysis_result.get("detections", []))
                detections = enrich_and_filter_detections(
                    image_bgr=image_bgr,
                    detections=analysis_result.get("detections", []),
                    clip_validator=clip_validator,
                    vit_classifier=vit_classifier,
                    sam_segmenter=sam_segmenter,
                    clip_threshold=clip_threshold,
                )
                analysis_result["detections"] = detections
                analysis_result["total_detections"] = len(detections)
                analysis_result["overall_score"] = round(
                    float(np.mean([d["score"] for d in detections])) if detections else 0.0,
                    2,
                )
                analysis_result["overall_severity"] = (
                    max(detections, key=lambda d: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(d["severity"]), 0))[
                        "severity"
                    ]
                    if detections
                    else "LOW"
                )
                uncertainty = evaluate_uncertainty(detections)

                annotated_bgr = fusion_engine.draw_results(image_bgr, analysis_result)
                for det in detections:
                    if det.get("mask") is None:
                        continue
                    color = np.array(det.get("color", [0, 200, 255]), dtype=np.uint8)
                    mask = det["mask"]
                    annotated_bgr[mask] = (0.7 * annotated_bgr[mask] + 0.3 * color).astype(np.uint8)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                consistency = {
                    "fraud_risk": "UNKNOWN",
                    "verdict": "Consistency check unavailable",
                    "consistency_score": -1,
                    "mismatch_reasons": [],
                    "clip_similarity_score": -1.0,
                }
                if len(detections) == 0:
                    consistency = _consistency_no_detection()
                elif claim_text.strip():
                    try:
                        clip_detections = [
                            {
                                "label": det["damage_type"],
                                "bbox": tuple(det["damage_bbox"]),
                                "location": det["part"],
                            }
                            for det in detections
                        ]
                        consistency = checker.check(str(temp_image_path), claim_text, clip_detections)
                    except Exception:
                        LOGGER.exception("Consistency stage failed for %s; fallback used", uploaded.name)

                decision_payload = {
                    "image_name": uploaded.name,
                    "claim_text": claim_text,
                    "fraud_risk": consistency.get("fraud_risk", "UNKNOWN"),
                    "verdict": consistency.get("verdict", ""),
                    "mismatch_reasons": consistency.get("mismatch_reasons", []),
                    "clip_similarity_score": consistency.get("clip_similarity_score", -1.0),
                    "total_detections": len(detections),
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
                        for d in detections[:5]
                    ],
                }
                llm_reasoning = llm_reasoner.reason(decision_payload)
                component_logs = _build_component_logs(
                    clip_validator=clip_validator,
                    vit_classifier=vit_classifier,
                    sam_segmenter=sam_segmenter,
                    use_sam=use_sam,
                    raw_count=raw_detection_count,
                    final_count=len(detections),
                    uncertainty=uncertainty,
                    consistency=consistency,
                    llm_reasoning=llm_reasoning,
                )

                with tabs[i]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="section-card"><h3 style="margin-top:0;">Original Image</h3></div>', unsafe_allow_html=True)
                        st.image(original, width="stretch")
                    with col2:
                        st.markdown('<div class="section-card"><h3 style="margin-top:0;">Annotated Detection</h3></div>', unsafe_allow_html=True)
                        st.image(annotated_rgb, width="stretch")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Detections", analysis_result.get("total_detections", 0))
                    m2.metric("Overall Severity", analysis_result.get("overall_severity", "LOW"))
                    m3.metric("Damage Score", f"{analysis_result.get('overall_score', 0.0):.2f}/10")
                    m4.metric("Fraud Risk", consistency.get("fraud_risk", "UNKNOWN"))
                    c1, c2 = st.columns(2)
                    c1.metric("Decision Confidence", f"{uncertainty.get('decision_confidence', 0.0):.2f}")
                    c2.metric("Uncertain", "YES" if uncertainty.get("uncertain") else "NO")
                    st.markdown("### LLM Reasoning")
                    st.write(f"**Provider:** {llm_reasoning.get('provider', 'unknown')} ({llm_reasoning.get('model', 'unknown')})")
                    st.write(f"**Final Verdict:** {llm_reasoning.get('final_verdict', 'N/A')}")
                    st.write(f"**Decision:** {llm_reasoning.get('decision', 'REVIEW')}")
                    st.write(f"**Summary:** {llm_reasoning.get('summary', 'N/A')}")
                    if llm_reasoning.get("recommended_actions"):
                        st.write("**Recommended Actions:**")
                        for act in llm_reasoning.get("recommended_actions", []):
                            st.markdown(f"- {act}")
                    with st.expander("Pipeline Runtime Logs", expanded=False):
                        st.code("\n".join(component_logs), language="text")

                for row_idx, det in enumerate(detections, start=1):
                    all_table_rows.append(
                        {
                            "Image": uploaded.name,
                            "#": row_idx,
                            "Damage Type": pretty_label(str(det["damage_type"])),
                            "Car Part": pretty_label(str(det["part"])),
                            "Severity": det["severity"],
                            "Action": det["action"],
                            "Score": det["score"],
                            "Final Label": pretty_label(str(det.get("fusion_label", det["damage_type"]))),
                            "Confidence": round(float(det["confidence"]), 3),
                            "YOLO/CLIP/ViT": f"{det.get('yolo_confidence', 0.0):.2f}/{det.get('clip_score', 0.0):.2f}/{det.get('vit_confidence', 0.0):.2f}",
                        }
                    )

                results_bundle.append(
                    {
                        "image_name": uploaded.name,
                        "detections": detections,
                        "consistency": consistency,
                        "uncertainty": uncertainty,
                        "llm_reasoning": llm_reasoning,
                    }
                )

            st.markdown('<div class="section-card"><h3 style="margin-top:0;">All Detections (Across Images)</h3></div>', unsafe_allow_html=True)
            df = pd.DataFrame(all_table_rows) if all_table_rows else pd.DataFrame(
                columns=["Image", "#", "Damage Type", "Final Label", "Car Part", "Severity", "Action", "Score", "Confidence", "YOLO/CLIP/ViT"]
            )
            def _severity_style(value: object) -> str:
                mapping = {"LOW": "color: #0c8f45;", "MEDIUM": "color: #cc7a00;", "HIGH": "color: #b00020;"}
                return mapping.get(str(value), "")
            if not df.empty:
                st.dataframe(df.style.map(_severity_style, subset=["Severity"]), width="stretch")
            else:
                st.dataframe(df, width="stretch")

            case_summary = build_case_summary(results_bundle, claim_text, llm_reasoner)
            case_reasoning = case_summary.get("reasoning", {})
            st.markdown('<div class="section-card"><h3 style="margin-top:0;">Final Case Verdict</h3></div>', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Case Risk", case_summary.get("final_case_risk", "UNKNOWN"))
            k2.metric("Avg Decision Confidence", f"{float(case_summary.get('avg_decision_confidence', 0.0)):.2f}")
            k3.metric("Uncertain Image Rate", f"{float(case_summary.get('uncertain_image_rate', 0.0)):.2f}")
            st.write(f"**LLM Provider:** {case_reasoning.get('provider', 'unknown')} ({case_reasoning.get('model', 'unknown')})")
            st.write(f"**Case Verdict:** {case_reasoning.get('final_verdict', 'N/A')}")
            st.write(f"**Case Decision:** {case_reasoning.get('decision', 'REVIEW')}")
            st.write(f"**Case Summary:** {case_reasoning.get('summary', 'N/A')}")
            if case_reasoning.get("recommended_actions"):
                st.write("**Case Recommended Actions:**")
                for action in case_reasoning.get("recommended_actions", []):
                    st.markdown(f"- {action}")

            report_text = build_report_text(claim_text, results_bundle, case_summary)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Full Report (TXT)",
                data=report_text,
                file_name=f"vehicle_damage_report_{timestamp}.txt",
                mime="text/plain",
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
