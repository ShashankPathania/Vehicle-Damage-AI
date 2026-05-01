# Vehicle Damage AI

Multi-model vehicle damage assessment and insurance-claim support system with:
- YOLOv8 damage detection
- CLIP validation for visual-text alignment
- ViT-based damage label refinement
- Optional SAM segmentation
- Uncertainty-aware decision layer
- LLM reasoning (Groq primary, Ollama fallback)
- Streamlit UI + CLI inference

The repository includes a trained checkpoint at `models/best.pt`, so evaluators can run inference directly without retraining.

## Features

- **Damage detection:** Detects damage regions and classes from car images.
- **Fusion pipeline:** Combines YOLO, CLIP, and ViT signals for better label quality and confidence calibration.
- **Post-processing:** Filters weak detections and merges overlap noise.
- **Uncertainty modeling:** Flags low-confidence / disagreeing predictions for manual review.
- **Claim consistency checks:** Compares visual evidence against claim text.
- **LLM report generation:** Produces structured, human-readable verdicts and recommendations.
- **Multi-image support:** Handles batch/case-level analysis in both CLI and Streamlit.

## Project Structure

```text
vehicle_damage_ai/
  app.py                     # Streamlit frontend
  inference.py               # CLI inference entry point
  train.py                   # YOLO training script
  consistency.py             # Claim consistency logic
  severity.py                # Severity scoring
  fusion.py                  # Base detector + localization pipeline
  models/
    best.pt                  # Trained YOLO checkpoint (included)
    clip_validator.py
    vit_classifier.py
    sam_segmenter.py
  utils/
    fusion.py                # Multi-model fusion + filtering
    uncertainty.py           # Decision confidence + uncertainty factors
    llm_reasoner.py          # Groq/Ollama reasoning + rule fallback
  requirements.txt
```

## Setup

### 1) Create environment and install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure environment variables

Create `.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1/chat/completions

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.1:latest

LLM_TIMEOUT_SEC=30
VIT_DAMAGE_MODEL=google/vit-base-patch16-224
SAM_CHECKPOINT=models/sam_vit_b_01ec64.pth
SAM_MODEL_TYPE=vit_b
```

Notes:
- Groq is optional but recommended for better reasoning quality.
- If Groq is unavailable, system falls back to Ollama, then rule-based reasoning.
- SAM is optional; pipeline still runs without it.

## Quick Start (No Training Required)

### CLI (single image)

```bash
python inference.py --image "samples/sample_01.jpg" --claim "rear bumper scratched in parking"
```

### CLI (multiple images)

```bash
python inference.py --images "samples/sample_01.jpg" "samples/sample_02.jpg" --claim "front side impact while parked"
```

### Streamlit UI

```bash
streamlit run app.py
```

Open the local URL, upload one or more images, enter claim text, and click **Analyze**.

## Output Summary

For each image:
- detections with final fused labels
- confidence and uncertainty indicators
- severity and risk signals
- LLM-generated verdict + recommended actions

Case-level (multi-image):
- aggregated risk
- average decision confidence
- uncertain-image rate
- case verdict and action guidance

## Training (Optional)

Retraining is not required for evaluation because `models/best.pt` is already included.

If you still want to retrain:

```bash
python train.py --epochs 50 --batch 16 --data "data/CarDD/dataset.yaml"
```

## Architecture

```text
YOLO Detection
    -> CLIP Validation
    -> ViT Refinement
    -> (Optional) SAM Segmentation
    -> Fusion + Post-processing
    -> Uncertainty + Consistency
    -> LLM Reasoning
```

## Limitations

- Performance depends on image quality, lighting, occlusions, and training data diversity.
- SAM and larger LLMs improve quality but increase runtime.
- Claim consistency is assistive, not legal/forensic proof.

## Recommended Evaluation Flow

1. Install dependencies.
2. Ensure `models/best.pt` exists (already included in this repo).
3. Run CLI on provided sample images.
4. Run Streamlit and test multi-image cases.
5. Validate outputs: detection quality, uncertainty flags, and reasoning consistency.
