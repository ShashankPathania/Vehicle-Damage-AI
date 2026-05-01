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

## Datasets Used

This project was developed using a merged vehicle-damage dataset setup:

- **CarDD** (primary benchmark dataset for car damage detection)
- **VehiDE** (additional vehicle damage samples for broader variation)

The training workflow combines these into a unified dataset, then uses YOLO-format labels for training/inference compatibility.

### Dataset Preparation Scripts

- `prepare_dataset.py` -> preprocess CarDD into YOLO structure
- `prepare_vehide_dataset.py` -> preprocess VehiDE into YOLO structure
- `merge_datasets.py` -> merge prepared datasets into one combined dataset
- `split_dataset.py` -> create deterministic `val/test` split for evaluation

### Expected Dataset Location

```text
data/
  Combined_Damage/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    dataset.yaml
```

### Notes for Evaluators

- Retraining is **not required** for this submission because `models/best.pt` is included.
- If you want to reproduce training, ensure dataset paths in `dataset.yaml` match your local layout.
- `split_dataset.py` is provided so test metrics can be computed on a held-out test split instead of only validation data.

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

### 1) Create environment and install Python dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Download required/optional model assets and place them correctly

This repo already includes:
- `models/best.pt` (trained YOLO damage detector, required for no-training evaluation)

#### SAM checkpoint (optional but recommended)

If you want segmentation overlays and SAM-assisted refinement, download the checkpoint:
- File: `sam_vit_b_01ec64.pth`
- Source: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Place at: `models/sam_vit_b_01ec64.pth`

Final expected model folder:

```text
models/
  best.pt
  sam_vit_b_01ec64.pth   # optional
```

#### ViT model (auto-download)

No manual download needed by default. On first run, Hugging Face downloads:
- `google/vit-base-patch16-224`

To use a different model, set `VIT_DAMAGE_MODEL` in `.env`.

### 3) Optional LLM runtime setup

#### Groq (cloud, recommended)
- Create API key from [https://console.groq.com/keys](https://console.groq.com/keys)
- Add it to `.env` as `GROQ_API_KEY`

#### Ollama fallback (local)
Install Ollama from [https://ollama.com/download](https://ollama.com/download), then pull the fallback model:

```bash
ollama pull llama3.1:latest
```

If Groq is unavailable, the pipeline tries Ollama automatically.

### 4) Configure environment variables

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

### 5) Quick dependency sanity checks (recommended)

```bash
python -c "import torch; print('torch ok:', torch.__version__)"
python -c "import clip; print('clip ok')"
python -c "from transformers import pipeline; print('transformers ok')"
```

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

## Metrics and Evaluation Guidance

- **Detection metrics (YOLO):** mAP50, mAP50-95, precision, recall
- **Operational metrics (pipeline):** uncertain rate, model-agreement ratio, average fused confidence
- **Decision quality checks:** consistency between computed fraud risk and final decision (`APPROVE` / `REVIEW` / `REJECT`)

If you retrain and evaluate with YOLO directly, use your dataset YAML with test split:

```bash
yolo detect val model=models/best.pt data=data/Combined_Damage/dataset.yaml split=test
```

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

1. Install Python dependencies from `requirements.txt`.
2. Verify `models/best.pt` is present (already included in this repo).
3. (Optional) Download SAM checkpoint to `models/sam_vit_b_01ec64.pth`.
4. Configure `.env` (Groq key and optional overrides).
5. (Optional) Install Ollama and run `ollama pull llama3.1:latest`.
6. Run CLI on sample images.
7. Run Streamlit for interactive multi-image evaluation.
8. Validate outputs: detection quality, uncertainty flags, and reasoning consistency.

## Troubleshooting

- **SAM not loading:** confirm `SAM_CHECKPOINT` points to `models/sam_vit_b_01ec64.pth`.
- **Groq not used:** ensure `.env` has `GROQ_API_KEY`, then restart CLI/Streamlit process.
- **Ollama fallback failing:** verify Ollama server is running and model `llama3.1:latest` is pulled.
- **Torch import/DLL errors on Windows:** use the project virtual environment interpreter for all commands.
