# AI-Based Vehicle Damage Assessment & Insurance Claim Consistency Checker

This capstone project detects vehicle damage from images, estimates severity, and cross-checks an insurance claim against visual evidence.  
It uses YOLOv8s for detection and CLIP for claim-image consistency validation.  
The system outputs annotated images, structured reports, and fraud risk levels for claim triage.  
A Streamlit app is included for a clean live demo workflow.

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Download and Prepare CarDD Dataset

Official dataset source: [CarDD GitHub](https://github.com/cardd-project/CarDD)

1. Clone or download CarDD.
2. Ensure raw dataset has:
   - `images/`
   - COCO annotation file (default assumed by script): `annotations/instances_default.json`
3. Run dataset preparation:

```bash
python prepare_dataset.py --source "C:\path\to\raw\CarDD" --output "data/CarDD"
```

Expected prepared structure:

```text
data/
  CarDD/
    images/
      train/
      val/
    labels/
      train/
      val/
    dataset.yaml
```

## 3) Train YOLOv8s

```bash
python train.py --epochs 50 --batch 16 --data "data/CarDD/dataset.yaml"
```

Smoke test:

```bash
python train.py --epochs 1 --batch 4 --data "data/CarDD/dataset.yaml"
```

Saved outputs:
- `models/best.pt`
- `logs/training_log.csv`
- `logs/confusion_matrix.png` and additional training curves

## 4) Run Inference (CLI)

```bash
python inference.py --image "samples/sample_01.jpg" --claim "rear bumper scratched in parking"
```

The script saves annotated output to:
- `outputs/result.jpg`

## 5) Launch Streamlit App

```bash
streamlit run app.py
```

Open the displayed local URL in your browser, upload an image, enter a claim, and click **Analyze**.

## 6) Fetch Demo Samples

```bash
python fetch_samples.py
```

Sample claims are available in:
- `samples/sample_claims.txt`

## 7) Expected Outputs

- Annotated damage image with bounding boxes and severity labels.
- Detection table: class, location, severity, action, score.
- Claim consistency details including CLIP similarity and mismatch reasons.
- Fraud risk level (`LOW`, `MEDIUM`, `HIGH`) and verdict summary.

## 8) Known Limitations

- Accuracy depends on CarDD training quality and annotation coverage.
- CLIP consistency scoring is heuristic and may be sensitive to image composition.
- CPU-only execution works but is slower than GPU acceleration.
- Coarse location mapping from bounding boxes may miss complex camera angles.

## Two-Stage Pipeline Architecture

This project uses a two-stage design to improve interpretability over single-model damage detection:

- **Stage 1: Surface damage detection**  
  `models/best.pt` (YOLOv8s fine-tuned on CarDD) detects damage classes such as dent, scratch, crack, glass breakage, lamp breakage, and tire flat.

- **Stage 2: Car part localization**  
  A geometric car-part mapping layer infers likely part regions (hood, bumper, windshield, doors, fenders, trunk, roof, headlights) directly from image-relative zones. This stage works with zero API keys and zero extra downloads.

- **Fusion: IoU-based damage-to-part assignment**  
  Each Stage 1 damage box is matched with Stage 2 part zones by IoU overlap. If overlap is low, a quadrant fallback assigns a best-effort part label.

- **Why two stages**  
  A single detector trained only on CarDD damage classes can identify *what* damage is present but often cannot reliably explain *which vehicle part* is affected. The two-stage pipeline addresses this dataset gap and provides better spatial context for claim verification and downstream fraud analysis.
