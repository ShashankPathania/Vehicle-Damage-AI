"""Training script for fine-tuning YOLOv8s on CarDD dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train YOLOv8s on CarDD.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/CarDD/dataset.yaml",
        help="Path to YOLO dataset.yaml file.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Training device ("auto", "cpu", "0", etc.).',
    )
    parser.add_argument(
        "--tune-large-damage",
        action="store_true",
        help="Apply training settings that improve sensitivity to large damage regions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="YOLO model checkpoint to fine-tune (e.g., yolov8s.pt, yolov8m.pt).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="vehicle_damage_yolov8s",
        help="Training run name under runs/ directory.",
    )
    return parser.parse_args()


def copy_training_artifacts(run_dir: Path, project_root: Path) -> None:
    """Copy best model and key plots/CSV logs into required output locations."""
    models_dir = project_root / "models"
    logs_dir = project_root / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, models_dir / "best.pt")

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, logs_dir / "training_log.csv")

    for image_name in ["confusion_matrix.png", "results.png", "PR_curve.png", "P_curve.png", "R_curve.png"]:
        src = run_dir / image_name
        if src.exists():
            shutil.copy2(src, logs_dir / image_name)


def main() -> None:
    """Run YOLOv8 training with a configurable set of options."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_path = (project_root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {data_path}. Run prepare_dataset.py first."
        )

    print("========================================")
    print("      VEHICLE DAMAGE MODEL TRAINING     ")
    print("========================================")
    print(f"Data: {data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Image size: {args.imgsz}")

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": "AdamW",
        "project": str(project_root / "runs"),
        "name": args.run_name,
        "exist_ok": True,
        "device": args.device,
    }

    # Defaults emphasize mixed-scale defects. This flag biases training toward
    # larger contiguous damage by reducing heavy mosaic and enabling multi-scale.
    if args.tune_large_damage:
        print("Applying large-damage tuning preset.")
        train_kwargs.update(
            {
                "multi_scale": True,
                "close_mosaic": 25,
                "mosaic": 0.2,
                "mixup": 0.0,
                "degrees": 2.0,
                "translate": 0.05,
                "scale": 0.25,
                "perspective": 0.0005,
            }
        )

    model = YOLO(args.model)
    results = model.train(**train_kwargs)

    run_dir = Path(results.save_dir)
    copy_training_artifacts(run_dir, project_root)
    print("Training complete.")
    print(f"Best weights: {project_root / 'models' / 'best.pt'}")
    print(f"Logs saved in: {project_root / 'logs'}")


if __name__ == "__main__":
    main()
