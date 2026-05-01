"""Convert VehiDE VIA polygon annotations into YOLO format with unified classes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image

CLASS_NAMES = [
    "scratch",
    "dent",
    "crack",
    "glass_damage",
    "tire_damage",
    "light_damage",
    "structural_damage",
    "missing_part",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

VEHIDE_TO_UNIFIED = {
    "broken_glass": "glass_damage",
    "broken_lights": "light_damage",
    "scratch": "scratch",
    "dents": "dent",
    "lost_parts": "missing_part",
    "torn": "structural_damage",
    "punctured": "structural_damage",
    "non-damaged": "ignore",
    # VehiDE labels observed in current dataset snapshot.
    "tray_son": "scratch",
    "mop_lom": "dent",
    "rach": "crack",
    "vo_kinh": "glass_damage",
    "be_den": "light_damage",
    "thung": "structural_damage",
    "mat_bo_phan": "missing_part",
}


def ensure_dirs(output_dir: Path) -> None:
    """Create output directory structure for YOLO dataset."""
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def polygon_to_bbox(all_x: List[float], all_y: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert polygon points to clipped XYXY bounding box."""
    x1 = max(0.0, min(float(x) for x in all_x))
    y1 = max(0.0, min(float(y) for y in all_y))
    x2 = min(float(width - 1), max(float(x) for x in all_x))
    y2 = min(float(height - 1), max(float(y) for y in all_y))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert XYXY bounding box to YOLO normalized XYWH."""
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    x_center = x1 + (w / 2.0)
    y_center = y1 + (h / 2.0)
    return x_center / width, y_center / height, w / width, h / height


def convert_split(
    split_name: str,
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
) -> Tuple[int, int]:
    """Convert one VehiDE split from VIA JSON into YOLO labels."""
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    converted_images = 0
    converted_labels = 0

    for image_name, item in data.items():
        source_image = images_dir / image_name
        if not source_image.exists():
            continue

        try:
            with Image.open(source_image) as image:
                width, height = image.size
        except Exception:
            continue

        lines: List[str] = []
        for region in item.get("regions", []):
            source_label = str(region.get("class", "")).strip().lower()
            unified_label = VEHIDE_TO_UNIFIED.get(source_label)
            if unified_label is None or unified_label == "ignore":
                continue

            all_x = region.get("all_x", [])
            all_y = region.get("all_y", [])
            if not all_x or not all_y or len(all_x) != len(all_y):
                continue

            x1, y1, x2, y2 = polygon_to_bbox(all_x, all_y, width, height)
            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                continue

            x_c, y_c, w, h = xyxy_to_yolo(x1, y1, x2, y2, width, height)
            class_id = CLASS_TO_ID[unified_label]
            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        destination_image = output_dir / "images" / split_name / image_name
        destination_label = output_dir / "labels" / split_name / f"{Path(image_name).stem}.txt"
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)
        destination_label.write_text("\n".join(lines), encoding="utf-8")
        converted_images += 1
        converted_labels += len(lines)

    return converted_images, converted_labels


def write_dataset_yaml(output_dir: Path) -> None:
    """Write YOLO dataset YAML for the converted VehiDE dataset."""
    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    """Run VehiDE VIA-to-YOLO conversion."""
    parser = argparse.ArgumentParser(description="Prepare VehiDE dataset in YOLO format with unified classes.")
    parser.add_argument("--source", type=str, default="data/vehiDE", help="Path to VehiDE root directory.")
    parser.add_argument("--output", type=str, default="data/VehiDE_YOLO", help="Output path for YOLO formatted dataset.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source = (root / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source).resolve()
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()

    train_json = source / "0Train_via_annos.json"
    val_json = source / "0Val_via_annos.json"
    train_images = source / "image" / "image"
    val_images = source / "validation" / "validation"

    for path in [train_json, val_json, train_images, val_images]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required VehiDE path: {path}")

    ensure_dirs(output)
    train_count, train_boxes = convert_split("train", train_json, train_images, output)
    val_count, val_boxes = convert_split("val", val_json, val_images, output)
    write_dataset_yaml(output)

    print(f"VehiDE conversion complete: {output}")
    print(f"Train images: {train_count} | Train boxes: {train_boxes}")
    print(f"Val images: {val_count} | Val boxes: {val_boxes}")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
