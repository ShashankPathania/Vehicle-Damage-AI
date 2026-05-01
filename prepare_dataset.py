"""Prepare CarDD dataset by converting COCO annotations into YOLO format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def ensure_dirs(base_dir: Path) -> None:
    """Create expected output directory structure."""
    for split in ["train", "val", "test"]:
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def coco_bbox_to_yolo(bbox: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] to YOLO normalized [x_center, y_center, w, h]."""
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    return x_center, y_center, w / width, h / height


def parse_coco(annotation_path: Path) -> Tuple[Dict[int, dict], Dict[int, List[dict]], Dict[int, str]]:
    """Load COCO JSON and index images, annotations and category names."""
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_map = {img["id"]: img for img in data.get("images", [])}
    ann_map: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        ann_map.setdefault(ann["image_id"], []).append(ann)

    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return image_map, ann_map, categories


def write_yolo_labels(
    labels_dir: Path,
    image_name: str,
    image_info: dict,
    anns: List[dict],
    category_to_index: Dict[int, int],
) -> None:
    """Write one YOLO label file for one image."""
    label_path = labels_dir / f"{Path(image_name).stem}.txt"
    width = int(image_info["width"])
    height = int(image_info["height"])
    lines: List[str] = []
    for ann in anns:
        class_id = category_to_index[ann["category_id"]]
        x_c, y_c, w, h = coco_bbox_to_yolo(ann["bbox"], width, height)
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Convert annotations (COCO splits) and generate YOLO dataset.yaml."""
    parser = argparse.ArgumentParser(description="Prepare CarDD dataset for YOLOv8 training.")
    parser.add_argument("--source", type=str, required=True, help="Path to CarDD_COCO directory.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/CarDD",
        help="Output directory where YOLO-ready dataset will be created.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset path does not exist: {source_dir}")

    ensure_dirs(output_dir)

    # COCO predefined splits
    splits = {
        "train": ("train2017", "instances_train2017.json"),
        "val": ("val2017", "instances_val2017.json"),
        "test": ("test2017", "instances_test2017.json"),
    }

    copied = {"train": 0, "val": 0, "test": 0}
    category_to_index = None
    class_names = None

    for split, (img_folder, ann_file) in splits.items():
        print(f"\nProcessing {split}...")

        annotation_path = source_dir / "annotations" / ann_file
        images_root = source_dir / img_folder

        if not annotation_path.exists():
            print(f"⚠️ Skipping {split}, annotation not found: {annotation_path}")
            continue

        image_map, ann_map, categories = parse_coco(annotation_path)

        # Initialize class mapping once
        if category_to_index is None:
            sorted_category_ids = sorted(categories.keys())
            category_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted_category_ids)}
            class_names = [categories[cid] for cid in sorted_category_ids]

        for image_id, image_info in image_map.items():
            filename = image_info["file_name"]
            source_image = images_root / filename

            if not source_image.exists():
                continue

            destination_image = output_dir / "images" / split / Path(filename).name
            destination_image.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_image, destination_image)

            write_yolo_labels(
                output_dir / "labels" / split,
                destination_image.name,
                image_info,
                ann_map.get(image_id, []),
                category_to_index,
            )

            copied[split] += 1

    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    yaml_path = output_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"\nDataset prepared at: {output_dir}")
    print(f"Train: {copied['train']} | Val: {copied['val']} | Test: {copied['test']}")
    print(f"Classes: {class_names}")
    print(f"dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()