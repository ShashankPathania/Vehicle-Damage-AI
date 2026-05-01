"""Create a YOLO test split by moving paired image/label files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Move a portion of YOLO split samples into images/test and labels/test."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/Combined_Damage",
        help="Path to dataset root containing images/, labels/, and dataset.yaml.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Split to sample from when creating test split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of source split to move into test (0, 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    return parser.parse_args()


def resolve_dataset_path(project_root: Path, dataset_arg: str) -> Path:
    """Resolve dataset path, supporting relative and absolute input."""
    dataset_path = Path(dataset_arg)
    if not dataset_path.is_absolute():
        dataset_path = (project_root / dataset_path).resolve()
    return dataset_path


def load_yaml(path: Path) -> dict:
    """Read dataset YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data: dict) -> None:
    """Write YAML while preserving key order."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    """Create test split and update dataset.yaml."""
    args = parse_args()
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be between 0 and 1 (exclusive).")

    project_root = Path(__file__).resolve().parent
    dataset_root = resolve_dataset_path(project_root, args.dataset)
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at: {yaml_path}")

    images_src = dataset_root / "images" / args.source_split
    labels_src = dataset_root / "labels" / args.source_split
    images_test = dataset_root / "images" / "test"
    labels_test = dataset_root / "labels" / "test"

    if not images_src.exists() or not labels_src.exists():
        raise FileNotFoundError(f"Missing source split directories: {images_src} and/or {labels_src}")

    images_test.mkdir(parents=True, exist_ok=True)
    labels_test.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in images_src.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    if not image_files:
        raise RuntimeError(f"No image files found in: {images_src}")

    image_files.sort(key=lambda p: p.name)
    random.seed(args.seed)
    move_count = max(1, int(len(image_files) * args.test_ratio))
    chosen = random.sample(image_files, k=move_count)

    moved = 0
    for image_path in chosen:
        src_label = labels_src / f"{image_path.stem}.txt"
        if not src_label.exists():
            continue

        dst_image = images_test / image_path.name
        dst_label = labels_test / src_label.name

        if dst_image.exists() or dst_label.exists():
            continue

        image_path.rename(dst_image)
        src_label.rename(dst_label)
        moved += 1

    if moved == 0:
        raise RuntimeError("No files were moved. Check label pairing and existing test files.")

    for cache_name in [f"{args.source_split}.cache", "test.cache"]:
        cache_file = dataset_root / "labels" / cache_name
        if cache_file.exists():
            cache_file.unlink()

    dataset_yaml = load_yaml(yaml_path)
    dataset_yaml["test"] = "images/test"
    save_yaml(yaml_path, dataset_yaml)

    print(f"Dataset root: {dataset_root}")
    print(f"Source split size: {len(image_files)}")
    print(f"Moved to test: {moved}")
    print("Updated dataset.yaml with: test: images/test")


if __name__ == "__main__":
    main()
