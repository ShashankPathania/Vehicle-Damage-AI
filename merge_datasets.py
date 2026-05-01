"""Merge CarDD and VehiDE YOLO datasets into a unified combined dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import yaml

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

CARDD_TO_UNIFIED = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "glass shatter": "glass_damage",
    "glass_shatter": "glass_damage",
    "lamp broken": "light_damage",
    "lamp_broken": "light_damage",
    "tire flat": "tire_damage",
    "tire_flat": "tire_damage",
}


def ensure_dirs(base_dir: Path) -> None:
    """Create combined dataset directory structure."""
    for split in ["train", "val"]:
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    """Normalize class names for robust mapping."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def build_mapping_from_yaml(yaml_path: Path, source_kind: str) -> Dict[int, int]:
    """Build class-id mapping from source dataset IDs to unified IDs."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    mapping: Dict[int, int] = {}

    for source_id, source_name in names.items():
        sid = int(source_id)
        sname = str(source_name).strip()
        if source_kind == "cardd":
            unified_name = CARDD_TO_UNIFIED.get(sname, CARDD_TO_UNIFIED.get(normalize_name(sname).replace("_", " ")))
        else:
            unified_name = sname
        if unified_name is None:
            continue
        uid = CLASS_TO_ID.get(normalize_name(unified_name))
        if uid is None:
            uid = CLASS_TO_ID.get(unified_name)
        if uid is not None:
            mapping[sid] = uid
    return mapping


def remap_label_file(source_label: Path, destination_label: Path, id_mapping: Dict[int, int]) -> None:
    """Remap one YOLO label file into unified class IDs."""
    if not source_label.exists():
        destination_label.write_text("", encoding="utf-8")
        return

    out_lines: List[str] = []
    for line in source_label.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            source_id = int(parts[0])
        except ValueError:
            continue
        unified_id = id_mapping.get(source_id)
        if unified_id is None:
            continue
        out_lines.append(f"{unified_id} {' '.join(parts[1:])}")
    destination_label.write_text("\n".join(out_lines), encoding="utf-8")


def merge_split(
    split: str,
    source_name: str,
    source_root: Path,
    out_root: Path,
    id_mapping: Dict[int, int],
) -> int:
    """Merge one split from one source dataset into output dataset."""
    image_dir = source_root / "images" / split
    label_dir = source_root / "labels" / split
    if not image_dir.exists():
        return 0

    copied = 0
    for image_path in image_dir.iterdir():
        if not image_path.is_file():
            continue
        new_stem = f"{source_name}_{image_path.stem}"
        dest_image = out_root / "images" / split / f"{new_stem}{image_path.suffix.lower()}"
        dest_label = out_root / "labels" / split / f"{new_stem}.txt"
        source_label = label_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, dest_image)
        remap_label_file(source_label, dest_label, id_mapping)
        copied += 1
    return copied


def write_dataset_yaml(output_dir: Path) -> None:
    """Write combined dataset.yaml with unified classes."""
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
    """Merge CarDD and VehiDE into one unified YOLO dataset."""
    parser = argparse.ArgumentParser(description="Merge CarDD and VehiDE YOLO datasets into Combined_Damage.")
    parser.add_argument("--cardd", type=str, default="data/CarDD", help="Path to CarDD YOLO dataset root.")
    parser.add_argument("--vehide", type=str, default="data/VehiDE_YOLO", help="Path to VehiDE YOLO dataset root.")
    parser.add_argument("--output", type=str, default="data/Combined_Damage", help="Output path for merged dataset.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cardd = (root / args.cardd).resolve() if not Path(args.cardd).is_absolute() else Path(args.cardd).resolve()
    vehide = (root / args.vehide).resolve() if not Path(args.vehide).is_absolute() else Path(args.vehide).resolve()
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()

    for required in [cardd / "dataset.yaml", vehide / "dataset.yaml"]:
        if not required.exists():
            raise FileNotFoundError(f"Missing dataset config: {required}")

    ensure_dirs(output)
    cardd_map = build_mapping_from_yaml(cardd / "dataset.yaml", source_kind="cardd")
    vehide_map = build_mapping_from_yaml(vehide / "dataset.yaml", source_kind="vehide")

    counts = {}
    counts["cardd_train"] = merge_split("train", "cardd", cardd, output, cardd_map)
    counts["cardd_val"] = merge_split("val", "cardd", cardd, output, cardd_map)
    counts["vehide_train"] = merge_split("train", "vehide", vehide, output, vehide_map)
    counts["vehide_val"] = merge_split("val", "vehide", vehide, output, vehide_map)

    write_dataset_yaml(output)
    print(f"Combined dataset ready at: {output}")
    print(
        "Merged images -> "
        f"CarDD(train={counts['cardd_train']}, val={counts['cardd_val']}), "
        f"VehiDE(train={counts['vehide_train']}, val={counts['vehide_val']})"
    )
    print(f"Unified classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
