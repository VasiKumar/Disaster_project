from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_structure(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def reset_output(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    ensure_structure(root)


def remap_label_line(line: str, class_map: Dict[int, int]) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    old_class = int(float(parts[0]))
    if old_class not in class_map:
        return None

    new_class = class_map[old_class]
    return " ".join([str(new_class), *parts[1:]])


def collect_pairs(image_dir: Path, label_dir: Path) -> List[tuple[Path, Path]]:
    pairs: List[tuple[Path, Path]] = []
    for image_path in image_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = image_path.relative_to(image_dir)
        label_path = (label_dir / rel).with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def write_pair(
    image_path: Path,
    label_path: Path,
    output_root: Path,
    split: str,
    class_map: Dict[int, int],
    prefix: str,
    index: int,
) -> None:
    img_dst = output_root / "images" / split / f"{prefix}_{index:06d}{image_path.suffix.lower()}"
    lbl_dst = output_root / "labels" / split / f"{prefix}_{index:06d}.txt"

    shutil.copy2(image_path, img_dst)

    remapped: List[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        mapped = remap_label_line(line, class_map)
        if mapped:
            remapped.append(mapped)

    lbl_dst.write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")


def write_image_with_lines(
    image_path: Path,
    output_root: Path,
    split: str,
    prefix: str,
    index: int,
    label_lines: List[str],
) -> None:
    img_dst = output_root / "images" / split / f"{prefix}_{index:06d}{image_path.suffix.lower()}"
    lbl_dst = output_root / "labels" / split / f"{prefix}_{index:06d}.txt"
    shutil.copy2(image_path, img_dst)
    lbl_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def split_samples(samples: List[Path], val_ratio: float, test_ratio: float):
    random.shuffle(samples)
    val_n = int(len(samples) * val_ratio)
    test_n = int(len(samples) * test_ratio)
    train_n = len(samples) - val_n - test_n
    return (
        ("train", samples[:train_n]),
        ("val", samples[train_n : train_n + val_n]),
        ("test", samples[train_n + val_n :]),
    )


@dataclass
class SourceStats:
    merged: int = 0
    positives: int = 0
    negatives: int = 0


def process_detection_source(source: dict, out_dir: Path, global_idx: int) -> tuple[int, SourceStats]:
    name = source["name"]
    image_dir = Path(source["image_dir"])
    label_dir = Path(source["label_dir"])
    class_map = {int(k): int(v) for k, v in source["class_map"].items()}
    val_ratio = float(source.get("val_ratio", 0.15))
    test_ratio = float(source.get("test_ratio", 0.05))

    stats = SourceStats()
    if not image_dir.exists() or not label_dir.exists():
        print(f"{name}: skipped (missing image_dir/label_dir)")
        return global_idx, stats

    pairs = collect_pairs(image_dir, label_dir)
    splits = split_samples([p for p, _ in pairs], val_ratio, test_ratio)

    pair_lookup = {p: l for p, l in pairs}
    for split_name, split_images in splits:
        for image_path in split_images:
            label_path = pair_lookup[image_path]
            write_pair(
                image_path=image_path,
                label_path=label_path,
                output_root=out_dir,
                split=split_name,
                class_map=class_map,
                prefix=name,
                index=global_idx,
            )
            stats.merged += 1
            stats.positives += 1
            global_idx += 1

    print(f"{name}: {stats.merged} detection samples merged")
    return global_idx, stats


def _parse_int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "skip", "-1"}:
        return None
    return int(value)


def _resolve_classification_target(
    class_name: str,
    explicit_map: Dict[str, Optional[int]],
    binary_target_id: Optional[int],
    positive_keywords: List[str],
    negative_keywords: List[str],
) -> Optional[int]:
    if class_name in explicit_map:
        return explicit_map[class_name]

    if binary_target_id is None:
        return None

    lower = class_name.lower()
    if any(neg in lower for neg in negative_keywords):
        return None
    if any(pos in lower for pos in positive_keywords):
        return binary_target_id
    return None


def process_classification_source(source: dict, out_dir: Path, global_idx: int) -> tuple[int, SourceStats]:
    name = source["name"]
    root_dir = Path(source["root_dir"])
    val_ratio = float(source.get("val_ratio", 0.15))
    test_ratio = float(source.get("test_ratio", 0.05))
    explicit_class_to_target = {
        str(k): _parse_int_or_none(v) for k, v in source.get("class_to_target", {}).items()
    }
    binary_target_id = _parse_int_or_none(source.get("binary_target_id"))
    positive_keywords = [str(v).lower() for v in source.get("positive_keywords", [])]
    negative_keywords = [str(v).lower() for v in source.get("negative_keywords", ["non", "normal", "safe"])]

    if binary_target_id is not None and not positive_keywords:
        # If no keywords provided, treat source name words as positive hints.
        positive_keywords = [w.lower() for w in str(name).replace("_", " ").split() if w]
    split_dirs = source.get("split_dirs", ["train", "val", "test"])

    stats = SourceStats()
    if not root_dir.exists():
        print(f"{name}: skipped (missing root_dir)")
        return global_idx, stats

    split_image_sets: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    if all((root_dir / split).exists() for split in split_dirs):
        # Respect existing train/val/test folders from source dataset.
        for src_split in split_dirs:
            mapped_split = "val" if str(src_split).lower() == "valid" else str(src_split).lower()
            if mapped_split not in split_image_sets:
                continue
            split_root = root_dir / src_split
            for image_path in split_root.rglob("*"):
                if image_path.suffix.lower() in IMAGE_EXTS:
                    split_image_sets[mapped_split].append(image_path)
    else:
        all_images: List[Path] = []
        for image_path in root_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTS:
                all_images.append(image_path)

        splits = split_samples(all_images, val_ratio, test_ratio)
        for split_name, split_images in splits:
            split_image_sets[split_name].extend(split_images)

    for split_name, split_images in split_image_sets.items():
        for image_path in split_images:
            class_name = image_path.parent.name
            target_id = _resolve_classification_target(
                class_name=class_name,
                explicit_map=explicit_class_to_target,
                binary_target_id=binary_target_id,
                positive_keywords=positive_keywords,
                negative_keywords=negative_keywords,
            )
            # For classification datasets, use a full-frame box as a weak localization.
            label_lines = [f"{target_id} 0.5 0.5 1.0 1.0"] if target_id is not None else []
            write_image_with_lines(
                image_path=image_path,
                output_root=out_dir,
                split=split_name,
                prefix=name,
                index=global_idx,
                label_lines=label_lines,
            )
            stats.merged += 1
            if target_id is None:
                stats.negatives += 1
            else:
                stats.positives += 1
            global_idx += 1

    print(
        f"{name}: {stats.merged} classification samples merged "
        f"(positives={stats.positives}, negatives={stats.negatives})"
    )
    return global_idx, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple datasets into one YOLO dataset")
    parser.add_argument("--config", default="configs/dataset_sources.yaml")
    parser.add_argument("--out", default="datasets/disaster")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(Path(args.config))
    out_dir = Path(args.out)
    reset_output(out_dir)

    global_idx = 0
    total_stats = SourceStats()
    for source in config.get("sources", []):
        source_type = str(source.get("type", "detection")).lower()
        if source_type == "classification":
            global_idx, stats = process_classification_source(source, out_dir, global_idx)
        else:
            global_idx, stats = process_detection_source(source, out_dir, global_idx)

        total_stats.merged += stats.merged
        total_stats.positives += stats.positives
        total_stats.negatives += stats.negatives

    if total_stats.merged == 0 or total_stats.positives == 0:
        raise SystemExit(
            "No usable training samples produced. Check downloaded datasets and configs/dataset_sources.yaml"
        )

    class_names = config["class_names"]
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print(f"\\nCreated YOLO dataset at {out_dir}")
    print(
        f"Merged total: {total_stats.merged} samples "
        f"(positives={total_stats.positives}, negatives={total_stats.negatives})"
    )
    print(f"Data config: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
