from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

import kagglehub


HUGE_DATASET_BLOCKLIST = {
    # Explicitly block very large sources that can exceed local budget.
    "awsaf49/coco-2017-dataset",
    "cocoapi/coco-2017-dataset",
}


# Curated small profiles with fallback candidates per category.
# Downloader tries each candidate in order until one works.
DATASET_PROFILES: Dict[str, Dict[str, Dict[str, object]]] = {
    "coverage": {
        "survivor_person": {
            "candidates": [
                "alincijov/person-detection-dataset",
                "andrewmvd/human-detection-dataset",
                "hmendonca/people-detection",
                "ankit2665/people-detection-images",
            ],
            "estimated_gb": 0.45,
        },
        "fire_smoke": {
            "candidates": [
                "phylake1337/fire-dataset",
                "kanakbagi/fire-smoke-images-dataset",
                "anirudhshankar/forest-fire-images",
                "abdelghaniaaba/wildfire-prediction-dataset",
                "elmadafri/the-wildfire-dataset",
            ],
            "estimated_gb": 0.40,
        },
        "smoke": {
            "candidates": [
                "deepcontractor/smoke-detection-dataset",
                "mayank1508/smoke-detection-dataset",
                "dgomonov/new-york-city-air-quality",
            ],
            "estimated_gb": 0.30,
        },
        "accident": {
            "candidates": [
                "ckay16/accident-detection-from-cctv-footage",
                "sid321axn/road-accident-detection-dataset",
                "mikoajd21/road-accident-detection",
            ],
            "estimated_gb": 0.30,
        },
        "crowd": {
            "candidates": [
                "fmena14/crowd-counting",
                "mohamedhanyyy/crowd-counting",
                "andrewmvd/crowd-counting",
                "saurabhshahane/crowd-counting-dataset",
            ],
            "estimated_gb": 0.50,
        },
        "fallen_person": {
            "candidates": [
                "hijest/fall-detection-dataset",
                "prudhvignv/fall-detection-dataset",
                "karthiknathan/fall-detection",
            ],
            "estimated_gb": 0.25,
        },
        "earthquake_damage": {
            "candidates": [
                "kmader/earthquake-damage-prediction",
                "sumitd3/earthquake-damage",
                "hamzamanssor/earthquake-damage-dataset",
            ],
            "estimated_gb": 0.35,
        },
        "unsafe_zone": {
            "candidates": [
                "andrewmvd/hard-hat-detection",
                "dataclusterlabs/construction-site-safety-image-dataset",
                "a2015003713/military-facility-object-detection",
            ],
            "estimated_gb": 0.45,
        },
        "flood": {
            "candidates": [
                "kabeer2004/flood-images-for-training-disaster-detection-model",
                "teajay/global-flood-dataset",
            ],
            "estimated_gb": 0.30,
        },
    },
    "tiny": {
        "fire_smoke": {
            "candidates": [
                "dhanu111/fire-detection-dataset",
                "shubhamgoel27/fire-detection-dataset",
                "phylake1337/fire-dataset",
                "kanakbagi/fire-smoke-images-dataset",
                "anirudhshankar/forest-fire-images",
            ],
            "estimated_gb": 0.18,
        },
        "accident": {
            "candidates": [
                "mikoajd21/road-accident-detection",
                "ckay16/accident-detection-from-cctv-footage",
                "sid321axn/road-accident-detection-dataset",
            ],
            "estimated_gb": 0.25,
        },
        "crowd": {
            "candidates": [
                "fmena14/crowd-counting",
                "mohamedhanyyy/crowd-counting",
                "saurabhshahane/crowd-counting-dataset",
                "tthien/shanghaitech-with-people-density-map",
            ],
            "estimated_gb": 0.30,
        },
        "earthquake_damage": {
            "candidates": [
                "sumitd3/earthquake-damage",
                "kmader/earthquake-damage-prediction",
                "hamzamanssor/earthquake-damage-dataset",
            ],
            "estimated_gb": 0.20,
        },
    },
    "small": {
        "fire_smoke": {
            "candidates": [
                "dhanu111/fire-detection-dataset",
                "shubhamgoel27/fire-detection-dataset",
                "abdelghaniaaba/wildfire-prediction-dataset",
            ],
            "estimated_gb": 0.20,
        },
        "accident": {
            "candidates": [
                "mikoajd21/road-accident-detection",
                "ckay16/accident-detection-from-cctv-footage",
                "sid321axn/road-accident-detection-dataset",
            ],
            "estimated_gb": 0.30,
        },
        "crowd": {
            "candidates": [
                "fmena14/crowd-counting",
                "mohamedhanyyy/crowd-counting",
                "andrewmvd/crowd-counting",
            ],
            "estimated_gb": 0.35,
        },
        "earthquake_damage": {
            "candidates": [
                "sumitd3/earthquake-damage",
                "kmader/earthquake-damage-prediction",
            ],
            "estimated_gb": 0.18,
        },
    }
}


def directory_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def copy_tree_limited(src: Path, dst: Path, max_files: int) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.is_file()]
    random.shuffle(files)
    selected = files[:max_files] if max_files > 0 else files

    for file_path in selected:
        rel = file_path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)


def choose_first_working_slug(candidates: List[str]) -> tuple[str | None, Path | None]:
    for slug in candidates:
        if slug in HUGE_DATASET_BLOCKLIST:
            print(f"  - candidate blocked (too large): {slug}")
            continue
        try:
            downloaded_path = Path(kagglehub.dataset_download(slug))
            return slug, downloaded_path
        except Exception as exc:
            print(f"  - candidate failed: {slug} ({type(exc).__name__})")
            continue
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public disaster datasets from internet")
    parser.add_argument("--out", default="datasets/raw", help="Output folder for raw datasets")
    parser.add_argument("--profile", default="coverage", choices=sorted(DATASET_PROFILES.keys()))
    parser.add_argument("--max-total-gb", type=float, default=3.0)
    parser.add_argument("--max-dataset-gb", type=float, default=1.2)
    parser.add_argument("--max-files-per-dataset", type=int, default=10000)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated categories to download (e.g. flood,fire_smoke)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = DATASET_PROFILES[args.profile]
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = {k: v for k, v in selected.items() if k in wanted}
        if not selected:
            raise SystemExit(f"No matching categories in profile '{args.profile}' for --only={args.only}")

    print(f"Downloading datasets into: {out_dir}")
    print(
        f"Limits: total<={args.max_total_gb:.2f} GB, per-dataset<={args.max_dataset_gb:.2f} GB, "
        f"max_files_per_dataset={args.max_files_per_dataset}"
    )

    total_bytes = 0
    max_total_bytes = int(args.max_total_gb * 1024**3)
    max_dataset_bytes = int(args.max_dataset_gb * 1024**3)

    for name, dataset_info in selected.items():
        candidates = [str(x) for x in dataset_info.get("candidates", [])]
        if not candidates:
            print(f"\n[{name}] Skipped: no candidates configured")
            continue
        estimated_gb = float(dataset_info.get("estimated_gb", 0.0))

        if estimated_gb > args.max_dataset_gb:
            print(
                f"\n[{name}] Skipped before download: estimated {estimated_gb:.2f} GB exceeds "
                f"per-dataset limit {args.max_dataset_gb:.2f} GB"
            )
            continue
        if total_bytes >= max_total_bytes:
            print(f"\n[{name}] Skipped: total budget already reached")
            continue

        print(f"\n[{name}] Trying lightweight alternatives ({len(candidates)} candidates)")
        try:
            slug, downloaded_path = choose_first_working_slug(candidates)
            if not slug or not downloaded_path:
                print(f"[{name}] Failed: no candidate dataset could be downloaded")
                continue

            print(f"[{name}] Using {slug}")
            target = out_dir / name

            copy_tree_limited(downloaded_path, target, args.max_files_per_dataset)

            current_size = directory_size_bytes(target)
            if current_size > max_dataset_bytes or (total_bytes + current_size) > max_total_bytes:
                shutil.rmtree(target)
                print(
                    f"[{name}] Removed after copy: size {current_size / 1024**3:.2f} GB exceeds configured limits"
                )
                continue

            total_bytes += current_size
            print(
                f"[{name}] Saved to {target} ({current_size / 1024**3:.2f} GB). "
                f"Running total: {total_bytes / 1024**3:.2f} GB"
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            print(f"[{name}] Failed: {exc}")

    print("\nDownload step complete.")
    print("Next: configure class remapping in configs/dataset_sources.yaml, then run scripts/build_yolo_dataset.py")


if __name__ == "__main__":
    main()
