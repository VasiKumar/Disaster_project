from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model for multi-disaster detection")
    parser.add_argument("--data", default="datasets/disaster/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="disk", choices=["false", "ram", "disk"])
    parser.add_argument("--amp", default="false", choices=["true", "false"])
    parser.add_argument("--project", default="runs/disaster")
    parser.add_argument("--name", default="yolo_disaster")
    parser.add_argument("--export", default="models/disaster_yolo.pt")
    args = parser.parse_args()

    device = args.device
    if str(device).lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    if str(device) == "0" and torch.cuda.is_available():
        print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")

    model = YOLO(args.model)
    cache_arg = False if args.cache == "false" else args.cache
    amp_arg = parse_bool(args.amp)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        cache=cache_arg,
        amp=amp_arg,
        project=args.project,
        name=args.name,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    export_path = Path(args.export)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, export_path)
        print(f"Best model exported to {export_path}")
    else:
        print("Training finished but best.pt not found.")


if __name__ == "__main__":
    main()
