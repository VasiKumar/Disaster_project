from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def load_confusion_matrix(model_path: str, data_yaml: str, split: str):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split, plots=False, verbose=False)

    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is None:
        raise RuntimeError("Validation did not return confusion matrix")

    matrix = getattr(cm_obj, "matrix", None)
    names = getattr(cm_obj, "names", None)
    if matrix is None or names is None:
        raise RuntimeError("Could not read confusion matrix data from validation results")

    return matrix, names


def compute_breakdown(matrix, names):
    nc = len(names)
    if matrix.shape[0] < nc or matrix.shape[1] < nc:
        raise ValueError("Confusion matrix shape is smaller than class count")

    total = float(matrix.sum())
    rows = []
    for i, class_name in enumerate(names):
        tp = float(matrix[i, i])
        fp = float(matrix[i, :].sum() - tp)
        fn = float(matrix[:, i].sum() - tp)
        tn = float(total - tp - fp - fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        rows.append(
            {
                "class": str(class_name),
                "TP": round(tp, 4),
                "FP": round(fp, 4),
                "FN": round(fn, 4),
                "TN": round(tn, 4),
                "precision_from_cm": round(precision, 6),
                "recall_from_cm": round(recall, 6),
            }
        )

    return pd.DataFrame(rows)


def plot_tp_fp_fn_tn(df: pd.DataFrame, out_path: Path) -> None:
    classes = df["class"].tolist()
    x = range(len(classes))

    plt.figure(figsize=(14, 7))
    plt.plot(x, df["TP"], marker="o", linewidth=2.2, label="TP")
    plt.plot(x, df["FP"], marker="o", linewidth=2.2, label="FP")
    plt.plot(x, df["FN"], marker="o", linewidth=2.2, label="FN")
    plt.plot(x, df["TN"], marker="o", linewidth=2.2, label="TN")

    plt.title("Confusion Matrix Breakdown by Class (TP/FP/FN/TN)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(list(x), classes, rotation=30, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TP/FP/FN/TN graph from YOLO confusion matrix")
    parser.add_argument("--model", default="runs/detect/runs/disaster/yolo_disaster5/weights/best.pt")
    parser.add_argument("--data", default="datasets/disaster/data.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="reports/graphs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, names = load_confusion_matrix(args.model, args.data, args.split)
    df = compute_breakdown(matrix, names)

    csv_path = out_dir / "confusion_breakdown_tp_fp_fn_tn.csv"
    df.to_csv(csv_path, index=False)

    plot_path = out_dir / "confusion_breakdown_tp_fp_fn_tn.png"
    plot_tp_fp_fn_tn(df, plot_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
