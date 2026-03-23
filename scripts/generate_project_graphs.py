from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required to generate graphs. Install with: pip install matplotlib"
    ) from exc


SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def load_incidents(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Incident file not found: {path}")
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Incident file must contain a JSON array")
    return content


def save_disaster_chance_plot(incidents: List[dict], out_path: Path) -> None:
    types = [str(item.get("disaster_type", "unknown")) for item in incidents]
    if not types:
        raise ValueError("No incidents available for disaster chance plot")

    counts = Counter(types)
    total = sum(counts.values())
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    percentages = [(v / total) * 100.0 for v in values]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, percentages, color="#f97316", edgecolor="#1f2937")
    plt.title("Disaster Case Chances (From Logged Incidents)")
    plt.xlabel("Disaster Type")
    plt.ylabel("Chance (%)")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(percentages) * 1.18)

    for bar, pct, count in zip(bars, percentages, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{pct:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_alert_trend_plot(incidents: List[dict], out_path: Path) -> None:
    rows = []
    for item in incidents:
        ts = item.get("created_at")
        sev = str(item.get("severity", "low")).lower()
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts))
        except ValueError:
            continue
        if sev not in SEVERITY_ORDER:
            sev = "low"
        rows.append((dt, sev))

    if not rows:
        raise ValueError("No timestamped incidents available for alert trend plot")

    rows.sort(key=lambda x: x[0])
    bucket_counts: Dict[datetime, Dict[str, int]] = defaultdict(lambda: {s: 0 for s in SEVERITY_ORDER})
    for dt, sev in rows:
        bucket = dt.replace(second=0, microsecond=0)
        bucket_counts[bucket][sev] += 1

    buckets = sorted(bucket_counts.keys())
    series = {s: [bucket_counts[b][s] for b in buckets] for s in SEVERITY_ORDER}

    x = list(range(len(buckets)))
    x_labels = [b.strftime("%m-%d %H:%M") for b in buckets]

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        x,
        series["low"],
        series["medium"],
        series["high"],
        series["critical"],
        labels=SEVERITY_ORDER,
        colors=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        alpha=0.85,
    )
    plt.title("Alert Severity Trend Over Time")
    plt.xlabel("Time")
    plt.ylabel("Incident Count")

    step = max(1, len(x_labels) // 8)
    tick_pos = x[::step]
    tick_lbl = x_labels[::step]
    plt.xticks(tick_pos, tick_lbl, rotation=25, ha="right")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def pick_best_results_csv(runs_root: Path) -> Path:
    csv_files = sorted(runs_root.glob("*/results.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No results.csv found under {runs_root}")

    best_csv = csv_files[0]
    best_score = -1.0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "metrics/mAP50-95(B)" not in df.columns or df.empty:
            continue
        score = float(df["metrics/mAP50-95(B)"].max())
        if score > best_score:
            best_score = score
            best_csv = csv_path
    return best_csv


def save_accuracy_plot(results_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"No rows in {results_csv}")

    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

    plt.figure(figsize=(11, 6))
    metrics = [
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/precision(B)", "precision"),
        ("metrics/recall(B)", "recall"),
    ]

    for col, label in metrics:
        if col in df.columns:
            plt.plot(df[epoch_col], df[col], marker="o", linewidth=2, label=label)

    plt.title(f"Training Accuracy Metrics Curve ({results_csv.parent.name})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project graphs for disaster chances, alerts, and accuracy")
    parser.add_argument("--incidents", default="data/incidents.json")
    parser.add_argument("--runs-root", default="runs/detect/runs/disaster")
    parser.add_argument("--out", default="reports/graphs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    incidents = load_incidents(Path(args.incidents))
    save_disaster_chance_plot(incidents, out_dir / "disaster_chance_plot.png")
    save_alert_trend_plot(incidents, out_dir / "alert_trend_plot.png")

    best_csv = pick_best_results_csv(Path(args.runs_root))
    save_accuracy_plot(best_csv, out_dir / "accuracy_metrics_plot.png")

    print(f"Graphs saved in: {out_dir.resolve()}")
    print(f"Accuracy graph source: {best_csv}")


if __name__ == "__main__":
    main()
