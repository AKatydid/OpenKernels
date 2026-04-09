#!/usr/bin/env python3
"""Plot SGEMM benchmark results from a CSV file.

CSV format:
variant,time_ms
naive,3.12
v1_tiling,1.45
v2_vectorize,0.98
v3_swizzle,0.81
v4_double_buffer,0.73
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path):
    variants = []
    times = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append(row["variant"])
            times.append(float(row["time_ms"]))
    return variants, times


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: plot_performance.py <results.csv>")
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Input file not found: {csv_path}")
        return 1

    variants, times = load_rows(csv_path)
    if not variants:
        print("No data rows in CSV")
        return 1

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(variants, times)
    plt.title("SGEMM Performance")
    plt.xlabel("Kernel Variant")
    plt.ylabel("Latency (ms)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = csv_path.with_suffix(".png")
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
