#!/usr/bin/env python3
"""Render route-level outcome differences from the three-seed table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CONTRASTS = (
    ("current_frame", "baseline", "Current - baseline",
     "current_frame_vs_baseline"),
    ("temporal", "baseline", "Temporal - baseline",
     "temporal_vs_baseline"),
)
PANELS = (
    ("collision", "Collision-route difference (pp)", 100.0),
    ("success", "Success Rate difference (pp)", 100.0),
    ("driving_score", "Driving Score difference (points)", 1.0),
)
COLORS = ("#6b7280", "#167c72")


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    for row in rows:
        for metric, _, _ in PANELS:
            row[metric] = float(row[metric])
    return rows


def route_deltas(rows, treatment, control, metric):
    values = defaultdict(list)
    for row in rows:
        values[(row["route"], row["arm"])].append(row[metric])
    routes = sorted({row["route"] for row in rows})
    deltas = np.asarray([
        np.mean(values[(route, treatment)])
        - np.mean(values[(route, control)])
        for route in routes
    ], dtype=np.float64)
    if len(deltas) != 185:
        raise ValueError("Route-level plot requires all 185 complete routes")
    return routes, deltas


def deterministic_jitter(routes, contrast_index):
    values = []
    for route in routes:
        digest = hashlib.sha256(
            (str(contrast_index) + ":" + route).encode("ascii")
        ).digest()
        unit = int.from_bytes(digest[:4], "big") / float(2**32 - 1)
        values.append((unit - 0.5) * 0.30)
    return np.asarray(values)


def render(job_outcomes: Path, registry_path: Path, outdir: Path) -> None:
    rows = load_rows(job_outcomes)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for ax, (metric, ylabel, scale) in zip(axes, PANELS):
        all_values = []
        for index, (
            treatment, control, label, registry_key
        ) in enumerate(CONTRASTS):
            routes, deltas = route_deltas(
                rows, treatment, control, metric
            )
            plotted = deltas * scale
            all_values.extend(plotted.tolist())
            reported = registry["contrasts"][registry_key][metric]
            expected_mean = reported["estimate"] * scale
            if not np.isclose(
                np.mean(plotted), expected_mean, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "{} {} mean does not reproduce registry".format(
                        registry_key, metric
                    )
                )
            if metric in ("collision", "success"):
                levels, counts = np.unique(
                    np.round(plotted, decimals=8), return_counts=True
                )
                ax.scatter(
                    np.full_like(levels, index), levels,
                    s=12 + 0.8 * counts, alpha=0.62,
                    color=COLORS[index], edgecolors="white",
                    linewidths=0.35, zorder=1,
                )
            else:
                x = index + deterministic_jitter(routes, index)
                ax.scatter(
                    x, plotted, s=10, alpha=0.58, color=COLORS[index],
                    linewidths=0, zorder=1,
                )
            lower = reported["ci95_lower"] * scale
            upper = reported["ci95_upper"] * scale
            ax.errorbar(
                index, expected_mean,
                yerr=[[expected_mean - lower], [upper - expected_mean]],
                fmt="D", markersize=4.3, color=COLORS[index],
                markeredgecolor="white", markeredgewidth=0.45,
                capsize=2.5, linewidth=1.1, zorder=3,
            )
        ax.axhline(0, color="#202124", linewidth=0.65, zorder=0)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks((0, 1))
        ax.set_xticklabels(
            ("Current\n- baseline", "Temporal\n- baseline"), fontsize=6.4
        )
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color="#d8dde0", linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)
        if metric in ("collision", "success"):
            limit = max(40.0, np.ceil(max(abs(np.asarray(all_values))) / 10) * 10)
            ax.set_ylim(-(limit + 5.0), limit + 5.0)

    fig.tight_layout(w_pad=1.0)
    for extension in ("pdf", "png"):
        fig.savefig(
            outdir / ("fig04_route_level_outcomes." + extension),
            dpi=300, bbox_inches="tight", pad_inches=0.025,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-outcomes", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    render(args.job_outcomes, args.registry, args.outdir)


if __name__ == "__main__":
    main()
