#!/usr/bin/env python3
"""Generate separate Chinese SVGs for the 48- and 36-pair SBAS networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network48", default="work/baselines/tongji_redundant_22_network48_qc.csv")
    p.add_argument("--network36", default="work/gamma_native_ipta_paperquality_stability36/pairs_stability_subset.csv")
    p.add_argument("--output48", default="picall/03_干涉对时空基线网络.svg")
    p.add_argument("--output36", default="picall/03a_稳定性检验时空基线网络.svg")
    return p.parse_args()


def acquisition_baselines(pairs: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(set(pairs["master"].astype(str)) | set(pairs["slave"].astype(str)))
    lookup = {date: index for index, date in enumerate(dates)}
    design = np.zeros((len(pairs) + 1, len(dates)), dtype=float)
    observed = np.zeros(len(pairs) + 1, dtype=float)
    for row_index, row in enumerate(pairs.itertuples(index=False)):
        design[row_index, lookup[str(row.master)]] = -1.0
        design[row_index, lookup[str(row.slave)]] = 1.0
        observed[row_index] = float(row.bperp_m)
    design[-1, 0] = 1.0
    values, *_ = np.linalg.lstsq(design, observed, rcond=None)
    values -= np.mean(values)
    return pd.DataFrame(
        {"date_key": dates, "date": pd.to_datetime(dates, format="%Y%m%d"), "bperp_m": values}
    )


def draw_network(
    pairs_path: str,
    output_path: str,
    title: str,
    line_color: str,
    note: str,
) -> None:
    pairs = pd.read_csv(pairs_path, dtype={"master": str, "slave": str})
    pairs["master"] = pairs["master"].str.zfill(8)
    pairs["slave"] = pairs["slave"].str.zfill(8)
    nodes = acquisition_baselines(pairs)
    heights = nodes.set_index("date_key")["bperp_m"].to_dict()

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for row in pairs.itertuples(index=False):
        ax.plot(
            [pd.to_datetime(row.master), pd.to_datetime(row.slave)],
            [heights[row.master], heights[row.slave]],
            color=line_color,
            linewidth=0.72,
            alpha=0.72,
            zorder=1,
        )
    ax.scatter(
        nodes["date"], nodes["bperp_m"], s=24, facecolor="#C74440",
        edgecolor="white", linewidth=0.48, zorder=3,
    )
    for row in nodes.itertuples(index=False):
        ax.text(row.date, row.bperp_m + 10.0, row.date_key[4:], ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.0, color="#A8ADB4", linewidth=0.65, zorder=0)
    ax.set_xlabel("成像日期")
    ax.set_ylabel("相对垂直基线（m）")
    ax.set_title(title, pad=7, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")
    ax.grid(True, color="#E4E7EB", linewidth=0.52, linestyle="--")
    ax.text(
        0.018, 0.975,
        f"{len(nodes)}景影像，{len(pairs)}个干涉对\n{note}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "alpha": 0.92, "pad": 2.4},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": title})
    plt.close(fig)
    print({"output": str(output), "acquisitions": len(nodes), "pairs": len(pairs)})


def main() -> None:
    args = parse_args()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    draw_network(
        args.network48, args.output48, "主反演48对SBAS时空基线网络", "#355F8A",
        "边连通度2，无桥边，用于正式GAMMA-SBAS反演",
    )
    draw_network(
        args.network36, args.output36, "稳定性检验36对SBAS时空基线网络", "#C67A2B",
        "独立删除12个低质量干涉对后仍保持边连通度2、无桥边",
    )


if __name__ == "__main__":
    main()
