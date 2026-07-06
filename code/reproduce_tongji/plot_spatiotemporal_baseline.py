#!/usr/bin/env python3
"""Plot SBAS temporal/perpendicular baseline network."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def solve_relative_baselines(pairs: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(set(pairs["master"].astype(str)) | set(pairs["slave"].astype(str)))
    index = {date: i for i, date in enumerate(dates)}
    # Fix the first acquisition to 0 m and solve B_slave - B_master = Bperp.
    rows = []
    y = []
    for row in pairs.itertuples(index=False):
        a = np.zeros(len(dates), dtype=float)
        a[index[str(row.slave)]] = 1.0
        a[index[str(row.master)]] = -1.0
        rows.append(a)
        y.append(float(row.bperp_m))
    a0 = np.zeros(len(dates), dtype=float)
    a0[0] = 1.0
    rows.append(a0)
    y.append(0.0)
    mat = np.vstack(rows)
    vec = np.asarray(y, dtype=float)
    sol, *_ = np.linalg.lstsq(mat, vec, rcond=None)
    nodes = pd.DataFrame(
        {
            "date": dates,
            "date_dt": pd.to_datetime(dates, format="%Y%m%d"),
            "relative_bperp_m": sol,
        }
    )
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--out-stem", default="results/pic_all/71_spatiotemporal_baseline_network")
    parser.add_argument("--summary", default="results/metadata/spatiotemporal_baseline_network_summary.json")
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    pairs = pd.read_csv(args.pairs)
    pairs["master"] = pairs["master"].astype(str)
    pairs["slave"] = pairs["slave"].astype(str)
    pairs["master_dt"] = pd.to_datetime(pairs["master"], format="%Y%m%d")
    pairs["slave_dt"] = pd.to_datetime(pairs["slave"], format="%Y%m%d")
    pairs["abs_bperp_m"] = pairs["bperp_m"].abs()
    nodes = solve_relative_baselines(pairs)
    node_lookup = nodes.set_index("date")["relative_bperp_m"].to_dict()

    fig = plt.figure(figsize=(11.2, 7.4), dpi=260)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], width_ratios=[1.45, 1.0, 1.0], hspace=0.32, wspace=0.28)
    ax_net = fig.add_subplot(gs[0, :])
    ax_dt = fig.add_subplot(gs[1, 0])
    ax_bp = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 2])

    norm = mpl.colors.Normalize(vmin=float(pairs["abs_bperp_m"].min()), vmax=float(pairs["abs_bperp_m"].max()))
    cmap = mpl.colormaps["viridis"]
    for row in pairs.itertuples(index=False):
        x = [row.master_dt, row.slave_dt]
        y = [node_lookup[row.master], node_lookup[row.slave]]
        color = cmap(norm(abs(float(row.bperp_m))))
        ax_net.plot(x, y, color=color, lw=1.15, alpha=0.78, zorder=1)
    ax_net.scatter(nodes["date_dt"], nodes["relative_bperp_m"], s=42, color="#222222", edgecolor="white", linewidth=0.5, zorder=3)
    for row in nodes.itertuples(index=False):
        ax_net.text(row.date_dt, row.relative_bperp_m + 18, row.date[4:], ha="center", va="bottom", fontsize=6.5, rotation=0)
    ax_net.axhline(0, color="0.55", lw=0.7, ls="--")
    ax_net.set_title(f"SBAS spatiotemporal baseline network ({len(nodes)} acquisitions, {len(pairs)} pairs)")
    ax_net.set_ylabel("Relative perpendicular baseline (m)")
    ax_net.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_net.grid(axis="y", color="0.9", lw=0.6)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_net, fraction=0.025, pad=0.012)
    cbar.set_label("|pair Bperp| (m)")

    bins_dt = np.arange(0, int(pairs["dt_days"].max()) + 12, 11)
    ax_dt.hist(pairs["dt_days"], bins=bins_dt, color="#4477aa", edgecolor="white")
    ax_dt.set_title("Temporal baseline")
    ax_dt.set_xlabel("Days")
    ax_dt.set_ylabel("Pairs")

    ax_bp.hist(pairs["bperp_m"], bins=18, color="#44aa99", edgecolor="white")
    ax_bp.axvline(0, color="0.25", lw=0.8)
    ax_bp.set_title("Perpendicular baseline")
    ax_bp.set_xlabel("Bperp (m)")
    ax_bp.set_ylabel("Pairs")

    sc = ax_scatter.scatter(pairs["dt_days"], pairs["bperp_m"], c=pairs["abs_bperp_m"], cmap=cmap, norm=norm, s=34, edgecolor="white", linewidth=0.35)
    ax_scatter.axhline(0, color="0.35", lw=0.8)
    ax_scatter.set_title("Pair baseline space")
    ax_scatter.set_xlabel("Temporal baseline (days)")
    ax_scatter.set_ylabel("Bperp (m)")
    ax_scatter.grid(color="0.92", lw=0.55)

    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    out_stem = Path(args.out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": f"{out_stem}.png",
        "svg": f"{out_stem}.svg",
        "pdf": f"{out_stem}.pdf",
    }
    fig.savefig(outputs["png"], dpi=320, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)

    nodes_out = out_stem.with_name(out_stem.name + "_nodes.csv")
    nodes.to_csv(nodes_out, index=False)
    summary = {
        "pairs_csv": args.pairs,
        "pair_count": int(len(pairs)),
        "acquisition_count": int(len(nodes)),
        "date_min": str(nodes["date"].min()),
        "date_max": str(nodes["date"].max()),
        "dt_days_min": int(pairs["dt_days"].min()),
        "dt_days_max": int(pairs["dt_days"].max()),
        "bperp_min_m": float(pairs["bperp_m"].min()),
        "bperp_max_m": float(pairs["bperp_m"].max()),
        "abs_bperp_median_m": float(pairs["abs_bperp_m"].median()),
        "relative_node_bperp_csv": str(nodes_out),
        "outputs": outputs,
        "note": "Node relative baselines are recovered by least squares from pair Bperp values with the first acquisition fixed to 0 m.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
