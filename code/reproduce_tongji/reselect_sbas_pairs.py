#!/usr/bin/env python3
"""Reselect SBAS pairs and plot the proposed baseline network.

No interferograms are generated. Pair Bperp values are approximate center-scene
estimates from RSLC parameter state vectors, intended for network design before
running precise GAMMA baseline products.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estimate_pair_baselines_from_par import estimate_bperp, height_ambiguity, parse_par


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y%m%d")


def all_candidate_pairs(rslc_dir: Path, max_abs_bperp: float) -> pd.DataFrame:
    pars = {p.name.split(".")[0]: parse_par(p) for p in sorted(rslc_dir.glob("*.rslc.par"))}
    dates = sorted(pars)
    rows = []
    c = 299792458.0
    for i, master in enumerate(dates):
        for slave in dates[i + 1 :]:
            dt_days = (parse_date(slave) - parse_date(master)).days
            bperp = estimate_bperp(pars[master], pars[slave])
            if abs(bperp) > max_abs_bperp:
                continue
            wavelength = c / pars[master]["radar_frequency"]
            hamb = height_ambiguity(wavelength, pars[master]["center_range_slc"], pars[master]["incidence_angle"], bperp)
            rows.append(
                {
                    "master": master,
                    "slave": slave,
                    "dt_days": int(dt_days),
                    "bperp_m": float(bperp),
                    "abs_bperp_m": abs(float(bperp)),
                    "height_ambiguity_m": hamb,
                }
            )
    return pd.DataFrame(rows)


def components(dates: list[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    graph: dict[str, list[str]] = {date: [] for date in dates}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    seen = set()
    comps = []
    for date in dates:
        if date in seen:
            continue
        comp = set()
        q: deque[str] = deque([date])
        seen.add(date)
        while q:
            cur = q.popleft()
            comp.add(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def degrees(dates: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    deg = {date: 0 for date in dates}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def edge_key(row: pd.Series) -> tuple[float, float, float]:
    return (float(row["dt_days"]), float(row["abs_bperp_m"]), abs(float(row["bperp_m"])))


def row_pair(row: pd.Series) -> tuple[str, str]:
    return str(row["master"]), str(row["slave"])


def select_pairs(candidates: pd.DataFrame, dates: list[str], min_degree: int, target_min: int, target_max: int) -> pd.DataFrame:
    stage1 = candidates[(candidates["dt_days"] <= 48) & (candidates["abs_bperp_m"] <= 300)].copy()
    stage2 = candidates[(candidates["dt_days"] <= 72) & (candidates["abs_bperp_m"] <= 400)].copy()
    bridge_pool = candidates.copy()
    selected: dict[tuple[str, str], str] = {}

    # Start from all preferred short/small-baseline pairs. This usually gives
    # better degree balance than a minimum spanning tree while staying in the
    # requested 70-120 pair range.
    for row in stage1.sort_values(["dt_days", "abs_bperp_m"]).itertuples(index=False):
        selected[(str(row.master), str(row.slave))] = "preferred_dt48_bperp300"

    def selected_edges() -> list[tuple[str, str]]:
        return list(selected)

    # Add stage-2 relaxed pairs only when they connect components or help
    # under-connected acquisitions.
    for pool, label in [(stage2, "relaxed_dt72_bperp400"), (bridge_pool, "connectivity_bridge_dt_override")]:
        while True:
            comps = components(dates, selected_edges())
            if len(comps) <= 1:
                break
            comp_id = {d: i for i, comp in enumerate(comps) for d in comp}
            eligible = []
            for idx, row in pool.iterrows():
                pair = row_pair(row)
                if pair in selected:
                    continue
                if comp_id[pair[0]] != comp_id[pair[1]]:
                    eligible.append((edge_key(row), idx))
            if not eligible:
                break
            _, idx = min(eligible)
            row = pool.loc[idx]
            selected[row_pair(row)] = label

    # Degree balancing.
    for pool, label in [
        (stage1, "preferred_dt48_bperp300"),
        (stage2, "relaxed_dt72_bperp400"),
        (bridge_pool, "degree_bridge_dt_override"),
    ]:
        changed = True
        while changed:
            changed = False
            deg = degrees(dates, selected_edges())
            low = {d for d, v in deg.items() if v < min_degree}
            if not low:
                break
            eligible = []
            for idx, row in pool.iterrows():
                pair = row_pair(row)
                if pair in selected:
                    continue
                if pair[0] in low or pair[1] in low:
                    eligible.append((edge_key(row), idx))
            if not eligible:
                break
            _, idx = min(eligible)
            row = pool.loc[idx]
            selected[row_pair(row)] = label
            changed = True
            if len(selected) >= target_max:
                break
        if len(selected) >= target_max:
            break

    # If still below target, add best remaining stage2 pairs first.
    for pool, label in [
        (stage2, "relaxed_dt72_bperp400"),
        (stage1, "preferred_dt48_bperp300"),
        (bridge_pool, "target_count_bridge_dt_override"),
    ]:
        if len(selected) >= target_min:
            break
        for row in pool.sort_values(["dt_days", "abs_bperp_m"]).itertuples(index=False):
            pair = (str(row.master), str(row.slave))
            if pair not in selected:
                selected[pair] = label
            if len(selected) >= target_min:
                break

    out = candidates.copy()
    out["selection_class"] = [selected.get((str(r.master), str(r.slave)), "") for r in out.itertuples(index=False)]
    return out[out["selection_class"] != ""].copy().sort_values(["master", "slave"]).reset_index(drop=True)


def solve_relative_baselines(pairs: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    idx = {d: i for i, d in enumerate(dates)}
    rows = []
    y = []
    for row in pairs.itertuples(index=False):
        a = np.zeros(len(dates), dtype=float)
        a[idx[str(row.slave)]] = 1.0
        a[idx[str(row.master)]] = -1.0
        rows.append(a)
        y.append(float(row.bperp_m))
    a0 = np.zeros(len(dates), dtype=float)
    a0[0] = 1.0
    rows.append(a0)
    y.append(0.0)
    sol, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(y), rcond=None)
    return pd.DataFrame({"date": dates, "date_dt": pd.to_datetime(dates, format="%Y%m%d"), "relative_bperp_m": sol})


def plot_network(selected: pd.DataFrame, dates: list[str], out_stem: Path, summary: dict) -> dict[str, str]:
    nodes = solve_relative_baselines(selected, dates)
    node_b = nodes.set_index("date")["relative_bperp_m"].to_dict()
    selected = selected.copy()
    selected["master_dt"] = pd.to_datetime(selected["master"], format="%Y%m%d")
    selected["slave_dt"] = pd.to_datetime(selected["slave"], format="%Y%m%d")
    colors = {
        "preferred_dt48_bperp300": "#117733",
        "relaxed_dt72_bperp400": "#4477aa",
        "connectivity_bridge_dt_override": "#cc6677",
        "degree_bridge_dt_override": "#ddaa33",
    }
    widths = defaultdict(lambda: 0.85, {"connectivity_bridge_dt_override": 1.65, "degree_bridge_dt_override": 1.35})
    fig = plt.figure(figsize=(12.2, 8.0), dpi=260)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], width_ratios=[1.5, 1.0, 1.0], hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax_dt = fig.add_subplot(gs[1, 0])
    ax_bp = fig.add_subplot(gs[1, 1])
    ax_deg = fig.add_subplot(gs[1, 2])
    for row in selected.itertuples(index=False):
        cls = str(row.selection_class)
        ax.plot(
            [row.master_dt, row.slave_dt],
            [node_b[row.master], node_b[row.slave]],
            color=colors.get(cls, "0.45"),
            lw=widths[cls],
            alpha=0.74,
            zorder=1,
        )
    ax.scatter(nodes["date_dt"], nodes["relative_bperp_m"], s=36, color="#222222", edgecolor="white", linewidth=0.45, zorder=3)
    for row in nodes.itertuples(index=False):
        ax.text(row.date_dt, row.relative_bperp_m + 15, row.date[4:], ha="center", va="bottom", fontsize=6.1)
    ax.axhline(0, color="0.55", lw=0.7, ls="--")
    ax.set_ylabel("Relative perpendicular baseline (m)")
    ax.set_title(f"Reselected SBAS baseline network ({len(dates)} acquisitions, {len(selected)} pairs)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", color="0.9", lw=0.6)
    handles = [mpl.lines.Line2D([0], [0], color=v, lw=2, label=k.replace("_", " ")) for k, v in colors.items() if (selected["selection_class"] == k).any()]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2)

    ax_dt.hist(selected["dt_days"], bins=np.arange(0, int(selected["dt_days"].max()) + 18, 12), color="#4477aa", edgecolor="white")
    ax_dt.axvline(48, color="#117733", ls="--", lw=1.0)
    ax_dt.axvline(72, color="#4477aa", ls="--", lw=1.0)
    ax_dt.set_title("Temporal baseline")
    ax_dt.set_xlabel("Days")
    ax_dt.set_ylabel("Pairs")
    ax_bp.hist(selected["bperp_m"], bins=22, color="#44aa99", edgecolor="white")
    ax_bp.axvline(-300, color="#117733", ls="--", lw=1.0)
    ax_bp.axvline(300, color="#117733", ls="--", lw=1.0)
    ax_bp.axvline(-400, color="#4477aa", ls=":", lw=1.0)
    ax_bp.axvline(400, color="#4477aa", ls=":", lw=1.0)
    ax_bp.set_title("Perpendicular baseline")
    ax_bp.set_xlabel("Bperp (m)")
    ax_bp.set_ylabel("Pairs")
    deg = pd.Series(degrees(dates, [(str(r.master), str(r.slave)) for r in selected.itertuples(index=False)]))
    ax_deg.bar(range(len(deg)), deg.sort_index().to_numpy(), color="#88ccee", width=0.8)
    ax_deg.axhline(2, color="#cc6677", ls="--", lw=1.0)
    ax_deg.set_title("Acquisition degree")
    ax_deg.set_xlabel("Acquisitions")
    ax_deg.set_ylabel("Pair count")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {"png": f"{out_stem}.png", "svg": f"{out_stem}.svg"}
    fig.savefig(outputs["png"], dpi=320, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    nodes.to_csv(out_stem.with_name(out_stem.name + "_nodes.csv"), index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--output-csv", default="work/baselines/reselected_sbas_pairs_approx.csv")
    parser.add_argument("--summary", default="results/metadata/reselected_sbas_pairs_summary.json")
    parser.add_argument("--out-stem", default="results/pic_all/72_reselected_sbas_baseline_network")
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--target-min", type=int, default=70)
    parser.add_argument("--target-max", type=int, default=120)
    parser.add_argument("--max-abs-bperp", type=float, default=500.0)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    dates = sorted(p.name.split(".")[0] for p in Path(args.rslc_dir).glob("*.rslc.par"))
    candidates = all_candidate_pairs(Path(args.rslc_dir), args.max_abs_bperp)
    selected = select_pairs(candidates, dates, args.min_degree, args.target_min, args.target_max)
    edges = [(str(r.master), str(r.slave)) for r in selected.itertuples(index=False)]
    comps = components(dates, edges)
    deg = degrees(dates, edges)
    outputs = plot_network(selected, dates, Path(args.out_stem), {})
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    summary = {
        "dates": len(dates),
        "candidate_pairs_after_bperp500": int(len(candidates)),
        "selected_pairs": int(len(selected)),
        "components": int(len(comps)),
        "connected": len(comps) == 1,
        "min_degree": int(min(deg.values())) if deg else 0,
        "degree_lt2_count": int(sum(v < args.min_degree for v in deg.values())),
        "dt_days_min": int(selected["dt_days"].min()),
        "dt_days_max": int(selected["dt_days"].max()),
        "bperp_min_m": float(selected["bperp_m"].min()),
        "bperp_max_m": float(selected["bperp_m"].max()),
        "abs_bperp_max_m": float(selected["abs_bperp_m"].max()),
        "selection_counts": {str(k): int(v) for k, v in selected["selection_class"].value_counts().items()},
        "output_csv": args.output_csv,
        "outputs": outputs,
        "note": "No interferograms were generated. Bperp values are approximate state-vector estimates for pair reselection; run precise GAMMA baseline estimation before interferogram generation.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
