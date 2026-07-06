#!/usr/bin/env python3
"""Select a reduced high-quality triangular SBAS network."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from estimate_pair_baselines_from_par import estimate_bperp, height_ambiguity, parse_par


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y%m%d")


def all_pairs(pars: dict[str, dict], dates: list[str]) -> pd.DataFrame:
    rows = []
    c = 299792458.0
    for i, master in enumerate(dates):
        for slave in dates[i + 1 :]:
            dt_days = (parse_date(slave) - parse_date(master)).days
            bperp = estimate_bperp(pars[master], pars[slave])
            wavelength = c / pars[master]["radar_frequency"]
            rows.append(
                {
                    "master": master,
                    "slave": slave,
                    "dt_days": int(dt_days),
                    "bperp_m": float(bperp),
                    "abs_bperp_m": abs(float(bperp)),
                    "height_ambiguity_m": height_ambiguity(
                        wavelength,
                        pars[master]["center_range_slc"],
                        pars[master]["incidence_angle"],
                        bperp,
                    ),
                }
            )
    return pd.DataFrame(rows)


def solve_node_baselines(pairs: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
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


def graph_degrees(dates: list[str], edges: set[tuple[str, str]]) -> dict[str, int]:
    deg = {d: 0 for d in dates}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def components(dates: list[str], edges: set[tuple[str, str]]) -> list[set[str]]:
    graph = {d: [] for d in dates}
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


def choose_best_window(pairs_all: pd.DataFrame, dates: list[str], min_images: int) -> list[str]:
    scored = []
    for n in range(min_images, min(31, len(dates)) + 1):
        for s in range(0, len(dates) - n + 1):
            sub = dates[s : s + n]
            p = pairs_all[pairs_all["master"].isin(sub) & pairs_all["slave"].isin(sub)]
            pref = p[(p["dt_days"] <= 72) & (p["abs_bperp_m"] <= 300)]
            if pref.empty:
                continue
            max_gap = max((parse_date(sub[i + 1]) - parse_date(sub[i])).days for i in range(n - 1))
            comps = components(sub, set(zip(pref["master"], pref["slave"])))
            scored.append(
                (
                    len(comps),
                    max_gap,
                    float(pref["abs_bperp_m"].quantile(0.90)),
                    -n,
                    s,
                    sub,
                )
            )
    scored.sort()
    return scored[0][-1]


def triangular_edges(nodes: pd.DataFrame, candidates: pd.DataFrame, min_degree: int, target_pairs: int) -> pd.DataFrame:
    dates = nodes["date"].tolist()
    x = (nodes["date_dt"] - nodes["date_dt"].min()).dt.days.to_numpy(dtype=float)
    y = nodes["relative_bperp_m"].to_numpy(dtype=float)
    # Scale axes so Delaunay balances temporal and baseline dimensions.
    coords = np.column_stack([x / max(np.nanstd(x), 1.0), y / max(np.nanstd(y), 1.0)])
    tri = Delaunay(coords)
    delaunay_pairs: set[tuple[str, str]] = set()
    for simplex in tri.simplices:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            a = dates[int(simplex[i])]
            b = dates[int(simplex[j])]
            if a > b:
                a, b = b, a
            delaunay_pairs.add((a, b))

    cand = candidates[(candidates["dt_days"] <= 72) & (candidates["abs_bperp_m"] <= 300)].copy()
    relaxed = candidates[(candidates["dt_days"] <= 72) & (candidates["abs_bperp_m"] <= 400)].copy()
    hard_limit = candidates[(candidates["dt_days"] <= 72) & (candidates["abs_bperp_m"] <= 500)].copy()
    cand["pair"] = list(zip(cand["master"], cand["slave"]))
    relaxed["pair"] = list(zip(relaxed["master"], relaxed["slave"]))
    hard_limit["pair"] = list(zip(hard_limit["master"], hard_limit["slave"]))
    cand_pairs = set(cand["pair"])
    edges = {p for p in delaunay_pairs if p in cand_pairs}
    edge_class = {p: "delaunay_dt72_bperp300" for p in edges}

    # Ensure temporal neighbor chain is present.
    for a, b in zip(dates[:-1], dates[1:]):
        p = (a, b)
        if p in cand_pairs:
            edges.add(p)
            edge_class.setdefault(p, "temporal_neighbor")

    def edge_score(row: pd.Series) -> tuple[float, float]:
        return float(row["dt_days"]), float(row["abs_bperp_m"])

    # Degree balancing: preferred candidates first, then relaxed candidates. The
    # final pool is only used to satisfy the hard min-degree constraint.
    for pool_df, label in [
        (cand, "degree_fill_dt72_bperp300"),
        (relaxed, "degree_fill_dt72_bperp400"),
        (hard_limit, "degree_fill_dt72_bperp500"),
    ]:
        while True:
            deg = graph_degrees(dates, edges)
            low = {d for d, v in deg.items() if v < min_degree}
            if not low:
                break
            pool = pool_df[~pool_df["pair"].isin(edges) & (pool_df["master"].isin(low) | pool_df["slave"].isin(low))]
            if pool.empty:
                break
            idx = min(pool.index, key=lambda i: edge_score(pool.loc[i]))
            p = pool_df.loc[idx, "pair"]
            edges.add(p)
            edge_class[p] = label

    for pool_df, label in [(cand, "density_fill_dt72_bperp300"), (relaxed, "density_fill_dt72_bperp400")]:
        while len(edges) < target_pairs:
            pool = pool_df[~pool_df["pair"].isin(edges)]
            if pool.empty:
                break
            idx = min(pool.index, key=lambda i: edge_score(pool.loc[i]))
            p = pool_df.loc[idx, "pair"]
            edges.add(p)
            edge_class[p] = label

    out = candidates[candidates.apply(lambda r: (r["master"], r["slave"]) in edges, axis=1)].copy()
    out["selection_class"] = [edge_class[(r.master, r.slave)] for r in out.itertuples(index=False)]
    return out.sort_values(["master", "slave"]).reset_index(drop=True)


def plot(nodes: pd.DataFrame, selected: pd.DataFrame, out_stem: Path) -> dict[str, str]:
    node_b = nodes.set_index("date")["relative_bperp_m"].to_dict()
    selected = selected.copy()
    selected["master_dt"] = pd.to_datetime(selected["master"], format="%Y%m%d")
    selected["slave_dt"] = pd.to_datetime(selected["slave"], format="%Y%m%d")
    colors = {
        "delaunay_dt72_bperp300": "#117733",
        "temporal_neighbor": "#4477aa",
        "degree_fill_dt72_bperp300": "#ddaa33",
        "degree_fill_dt72_bperp400": "#cc6677",
        "degree_fill_dt72_bperp500": "#882255",
        "density_fill_dt72_bperp300": "#88ccee",
        "density_fill_dt72_bperp400": "#aa4499",
    }
    fig = plt.figure(figsize=(11.5, 7.5), dpi=260)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], width_ratios=[1.4, 1.0, 1.0], hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax_dt = fig.add_subplot(gs[1, 0])
    ax_bp = fig.add_subplot(gs[1, 1])
    ax_deg = fig.add_subplot(gs[1, 2])
    for row in selected.itertuples(index=False):
        cls = str(row.selection_class)
        ax.plot(
            [row.master_dt, row.slave_dt],
            [node_b[row.master], node_b[row.slave]],
            color=colors.get(cls, "0.5"),
            lw=1.0 if cls != "temporal_neighbor" else 1.25,
            alpha=0.74,
        )
    ax.scatter(nodes["date_dt"], nodes["relative_bperp_m"], s=38, color="#222222", edgecolor="white", linewidth=0.45, zorder=3)
    for row in nodes.itertuples(index=False):
        ax.text(row.date_dt, row.relative_bperp_m + 11, row.date[4:], ha="center", va="bottom", fontsize=6.2)
    ax.axhline(0, color="0.55", lw=0.7, ls="--")
    ax.set_title(f"Reduced triangular SBAS network ({len(nodes)} acquisitions, {len(selected)} pairs)")
    ax.set_ylabel("Relative perpendicular baseline (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", color="0.9", lw=0.6)
    handles = [mpl.lines.Line2D([0], [0], color=v, lw=2.0, label=k.replace("_", " ")) for k, v in colors.items() if (selected["selection_class"] == k).any()]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2)

    ax_dt.hist(selected["dt_days"], bins=np.arange(0, int(selected["dt_days"].max()) + 12, 12), color="#4477aa", edgecolor="white")
    ax_dt.axvline(48, color="#117733", ls="--", lw=1.0)
    ax_dt.axvline(72, color="#4477aa", ls=":", lw=1.0)
    ax_dt.set_title("Temporal baseline")
    ax_dt.set_xlabel("Days")
    ax_dt.set_ylabel("Pairs")
    ax_bp.hist(selected["bperp_m"], bins=16, color="#44aa99", edgecolor="white")
    ax_bp.axvline(-300, color="#117733", ls="--", lw=1.0)
    ax_bp.axvline(300, color="#117733", ls="--", lw=1.0)
    ax_bp.set_title("Perpendicular baseline")
    ax_bp.set_xlabel("Bperp (m)")
    ax_bp.set_ylabel("Pairs")
    deg = graph_degrees(nodes["date"].tolist(), set(zip(selected["master"], selected["slave"])))
    ax_deg.bar(range(len(deg)), pd.Series(deg).sort_index().to_numpy(), color="#88ccee", width=0.8)
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
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--target-pairs", type=int, default=55)
    parser.add_argument("--output-csv", default="work/baselines/reselected_sbas_pairs_triangular_22_approx.csv")
    parser.add_argument("--summary", default="results/metadata/reselected_sbas_pairs_triangular_summary.json")
    parser.add_argument("--out-stem", default="results/pic_all/73_reselected_sbas_triangular_network")
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
    rslc = Path(args.rslc_dir)
    pars = {p.name.split(".")[0]: parse_par(p) for p in sorted(rslc.glob("*.rslc.par"))}
    all_dates = sorted(pars)
    allp = all_pairs(pars, all_dates)
    selected_dates = choose_best_window(allp, all_dates, args.min_images)
    subp = allp[allp["master"].isin(selected_dates) & allp["slave"].isin(selected_dates)].copy()
    nodes = solve_node_baselines(subp, selected_dates)
    selected = triangular_edges(nodes, subp, args.min_degree, args.target_pairs)
    edges = set(zip(selected["master"], selected["slave"]))
    deg = graph_degrees(selected_dates, edges)
    comps = components(selected_dates, edges)
    outputs = plot(nodes, selected, Path(args.out_stem))
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    nodes_csv = Path(args.out_stem).with_name(Path(args.out_stem).name + "_nodes.csv")
    nodes.to_csv(nodes_csv, index=False)
    summary = {
        "selected_dates": selected_dates,
        "acquisitions": len(selected_dates),
        "selected_pairs": int(len(selected)),
        "connected": len(comps) == 1,
        "components": len(comps),
        "min_degree": int(min(deg.values())),
        "degree_lt2_count": int(sum(v < args.min_degree for v in deg.values())),
        "dt_days_min": int(selected["dt_days"].min()),
        "dt_days_max": int(selected["dt_days"].max()),
        "abs_bperp_max_m": float(selected["abs_bperp_m"].max()),
        "abs_bperp_median_m": float(selected["abs_bperp_m"].median()),
        "selection_counts": {str(k): int(v) for k, v in selected["selection_class"].value_counts().items()},
        "output_csv": args.output_csv,
        "nodes_csv": str(nodes_csv),
        "outputs": outputs,
        "note": "Reduced 22-acquisition approximate triangular SBAS network. No interferograms were generated; run precise GAMMA baseline estimation before production interferograms.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
