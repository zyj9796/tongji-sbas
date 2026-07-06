#!/usr/bin/env python3
"""Audit SBAS interferogram phase consistency with triangular closure residuals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_gamma_differential_island_observations import read_float, read_fcomplex


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def pair_key(a: str, b: str) -> str:
    return f"{a}_{b}"


def robust_mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def add_phase(
    accum: np.ndarray,
    phase_by_pair: dict[str, np.ndarray],
    a: str,
    b: str,
    sign: float,
) -> tuple[np.ndarray, str]:
    direct = pair_key(a, b)
    reverse = pair_key(b, a)
    if direct in phase_by_pair:
        return accum + sign * phase_by_pair[direct], direct
    if reverse in phase_by_pair:
        return accum - sign * phase_by_pair[reverse], reverse
    raise KeyError((a, b))


def plot_outputs(
    closure_mad: np.ndarray,
    triangle_stats: pd.DataFrame,
    pair_stats: pd.DataFrame,
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), dpi=220)
    finite = closure_mad[np.isfinite(closure_mad)]
    lo, hi = (0.0, 1.0) if finite.size == 0 else np.nanpercentile(finite, [2, 98])
    if lo == hi:
        hi = lo + 1.0
    im = axes[0, 0].imshow(closure_mad, cmap="magma", vmin=lo, vmax=hi)
    axes[0, 0].set_title("Per-pixel closure MAD, rad")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.02)

    axes[0, 1].hist(finite, bins=80, color="#4477aa", edgecolor="white")
    axes[0, 1].set_title("Closure MAD distribution")
    axes[0, 1].set_xlabel("rad")
    axes[0, 1].set_ylabel("pixels")

    tri = triangle_stats.sort_values("closure_mad_rad")
    axes[1, 0].scatter(range(len(tri)), tri["closure_mad_rad"], c=tri["mean_coherence"], cmap="viridis", s=24)
    axes[1, 0].set_title("Triangle closure quality")
    axes[1, 0].set_xlabel("triangles sorted by MAD")
    axes[1, 0].set_ylabel("closure MAD, rad")

    pr = pair_stats.sort_values("median_triangle_closure_mad_rad", ascending=False).head(20)
    axes[1, 1].barh(range(len(pr)), pr["median_triangle_closure_mad_rad"], color="#cc6677")
    axes[1, 1].set_yticks(range(len(pr)))
    axes[1, 1].set_yticklabels(pr["pair"], fontsize=7)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Worst pairs by linked closure MAD")
    axes[1, 1].set_xlabel("rad")

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--intf-root", required=True)
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--min-coherence", type=float, default=0.35)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for sub in ["arrays", "figures", "tables", "metadata"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv)
    dates = sorted(set(pairs["master"].astype(str)).union(set(pairs["slave"].astype(str))))
    available = {pair_key(str(r.master), str(r.slave)): r for r in pairs.itertuples(index=False)}

    phase_by_pair: dict[str, np.ndarray] = {}
    coh_by_pair: dict[str, np.ndarray] = {}
    for row in pairs.itertuples(index=False):
        master = str(row.master)
        slave = str(row.slave)
        pair = pair_key(master, slave)
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        valid = np.isfinite(cc) & (cc >= args.min_coherence) & (np.abs(diff) > 0.0)
        phase_by_pair[pair] = np.where(valid, np.angle(diff), np.nan).astype(np.float32)
        coh_by_pair[pair] = np.where(valid, cc, np.nan).astype(np.float32)

    closure_abs_sum = np.zeros((args.rows, args.cols), dtype=np.float64)
    closure_count = np.zeros((args.rows, args.cols), dtype=np.int16)
    triangle_rows = []
    pair_to_closures: dict[str, list[float]] = {k: [] for k in available}

    for i, a in enumerate(dates):
        for j in range(i + 1, len(dates)):
            b = dates[j]
            for k in range(j + 1, len(dates)):
                c = dates[k]
                if not (
                    (pair_key(a, b) in available or pair_key(b, a) in available)
                    and (pair_key(b, c) in available or pair_key(c, b) in available)
                    and (pair_key(a, c) in available or pair_key(c, a) in available)
                ):
                    continue
                zero = np.zeros((args.rows, args.cols), dtype=np.float32)
                accum, p_ab = add_phase(zero, phase_by_pair, a, b, 1.0)
                accum, p_bc = add_phase(accum, phase_by_pair, b, c, 1.0)
                accum, p_ac = add_phase(accum, phase_by_pair, a, c, -1.0)
                closure = wrap_phase(accum)
                valid = np.isfinite(closure)
                if int(valid.sum()) == 0:
                    continue
                closure_abs = np.abs(closure)
                closure_abs_sum[valid] += closure_abs[valid]
                closure_count[valid] += 1
                mean_coh = np.nanmean(np.stack([coh_by_pair[p_ab], coh_by_pair[p_bc], coh_by_pair[p_ac]]), axis=0)
                mad = robust_mad(closure[valid])
                row = {
                    "date_a": a,
                    "date_b": b,
                    "date_c": c,
                    "pair_ab": p_ab,
                    "pair_bc": p_bc,
                    "pair_ac": p_ac,
                    "valid_pixels": int(valid.sum()),
                    "closure_median_abs_rad": float(np.nanmedian(closure_abs[valid])),
                    "closure_mad_rad": mad,
                    "closure_p90_abs_rad": float(np.nanpercentile(closure_abs[valid], 90)),
                    "mean_coherence": float(np.nanmean(mean_coh[valid])),
                }
                triangle_rows.append(row)
                for p in [p_ab, p_bc, p_ac]:
                    pair_to_closures[p].append(mad)

    closure_mean_abs = np.full((args.rows, args.cols), np.nan, dtype=np.float32)
    ok = closure_count > 0
    closure_mean_abs[ok] = (closure_abs_sum[ok] / closure_count[ok]).astype(np.float32)
    np.save(out_dir / "arrays/phase_closure_mean_abs_rad.npy", closure_mean_abs)
    np.save(out_dir / "arrays/phase_closure_count.npy", closure_count)

    triangle_stats = pd.DataFrame(triangle_rows)
    pair_rows = []
    for pair in available:
        vals = np.array(pair_to_closures[pair], dtype=float)
        pair_rows.append(
            {
                "pair": pair,
                "triangle_count": int(np.isfinite(vals).sum()),
                "median_triangle_closure_mad_rad": float(np.nanmedian(vals)) if vals.size else np.nan,
                "p90_triangle_closure_mad_rad": float(np.nanpercentile(vals, 90)) if vals.size else np.nan,
            }
        )
    pair_stats = pd.DataFrame(pair_rows)
    triangle_stats.to_csv(out_dir / "tables/triangle_phase_closure_summary.csv", index=False)
    pair_stats.to_csv(out_dir / "tables/pair_phase_closure_summary.csv", index=False)
    plot_outputs(closure_mean_abs, triangle_stats, pair_stats, out_dir / "figures/sbas_phase_closure_audit")

    summary = {
        "pairs_csv": args.pairs_csv,
        "intf_root": args.intf_root,
        "dates": dates,
        "n_pairs": int(len(pairs)),
        "n_triangles": int(len(triangle_stats)),
        "min_coherence": args.min_coherence,
        "closure_mean_abs_median_rad": float(np.nanmedian(closure_mean_abs)),
        "closure_mean_abs_p90_rad": float(np.nanpercentile(closure_mean_abs, 90)),
        "triangle_closure_mad_median_rad": float(triangle_stats["closure_mad_rad"].median()) if not triangle_stats.empty else None,
        "triangle_closure_mad_p90_rad": float(triangle_stats["closure_mad_rad"].quantile(0.9)) if not triangle_stats.empty else None,
        "worst_pairs": pair_stats.sort_values("median_triangle_closure_mad_rad", ascending=False).head(10).to_dict(orient="records"),
    }
    (out_dir / "metadata/phase_closure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
