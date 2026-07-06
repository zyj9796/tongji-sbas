#!/usr/bin/env python3
"""Benchmark paper-like KNN/Bisquare/LSMR island unwrap against skimage unwrap."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from scipy.spatial import Delaunay, cKDTree
from skimage.restoration import unwrap_phase

from extract_gamma_differential_island_observations import read_fcomplex, read_float
from inventory_data import parse_gamma_par


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def design(bperp: np.ndarray, dt_days: np.ndarray, wavelength: float, range_m: float, inc_deg: float) -> np.ndarray:
    return np.column_stack(
        [
            4.0 * math.pi * bperp / (wavelength * range_m * math.sin(math.radians(inc_deg))),
            4.0 * math.pi * dt_days / wavelength,
        ]
    )


def robust_bisquare_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 30, tune: float = 4.685) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[keep]
    y = y[keep]
    if len(y) < x.shape[1] + 1:
        return np.full(x.shape[1], np.nan), np.full(len(keep), np.nan)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    for _ in range(max_iter):
        resid = y - x @ coef
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = max(1.4826 * mad, 1e-6)
        u = resid / (tune * scale)
        w = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if np.count_nonzero(w > 0) < x.shape[1] + 1:
            break
        xw = x * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        new_coef, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        if np.linalg.norm(new_coef - coef) < 1e-6:
            coef = new_coef
            break
        coef = new_coef
    full_resid = np.full(len(keep), np.nan)
    full_resid[keep] = y - x @ coef
    return coef, full_resid


def local_arcs(coords: np.ndarray, n_knn: int) -> np.ndarray:
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(n_knn, n)
    _, idx = cKDTree(coords).query(coords, k=k)
    idx = np.atleast_2d(idx)
    arcs = []
    for i in range(n):
        for j in idx[i, 1:]:
            a, b = sorted((i, int(j)))
            if a != b:
                arcs.append((a, b))
    if n >= 4:
        try:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                for a, b in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])):
                    a, b = sorted((int(a), int(b)))
                    if a != b:
                        arcs.append((a, b))
        except Exception:
            pass
    if not arcs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.asarray(arcs, dtype=np.int64), axis=0)


def paper_like_unwrap(phase_patch: np.ndarray, valid_mask: np.ndarray, wavelength_m: float) -> tuple[np.ndarray, dict[str, float | int]]:
    rr, cc = np.nonzero(valid_mask & np.isfinite(phase_patch))
    n = len(rr)
    out = np.full_like(phase_patch, np.nan, dtype=np.float32)
    if n < 10 or phase_patch.shape[0] == 1 or phase_patch.shape[1] <= 4:
        out[rr, cc] = phase_patch[rr, cc]
        return out, {"points": int(n), "arcs": 0, "kept_arcs": 0, "status": "too_small"}

    coords = np.column_stack([cc.astype(np.float64), rr.astype(np.float64)])
    obs = phase_patch[rr, cc].astype(np.float64)
    tree = cKDTree(coords)
    k_nearest = min(9, n)
    _, clusters = tree.query(coords, k=k_nearest)
    clusters = np.atleast_2d(clusters)
    threshold = 4.0 * math.pi * (math.sqrt(2.0) * 1.0) / (wavelength_m * 1000.0)

    arc_from = []
    arc_to = []
    arc_obs = []
    arc_resid = []
    total_arcs = 0
    for cluster in clusters:
        cluster = np.asarray(cluster, dtype=np.int64)
        ccoords = coords[cluster]
        cobs = obs[cluster]
        arcs = local_arcs(ccoords, min(4, len(cluster)))
        if len(arcs) == 0:
            continue
        total_arcs += len(arcs)
        x0 = ccoords[:, 0]
        y0 = ccoords[:, 1]
        xr = (x0 - x0.min()) / max(float(x0.max() - x0.min()), 1e-6)
        yr = (y0 - y0.min()) / max(float(y0.max() - y0.min()), 1e-6)
        f = arcs[:, 0]
        t = arcs[:, 1]
        poly = np.column_stack([xr[t] - xr[f], yr[t] - yr[f], xr[t] * yr[t] - xr[f] * yr[f]])
        y = wrap_phase(cobs[t] - cobs[f])
        coef, resid = robust_bisquare_fit(poly, y)
        if not np.isfinite(coef).all():
            continue
        pred = poly @ coef
        keep = np.isfinite(resid) & (np.abs(resid) < threshold)
        if not np.any(keep):
            continue
        arc_from.extend(cluster[f[keep]].tolist())
        arc_to.extend(cluster[t[keep]].tolist())
        arc_obs.extend(pred[keep].tolist())
        arc_resid.extend(resid[keep].tolist())

    if not arc_obs:
        return out, {"points": int(n), "arcs": int(total_arcs), "kept_arcs": 0, "status": "no_arcs"}

    arc_from_arr = np.asarray(arc_from, dtype=np.int64)
    arc_to_arr = np.asarray(arc_to, dtype=np.int64)
    b = np.asarray(arc_obs, dtype=np.float64)
    m = len(b)
    rows = np.arange(m)
    mat = coo_matrix(
        (
            np.r_[np.full(m, -1.0), np.full(m, 1.0)],
            (np.r_[rows, rows], np.r_[arc_from_arr, arc_to_arr]),
        ),
        shape=(m, n),
    ).tocsr()
    active = np.flatnonzero(np.asarray(np.abs(mat).sum(axis=0)).ravel() > 0)
    if len(active) < 2:
        return out, {"points": int(n), "arcs": int(total_arcs), "kept_arcs": int(m), "status": "rank_fail"}
    reduced = mat[:, active[1:]]
    sol = lsmr(reduced, b, atol=1e-7, btol=1e-7, maxiter=500)[0]
    full = np.full(n, np.nan, dtype=np.float64)
    full[active] = np.r_[0.0, sol]
    out[rr, cc] = full.astype(np.float32)
    return out, {
        "points": int(n),
        "arcs": int(total_arcs),
        "kept_arcs": int(m),
        "status": "ok",
        "median_abs_local_resid": float(np.nanmedian(np.abs(arc_resid))) if arc_resid else np.nan,
    }


def solve_dem(phase_rows: list[np.ndarray], coh_rows: list[np.ndarray], a: np.ndarray, min_pairs: int) -> np.ndarray:
    phases = np.stack(phase_rows, axis=0)
    cohs = np.stack(coh_rows, axis=0)
    n_pairs, n_pix = phases.shape
    dem = np.full(n_pix, np.nan, dtype=np.float32)
    for j in range(n_pix):
        valid = np.isfinite(phases[:, j]) & np.isfinite(cohs[:, j]) & (cohs[:, j] > 0)
        if int(np.sum(valid)) < min_pairs:
            continue
        w = np.clip(cohs[valid, j], 0.05, 1.0)
        aw = a[valid] * w[:, None]
        yw = phases[valid, j] * w
        coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        dem[j] = coef[0]
    return dem


def height_range(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return float("nan")
    p05, p95 = np.percentile(vals, [5, 95])
    return float(p95 - p05)


def plot_results(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=220)
    ok = df[df["paper_status"] == "ok"].copy()
    axes[0].hist(ok["phase_diff_std_rad"].dropna(), bins=30, color="#4477aa", edgecolor="white")
    axes[0].set_title("Paper-like vs skimage phase diff")
    axes[0].set_xlabel("std rad after median removal")
    axes[1].scatter(ok["skimage_height_m"], ok["paper_height_m"], s=22, alpha=0.75, color="#117733")
    lim = float(np.nanmax([ok["skimage_height_m"].max(), ok["paper_height_m"].max(), 1.0])) if not ok.empty else 1.0
    axes[1].plot([0, lim], [0, lim], color="#333333", linewidth=0.8)
    axes[1].set_title("Selected-island height range")
    axes[1].set_xlabel("skimage m")
    axes[1].set_ylabel("paper-like m")
    status_counts = df["paper_status"].value_counts()
    axes[2].bar(status_counts.index, status_counts.values, color="#cc6677")
    axes[2].set_title("Paper-like unwrap status")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outliers", default="results/tables/tongji_building_height_qc_outliers_dsm_review.csv")
    parser.add_argument("--pairs-csv", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--intf-root", default="work/gamma_sbas/intf")
    parser.add_argument("--island-label", default="work/masks/island_label_touying_blue_bottom.npy")
    parser.add_argument("--fid-mask", default="work/masks/building_fid_mask_touying_blue_bottom.npy")
    parser.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--top-islands", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=0.2)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--output-csv", default="results/tables/paper_unwrap_benchmark.csv")
    parser.add_argument("--summary", default="results/metadata/paper_unwrap_benchmark_summary.json")
    parser.add_argument("--figure", default="results/pic_all/27_paper_unwrap_benchmark.png")
    args = parser.parse_args()

    outliers = pd.read_csv(args.outliers)
    island_ids = []
    for value in outliers.sort_values("dsm_review_priority_rank")["qc_island_ids"]:
        for part in str(value).split(";"):
            if part and part != "nan":
                island_id = int(float(part))
                if island_id not in island_ids:
                    island_ids.append(island_id)
        if len(island_ids) >= args.top_islands:
            break

    pairs = pd.read_csv(args.pairs_csv).head(args.max_pairs).copy()
    label = np.load(args.island_label)
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    a = design(
        pairs["bperp_m"].to_numpy(dtype=np.float64),
        pairs["dt_days"].to_numpy(dtype=np.float64),
        wavelength,
        float(par["center_range_slc"]),
        float(par["incidence_angle"]),
    )
    phase_stack = []
    coh_stack = []
    for row in pairs.itertuples(index=False):
        pair = f"{row.master}_{row.slave}"
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        phase_stack.append(np.angle(diff).astype(np.float32))
        coh_stack.append(cc.astype(np.float32))

    rows = []
    for island_id in island_ids:
        keep = label == island_id
        rr, cc = np.nonzero(keep)
        if len(rr) < 20:
            continue
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        patch_keep = keep[r0:r1, c0:c1]
        pix = np.nonzero(patch_keep.ravel())[0]
        sk_phase_rows = []
        paper_phase_rows = []
        coh_rows = []
        pair_status = []
        pair_phase_diffs = []
        for k, pair_row in enumerate(pairs.itertuples(index=False)):
            phase_patch = phase_stack[k][r0:r1, c0:c1]
            coh_patch = coh_stack[k][r0:r1, c0:c1]
            valid = patch_keep & np.isfinite(phase_patch) & np.isfinite(coh_patch) & (coh_patch >= args.min_coherence)
            if int(np.sum(valid)) < 20:
                sk = np.full_like(phase_patch, np.nan, dtype=np.float32)
                paper = np.full_like(phase_patch, np.nan, dtype=np.float32)
                info = {"status": "too_few_valid", "kept_arcs": 0, "arcs": 0}
            else:
                try:
                    sk = unwrap_phase(np.ma.array(phase_patch, mask=~valid)).filled(np.nan).astype(np.float32)
                except Exception:
                    sk = np.full_like(phase_patch, np.nan, dtype=np.float32)
                paper, info = paper_like_unwrap(phase_patch, valid, wavelength)
            sk_vec = sk.ravel()[pix]
            paper_vec = paper.ravel()[pix]
            coh_vec = coh_patch.ravel()[pix]
            both = np.isfinite(sk_vec) & np.isfinite(paper_vec)
            if int(np.sum(both)) > 0:
                diff = paper_vec[both] - sk_vec[both]
                diff = diff - np.nanmedian(diff)
                pair_phase_diffs.append(float(np.nanstd(diff)))
            else:
                pair_phase_diffs.append(np.nan)
            sk_phase_rows.append(sk_vec)
            paper_phase_rows.append(paper_vec)
            coh_rows.append(coh_vec)
            pair_status.append(str(info.get("status", "unknown")))

        sk_dem = solve_dem(sk_phase_rows, coh_rows, a, args.min_pairs)
        paper_dem = solve_dem(paper_phase_rows, coh_rows, a, args.min_pairs)
        rows.append(
            {
                "island_id": island_id,
                "pixel_count": int(len(pix)),
                "pairs": int(len(pairs)),
                "paper_ok_pairs": int(sum(s == "ok" for s in pair_status)),
                "paper_status": "ok" if any(s == "ok" for s in pair_status) else ";".join(sorted(set(pair_status))),
                "phase_diff_std_rad": float(np.nanmedian(pair_phase_diffs)),
                "skimage_height_m": height_range(sk_dem),
                "paper_height_m": height_range(paper_dem),
                "height_delta_m": height_range(paper_dem) - height_range(sk_dem),
                "selected_pair_statuses": ";".join(pair_status),
            }
        )

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    plot_results(df, Path(args.figure))
    summary = {
        "selected_islands": island_ids,
        "pairs_used": int(len(pairs)),
        "rows": int(len(df)),
        "paper_ok_rows": int((df["paper_status"] == "ok").sum()) if not df.empty else 0,
        "median_phase_diff_std_rad": float(df["phase_diff_std_rad"].median()) if not df.empty else None,
        "median_height_delta_m": float(df["height_delta_m"].median()) if not df.empty else None,
        "output_csv": args.output_csv,
        "figure": args.figure,
        "note": "Diagnostic only: paper-like unwrap approximates the MATLAB KNN/Bisquare/LSMR method on selected priority outlier islands and selected GAMMA pairs; it is not yet used in the production height stack.",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
