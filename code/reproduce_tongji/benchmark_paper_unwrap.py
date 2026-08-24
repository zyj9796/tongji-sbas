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
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr
from scipy.spatial import Delaunay, cKDTree
from skimage.restoration import unwrap_phase

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "DejaVu Sans"],
        "svg.fonttype": "none",
    }
)

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


def bisquare_weights(residual: np.ndarray, tune: float = 4.685) -> np.ndarray:
    """Return Tukey-Bisquare weights using the paper's 4.685 tuning constant."""
    finite = np.isfinite(residual)
    result = np.zeros_like(residual, dtype=np.float64)
    if not np.any(finite):
        return result
    centre = float(np.nanmedian(residual[finite]))
    mad = float(np.nanmedian(np.abs(residual[finite] - centre)))
    scale = max(1.4826 * mad, 1.0e-6)
    u = (residual - centre) / (tune * scale)
    inside = finite & (np.abs(u) < 1.0)
    result[inside] = (1.0 - u[inside] ** 2) ** 2
    return result


def robust_bisquare_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 30,
    tune: float = 4.685,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[keep]
    y = y[keep]
    if len(y) < x.shape[1] + 1:
        missing = np.full(len(keep), np.nan)
        return np.full(x.shape[1], np.nan), missing, np.zeros(len(keep), dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    for _ in range(max_iter):
        resid = y - x @ coef
        w = bisquare_weights(resid, tune=tune)
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
    full_weight = np.zeros(len(keep), dtype=np.float64)
    full_weight[keep] = bisquare_weights(full_resid[keep], tune=tune)
    return coef, full_resid, full_weight


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


def paper_like_unwrap(
    phase_patch: np.ndarray,
    valid_mask: np.ndarray,
    wavelength_m: float,
    coherence_patch: np.ndarray | None = None,
    amplitude_dispersion_patch: np.ndarray | None = None,
    use_quality_weights: bool = True,
    preserve_wrapped_reference: bool = True,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Unwrap one vector-isolated island with the paper's local robust network.

    Only valid island pixels become graph nodes.  Edge equations are weighted by
    interferometric coherence, amplitude dispersion, and the local Bisquare
    residual.  Each disconnected graph component retains one observed wrapped
    phase as its datum instead of being reset to an arbitrary zero.
    """
    rr, cc = np.nonzero(valid_mask & np.isfinite(phase_patch))
    n = len(rr)
    out = np.full_like(phase_patch, np.nan, dtype=np.float32)
    if n < 10 or phase_patch.shape[0] == 1 or phase_patch.shape[1] <= 4:
        out[rr, cc] = phase_patch[rr, cc]
        return out, {"points": int(n), "arcs": 0, "kept_arcs": 0, "status": "too_small"}

    coords = np.column_stack([cc.astype(np.float64), rr.astype(np.float64)])
    obs = phase_patch[rr, cc].astype(np.float64)
    if coherence_patch is None:
        node_coherence = np.ones(n, dtype=np.float64)
    else:
        node_coherence = np.clip(coherence_patch[rr, cc].astype(np.float64), 1.0e-3, 1.0)
        node_coherence[~np.isfinite(node_coherence)] = 1.0e-3
    if amplitude_dispersion_patch is None:
        node_da = np.zeros(n, dtype=np.float64)
    else:
        node_da = amplitude_dispersion_patch[rr, cc].astype(np.float64)
        node_da[~np.isfinite(node_da)] = 1.0
    tree = cKDTree(coords)
    k_nearest = min(9, n)
    _, clusters = tree.query(coords, k=k_nearest)
    clusters = np.atleast_2d(clusters)
    threshold = 4.0 * math.pi * (math.sqrt(2.0) * 1.0) / (wavelength_m * 1000.0)

    arc_from = []
    arc_to = []
    arc_obs = []
    arc_resid = []
    arc_weight = []
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
        coef, resid, residual_weight = robust_bisquare_fit(poly, y)
        if not np.isfinite(coef).all():
            continue
        pred = poly @ coef
        edge_coherence = np.sqrt(node_coherence[cluster[f]] * node_coherence[cluster[t]])
        edge_da = 0.5 * (node_da[cluster[f]] + node_da[cluster[t]])
        da_weight = np.exp(-np.square(edge_da / 0.40))
        combined_weight = edge_coherence * da_weight * residual_weight
        keep = np.isfinite(resid) & (np.abs(resid) < threshold)
        if use_quality_weights:
            keep &= combined_weight > 1.0e-6
        if not np.any(keep):
            continue
        arc_from.extend(cluster[f[keep]].tolist())
        arc_to.extend(cluster[t[keep]].tolist())
        arc_obs.extend(pred[keep].tolist())
        arc_resid.extend(resid[keep].tolist())
        arc_weight.extend((combined_weight[keep] if use_quality_weights else np.ones(np.sum(keep))).tolist())

    if not arc_obs:
        return out, {"points": int(n), "arcs": int(total_arcs), "kept_arcs": 0, "status": "no_arcs"}

    arc_from_arr = np.asarray(arc_from, dtype=np.int64)
    arc_to_arr = np.asarray(arc_to, dtype=np.int64)
    b = np.asarray(arc_obs, dtype=np.float64)
    edge_weight = np.clip(np.asarray(arc_weight, dtype=np.float64), 1.0e-6, 1.0)
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
    adjacency = coo_matrix(
        (
            np.r_[edge_weight, edge_weight],
            (np.r_[arc_from_arr, arc_to_arr], np.r_[arc_to_arr, arc_from_arr]),
        ),
        shape=(n, n),
    ).tocsr()
    component_count, component_label = connected_components(adjacency, directed=False)

    # Weighted edge equations plus one strong observed-phase datum per graph
    # component.  The datum preserves the phase modulo 2pi; it does not invent
    # an absolute integer cycle.
    sqrt_weight = np.sqrt(edge_weight)
    weighted_mat = mat.multiply(sqrt_weight[:, None])
    weighted_rhs = b * sqrt_weight
    anchor_nodes: list[int] = []
    for component in range(component_count):
        candidates = np.flatnonzero((component_label == component) & np.isin(np.arange(n), active))
        if len(candidates):
            quality = node_coherence[candidates] * np.exp(-np.square(node_da[candidates] / 0.40))
            anchor_nodes.append(int(candidates[int(np.nanargmax(quality))]))
    if not anchor_nodes:
        return out, {"points": int(n), "arcs": int(total_arcs), "kept_arcs": int(m), "status": "no_anchor"}
    anchor_scale = 1.0e3
    anchor_rows = coo_matrix(
        (
            np.full(len(anchor_nodes), anchor_scale),
            (np.arange(len(anchor_nodes)), np.asarray(anchor_nodes, dtype=np.int64)),
        ),
        shape=(len(anchor_nodes), n),
    ).tocsr()
    system = vstack([weighted_mat, anchor_rows], format="csr")
    anchor_rhs = (
        obs[np.asarray(anchor_nodes, dtype=np.int64)]
        if preserve_wrapped_reference
        else np.zeros(len(anchor_nodes), dtype=np.float64)
    )
    rhs = np.r_[weighted_rhs, anchor_scale * anchor_rhs]
    full = lsmr(system, rhs, atol=1e-7, btol=1e-7, maxiter=1000)[0]
    inactive = np.ones(n, dtype=bool)
    inactive[active] = False
    full[inactive] = np.nan
    out[rr, cc] = full.astype(np.float32)
    return out, {
        "points": int(n),
        "arcs": int(total_arcs),
        "kept_arcs": int(m),
        "status": "ok",
        "components": int(len(anchor_nodes)),
        "reference_policy": (
            "one observed wrapped-phase datum per connected component"
            if preserve_wrapped_reference
            else "arbitrary zero datum per connected component"
        ),
        "quality_weighting": "coherence_x_DA_x_Bisquare" if use_quality_weights else "equal",
        "median_edge_weight": float(np.nanmedian(edge_weight)),
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
    axes[0].set_title("质量加权前后相位差")
    axes[0].set_xlabel("去中位数后的标准差（弧度）")
    axes[1].scatter(ok["paper_equal_height_m"], ok["paper_weighted_height_m"], s=22, alpha=0.75, color="#117733")
    lim = float(np.nanmax([ok["paper_equal_height_m"].max(), ok["paper_weighted_height_m"].max(), 1.0])) if not ok.empty else 1.0
    axes[1].plot([0, lim], [0, lim], color="#333333", linewidth=0.8)
    axes[1].set_title("论文等权与质量加权高程范围")
    axes[1].set_xlabel("等权网络（米）")
    axes[1].set_ylabel("质量加权网络（米）")
    status_counts = df["paper_status"].value_counts()
    status_labels = ["成功" if value == "ok" else str(value) for value in status_counts.index]
    axes[2].bar(status_labels, status_counts.values, color="#cc6677")
    axes[2].set_title("论文加权解缠状态")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outliers", default="results/tables/tongji_building_height_qc_outliers_dsm_review.csv")
    parser.add_argument("--island-id-file", default="", help="Optional frozen CSV containing island_id values")
    parser.add_argument("--pairs-csv", default="work/baselines/tongji_redundant_22_network48_qc.csv")
    parser.add_argument("--intf-root", default="work/gamma_native_ipta_sbas/interferograms")
    parser.add_argument("--island-label", default="work/roof_sbas_adaptive_window_paper_quality_network48/roof_core_island_label.npy")
    parser.add_argument("--fid-mask", default="work/masks/building_fid_mask_touying_blue_bottom.npy")
    parser.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--top-islands", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=0.2)
    parser.add_argument("--amplitude-dispersion", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--output-csv", default="results/tables/paper_unwrap_benchmark.csv")
    parser.add_argument("--summary", default="results/metadata/paper_unwrap_benchmark_summary.json")
    parser.add_argument("--figure", default="results/pic_all/27_论文加权孤岛解缠消融.svg")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_csv).head(args.max_pairs).copy()
    label = np.load(args.island_label)
    if args.island_id_file:
        frozen = pd.read_csv(args.island_id_file)
        island_ids = [int(value) for value in frozen["island_id"].head(args.top_islands)]
    elif Path(args.outliers).exists():
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
    else:
        ids, counts = np.unique(label[label > 0], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        island_ids = [int(value) for value in ids[order[: args.top_islands]]]
    amplitude_dispersion = np.load(args.amplitude_dispersion).astype(np.float32)
    if amplitude_dispersion.shape != label.shape:
        raise ValueError(
            f"amplitude-dispersion shape {amplitude_dispersion.shape} does not match label shape {label.shape}"
        )
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
        diff_path = pair_dir / f"{pair}.adf.diff"
        if not diff_path.exists():
            diff_path = pair_dir / f"{pair}.diff"
        diff = read_fcomplex(diff_path, args.rows, args.cols)
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
        paper_equal_phase_rows = []
        paper_weighted_phase_rows = []
        coh_rows = []
        pair_status = []
        pair_phase_diffs = []
        for k, pair_row in enumerate(pairs.itertuples(index=False)):
            phase_patch = phase_stack[k][r0:r1, c0:c1]
            coh_patch = coh_stack[k][r0:r1, c0:c1]
            da_patch = amplitude_dispersion[r0:r1, c0:c1]
            valid = patch_keep & np.isfinite(phase_patch) & np.isfinite(coh_patch) & (coh_patch >= args.min_coherence)
            if int(np.sum(valid)) < 20:
                sk = np.full_like(phase_patch, np.nan, dtype=np.float32)
                paper_equal = np.full_like(phase_patch, np.nan, dtype=np.float32)
                paper_weighted = np.full_like(phase_patch, np.nan, dtype=np.float32)
                info = {"status": "too_few_valid", "kept_arcs": 0, "arcs": 0}
            else:
                try:
                    sk = unwrap_phase(np.ma.array(phase_patch, mask=~valid)).filled(np.nan).astype(np.float32)
                except Exception:
                    sk = np.full_like(phase_patch, np.nan, dtype=np.float32)
                paper_equal, _ = paper_like_unwrap(
                    phase_patch,
                    valid,
                    wavelength,
                    use_quality_weights=False,
                    preserve_wrapped_reference=False,
                )
                paper_weighted, info = paper_like_unwrap(
                    phase_patch,
                    valid,
                    wavelength,
                    coherence_patch=coh_patch,
                    amplitude_dispersion_patch=da_patch,
                )
            sk_vec = sk.ravel()[pix]
            paper_equal_vec = paper_equal.ravel()[pix]
            paper_weighted_vec = paper_weighted.ravel()[pix]
            coh_vec = coh_patch.ravel()[pix]
            both = np.isfinite(paper_equal_vec) & np.isfinite(paper_weighted_vec)
            if int(np.sum(both)) > 0:
                diff = paper_weighted_vec[both] - paper_equal_vec[both]
                diff = diff - np.nanmedian(diff)
                pair_phase_diffs.append(float(np.nanstd(diff)))
            else:
                pair_phase_diffs.append(np.nan)
            sk_phase_rows.append(sk_vec)
            paper_equal_phase_rows.append(paper_equal_vec)
            paper_weighted_phase_rows.append(paper_weighted_vec)
            coh_rows.append(coh_vec)
            pair_status.append(str(info.get("status", "unknown")))

        sk_dem = solve_dem(sk_phase_rows, coh_rows, a, args.min_pairs)
        paper_equal_dem = solve_dem(paper_equal_phase_rows, coh_rows, a, args.min_pairs)
        paper_weighted_dem = solve_dem(paper_weighted_phase_rows, coh_rows, a, args.min_pairs)
        rows.append(
            {
                "island_id": island_id,
                "pixel_count": int(len(pix)),
                "pairs": int(len(pairs)),
                "paper_ok_pairs": int(sum(s == "ok" for s in pair_status)),
                "paper_status": "ok" if any(s == "ok" for s in pair_status) else ";".join(sorted(set(pair_status))),
                "phase_diff_std_rad": float(np.nanmedian(pair_phase_diffs)),
                "skimage_height_m": height_range(sk_dem),
                "paper_equal_height_m": height_range(paper_equal_dem),
                "paper_weighted_height_m": height_range(paper_weighted_dem),
                "height_delta_m": height_range(paper_weighted_dem) - height_range(paper_equal_dem),
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
        "note": "仅作冻结消融诊断：对比等权零基准网络与相干性、振幅离差、Bisquare联合加权且保留缠绕相位基准的网络；尚未进入正式高度产品。",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
