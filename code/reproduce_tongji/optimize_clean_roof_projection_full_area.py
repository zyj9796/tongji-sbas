#!/usr/bin/env python3
"""Layer-constrained full-area correction for clean roof projections in SAR coordinates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_dilation
from shapely import affinity


def stretch(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    lo, hi = np.percentile(arr[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def edge_map(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img.astype(np.float32))
    edge = np.hypot(gx, gy)
    valid = np.isfinite(edge)
    hi = np.percentile(edge[valid], 98) if np.any(valid) else 1.0
    return np.clip(edge / max(float(hi), 1e-6), 0.0, 1.0).astype(np.float32)


def polygon_xy(geom) -> np.ndarray | None:
    if geom is None or geom.is_empty:
        return None
    poly = geom
    if geom.geom_type == "MultiPolygon":
        poly = max(list(geom.geoms), key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        return None
    xy = np.asarray(poly.exterior.coords, dtype=np.float64)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
        return None
    return xy


def local_pixels(
    xy: np.ndarray, rows: int, cols: int, pad: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin = max(0, int(np.floor(np.min(xy[:, 0]))) - pad)
    xmax = min(cols - 1, int(np.ceil(np.max(xy[:, 0]))) + pad)
    ymin = max(0, int(np.floor(np.min(xy[:, 1]))) - pad)
    ymax = min(rows - 1, int(np.ceil(np.max(xy[:, 1]))) + pad)
    empty = np.zeros((0,), dtype=np.int64)
    if xmax <= xmin or ymax <= ymin:
        return empty, empty, empty, empty, empty, empty
    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = MplPath(xy).contains_points(pts).reshape(yy.shape)
    if int(np.sum(inside)) < 4:
        return empty, empty, empty, empty, empty, empty
    dil1 = binary_dilation(inside, iterations=1)
    dil5 = binary_dilation(inside, iterations=5)
    ring = dil5 & ~dil1
    boundary = dil1 & ~inside
    rr_in, cc_in = np.nonzero(inside)
    rr_ring, cc_ring = np.nonzero(ring)
    rr_boundary, cc_boundary = np.nonzero(boundary)
    return rr_in + ymin, cc_in + xmin, rr_ring + ymin, cc_ring + xmin, rr_boundary + ymin, cc_boundary + xmin


def shifted_values(img: np.ndarray, rr: np.ndarray, cc: np.ndarray, dr: int, dc: int) -> np.ndarray:
    r = rr + dr
    c = cc + dc
    ok = (r >= 0) & (c >= 0) & (r < img.shape[0]) & (c < img.shape[1])
    if not np.any(ok):
        return np.zeros((0,), dtype=np.float32)
    return img[r[ok], c[ok]]


def score_shift(
    amp: np.ndarray,
    edge: np.ndarray,
    rr_in: np.ndarray,
    cc_in: np.ndarray,
    rr_ring: np.ndarray,
    cc_ring: np.ndarray,
    rr_boundary: np.ndarray,
    cc_boundary: np.ndarray,
    dr: int,
    dc: int,
    max_shift: int,
) -> dict[str, float]:
    inside_amp = shifted_values(amp, rr_in, cc_in, dr, dc)
    if inside_amp.size < 4:
        return {"score": -1e9, "inside_amp": 0.0, "ring_amp": 0.0, "inside_edge": 0.0, "ring_edge": 0.0}
    ring_amp = shifted_values(amp, rr_ring, cc_ring, dr, dc)
    boundary_edge = shifted_values(edge, rr_boundary, cc_boundary, dr, dc)
    ring_edge = shifted_values(edge, rr_ring, cc_ring, dr, dc)
    ia = float(np.mean(inside_amp))
    ra = float(np.mean(ring_amp)) if ring_amp.size else float(np.mean(amp))
    be = float(np.mean(boundary_edge)) if boundary_edge.size else 0.0
    re = float(np.mean(ring_edge)) if ring_edge.size else float(np.mean(edge))
    bright = float(np.mean(inside_amp >= np.percentile(ring_amp, 85))) if ring_amp.size else 0.0
    offset_penalty = 0.004 * (dr * dr + dc * dc) / max(max_shift, 1)
    score = 92.0 * (ia - ra) + 32.0 * (be - re) + 9.0 * bright - offset_penalty
    return {"score": float(score), "inside_amp": ia, "ring_amp": ra, "inside_edge": be, "ring_edge": re}


def max_shift_for_height(height_m: float) -> int:
    if height_m < 12:
        return 4
    if height_m < 24:
        return 6
    if height_m < 30:
        return 8
    return int(np.clip(4 + height_m / 3.5, 10, 28))


def gates_for_height(height_m: float) -> tuple[float, float, float]:
    if height_m < 12:
        return 12.0, 0.035, 4.5
    if height_m < 24:
        return 10.0, 0.030, 5.5
    if height_m < 30:
        return 8.0, 0.025, 6.5
    if height_m < 45:
        return 5.0, 0.020, 9.0
    return 4.0, 0.015, 12.0


def optimize_one(amp: np.ndarray, edge: np.ndarray, xy: np.ndarray, height_m: float) -> dict[str, float | int]:
    max_shift = max_shift_for_height(height_m)
    step = 2
    rr_in, cc_in, rr_ring, cc_ring, rr_boundary, cc_boundary = local_pixels(
        xy, amp.shape[0], amp.shape[1], pad=max_shift + 6
    )
    if rr_in.size < 4:
        return {
            "ok": 0,
            "raw_row_shift": 0,
            "raw_col_shift": 0,
            "max_shift": max_shift,
            "pixels": int(rr_in.size),
            "score_gain": 0.0,
            "base_score": -1e9,
            "best_score": -1e9,
            "inside_amp": 0.0,
            "ring_amp": 0.0,
            "inside_edge": 0.0,
        }
    base = score_shift(amp, edge, rr_in, cc_in, rr_ring, cc_ring, rr_boundary, cc_boundary, 0, 0, max_shift)
    best = {**base, "row_shift": 0, "col_shift": 0}
    for dr in range(-max_shift, max_shift + 1, step):
        for dc in range(-max_shift, max_shift + 1, step):
            s = score_shift(amp, edge, rr_in, cc_in, rr_ring, cc_ring, rr_boundary, cc_boundary, dr, dc, max_shift)
            if s["score"] > best["score"]:
                best = {**s, "row_shift": dr, "col_shift": dc}
    br, bc = int(best["row_shift"]), int(best["col_shift"])
    for dr in range(br - step, br + step + 1):
        for dc in range(bc - step, bc + step + 1):
            if abs(dr) > max_shift or abs(dc) > max_shift:
                continue
            s = score_shift(amp, edge, rr_in, cc_in, rr_ring, cc_ring, rr_boundary, cc_boundary, dr, dc, max_shift)
            if s["score"] > best["score"]:
                best = {**s, "row_shift": dr, "col_shift": dc}
    return {
        "ok": 1,
        "raw_row_shift": int(best["row_shift"]),
        "raw_col_shift": int(best["col_shift"]),
        "max_shift": max_shift,
        "pixels": int(rr_in.size),
        "base_score": float(base["score"]),
        "best_score": float(best["score"]),
        "score_gain": float(best["score"] - base["score"]),
        "inside_amp": float(best["inside_amp"]),
        "ring_amp": float(best["ring_amp"]),
        "inside_edge": float(best["inside_edge"]),
    }


def is_reliable(row: pd.Series) -> bool:
    if int(row["ok"]) != 1:
        return False
    gain_gate, contrast_gate, _ = gates_for_height(float(row["height_m"]))
    contrast = float(row["inside_amp"] - row["ring_amp"])
    if float(row["score_gain"]) < gain_gate or contrast < contrast_gate:
        return False
    if int(row["raw_row_shift"]) == 0 and int(row["raw_col_shift"]) == 0:
        return False
    if float(row["height_m"]) < 30 and abs(int(row["raw_row_shift"])) > abs(int(row["raw_col_shift"])) + 2:
        return False
    return True


def local_field(metrics: pd.DataFrame, reliable: pd.DataFrame, radius: float) -> pd.DataFrame:
    out = []
    src = reliable[["cx", "cy", "raw_row_shift", "raw_col_shift", "score_gain", "height_m"]].to_numpy(dtype=float)
    for row in metrics.itertuples(index=False):
        if src.size == 0:
            out.append({"field_row_shift": 0.0, "field_col_shift": 0.0, "field_support": 0, "field_distance_median": np.nan})
            continue
        dx = src[:, 0] - float(row.cx)
        dy = src[:, 1] - float(row.cy)
        dist = np.hypot(dx, dy)
        mask = dist <= radius
        if not np.any(mask):
            out.append({"field_row_shift": 0.0, "field_col_shift": 0.0, "field_support": 0, "field_distance_median": np.nan})
            continue
        near = src[mask]
        d = dist[mask]
        weights = np.maximum(near[:, 4], 1.0) / np.maximum(d + 20.0, 20.0)
        row_shift = float(np.average(near[:, 2], weights=weights))
        col_shift = float(np.average(near[:, 3], weights=weights))
        out.append(
            {
                "field_row_shift": row_shift,
                "field_col_shift": col_shift,
                "field_support": int(np.sum(mask)),
                "field_distance_median": float(np.median(d)),
            }
        )
    return pd.DataFrame(out)


def choose_final(row: pd.Series) -> tuple[int, int, str]:
    if int(row["ok"]) != 1:
        return 0, 0, "invalid_geometry"
    gain_gate, contrast_gate, tolerance = gates_for_height(float(row["height_m"]))
    contrast = float(row["inside_amp"] - row["ring_amp"])
    raw_dr = int(row["raw_row_shift"])
    raw_dc = int(row["raw_col_shift"])
    field_dr = int(round(float(row["field_row_shift"])))
    field_dc = int(round(float(row["field_col_shift"])))
    support = int(row["field_support"])
    dist_to_field = float(np.hypot(raw_dr - float(row["field_row_shift"]), raw_dc - float(row["field_col_shift"])))
    strong_local = float(row["score_gain"]) >= gain_gate and contrast >= contrast_gate
    if float(row["height_m"]) >= 30:
        if strong_local:
            return raw_dr, raw_dc, "applied_highrise_local"
        return 0, 0, "rejected_highrise_score"
    if strong_local and support >= 3 and dist_to_field <= tolerance:
        return raw_dr, raw_dc, "applied_lowrise_local_field_consistent"
    return 0, 0, "kept_baseline"


def plot_overlay(original: gpd.GeoDataFrame, optimized: gpd.GeoDataFrame, amp: np.ndarray, out_png: Path, out_svg: Path) -> None:
    lo, hi = np.percentile(amp[np.isfinite(amp)], [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=260, sharex=True, sharey=True)
    for ax in axes:
        ax.imshow(amp, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
        ax.set_xlim(0, amp.shape[1] - 1)
        ax.set_ylim(amp.shape[0] - 1, 0)
        ax.set_xlabel("SAR range column")
        ax.set_ylabel("SAR azimuth row")
    original.boundary.plot(ax=axes[0], color="#ff2b2b", linewidth=0.28, alpha=0.86)
    axes[0].set_title("Before: clean roof projection")
    optimized.plot(ax=axes[1], facecolor="#00bcd4", edgecolor="none", alpha=0.14)
    optimized.boundary.plot(ax=axes[1], color="#00e676", linewidth=0.30, alpha=0.90)
    moved = original[(optimized["final_row_shift"].ne(0)) | (optimized["final_col_shift"].ne(0))]
    if not moved.empty:
        moved.boundary.plot(ax=axes[1], color="#fff176", linewidth=0.34, alpha=0.80)
    axes[1].text(
        0.012,
        0.02,
        "green: full-area corrected projection\nyellow: shifted roofs\nlow-rise shifts require local amplitude gain + neighborhood support",
        transform=axes[1].transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#c8cdd2", "alpha": 0.86, "pad": 5},
    )
    axes[1].set_title("After: layer-constrained full-area correction")
    fig.suptitle("Full-area building-vector projection correction in SAR image coordinates", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="work/projection/20200708_clean_equal_height_roof_projection_sar.geojson")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--out-geojson", default="work/projection/20200708_clean_equal_height_roof_projection_sar_full_area_layer_constrained.geojson")
    parser.add_argument("--metrics-csv", default="work/projection/20200708_clean_equal_height_roof_projection_sar_full_area_layer_constrained_metrics.csv")
    parser.add_argument("--summary", default="results/metadata/clean_equal_height_roof_projection_full_area_layer_constrained_summary.json")
    parser.add_argument("--overlay-png", default="results/pic_all/png/current_strict_clean_equal_height_full/128_dsm_clean_building_vector_projected_on_sar_amplitude_full_area_layer_constrained.png")
    parser.add_argument("--overlay-svg", default="results/pic_all/svg/current_strict_clean_equal_height_full/128_dsm_clean_building_vector_projected_on_sar_amplitude_full_area_layer_constrained.svg")
    parser.add_argument("--field-radius", type=float, default=150.0)
    args = parser.parse_args()

    amp_raw = np.load(args.amp_npy).astype(np.float32)
    amp = stretch(amp_raw)
    edge = edge_map(amp)
    gdf = gpd.read_file(args.input)

    rows = []
    for idx, row in gdf.iterrows():
        xy = polygon_xy(row.geometry)
        height_m = float(row.get("height_m", 0.0) or 0.0)
        centroid = row.geometry.centroid
        if xy is None:
            result = {
                "ok": 0,
                "raw_row_shift": 0,
                "raw_col_shift": 0,
                "max_shift": 0,
                "pixels": 0,
                "score_gain": 0.0,
                "base_score": -1e9,
                "best_score": -1e9,
                "inside_amp": 0.0,
                "ring_amp": 0.0,
                "inside_edge": 0.0,
            }
        else:
            result = optimize_one(amp, edge, xy, height_m)
        rows.append({"index": idx, "clean_id": int(row.clean_id), "height_m": height_m, "cx": centroid.x, "cy": centroid.y, **result})

    metrics = pd.DataFrame(rows)
    metrics["raw_reliable"] = metrics.apply(is_reliable, axis=1)
    reliable = metrics[metrics["raw_reliable"]].copy()
    fields = local_field(metrics, reliable, args.field_radius)
    metrics = pd.concat([metrics.reset_index(drop=True), fields], axis=1)
    finals = metrics.apply(choose_final, axis=1)
    metrics["final_row_shift"] = [int(x[0]) for x in finals]
    metrics["final_col_shift"] = [int(x[1]) for x in finals]
    metrics["final_status"] = [str(x[2]) for x in finals]

    optimized = gdf.copy()
    for _, item in metrics.iterrows():
        dr, dc = int(item["final_row_shift"]), int(item["final_col_shift"])
        idx = int(item["index"])
        if dr or dc:
            optimized.at[idx, "geometry"] = affinity.translate(gdf.at[idx, "geometry"], xoff=dc, yoff=dr)
        optimized.at[idx, "raw_row_shift"] = int(item["raw_row_shift"])
        optimized.at[idx, "raw_col_shift"] = int(item["raw_col_shift"])
        optimized.at[idx, "field_row_shift"] = float(item["field_row_shift"])
        optimized.at[idx, "field_col_shift"] = float(item["field_col_shift"])
        optimized.at[idx, "field_support"] = int(item["field_support"])
        optimized.at[idx, "final_row_shift"] = dr
        optimized.at[idx, "final_col_shift"] = dc
        optimized.at[idx, "final_status"] = str(item["final_status"])
        optimized.at[idx, "local_score_gain"] = float(item["score_gain"])

    Path(args.out_geojson).parent.mkdir(parents=True, exist_ok=True)
    optimized.to_file(args.out_geojson, driver="GeoJSON")
    Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics.drop(columns=["index"]).to_csv(args.metrics_csv, index=False)
    plot_overlay(gdf, optimized, amp_raw, Path(args.overlay_png), Path(args.overlay_svg))

    moved = metrics[(metrics["final_row_shift"].ne(0)) | (metrics["final_col_shift"].ne(0))]
    low = metrics[metrics["height_m"] < 30]
    low_moved = low[(low["final_row_shift"].ne(0)) | (low["final_col_shift"].ne(0))]
    high = metrics[metrics["height_m"] >= 30]
    high_moved = high[(high["final_row_shift"].ne(0)) | (high["final_col_shift"].ne(0))]
    summary = {
        "input_projection": args.input,
        "optimized_projection": args.out_geojson,
        "metrics_csv": args.metrics_csv,
        "building_count": int(len(gdf)),
        "optimized_ok": int(metrics["ok"].sum()),
        "raw_reliable_candidates": int(metrics["raw_reliable"].sum()),
        "moved_buildings": int(len(moved)),
        "lowrise_height_lt_30m": int(len(low)),
        "lowrise_moved": int(len(low_moved)),
        "highrise_height_ge_30m": int(len(high)),
        "highrise_moved": int(len(high_moved)),
        "median_row_shift_applied": float(moved["final_row_shift"].median()) if not moved.empty else 0.0,
        "median_col_shift_applied": float(moved["final_col_shift"].median()) if not moved.empty else 0.0,
        "final_status_counts": {str(k): int(v) for k, v in metrics["final_status"].value_counts().sort_index().items()},
        "outputs": {"png": args.overlay_png, "svg": args.overlay_svg},
        "note": "Extends the high-rise local-amplitude correction to the full area. Low-rise buildings can move, but only when their own local amplitude/edge gain is consistent with the spatial neighborhood shift field; the field is not applied by interpolation alone.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
