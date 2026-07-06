#!/usr/bin/env python3
"""Optimize per-building SAR-coordinate projection shifts against amplitude."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import binary_dilation


def polygon_array(feat: dict) -> np.ndarray | None:
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [])
    if geom.get("type") != "Polygon" or not coords:
        return None
    xy = np.asarray(coords[0], dtype=np.float64)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
        return None
    return xy


def stretch_amp(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    p2, p98 = np.percentile(arr[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    return np.clip((arr - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)


def edge_map(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img.astype(np.float32))
    edge = np.hypot(gx, gy)
    valid = np.isfinite(edge)
    p98 = np.percentile(edge[valid], 98) if np.any(valid) else 1.0
    return np.clip(edge / max(float(p98), 1e-6), 0.0, 1.0)


def rasterize_local_polygon(xy: np.ndarray, image_rows: int, image_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin = max(0, int(np.floor(np.min(xy[:, 0]))) - 8)
    xmax = min(image_cols - 1, int(np.ceil(np.max(xy[:, 0]))) + 8)
    ymin = max(0, int(np.floor(np.min(xy[:, 1]))) - 8)
    ymax = min(image_rows - 1, int(np.ceil(np.max(xy[:, 1]))) + 8)
    if xmax < xmin or ymax < ymin:
        z = np.zeros((0,), dtype=np.int64)
        return z, z, z, z
    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    inside = MplPath(xy).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
    if not np.any(inside):
        z = np.zeros((0,), dtype=np.int64)
        return z, z, z, z
    ring = binary_dilation(inside, iterations=5) & ~binary_dilation(inside, iterations=1)
    rr_in, cc_in = np.nonzero(inside)
    rr_ring, cc_ring = np.nonzero(ring)
    return rr_in + ymin, cc_in + xmin, rr_ring + ymin, cc_ring + xmin


def shifted_values(img: np.ndarray, rr: np.ndarray, cc: np.ndarray, dr: int, dc: int) -> np.ndarray:
    r = rr + dr
    c = cc + dc
    ok = (r >= 0) & (c >= 0) & (r < img.shape[0]) & (c < img.shape[1])
    if not np.any(ok):
        return np.zeros((0,), dtype=np.float32)
    return img[r[ok], c[ok]]


def score_shift(
    amp: np.ndarray,
    edges: np.ndarray,
    rr_in: np.ndarray,
    cc_in: np.ndarray,
    rr_ring: np.ndarray,
    cc_ring: np.ndarray,
    dr: int,
    dc: int,
    max_shift: int,
) -> dict:
    inside_amp_vals = shifted_values(amp, rr_in, cc_in, dr, dc)
    if inside_amp_vals.size < 4:
        return {"score": -1e9, "inside_amp": 0.0, "ring_amp": 0.0, "inside_edge": 0.0, "ring_edge": 0.0}
    ring_amp_vals = shifted_values(amp, rr_ring, cc_ring, dr, dc)
    inside_edge_vals = shifted_values(edges, rr_in, cc_in, dr, dc)
    ring_edge_vals = shifted_values(edges, rr_ring, cc_ring, dr, dc)
    inside_amp = float(np.mean(inside_amp_vals))
    ring_amp = float(np.mean(ring_amp_vals)) if ring_amp_vals.size else float(np.mean(amp))
    inside_edge = float(np.mean(inside_edge_vals)) if inside_edge_vals.size else 0.0
    ring_edge = float(np.mean(ring_edge_vals)) if ring_edge_vals.size else float(np.mean(edges))
    offset_penalty = 0.0015 * (dr * dr + dc * dc) / max(max_shift, 1)
    score = 100.0 * (inside_amp - ring_amp) + 35.0 * (inside_edge - ring_edge) - offset_penalty
    return {
        "score": score,
        "inside_amp": inside_amp,
        "ring_amp": ring_amp,
        "inside_edge": inside_edge,
        "ring_edge": ring_edge,
    }


def optimize_one(amp: np.ndarray, edges: np.ndarray, xy: np.ndarray, max_shift: int, coarse_step: int) -> dict:
    rr_in, cc_in, rr_ring, cc_ring = rasterize_local_polygon(xy, amp.shape[0], amp.shape[1])
    if rr_in.size < 4:
        return {"ok": 0, "row_shift": 0, "col_shift": 0, "pixels": int(rr_in.size), "base_score": -1e9, "best_score": -1e9}
    base = score_shift(amp, edges, rr_in, cc_in, rr_ring, cc_ring, 0, 0, max_shift)
    best = {**base, "row_shift": 0, "col_shift": 0}
    for dr in range(-max_shift, max_shift + 1, coarse_step):
        for dc in range(-max_shift, max_shift + 1, coarse_step):
            s = score_shift(amp, edges, rr_in, cc_in, rr_ring, cc_ring, dr, dc, max_shift)
            if s["score"] > best["score"]:
                best = {**s, "row_shift": dr, "col_shift": dc}
    br = int(best["row_shift"])
    bc = int(best["col_shift"])
    for dr in range(br - coarse_step, br + coarse_step + 1):
        for dc in range(bc - coarse_step, bc + coarse_step + 1):
            if abs(dr) > max_shift or abs(dc) > max_shift:
                continue
            s = score_shift(amp, edges, rr_in, cc_in, rr_ring, cc_ring, dr, dc, max_shift)
            if s["score"] > best["score"]:
                best = {**s, "row_shift": dr, "col_shift": dc}
    return {
        "ok": 1,
        "pixels": int(rr_in.size),
        "base_score": float(base["score"]),
        "best_score": float(best["score"]),
        "score_gain": float(best["score"] - base["score"]),
        "row_shift": int(best["row_shift"]),
        "col_shift": int(best["col_shift"]),
        "inside_amp": float(best["inside_amp"]),
        "ring_amp": float(best["ring_amp"]),
        "inside_edge": float(best["inside_edge"]),
        "ring_edge": float(best["ring_edge"]),
    }


def shift_feature(feat: dict, row_shift: int, col_shift: int) -> None:
    geom = feat.get("geometry", {})
    if geom.get("type") != "Polygon":
        return
    rings = []
    for ring in geom.get("coordinates", []):
        rings.append([[float(x) + col_shift, float(y) + row_shift] for x, y, *rest in ring])
    geom["coordinates"] = rings
    props = feat.setdefault("properties", {})
    props["local_opt_row_shift"] = row_shift
    props["local_opt_col_shift"] = col_shift


def plot_overlay(path: Path, data: dict, amp: np.ndarray, max_draw: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=220)
    ax.imshow(amp, cmap="gray", vmin=0, vmax=1)
    count = 0
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("surface") != "layover":
            continue
        xy = polygon_array(feat)
        if xy is None:
            continue
        if max_draw <= 0 or count < max_draw:
            ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor="#ffb000", linewidth=0.22, alpha=0.7))
            count += 1
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Locally optimized layover projection")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-geojson", default="work/projection/20200708_building_projection_sar.geojson")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--out-geojson", default="work/projection/20200708_building_projection_sar_local_optimized.geojson")
    parser.add_argument("--metrics-csv", default="work/projection/20200708_projection_local_shift_metrics.csv")
    parser.add_argument("--summary", default="results/metadata/projection_local_optimization_summary.json")
    parser.add_argument("--overlay-png", default="work/projection/20200708_building_projection_sar_local_optimized_overlay.png")
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--coarse-step", type=int, default=2)
    parser.add_argument("--min-mask-pixels", type=int, default=4)
    parser.add_argument("--max-draw", type=int, default=0)
    args = parser.parse_args()

    amp_raw = np.load(args.amp_npy)
    amp = stretch_amp(amp_raw)
    edges = edge_map(amp)
    data = json.loads(Path(args.input_geojson).read_text(encoding="utf-8"))

    by_uid: dict[int, list[dict]] = defaultdict(list)
    layover_by_uid: dict[int, dict] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        uid = int(props.get("uid", -1))
        if uid < 0:
            continue
        by_uid[uid].append(feat)
        if props.get("surface") == "layover" and int(float(props.get("mask_pixels", 0))) >= args.min_mask_pixels:
            layover_by_uid[uid] = feat

    rows_out = []
    for uid, feat in sorted(layover_by_uid.items()):
        xy = polygon_array(feat)
        if xy is None:
            result = {"ok": 0, "row_shift": 0, "col_shift": 0, "pixels": 0, "base_score": -1e9, "best_score": -1e9}
        else:
            result = optimize_one(amp, edges, xy, args.max_shift, args.coarse_step)
        for item in by_uid.get(uid, []):
            shift_feature(item, int(result["row_shift"]), int(result["col_shift"]))
        rows_out.append({"uid": uid, "mask_pixels": int(feat.get("properties", {}).get("mask_pixels", 0)), **result})

    data["local_projection_optimization"] = {
        "surface": "layover",
        "max_shift": args.max_shift,
        "coarse_step": args.coarse_step,
        "optimized_uids": len(rows_out),
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
    }

    out_geojson = Path(args.out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0].keys()) if rows_out else []
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    ok_rows = [r for r in rows_out if int(r["ok"]) == 1]
    summary = {
        "input_geojson": args.input_geojson,
        "optimized_geojson": args.out_geojson,
        "optimized_uids": len(rows_out),
        "ok_uids": len(ok_rows),
        "median_row_shift": float(np.median([r["row_shift"] for r in ok_rows])) if ok_rows else 0.0,
        "median_col_shift": float(np.median([r["col_shift"] for r in ok_rows])) if ok_rows else 0.0,
        "median_score_gain": float(np.median([r["score_gain"] for r in ok_rows])) if ok_rows else 0.0,
        "mean_score_gain": float(np.mean([r["score_gain"] for r in ok_rows])) if ok_rows else 0.0,
        "metrics_csv": args.metrics_csv,
        "note": "Per-building integer SAR-coordinate shifts optimized against the mean BMP amplitude proxy.",
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_overlay(Path(args.overlay_png), data, amp, args.max_draw)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
