#!/usr/bin/env python3
"""Adopt the current final projection from `touying_roof_workflow`.

This script deliberately does not recompute, merge, or reshape the reference
projection. The canonical geometry remains the final
`results/blue_aligned/20200708_full_area_projection_blue_aligned_bottom_only.geojson`
produced by `touying_roof_workflow`: SAR image coordinates with x=range column
and y=azimuth row, bottom polygons only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from skimage.draw import polygon as draw_polygon


def stretch(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [2, 98])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def polygon_xy(feature: dict) -> np.ndarray | None:
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    if geom.get("type") != "Polygon" or not coords:
        return None
    xy = np.asarray(coords[0], dtype=np.float64)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
        return None
    return xy


def feature_metrics(feature: dict, rows: int, cols: int) -> dict:
    props = dict(feature.get("properties", {}))
    xy = polygon_xy(feature)
    if xy is None:
        return {**props, "vertices": 0, "vertices_in_frame": 0, "raster_pixels": 0}
    in_frame = (xy[:, 0] >= 0) & (xy[:, 0] < cols) & (xy[:, 1] >= 0) & (xy[:, 1] < rows)
    rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=(rows, cols))
    return {
        **props,
        "vertices": int(xy.shape[0]),
        "vertices_in_frame": int(np.sum(in_frame)),
        "raster_pixels": int(rr.size),
        "row_min": float(np.min(xy[:, 1])),
        "row_max": float(np.max(xy[:, 1])),
        "col_min": float(np.min(xy[:, 0])),
        "col_max": float(np.max(xy[:, 0])),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_bottoms(features: list[dict], amp_npy: Path, out_png: Path) -> None:
    amp = np.load(amp_npy)
    bg = stretch(amp)
    polygons = [xy for feat in features if (xy := polygon_xy(feat)) is not None]
    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=320)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    for xy in polygons:
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor="#00d4ff", linewidth=0.34, alpha=0.62))
    if polygons:
        xy_all = np.vstack(polygons)
        ax.set_xlim(max(0, float(np.nanmin(xy_all[:, 0])) - 80), min(amp.shape[1], float(np.nanmax(xy_all[:, 0])) + 80))
        ax.set_ylim(min(amp.shape[0], float(np.nanmax(xy_all[:, 1])) + 80), max(0, float(np.nanmin(xy_all[:, 1])) - 80))
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Tongji blue-aligned bottom projection (touying_roof_workflow)")
    ax.text(
        0.012,
        0.988,
        f"bottom projections: {len(features)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-geojson", default="tmp/touying_roof_workflow/results/blue_aligned/20200708_full_area_projection_blue_aligned_bottom_only.geojson")
    parser.add_argument("--source-summary", default="tmp/touying_roof_workflow/results/blue_aligned/20200708_sar_brightness_projection_optimization_summary.json")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--out-geojson", default="work/projection/20200708_blue_aligned_bottom_touying.geojson")
    parser.add_argument("--metrics-csv", default="work/projection/20200708_blue_aligned_bottom_touying_metrics.csv")
    parser.add_argument("--summary", default="results/metadata/touying_blue_aligned_projection_adoption_summary.json")
    parser.add_argument("--overlay-png", default="results/pic_all/04_touying_blue_aligned_bottom_overlay.png")
    args = parser.parse_args()

    source = Path(args.source_geojson)
    source_text = source.read_text(encoding="utf-8")
    payload = json.loads(source_text)
    features = [feat for feat in payload.get("features", []) if feat.get("properties", {}).get("surface") == "bottom"]
    if not features:
        raise RuntimeError(f"No bottom features found in {source}")

    out_geojson = Path(args.out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(source_text, encoding="utf-8")

    metrics = [feature_metrics(feat, args.rows, args.cols) for feat in features]
    write_csv(Path(args.metrics_csv), metrics)
    plot_bottoms(features, Path(args.amp_npy), Path(args.overlay_png))

    source_summary = json.loads(Path(args.source_summary).read_text(encoding="utf-8")) if Path(args.source_summary).exists() else {}
    summary = {
        "source": "https://github.com/aaaroger/touying_roof_workflow",
        "canonical_source_geojson": str(source),
        "adopted_geojson": str(out_geojson),
        "bottom_features": len(features),
        "coordinate_system": payload.get("coordinate_system", "SAR image coordinates: x=range column, y=azimuth row"),
        "geometry_policy": "Exact blue-aligned bottom-only geometry copied from touying_roof_workflow; no roof/layover reconstruction and no local reprojection.",
        "source_summary": source_summary,
        "metrics_csv": args.metrics_csv,
        "overlay_png": args.overlay_png,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
