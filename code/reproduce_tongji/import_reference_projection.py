#!/usr/bin/env python3
"""Import `touying_roof_workflow` projection results and map them to local UIDs.

The reference workflow uses source FIDs from the full `tongji_clip.shp` in
`data/aaa.zip`. The local `tongji_clip_rslc_extent.shp` is the intersection
subset of that full shapefile, in sorted source-FID order. This script restores
`source_fid` for local `uid` values and filters the reference SAR-coordinate
projection to the local buildings.
"""

from __future__ import annotations

import argparse
import csv
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
from matplotlib.patches import Polygon as MplPolygon
from skimage.draw import polygon as draw_polygon


def build_uid_source_fid_map(full_shp: Path, extent_geojson: Path, local_shp: Path) -> list[dict]:
    full = gpd.read_file(full_shp).to_crs("EPSG:4326")
    extent = gpd.read_file(extent_geojson).to_crs("EPSG:4326")
    local = gpd.read_file(local_shp).to_crs("EPSG:4326")
    inter = (
        gpd.sjoin(full.reset_index().rename(columns={"index": "source_fid"}), extent[["geometry"]], predicate="intersects", how="inner")
        .sort_values("source_fid")
        .reset_index(drop=True)
    )
    if len(inter) != len(local):
        raise RuntimeError(f"Local subset length mismatch: local={len(local)} intersects={len(inter)}")
    rows = []
    for idx in range(len(local)):
        rows.append(
            {
                "uid": idx + 1,
                "source_fid": int(inter.iloc[idx]["source_fid"]),
                "height_prior_m": float(local.iloc[idx]["height"]),
                "floor_prior": float(local.iloc[idx]["Floor"]),
            }
        )
    return rows


def xy_from_feature(feat: dict) -> np.ndarray | None:
    coords = feat.get("geometry", {}).get("coordinates", [])
    if not coords:
        return None
    xy = np.asarray(coords[0], dtype=np.float64)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    return xy if xy.shape[0] >= 3 else None


def polygon_metric(xy: np.ndarray, rows: int, cols: int) -> dict:
    finite = np.all(np.isfinite(xy), axis=1)
    in_frame = finite & (xy[:, 0] >= 0) & (xy[:, 0] < cols) & (xy[:, 1] >= 0) & (xy[:, 1] < rows)
    mask_pixels = 0
    if np.sum(finite) >= 3:
        rr, cc = draw_polygon(xy[finite, 1], xy[finite, 0], shape=(rows, cols))
        mask_pixels = int(rr.size)
    return {
        "projected_vertices": int(np.sum(finite)),
        "vertices_in_frame": int(np.sum(in_frame)),
        "mask_pixels": mask_pixels,
        "row_min": float(np.nanmin(xy[:, 1])) if xy.size else None,
        "row_max": float(np.nanmax(xy[:, 1])) if xy.size else None,
        "col_min": float(np.nanmin(xy[:, 0])) if xy.size else None,
        "col_max": float(np.nanmax(xy[:, 0])) if xy.size else None,
    }


def import_projection(args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    uid_map_rows = build_uid_source_fid_map(Path(args.full_shp), Path(args.extent_geojson), Path(args.local_shp))
    source_to_uid = {row["source_fid"]: row for row in uid_map_rows}
    ref = json.loads(Path(args.reference_projection).read_text(encoding="utf-8"))
    by_fid_surface = {}
    for feat in ref.get("features", []):
        props = feat.get("properties", {})
        fid = int(props.get("fid", -1))
        surface = props.get("surface")
        if fid in source_to_uid and surface in {"bottom", "roof"}:
            by_fid_surface[(fid, surface)] = feat

    features = []
    metrics = []
    missing = []
    for source_fid, local_row in sorted(source_to_uid.items(), key=lambda x: x[1]["uid"]):
        bottom = by_fid_surface.get((source_fid, "bottom"))
        roof = by_fid_surface.get((source_fid, "roof"))
        if bottom is None or roof is None:
            missing.append({"uid": local_row["uid"], "source_fid": source_fid, "reason": "missing_reference_projection"})
            continue
        bottom_xy = xy_from_feature(bottom)
        roof_xy = xy_from_feature(roof)
        if bottom_xy is None or roof_xy is None:
            missing.append({"uid": local_row["uid"], "source_fid": source_fid, "reason": "invalid_reference_geometry"})
            continue
        layover_xy = np.vstack([bottom_xy, roof_xy[::-1]])
        for surface, xy, ref_feat in [("bottom", bottom_xy, bottom), ("roof", roof_xy, roof), ("layover", layover_xy, roof)]:
            ref_props = ref_feat.get("properties", {})
            props = {
                "uid": local_row["uid"],
                "source_fid": source_fid,
                "surface": surface,
                "height_prior_m": local_row["height_prior_m"],
                "floor_prior": local_row["floor_prior"],
                "reference_height_m": float(ref_props.get("height_m", local_row["height_prior_m"])),
                "base_height_m": float(ref_props.get("base_height_m", 0.0)),
                "top_height_m": float(ref_props.get("top_height_m", 0.0)),
                "reference_mask0_pixels": int(float(ref_props.get("mask0_pixels", 0))),
                "reference_mask_pixels": int(float(ref_props.get("mask_pixels", 0))),
                "projection_score": float(ref_props.get("projection_score", -1000000.0)),
                "correction_row_shift": float(ref_props.get("correction_row_shift", 0.0)),
                "correction_col_shift": float(ref_props.get("correction_col_shift", 0.0)),
                "sar_brightness_opt_row_shift": int(float(ref_props.get("sar_brightness_opt_row_shift", 0))),
                "sar_brightness_opt_col_shift": int(float(ref_props.get("sar_brightness_opt_col_shift", 0))),
                "local_opt_row_shift": int(float(ref_props.get("local_opt_row_shift", 0))),
                "local_opt_col_shift": int(float(ref_props.get("local_opt_col_shift", 0))),
                **polygon_metric(xy, args.rows, args.cols),
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Polygon", "coordinates": [xy.tolist() + [xy[0].tolist()]]},
                }
            )
            metrics.append(props)
    payload = {
        "type": "FeatureCollection",
        "name": "tongji_reference_projection_mapped_to_local_uid",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "source": "https://github.com/aaaroger/touying_roof_workflow",
        "features": features,
    }
    summary = {
        "reference_projection": args.reference_projection,
        "local_buildings": len(uid_map_rows),
        "mapped_buildings": len({f["properties"]["uid"] for f in features}),
        "mapped_features": len(features),
        "missing_buildings": len(missing),
        "missing": missing,
        "note": "Reference projection imported from touying_roof_workflow and filtered to local rslc_extent buildings by restored source_fid.",
    }
    return payload, metrics, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_overlay(geojson_path: Path, amp_npy: Path, out_png: Path) -> None:
    if not amp_npy.exists():
        return
    amp = np.load(amp_npy)
    valid = np.isfinite(amp)
    p2, p98 = np.percentile(amp[valid], [2, 98])
    bg = np.clip((amp - p2) / max(float(p98 - p2), 1e-6), 0, 1)
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(11, 8), dpi=240)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    count = 0
    for feat in data.get("features", []):
        if feat.get("properties", {}).get("surface") != "layover":
            continue
        xy = xy_from_feature(feat)
        if xy is None:
            continue
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor="#ffb000", linewidth=0.18, alpha=0.65))
        count += 1
    ax.set_title(f"Reference workflow layover projection mapped to local UID ({count} buildings)")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-projection", default="tmp/touying_roof_workflow/results/20200708_full_area_projection_sar_col_row_local_optimized.geojson")
    parser.add_argument("--full-shp", default="tmp/reference_inputs/shp/tongji_clip.shp")
    parser.add_argument("--extent-geojson", default="tmp/reference_inputs/rslc_extent/tongji_rslc_extent_wgs84.geojson")
    parser.add_argument("--local-shp", default="data/shp/tongji_clip_rslc_extent.shp")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--output-geojson", default="work/projection/20200708_building_projection_reference.geojson")
    parser.add_argument("--metrics-csv", default="work/projection/20200708_building_projection_reference_metrics.csv")
    parser.add_argument("--summary", default="results/metadata/reference_projection_import_summary.json")
    parser.add_argument("--overlay-png", default="results/pic_all/10_reference_projection_layover_overlay.png")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    args = parser.parse_args()

    payload, metrics, summary = import_projection(args)
    out_geojson = Path(args.output_geojson)
    out_summary = Path(args.summary)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(Path(args.metrics_csv), metrics)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_overlay(out_geojson, Path(args.amp_npy), Path(args.overlay_png))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
