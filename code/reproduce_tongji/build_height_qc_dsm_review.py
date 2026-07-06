#!/usr/bin/env python3
"""Add DSM and island context to focused building-height QC outliers."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated.*", category=DeprecationWarning)

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds


def stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [p_low, p_high])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0, 1)


def valid_values(ds: rasterio.io.DatasetReader, geom, pad_m: float = 0.0) -> np.ndarray:
    target = geom.buffer(pad_m) if pad_m else geom
    if target.is_empty:
        return np.array([], dtype=np.float32)
    minx, miny, maxx, maxy = target.bounds
    try:
        window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform).round_offsets().round_lengths()
        arr = ds.read(1, window=window, boundless=True, fill_value=ds.nodata).astype(np.float32)
    except Exception:
        return np.array([], dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    transform = ds.window_transform(window)
    mask = geometry_mask([target], out_shape=arr.shape, transform=transform, invert=True)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if ds.nodata is not None:
        vals = vals[vals != float(ds.nodata)]
    return vals


def percentile(vals: np.ndarray, q: float) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def consistency_bucket(row: pd.Series) -> str:
    dsm = row.get("dsm_height_proxy_m", np.nan)
    prior = row.get("height_prior_m", np.nan)
    insar = row.get("height_insar_m", np.nan)
    if not np.isfinite(dsm) or not np.isfinite(prior) or not np.isfinite(insar):
        return "insufficient_dsm"
    prior_err = abs(float(dsm) - float(prior))
    insar_err = abs(float(dsm) - float(insar))
    if prior_err <= 0.05:
        return "dsm_proxy_matches_prior"
    if prior_err + 5.0 < insar_err:
        return "dsm_proxy_closer_to_prior"
    if insar_err + 5.0 < prior_err:
        return "dsm_proxy_closer_to_insar"
    return "dsm_proxy_ambiguous"


def attach_island_context(outliers: pd.DataFrame, height_points: Path, island_heights: Path) -> pd.DataFrame:
    pts = pd.read_csv(height_points)
    islands = pd.read_csv(island_heights)
    island_cols = ["island_id", "uid_count", "pixel_count", "pixel_count_used", "median_coherence"]
    pts = pts.merge(islands[island_cols], on="island_id", how="left", suffixes=("_building", "_island"))
    if "pixel_count_used_building" not in pts.columns and "pixel_count_used" in pts.columns:
        pts["pixel_count_used_building"] = pts["pixel_count_used"]
    if "pixel_count_used_island" not in pts.columns:
        pts["pixel_count_used_island"] = np.nan

    def join_vals(series: pd.Series) -> str:
        vals = [str(int(v)) for v in series.dropna().unique()]
        return ";".join(vals)

    grouped = (
        pts.groupby("uid", as_index=False)
        .agg(
            qc_island_ids=("island_id", join_vals),
            qc_island_count=("island_id", "nunique"),
            qc_height_point_count=("height_m", "count"),
            qc_height_points_median_m=("height_m", "median"),
            qc_height_points_min_m=("height_m", "min"),
            qc_height_points_max_m=("height_m", "max"),
            qc_island_uid_count_max=("uid_count", "max"),
            qc_island_pixels_sum=("pixel_count", "sum"),
            qc_pixel_count_used_sum=("pixel_count_used_building", "sum"),
            qc_island_pixel_count_used_sum=("pixel_count_used_island", "sum"),
            qc_island_median_coh_median=("median_coherence", "median"),
        )
    )
    return outliers.merge(grouped, on="uid", how="left")


def dsm_stats(outliers: gpd.GeoDataFrame, dsm_path: Path, ring_outer_m: float, ring_inner_m: float) -> gpd.GeoDataFrame:
    with rasterio.open(dsm_path) as ds:
        projected = outliers.to_crs(ds.crs)
        rows = []
        for geom in projected.geometry:
            footprint = valid_values(ds, geom)
            ring_geom = geom.buffer(ring_outer_m).difference(geom.buffer(ring_inner_m))
            ring = valid_values(ds, ring_geom)
            fp_p05 = percentile(footprint, 5)
            fp_p50 = percentile(footprint, 50)
            fp_p95 = percentile(footprint, 95)
            ground_p10 = percentile(ring, 10)
            rows.append(
                {
                    "dsm_footprint_pixels": int(footprint.size),
                    "dsm_ring_pixels": int(ring.size),
                    "dsm_footprint_p05_m": fp_p05,
                    "dsm_footprint_median_m": fp_p50,
                    "dsm_footprint_p95_m": fp_p95,
                    "dsm_ring_ground_p10_m": ground_p10,
                    "dsm_height_proxy_m": fp_p95 - ground_p10 if np.isfinite(fp_p95) and np.isfinite(ground_p10) else np.nan,
                    "dsm_relief_p95_p05_m": fp_p95 - fp_p05 if np.isfinite(fp_p95) and np.isfinite(fp_p05) else np.nan,
                }
            )
    stats = pd.DataFrame(rows)
    out = outliers.copy()
    for col in stats.columns:
        out[col] = stats[col].to_numpy()
    out["dsm_minus_prior_m"] = out["dsm_height_proxy_m"] - out["height_prior_m"]
    out["dsm_minus_insar_m"] = out["dsm_height_proxy_m"] - out["height_insar_m"]
    out["dsm_insar_abs_gap_m"] = out["dsm_minus_insar_m"].abs()
    out["dsm_matches_prior_exact"] = out["dsm_minus_prior_m"].abs() <= 0.05
    out["dsm_consistency_bucket"] = out.apply(consistency_bucket, axis=1)
    out = out.sort_values(["dsm_insar_abs_gap_m", "qc_severity"], ascending=False).reset_index(drop=True)
    out["dsm_review_priority_rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def plot_review(outliers: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=220)
    valid = outliers[np.isfinite(outliers["dsm_height_proxy_m"])].copy()
    colors = {
        "dsm_proxy_matches_prior": "#cc6677",
        "dsm_proxy_closer_to_prior": "#aa4499",
        "dsm_proxy_closer_to_insar": "#117733",
        "dsm_proxy_ambiguous": "#ddaa33",
        "insufficient_dsm": "#999999",
    }

    for bucket, sub in valid.groupby("dsm_consistency_bucket"):
        axes[0, 0].scatter(
            sub["height_prior_m"],
            sub["dsm_height_proxy_m"],
            s=18,
            alpha=0.75,
            label=bucket,
            color=colors.get(bucket, "#666666"),
            edgecolors="none",
        )
    max_v = float(np.nanmax([valid["height_prior_m"].max(), valid["dsm_height_proxy_m"].max(), 1.0])) if not valid.empty else 1.0
    axes[0, 0].plot([0, max_v], [0, max_v], color="#333333", linewidth=0.8)
    axes[0, 0].set_title("DSM proxy vs prior height")
    axes[0, 0].set_xlabel("Prior height m")
    axes[0, 0].set_ylabel("DSM proxy height m")
    axes[0, 0].legend(frameon=False, fontsize=7)

    for bucket, sub in valid.groupby("dsm_consistency_bucket"):
        axes[0, 1].scatter(
            sub["height_insar_m"],
            sub["dsm_height_proxy_m"],
            s=18,
            alpha=0.75,
            label=bucket,
            color=colors.get(bucket, "#666666"),
            edgecolors="none",
        )
    max_v = float(np.nanmax([valid["height_insar_m"].max(), valid["dsm_height_proxy_m"].max(), 1.0])) if not valid.empty else 1.0
    axes[0, 1].plot([0, max_v], [0, max_v], color="#333333", linewidth=0.8)
    axes[0, 1].set_title("DSM proxy vs InSAR height")
    axes[0, 1].set_xlabel("InSAR height m")
    axes[0, 1].set_ylabel("DSM proxy height m")

    counts = outliers["dsm_consistency_bucket"].value_counts()
    axes[1, 0].barh(counts.index, counts.values, color=[colors.get(idx, "#666666") for idx in counts.index])
    axes[1, 0].set_title("DSM consistency buckets")
    axes[1, 0].set_xlabel("Buildings")

    axes[1, 1].hist(valid["dsm_minus_insar_m"].dropna(), bins=35, color="#4477aa", alpha=0.75, label="DSM - InSAR")
    axes[1, 1].hist(valid["dsm_minus_prior_m"].dropna(), bins=35, color="#cc6677", alpha=0.55, label="DSM - prior")
    axes[1, 1].axvline(0, color="#333333", linewidth=0.8)
    axes[1, 1].set_title("DSM residuals")
    axes[1, 1].set_xlabel("m")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_chips(
    outliers: gpd.GeoDataFrame,
    all_buildings: gpd.GeoDataFrame,
    dsm_path: Path,
    chip_dir: Path,
    limit: int,
    pad_m: float,
) -> list[str]:
    chip_dir.mkdir(parents=True, exist_ok=True)
    for stale in chip_dir.glob("rank_*.png"):
        stale.unlink()
    made = []
    with rasterio.open(dsm_path) as ds:
        out_utm = outliers.to_crs(ds.crs)
        all_utm = all_buildings.to_crs(ds.crs)
        ordered = out_utm.sort_values(["dsm_review_priority_rank"], ascending=True).head(limit)
        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            geom = row.geometry
            minx, miny, maxx, maxy = geom.buffer(pad_m).bounds
            window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform).round_offsets().round_lengths()
            arr = ds.read(1, window=window, boundless=True, fill_value=ds.nodata).astype(np.float32)
            arr[arr == ds.nodata] = np.nan
            transform = ds.window_transform(window)
            extent = [minx, maxx, miny, maxy]

            neighbors = all_utm.cx[minx:maxx, miny:maxy]
            target = gpd.GeoDataFrame([row._asdict()], geometry=[geom], crs=ds.crs)

            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=180)
            axes[0].imshow(stretch(arr), cmap="gray", extent=extent, origin="upper")
            if not neighbors.empty:
                neighbors.boundary.plot(ax=axes[0], color="#999999", linewidth=0.4)
            target.boundary.plot(ax=axes[0], color="#ff2d55", linewidth=1.4)
            axes[0].set_title("DSM chip and footprint")
            axes[0].set_xlim(minx, maxx)
            axes[0].set_ylim(miny, maxy)
            axes[0].set_aspect("equal")

            labels = ["prior", "insar", "dsm"]
            vals = [row.height_prior_m, row.height_insar_m, row.dsm_height_proxy_m]
            axes[1].bar(labels, vals, color=["#cc6677", "#4477aa", "#117733"])
            axes[1].set_title(f"uid {int(row.uid)} | {row.height_qc_class}")
            axes[1].set_ylabel("Height m")
            axes[1].text(
                0.02,
                0.98,
                f"review: {row.recommended_review}\nDSM consistency: {row.dsm_consistency_bucket}\ncoh: {row.median_coh:.3f}",
                transform=axes[1].transAxes,
                va="top",
                ha="left",
                fontsize=7,
            )
            fig.tight_layout()
            out = chip_dir / f"rank_{rank:02d}_uid_{int(row.uid)}_{row.height_qc_class}.png"
            fig.savefig(out)
            plt.close(fig)
            made.append(str(out))
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outliers", default="results/geodata/tongji_building_height_qc_outliers.geojson")
    parser.add_argument("--all-buildings", default="results/geodata/tongji_building_height_qc.geojson")
    parser.add_argument("--dsm", default="data/dsm/tongji_real_dsm_1m_rslc_extent.tif")
    parser.add_argument("--height-points", default="work/height/height_points.csv")
    parser.add_argument("--island-heights", default="work/height/island_pixel_lgr_heights.csv")
    parser.add_argument("--ring-outer-m", type=float, default=25.0)
    parser.add_argument("--ring-inner-m", type=float, default=2.0)
    parser.add_argument("--chip-limit", type=int, default=24)
    parser.add_argument("--chip-pad-m", type=float, default=45.0)
    parser.add_argument("--out-csv", default="results/tables/tongji_building_height_qc_outliers_dsm_review.csv")
    parser.add_argument("--out-geojson", default="results/geodata/tongji_building_height_qc_outliers_dsm_review.geojson")
    parser.add_argument("--summary", default="results/metadata/building_height_qc_outliers_dsm_review_summary.json")
    parser.add_argument("--figure", default="results/pic_all/26_building_height_qc_outliers_dsm_review.png")
    parser.add_argument("--chip-dir", default="results/pic_all/qc_outlier_dsm_chips")
    args = parser.parse_args()

    outliers = gpd.read_file(args.outliers)
    reviewed = dsm_stats(outliers, Path(args.dsm), args.ring_outer_m, args.ring_inner_m)
    table = attach_island_context(pd.DataFrame(reviewed.drop(columns="geometry")), Path(args.height_points), Path(args.island_heights))
    for col in table.columns:
        if col not in reviewed.columns:
            reviewed[col] = table[col].to_numpy()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_geojson).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reviewed.drop(columns="geometry")).to_csv(args.out_csv, index=False)
    reviewed.to_file(args.out_geojson, driver="GeoJSON")
    plot_review(pd.DataFrame(reviewed.drop(columns="geometry")), Path(args.figure))
    chips = plot_chips(reviewed, gpd.read_file(args.all_buildings), Path(args.dsm), Path(args.chip_dir), args.chip_limit, args.chip_pad_m)

    summary = {
        "input_outliers": args.outliers,
        "outlier_buildings": int(len(reviewed)),
        "dsm_consistency_counts": {str(k): int(v) for k, v in reviewed["dsm_consistency_bucket"].value_counts().sort_index().items()},
        "dsm_matches_prior_exact_count": int(reviewed["dsm_matches_prior_exact"].sum()),
        "dsm_valid_buildings": int(np.isfinite(reviewed["dsm_height_proxy_m"]).sum()),
        "median_dsm_height_proxy_m": float(np.nanmedian(reviewed["dsm_height_proxy_m"])),
        "median_dsm_minus_prior_m": float(np.nanmedian(reviewed["dsm_minus_prior_m"])),
        "median_dsm_minus_insar_m": float(np.nanmedian(reviewed["dsm_minus_insar_m"])),
        "ring_outer_m": args.ring_outer_m,
        "ring_inner_m": args.ring_inner_m,
        "out_csv": args.out_csv,
        "out_geojson": args.out_geojson,
        "figure": args.figure,
        "chip_count": len(chips),
        "chip_dir": args.chip_dir,
        "interpretation_note": "DSM proxy is a consistency aid, not independent validation. In this run all outlier DSM proxies match the shapefile prior height within 0.05 m, so the DSM likely shares the same building-height source or encodes the same prior over these footprints.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
