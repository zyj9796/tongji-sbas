#!/usr/bin/env python3
"""Create diagnostic figures for the Tongji reproduction workflow."""

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
from matplotlib.patches import Patch, Polygon as MplPolygon


def stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [p_low, p_high])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0, 1)


def projection_polygons(path: Path, surface: str) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if surface and props.get("surface") != surface:
            continue
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        xy = np.asarray(coords[0], dtype=np.float64)
        if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if xy.shape[0] >= 3:
            out.append(xy)
    return out


def fig_buildings_map(buildings: Path, out: Path) -> None:
    gdf = gpd.read_file(buildings)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    gdf.plot(ax=ax, column="height_prior_m", cmap="viridis", linewidth=0.15, edgecolor="black", legend=True)
    ax.set_title("Tongji buildings in WGS84 colored by prior height")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_height_area(buildings_csv: Path, out: Path) -> None:
    df = pd.read_csv(buildings_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=220)
    axes[0].hist(df["height_prior_m"].dropna(), bins=40, color="#4477aa", edgecolor="white")
    axes[0].set_title("Prior building height")
    axes[0].set_xlabel("height_prior_m")
    axes[0].set_ylabel("count")
    axes[1].scatter(df["area_m2"], df["height_prior_m"], s=5, alpha=0.45, color="#cc6677")
    axes[1].set_xscale("log")
    axes[1].set_title("Area vs prior height")
    axes[1].set_xlabel("area_m2 log scale")
    axes[1].set_ylabel("height_prior_m")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_amplitude(amp_npy: Path, out: Path) -> None:
    amp = np.load(amp_npy)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=220)
    im = ax.imshow(stretch(amp), cmap="gray", vmin=0, vmax=1)
    ax.set_title("Mean SAR amplitude proxy from 43 crop BMPs")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_projection_overlay(amp_npy: Path, projection: Path, out: Path) -> None:
    amp = np.load(amp_npy)
    bg = stretch(amp)
    polys = projection_polygons(projection, "bottom")
    fig, ax = plt.subplots(figsize=(11, 8), dpi=240)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    for xy in polys:
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor="#00d4ff", linewidth=0.24, alpha=0.62))
    ax.set_title(f"Touying blue-aligned bottom projection ({len(polys)} features)")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_touying_projection_metrics(metrics: Path, out: Path) -> None:
    df = pd.read_csv(metrics)
    bottom = df[df["surface"] == "bottom"].copy() if "surface" in df.columns else df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)
    axes[0].hist(bottom["raster_pixels"], bins=50, color="#44aa99", edgecolor="white")
    axes[0].set_title("Touying bottom raster pixels")
    axes[0].set_xlabel("pixels")
    axes[1].scatter(bottom["col_min"], bottom["row_min"], s=np.clip(bottom["raster_pixels"] / 20, 2, 40), alpha=0.45, color="#88ccee")
    axes[1].invert_yaxis()
    axes[1].set_title("Touying bottom location")
    axes[1].set_xlabel("col_min")
    axes[1].set_ylabel("row_min")
    axes[2].bar(["row", "col"], [float(bottom["sar_brightness_opt_row_shift"].iloc[0]), float(bottom["sar_brightness_opt_col_shift"].iloc[0])], color=["#117733", "#332288"])
    axes[2].set_title("Blue alignment shift")
    axes[2].set_ylabel("pixels")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_masks(amp_npy: Path, uid_mask: Path, island_label: Path, out: Path) -> None:
    amp = stretch(np.load(amp_npy))
    uid = np.load(uid_mask)
    islands = np.load(island_label)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=220)
    axes[0].imshow(amp, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Mean amplitude")
    overlay = np.ma.masked_where(uid <= 0, uid % 251)
    axes[1].imshow(amp, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(overlay, cmap="turbo", alpha=0.55)
    axes[1].set_title("Building UID mask")
    island_overlay = np.ma.masked_where(islands <= 0, islands % 251)
    axes[2].imshow(amp, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(island_overlay, cmap="nipy_spectral", alpha=0.6)
    axes[2].set_title("Extracted islands")
    for ax in axes:
        ax.set_xlabel("Range column")
        ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_island_stats(islands_csv: Path, out: Path) -> None:
    df = pd.read_csv(islands_csv)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)
    axes[0].hist(df["pixel_count"], bins=50, color="#117733", edgecolor="white")
    axes[0].set_title("Island pixel count")
    axes[0].set_xlabel("pixels")
    axes[1].hist(df["uid_count"], bins=np.arange(0.5, df["uid_count"].max() + 1.5, 1), color="#aa4499", edgecolor="white")
    axes[1].set_title("UIDs per island")
    axes[1].set_xlabel("uid_count")
    axes[2].scatter(df["col_min"], df["row_min"], s=np.clip(df["pixel_count"] / 15, 2, 60), alpha=0.5, color="#882255")
    axes[2].invert_yaxis()
    axes[2].set_title("Island location by bbox min")
    axes[2].set_xlabel("col_min")
    axes[2].set_ylabel("row_min")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_temporal_pairs(pairs_csv: Path, out: Path) -> None:
    df = pd.read_csv(pairs_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=220)
    ax.hist(df["dt_days"], bins=np.arange(0, df["dt_days"].max() + 3, 2), color="#332288", edgecolor="white")
    ax.set_title(f"Temporal candidate interferogram pairs ({len(df)})")
    ax.set_xlabel("dt_days")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_touying_mapping(buildings: Path, out: Path) -> None:
    gdf = gpd.read_file(buildings)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=220)
    gdf.plot(ax=axes[0], color="#d8d8d8", linewidth=0.08, edgecolor="#666666")
    axes[0].set_title("Local buildings")
    gdf.plot(ax=axes[1], color="#eeeeee", linewidth=0.05, edgecolor="#999999")
    gdf[gdf["in_touying_blue_projection"]].plot(ax=axes[1], color="#00a6c8", linewidth=0.05, edgecolor="#005d73")
    axes[1].set_title("Mapped to Touying blue projection")
    gdf.plot(ax=axes[2], color="#eeeeee", linewidth=0.05, edgecolor="#999999")
    gdf[gdf["in_touying_blue_raster"]].plot(ax=axes[2], color="#117733", linewidth=0.05, edgecolor="#06411e")
    axes[2].set_title("Primary FID in island raster")
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_final_building_heights(buildings: Path, out: Path) -> None:
    gdf = gpd.read_file(buildings)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=220)
    gdf.plot(ax=axes[0], column="height_final_m", cmap="viridis", linewidth=0.06, edgecolor="#555555", legend=True)
    axes[0].set_title("Final building height")
    insar = gdf[gdf["height_source"] == "insar"]
    gdf.plot(ax=axes[1], color="#e6e6e6", linewidth=0.04, edgecolor="#999999")
    if not insar.empty:
        insar.plot(ax=axes[1], column="height_insar_m", cmap="magma", linewidth=0.06, edgecolor="#333333", legend=True)
    axes[1].set_title("Buildings with GAMMA/InSAR height")
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/pic_all")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = [
        (fig_buildings_map, (Path("results/geodata/tongji_buildings_prepared.gpkg"), out / "01_buildings_wgs84_prior_height.png")),
        (fig_height_area, (Path("results/tables/tongji_buildings_prepared.csv"), out / "02_height_prior_area_stats.png")),
        (fig_amplitude, (Path("work/mli/mean_crop_bmp_amplitude.npy"), out / "03_mean_sar_amplitude.png")),
        (fig_projection_overlay, (Path("work/mli/mean_crop_bmp_amplitude.npy"), Path("work/projection/20200708_blue_aligned_bottom_touying.geojson"), out / "04_touying_blue_aligned_bottom_overlay.png")),
        (fig_touying_projection_metrics, (Path("work/projection/20200708_blue_aligned_bottom_touying_metrics.csv"), out / "05_touying_blue_aligned_bottom_stats.png")),
        (fig_masks, (Path("work/mli/mean_crop_bmp_amplitude.npy"), Path("work/masks/building_fid_mask_touying_blue_bottom.npy"), Path("work/masks/island_label_touying_blue_bottom.npy"), out / "06_touying_blue_bottom_masks_islands_overlay.png")),
        (fig_island_stats, (Path("work/masks/islands_touying_blue_bottom.csv"), out / "07_touying_blue_bottom_island_stats.png")),
        (fig_temporal_pairs, (Path("work/baselines/temporal_candidate_pairs.csv"), out / "08_temporal_candidate_pairs.png")),
        (fig_touying_mapping, (Path("results/geodata/touying_fid_uid_buildings.geojson"), out / "13_touying_fid_uid_coverage.png")),
        (fig_final_building_heights, (Path("results/geodata/tongji_building_height_insar.geojson"), out / "21_final_building_heights_gamma_lgr.png")),
    ]
    made = []
    for func, params in tasks:
        func(*params)
        made.append(str(params[-1]))
    print(json.dumps({"created": made}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    main()
