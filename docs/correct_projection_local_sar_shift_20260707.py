#!/usr/bin/env python3
"""Create an isolated SAR-local projection correction branch.

The correction uses only SAR amplitude/edge evidence and the current clean_id
roof mask geometry. It does not use the building-vector height attribute.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tongji")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage
from shapely import affinity

BASE = Path("/home/u/geocoding/tongji_sbas")
CODE = BASE / "code/reproduce_tongji"
sys.path.insert(0, str(CODE))

from optimize_red_building_mask_shifts import best_shift_for_building, shift_bool, robust01  # noqa: E402

OUT = BASE / "results/projection_correction_20260707"
FIG = OUT / "figures"
BASELINE = OUT / "baseline_current"


def centroid_table(uid_mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for clean_id in sorted(int(v) for v in np.unique(uid_mask) if int(v) > 0):
        rr, cc = np.nonzero(uid_mask == clean_id)
        if rr.size == 0:
            continue
        rows.append(
            {
                "clean_id": clean_id,
                "centroid_row": float(np.mean(rr)),
                "centroid_col": float(np.mean(cc)),
                "mask_pixels": int(rr.size),
            }
        )
    return pd.DataFrame(rows)


def add_field_consistency(metrics: pd.DataFrame, radius: float = 130.0) -> pd.DataFrame:
    out = metrics.copy()
    raw = out[out["raw_candidate"]].copy()
    if raw.empty:
        out["neighbor_support"] = 0
        out["neighbor_median_row_shift"] = 0.0
        out["neighbor_median_col_shift"] = 0.0
        out["shift_distance_to_neighbor_median"] = np.nan
        return out
    src = raw[["centroid_row", "centroid_col", "row_shift", "col_shift"]].to_numpy(dtype=float)
    support = []
    med_r = []
    med_c = []
    dist_to_med = []
    for row in out.itertuples(index=False):
        dr = src[:, 0] - float(row.centroid_row)
        dc = src[:, 1] - float(row.centroid_col)
        dist = np.hypot(dr, dc)
        keep = (dist <= radius) & (dist > 0)
        if not np.any(keep):
            support.append(0)
            med_r.append(0.0)
            med_c.append(0.0)
            dist_to_med.append(np.nan)
            continue
        mr = float(np.median(src[keep, 2]))
        mc = float(np.median(src[keep, 3]))
        support.append(int(np.sum(keep)))
        med_r.append(mr)
        med_c.append(mc)
        dist_to_med.append(float(np.hypot(float(row.row_shift) - mr, float(row.col_shift) - mc)))
    out["neighbor_support"] = support
    out["neighbor_median_row_shift"] = med_r
    out["neighbor_median_col_shift"] = med_c
    out["shift_distance_to_neighbor_median"] = dist_to_med
    return out


def apply_selected_shifts(uid_mask: np.ndarray, metrics: pd.DataFrame, min_pixels: int) -> np.ndarray:
    shifted = uid_mask.copy()
    accepted = metrics[metrics["accepted_final"]].sort_values("score_gain", ascending=False)
    for row in accepted.itertuples(index=False):
        clean_id = int(row.clean_id)
        original = uid_mask == clean_id
        candidate = shift_bool(original, int(row.row_shift), int(row.col_shift))
        candidate = candidate & ((shifted == 0) | (shifted == clean_id))
        if int(candidate.sum()) < min_pixels:
            continue
        shifted[shifted == clean_id] = 0
        shifted[candidate] = clean_id
    return shifted.astype(np.int32)


def translate_projection(gdf: gpd.GeoDataFrame, metrics: pd.DataFrame) -> gpd.GeoDataFrame:
    shift_map = {
        int(row.clean_id): (int(row.row_shift), int(row.col_shift), str(row.accept_reason))
        for row in metrics[metrics["accepted_final"]].itertuples(index=False)
    }
    out = gdf.copy()
    out["local_sar_row_shift"] = 0
    out["local_sar_col_shift"] = 0
    out["local_sar_shift_status"] = "baseline"
    out["local_sar_shift_branch"] = "projection_correction_20260707"
    for idx, row in out.iterrows():
        clean_id = int(row["clean_id"])
        if clean_id not in shift_map:
            continue
        dr, dc, status = shift_map[clean_id]
        out.at[idx, "geometry"] = affinity.translate(row.geometry, xoff=dc, yoff=dr)
        out.at[idx, "local_sar_row_shift"] = dr
        out.at[idx, "local_sar_col_shift"] = dc
        out.at[idx, "local_sar_shift_status"] = status
    return out


def save_preview(amp: np.ndarray, original: np.ndarray, corrected: np.ndarray, metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    amp_show = robust01(np.log1p(np.maximum(amp, 0)))
    changed = np.zeros(original.shape, dtype=np.uint8)
    changed[(original > 0) & (corrected > 0)] = 1
    changed[(original > 0) & (corrected == 0)] = 2
    changed[(original == 0) & (corrected > 0)] = 3
    accepted_ids = metrics.loc[metrics["accepted_final"], "clean_id"].astype(int).tolist()
    accepted_mask = np.isin(corrected, accepted_ids) if accepted_ids else np.zeros(corrected.shape, dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), dpi=260)
    axes[0, 0].imshow(amp_show, cmap="gray", interpolation="nearest")
    axes[0, 0].imshow(np.ma.masked_where(original <= 0, original % 251), cmap="turbo", alpha=0.42, interpolation="nearest")
    axes[0, 0].set_title("Baseline audited roof mask")
    axes[0, 1].imshow(amp_show, cmap="gray", interpolation="nearest")
    axes[0, 1].imshow(np.ma.masked_where(corrected <= 0, corrected % 251), cmap="turbo", alpha=0.42, interpolation="nearest")
    axes[0, 1].imshow(np.ma.masked_where(~accepted_mask, accepted_mask), cmap=ListedColormap(["#ffcc00"]), alpha=0.75, interpolation="nearest")
    axes[0, 1].set_title("Corrected mask; yellow = shifted buildings")
    axes[1, 0].imshow(changed, cmap=ListedColormap(["#ffffff", "#d9d9d9", "#d73027", "#1a9850"]), interpolation="nearest", vmin=0, vmax=3)
    axes[1, 0].set_title("Pixel changes; red = moved out, green = moved in")
    ax = axes[1, 1]
    raw = metrics[metrics["raw_candidate"]]
    acc = metrics[metrics["accepted_final"]]
    ax.scatter(raw["col_shift"], -raw["row_shift"], s=18, color="#b8c0cc", alpha=0.6, label="raw candidate")
    ax.scatter(acc["col_shift"], -acc["row_shift"], s=34, color="#1a9850", edgecolor="white", linewidth=0.4, label="accepted")
    ax.axhline(0, color="#d7dde4", lw=0.8)
    ax.axvline(0, color="#d7dde4", lw=0.8)
    ax.set_xlabel("Range shift (px)")
    ax.set_ylabel("Azimuth shift (px)")
    ax.set_title("Accepted local SAR shifts")
    ax.legend(loc="best", fontsize=7)
    for ax in axes.ravel()[:3]:
        ax.set_axis_off()
    fig.suptitle("Local SAR-amplitude projection correction branch", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "projection_correction_overlay.png", bbox_inches="tight")
    fig.savefig(FIG / "projection_correction_overlay.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BASELINE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    current_projection = BASE / "work/projection/20200708_clean_equal_height_roof_projection_sar.geojson"
    current_mask = BASE / "work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy"
    current_islands = BASE / "work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy"
    current_islands_csv = BASE / "work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv"
    current_shift_metrics = BASE / "work/projection/cleanid_split_red_building_mask_shift_metrics_audited.csv"
    amp_path = BASE / "work/mli/mean_crop_bmp_amplitude.npy"

    for src in [current_projection, current_mask, current_islands, current_islands_csv, current_shift_metrics]:
        shutil.copy2(src, BASELINE / src.name)

    uid_mask = np.load(current_mask).astype(np.int32)
    amp = np.load(amp_path).astype(np.float32)
    centroids = centroid_table(uid_mask)
    clean_ids = centroids["clean_id"].astype(int).tolist()
    scores = [
        best_shift_for_building(
            clean_id,
            uid_mask,
            amp,
            max_shift=8,
            min_pixels=20,
            min_score_gain=0.08,
            min_contrast_gain=0.025,
            min_edge_gain=0.025,
            max_overlap_frac=0.10,
            min_kept_frac=0.90,
        )
        for clean_id in clean_ids
    ]
    metrics = pd.DataFrame([s.__dict__ for s in scores]).merge(centroids, on="clean_id", how="left")
    metrics["raw_candidate"] = metrics["accepted"].astype(bool)
    metrics = add_field_consistency(metrics)
    metrics["accepted_final"] = False
    metrics["accept_reason"] = metrics["reject_reason"].astype(str)
    # Strong candidates are accepted directly; moderate candidates need local
    # shift-field consistency to reduce isolated false-positive moves.
    strong = metrics["raw_candidate"] & (metrics["score_gain"] >= 0.16)
    field_consistent = (
        metrics["raw_candidate"]
        & (metrics["score_gain"] >= 0.08)
        & (metrics["neighbor_support"] >= 2)
        & (metrics["shift_distance_to_neighbor_median"].fillna(999) <= 3.0)
    )
    metrics.loc[strong | field_consistent, "accepted_final"] = True
    metrics.loc[strong, "accept_reason"] = "accepted_strong_sar_gain"
    metrics.loc[field_consistent & ~strong, "accept_reason"] = "accepted_local_field_consistent"
    metrics.loc[metrics["raw_candidate"] & ~metrics["accepted_final"], "accept_reason"] = "rejected_isolated_candidate"

    corrected_mask = apply_selected_shifts(uid_mask, metrics, min_pixels=20)
    corrected_mask_path = OUT / "building_fid_mask_clean_equal_height_roof_only_full_area_128_local_sar_corrected.npy"
    np.save(corrected_mask_path, corrected_mask)

    gdf = gpd.read_file(current_projection)
    corrected_projection = translate_projection(gdf, metrics)
    corrected_projection_path = OUT / "20200708_clean_equal_height_roof_projection_sar_local_sar_corrected.geojson"
    corrected_projection.to_file(corrected_projection_path, driver="GeoJSON")

    metrics_path = OUT / "local_sar_projection_shift_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    save_preview(amp, uid_mask, corrected_mask, metrics)

    accepted = metrics[metrics["accepted_final"]].copy()
    summary = {
        "method": "Conservative full-area local SAR-amplitude/edge mask correction. No building-vector height attribute is used.",
        "baseline_saved_dir": str(BASELINE.relative_to(BASE)),
        "input_projection": str(current_projection.relative_to(BASE)),
        "input_mask": str(current_mask.relative_to(BASE)),
        "searched_buildings": int(len(metrics)),
        "raw_candidates": int(metrics["raw_candidate"].sum()),
        "accepted_final": int(metrics["accepted_final"].sum()),
        "changed_pixels": int(np.sum(uid_mask != corrected_mask)),
        "accepted_clean_ids": [int(v) for v in accepted["clean_id"].sort_values().tolist()],
        "accept_reason_counts": {str(k): int(v) for k, v in metrics["accept_reason"].value_counts().sort_index().items()},
        "median_accepted_row_shift": float(accepted["row_shift"].median()) if not accepted.empty else 0.0,
        "median_accepted_col_shift": float(accepted["col_shift"].median()) if not accepted.empty else 0.0,
        "outputs": {
            "corrected_projection": str(corrected_projection_path.relative_to(BASE)),
            "corrected_mask": str(corrected_mask_path.relative_to(BASE)),
            "metrics_csv": str(metrics_path.relative_to(BASE)),
            "overlay_png": str((FIG / "projection_correction_overlay.png").relative_to(BASE)),
            "overlay_svg": str((FIG / "projection_correction_overlay.svg").relative_to(BASE)),
        },
        "height_field_use": "not_read_for_fitting_filtering_calibration_selection_or_qc",
    }
    summary_path = OUT / "local_sar_projection_correction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
