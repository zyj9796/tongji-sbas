#!/usr/bin/env python3
"""Aggregate audited pixel solutions and export building-level CSV/GPKG."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


WIDTH = 10_000
LINES = 7_000


def metrics(path: Path) -> tuple[pd.DataFrame, np.lib.npyio.NpzFile]:
    data = np.load(path)
    uid = data["building_uid"].astype(np.int64)
    height = data["dem_error_or_height_above_anchor_m"].astype(np.float64)
    rms = data["phase_residual_rms_rad"].astype(np.float64)
    sigma = data["height_sigma_m"].astype(np.float64)
    rows = []
    for building in np.unique(uid):
        member = (uid == building) & np.isfinite(height)
        values = height[member]
        if not len(values):
            continue
        q05, q25, q50, q75, q95 = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
        iqr = q75 - q25
        keep = member & (height >= q25 - 1.5 * iqr) & (height <= q75 + 1.5 * iqr)
        rows.append({
            "building_uid": int(building),
            "selected_pixel_count": int(member.sum()),
            "iqr_retained_pixel_count": int(keep.sum()),
            "paper_text_median_m": float(np.median(height[keep])),
            "original_code_range_m": float(np.max(values) - np.min(values)),
            "robust_q95_q05_height_m": float(q95 - q05),
            "relative_height_q05_m": float(q05),
            "relative_height_q95_m": float(q95),
            "median_phase_residual_rms_rad": float(np.nanmedian(rms[member])),
            "median_height_sigma_m": float(np.nanmedian(sigma[member])),
        })
    return pd.DataFrame(rows), data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-48", type=Path, required=True)
    parser.add_argument("--pixel-45", type=Path, required=True)
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--ground-points", type=Path, required=True)
    parser.add_argument("--ground-height-rdc", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--gpkg-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    result48, _ = metrics(args.pixel_48)
    result45, _ = metrics(args.pixel_45)
    result45 = result45[["building_uid", "paper_text_median_m", "robust_q95_q05_height_m"]].rename(
        columns={
            "paper_text_median_m": "paper_text_median_45_m",
            "robust_q95_q05_height_m": "robust_height_45_m",
        }
    )
    result = result48.merge(result45, on="building_uid", how="left")
    result["robust_height_network_difference_m"] = (
        result["robust_q95_q05_height_m"] - result["robust_height_45_m"]
    )

    ground_points = np.load(args.ground_points)
    ground_raster = np.memmap(
        args.ground_height_rdc, dtype=">f4", mode="r", shape=(LINES, WIDTH)
    )
    ground_rows = []
    for building in np.unique(ground_points["building_uid"]):
        member = ground_points["building_uid"] == building
        value = ground_raster[ground_points["row"][member], ground_points["col"][member]]
        ground_rows.append({
            "building_uid": int(building),
            "ground_ellipsoid_height_m": float(np.nanmedian(value)),
        })
    result = result.merge(pd.DataFrame(ground_rows), on="building_uid", how="left")
    result["recommended_building_height_m"] = result["robust_q95_q05_height_m"]
    result["roof_ellipsoid_height_m"] = (
        result["ground_ellipsoid_height_m"] + result["recommended_building_height_m"]
    )
    result["quality_pass"] = (
        (result["selected_pixel_count"] >= 5)
        & (result["median_height_sigma_m"] <= 10.0)
        & (result["recommended_building_height_m"] >= 0.0)
        & (result["recommended_building_height_m"] <= 400.0)
        & (result["robust_height_network_difference_m"].abs() <= 10.0)
    )
    result["height_definition"] = "SBAS像元相对高差的P95-P05；非先验填充"

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("building_uid").to_csv(args.csv_output, index=False, encoding="utf-8-sig")

    buildings = gpd.read_file(args.buildings)
    output = buildings.merge(result, on="building_uid", how="left")
    output["solution_status"] = np.where(
        output["recommended_building_height_m"].isna(), "无解",
        np.where(output["quality_pass"].fillna(False), "通过", "低可信"),
    )
    if args.gpkg_output.exists():
        args.gpkg_output.unlink()
    output.to_file(args.gpkg_output, layer="building_height", driver="GPKG")

    # Floor is introduced only after all InSAR products are frozen, in a
    # separate audit table.  It is never merged back into recommended height.
    audit = output.loc[output["recommended_building_height_m"].notna(), [
        "building_uid", "Floor", "recommended_building_height_m",
        "paper_text_median_m", "original_code_range_m", "quality_pass",
    ]].copy()
    audit["floor_nominal_3m_audit_only"] = audit["Floor"] * 3.0
    audit["recommended_minus_nominal_m"] = (
        audit["recommended_building_height_m"] - audit["floor_nominal_3m_audit_only"]
    )
    audit.drop(columns="geometry", errors="ignore").to_csv(
        args.audit_output, index=False, encoding="utf-8-sig"
    )

    solved = result["recommended_building_height_m"].to_numpy()
    summary = {
        "candidate_highrise_building_count": int(len(np.unique(ground_points["building_uid"]))),
        "solved_building_count": int(len(result)),
        "quality_pass_building_count": int(result["quality_pass"].sum()),
        "recommended_height_definition": "pixel SBAS relative height P95-P05",
        "recommended_height_m_quantiles": {
            str(q): float(np.quantile(solved, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "ground_ellipsoid_height_m_quantiles": {
            str(q): float(np.nanquantile(result["ground_ellipsoid_height_m"], q))
            for q in (0.05, 0.5, 0.95)
        },
        "network_48_minus_45_robust_height_m_quantiles": {
            str(q): float(np.nanquantile(result["robust_height_network_difference_m"], q))
            for q in (0.05, 0.5, 0.95)
        },
        "prior_policy": "Floor is absent from inversion and aggregation; nominal 3m/floor appears only in separate post hoc audit CSV",
        "unsolved_policy": "null; never filled",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
