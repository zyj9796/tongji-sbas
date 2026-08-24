#!/usr/bin/env python3
"""Select temporally repeatable rooftop PS candidates without height priors.

The input point heights come from the per-building GAMMA ``mb_pt`` run.  A
candidate is retained only if independent date and baseline subsets reproduce
its roof-minus-local-ground height.  The highest retained candidate represents
the roof; missing buildings remain unsolved.  Vector heights are merged only
after selection for an audit comparison and are never used for fitting/filling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def fit_height(y: np.ndarray, k: np.ndarray, q: np.ndarray, w: np.ndarray, select: np.ndarray) -> tuple[float, float]:
    ok = select & np.isfinite(y) & np.isfinite(k) & np.isfinite(q) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < 8:
        return np.nan, np.nan
    a = np.column_stack([k[ok], q[ok]])
    sw = np.sqrt(np.clip(w[ok], 1.0e-3, 1.0))
    coef, *_ = np.linalg.lstsq(a * sw[:, None], y[ok] * sw, rcond=None)
    resid = y[ok] - a @ coef
    return float(coef[0]), float(np.sqrt(np.average(resid**2, weights=w[ok])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", default="work/gamma_building_roofcore_closurecorrected_gamma100_full")
    p.add_argument("--ipta", default="work/gamma_native_ipta_sbas/ipta")
    p.add_argument("--pairs", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--max-point-sigma", type=float, default=0.75)
    p.add_argument("--max-fit-rms", type=float, default=1.20)
    p.add_argument("--max-subset-spread", type=float, default=8.0)
    p.add_argument("--max-mb-fit-difference", type=float, default=8.0)
    p.add_argument("--output", default="results/tables/tongji_building_height_stable_rooftop_ps_insar_only.csv")
    p.add_argument("--geojson", default="results/geodata/tongji_building_height_stable_rooftop_ps_insar_only.geojson")
    p.add_argument("--summary", default="results/metadata/tongji_building_height_stable_rooftop_ps_summary.json")
    args = p.parse_args()

    ipta = Path(args.ipta)
    metadata = pd.read_csv(ipta / "point_metadata.csv").sort_values("point_index").reset_index(drop=True)
    pairs = pd.read_csv(args.pairs, dtype={"master": str, "slave": str}).drop_duplicates(["master", "slave"])
    npoints, npairs = len(metadata), len(pairs)
    punw = np.memmap(ipta / "punw_closure_corrected", dtype=">f4", mode="r", shape=(npairs, npoints))
    sensitivity = np.load(ipta / "phase_height_sensitivity_rad_per_m.npy", mmap_mode="r")
    pmask = np.fromfile(ipta / "pmask", dtype=np.uint8)[:npoints] > 0
    pcc = np.memmap(Path(args.work) / "pcc", dtype=">f4", mode="r", shape=(npairs, npoints))
    wavelength = 299792458.0 / 5.4050005e9
    dt_year = pairs["dt_days"].to_numpy(float) / 365.25
    q = -4.0 * np.pi * dt_year / wavelength
    early = pd.to_datetime(pairs["slave"], format="%Y%m%d").to_numpy() <= np.datetime64("2022-12-31")
    late = pd.to_datetime(pairs["master"], format="%Y%m%d").to_numpy() >= np.datetime64("2022-11-01")
    long_b = pairs["bperp_m"].abs().to_numpy(float) >= 100.0
    short_b = ~long_b
    all_pairs = np.ones(npairs, dtype=bool)

    ground = metadata[metadata["point_class"].eq("ground") & pmask].copy()
    ground_ids = ground["point_index"].to_numpy(int)
    ground_tree = cKDTree(ground[["range_pixel", "azimuth_pixel"]].to_numpy(float))
    rows, point_rows = [], []
    shard_root = Path(args.work) / "building_shards"
    for shard in sorted(shard_root.glob("[0-9]*_points.npz"), key=lambda x: int(x.stem.split("_")[0])):
        clean_id = int(shard.stem.split("_")[0])
        data = np.load(shard)
        point_ids = data["point_index"].astype(int)
        mb_height = data["height_m"].astype(float)
        point_sigma = data["phase_sigma_rad"].astype(float)
        if not len(point_ids):
            continue
        xy = metadata.iloc[point_ids][["range_pixel", "azimuth_pixel"]].to_numpy(float)
        centroid = xy.mean(axis=0)
        _, near = ground_tree.query(centroid, k=min(40, len(ground_ids)))
        local_ground_ids = ground_ids[np.atleast_1d(near)]
        ground_phase = np.nanmedian(np.asarray(punw[:, local_ground_ids], dtype=float), axis=1)
        ground_coh = np.nanmedian(np.asarray(pcc[:, local_ground_ids], dtype=float), axis=1)
        candidates = []
        for point_id, height_mb, sigma in zip(point_ids, mb_height, point_sigma):
            y = np.asarray(punw[:, point_id], dtype=float) - ground_phase
            k = np.asarray(sensitivity[:, point_id], dtype=float)
            w = np.sqrt(np.clip(np.asarray(pcc[:, point_id], dtype=float), 0, 1) * np.clip(ground_coh, 0, 1))
            fits = [fit_height(y, k, q, w, subset) for subset in [all_pairs, early, late, long_b, short_b]]
            heights = np.asarray([x[0] for x in fits])
            rms = fits[0][1]
            subset_spread = float(np.nanmax(heights[1:]) - np.nanmin(heights[1:]))
            stable = bool(
                np.all(np.isfinite(heights))
                and np.isfinite(height_mb) and 0.0 <= height_mb <= 180.0
                and sigma <= args.max_point_sigma and rms <= args.max_fit_rms
                and subset_spread <= args.max_subset_spread
                and abs(height_mb - heights[0]) <= args.max_mb_fit_difference
            )
            record = {
                "clean_id": clean_id, "point_index": int(point_id), "mb_height_m": float(height_mb),
                "phase_sigma_rad": float(sigma), "fit_height_all_m": float(heights[0]), "fit_rms_rad": float(rms),
                "height_early_m": float(heights[1]), "height_late_m": float(heights[2]),
                "height_long_baseline_m": float(heights[3]), "height_short_baseline_m": float(heights[4]),
                "subset_height_spread_m": subset_spread, "stable": stable,
            }
            point_rows.append(record)
            if stable:
                candidates.append(record)
        if candidates:
            winner = max(candidates, key=lambda x: x["mb_height_m"])
            rows.append({
                "clean_id": clean_id, "insar_height_m": winner["mb_height_m"],
                "roof_point_index": winner["point_index"], "stable_roof_candidates": len(candidates),
                "phase_sigma_rad": winner["phase_sigma_rad"], "fit_rms_rad": winner["fit_rms_rad"],
                "subset_height_spread_m": winner["subset_height_spread_m"],
                "solution_source": "GAMMA_mb_pt_temporally_stable_rooftop_PS", "filled_from_prior": False,
            })

    result = pd.DataFrame(rows)
    point_audit = Path(args.work) / "stable_rooftop_point_audit.csv"
    pd.DataFrame(point_rows).to_csv(point_audit, index=False)
    buildings = gpd.read_file(args.buildings)
    prior = buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"})
    result = result.merge(prior, on="clean_id", how="left")
    result["difference_to_prior_m"] = result["insar_height_m"] - result["prior_height_m"]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    geo = buildings.merge(result, on="clean_id", how="left"); geo_path = Path(args.geojson); geo_path.parent.mkdir(parents=True, exist_ok=True); geo.to_file(geo_path, driver="GeoJSON")
    compared = result[["insar_height_m", "prior_height_m"]].dropna()
    summary = {
        "method": "highest GAMMA mb_pt rooftop PS passing early/late and long/short baseline repeatability",
        "prior_used_in_selection_or_filling": False,
        "thresholds": {"point_sigma_rad": args.max_point_sigma, "fit_rms_rad": args.max_fit_rms, "subset_spread_m": args.max_subset_spread, "mb_fit_difference_m": args.max_mb_fit_difference},
        "buildings_solved": len(result), "median_height_m": float(result.insar_height_m.median()) if len(result) else None,
        "p05_p95_height_m": [float(result.insar_height_m.quantile(.05)), float(result.insar_height_m.quantile(.95))] if len(result) else None,
        "post_selection_prior_comparison": {"count": len(compared), "mae_m": float((compared.insar_height_m-compared.prior_height_m).abs().mean()), "correlation": float(compared.corr().iloc[0,1]) if len(compared)>1 else None},
        "artifacts": {"result_csv": str(output), "result_geojson": str(geo_path), "point_audit": str(point_audit)},
    }
    summary_path = Path(args.summary); summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
