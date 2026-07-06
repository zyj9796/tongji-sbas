#!/usr/bin/env python3
"""Estimate building heights from APS-corrected SBAS deformation residual phase."""

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
import numpy as np
import pandas as pd

from estimate_pixel_lgr_building_heights import plot_qc, summarize_height, unwrap_patch
from extract_gamma_differential_island_observations import read_float, read_fcomplex
from inventory_data import parse_gamma_par


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def height_design(bperp: np.ndarray, wavelength: float, range_m: float, inc_deg: float) -> np.ndarray:
    return 4.0 * math.pi * bperp / (wavelength * range_m * math.sin(math.radians(inc_deg)))


def solve_dem_residual_pixels(
    residual_phases: np.ndarray,
    cohs: np.ndarray,
    a: np.ndarray,
    min_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pairs, n_pix = residual_phases.shape
    dem = np.full(n_pix, np.nan, dtype=np.float32)
    rmse = np.full(n_pix, np.nan, dtype=np.float32)
    n_valid = np.zeros(n_pix, dtype=np.int16)
    for j in range(n_pix):
        valid = np.isfinite(residual_phases[:, j]) & np.isfinite(cohs[:, j]) & np.isfinite(a) & (cohs[:, j] > 0)
        n_valid[j] = int(np.sum(valid))
        if n_valid[j] < min_pairs:
            continue
        w = np.clip(cohs[valid, j], 0.05, 1.0) ** 2
        av = a[valid]
        yv = residual_phases[valid, j]
        denom = float(np.sum(w * av * av))
        if not np.isfinite(denom) or denom <= 1e-8:
            continue
        coef = float(np.sum(w * av * yv) / denom)
        pred = av * coef
        dem[j] = coef
        rmse[j] = float(np.sqrt(np.mean((yv - pred) ** 2)))
    return dem, rmse, n_valid


def solve_wrapped_dem_residual_pixels(
    residual_wrapped: np.ndarray,
    cohs: np.ndarray,
    a: np.ndarray,
    min_pairs: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pairs, n_pix = residual_wrapped.shape
    dem = np.full(n_pix, np.nan, dtype=np.float32)
    rmse = np.full(n_pix, np.nan, dtype=np.float32)
    n_valid = np.zeros(n_pix, dtype=np.int16)
    for j in range(n_pix):
        valid = np.isfinite(residual_wrapped[:, j]) & np.isfinite(cohs[:, j]) & np.isfinite(a) & (cohs[:, j] > 0)
        n_valid[j] = int(np.sum(valid))
        if n_valid[j] < min_pairs:
            continue
        av = a[valid]
        y_wrapped = residual_wrapped[valid, j].astype(np.float64)
        w = np.clip(cohs[valid, j], 0.05, 1.0).astype(np.float64) ** 2
        coef = 0.0
        y_unwrapped = y_wrapped.copy()
        for _ in range(max(1, iterations)):
            denom = float(np.sum(w * av * av))
            if not np.isfinite(denom) or denom <= 1e-8:
                coef = np.nan
                break
            coef = float(np.sum(w * av * y_unwrapped) / denom)
            pred = av * coef
            ambiguity = np.rint((pred - y_wrapped) / (2.0 * np.pi))
            y_unwrapped = y_wrapped + 2.0 * np.pi * ambiguity
        if not np.isfinite(coef):
            continue
        pred = av * coef
        dem[j] = coef
        rmse[j] = float(np.sqrt(np.mean((y_unwrapped - pred) ** 2)))
    return dem, rmse, n_valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--intf-root", required=True)
    parser.add_argument("--aps-run-dir", required=True)
    parser.add_argument("--island-label", required=True)
    parser.add_argument("--fid-mask", required=True)
    parser.add_argument("--islands-csv", required=True)
    parser.add_argument("--fid-uid-map", required=True)
    parser.add_argument("--reference-height-rdc", required=True)
    parser.add_argument("--ground-dem-m", type=float, default=4.0)
    parser.add_argument("--residual-sign", type=float, default=-1.0)
    parser.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=0.60)
    parser.add_argument("--amplitude-dispersion-npy", default="")
    parser.add_argument("--max-amplitude-dispersion", type=float, default=0.35)
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--max-pixel-rmse-rad", type=float, default=1.25)
    parser.add_argument("--grubbs-alpha", type=float, default=0.20)
    parser.add_argument("--unwrap-method", choices=["temporal", "skimage", "paper"], default="temporal")
    parser.add_argument("--temporal-unwrap-iterations", type=int, default=6)
    parser.add_argument("--max-islands", type=int, default=0)
    parser.add_argument("--output-islands", required=True)
    parser.add_argument("--output-points", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--figure-svg", default="")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_csv)
    islands_info = pd.read_csv(args.islands_csv)
    if args.max_islands > 0:
        islands_info = islands_info.sort_values("pixel_count", ascending=False).head(args.max_islands).copy()
    fid_map = pd.read_csv(args.fid_uid_map)
    fid_to_uid = dict(zip(fid_map["touying_fid"].astype(int), fid_map["uid"].astype(int)))
    label = np.load(args.island_label)
    fid_mask = np.load(args.fid_mask)
    reference_height = read_float(Path(args.reference_height_rdc), args.rows, args.cols)
    amp_disp = None
    if args.amplitude_dispersion_npy:
        amp_disp = np.load(args.amplitude_dispersion_npy).astype(np.float32)
        if amp_disp.shape != label.shape:
            raise ValueError(f"amplitude dispersion shape {amp_disp.shape} does not match island label shape {label.shape}")

    aps_stack_mm = np.load(Path(args.aps_run_dir) / "arrays/los_displacement_timeseries_mm.npy").astype(np.float32)
    aps_summary = json.loads((Path(args.aps_run_dir) / "metadata/summary.json").read_text(encoding="utf-8"))
    dates = [str(d) for d in aps_summary.get("dates", [])]
    if not dates:
        # APS summary is compact; fall back to source ref_opt summary.
        source_run = Path(aps_summary["source_run"])
        source_summary = json.loads((source_run / "metadata/summary.json").read_text(encoding="utf-8"))
        dates = [str(d) for d in source_summary["dates"]]
    date_to_idx = {d: i for i, d in enumerate(dates)}
    if aps_stack_mm.shape[1:] != label.shape:
        raise ValueError(f"APS stack shape {aps_stack_mm.shape} does not match label shape {label.shape}")

    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    phase_scale = 4.0 * math.pi / wavelength
    a = height_design(
        pairs["bperp_m"].to_numpy(dtype=np.float64),
        wavelength,
        float(par["center_range_slc"]),
        float(par["incidence_angle"]),
    )

    residual_stack = []
    coh_stack = []
    for row in pairs.itertuples(index=False):
        master = str(row.master)
        slave = str(row.slave)
        pair = f"{master}_{slave}"
        if master not in date_to_idx or slave not in date_to_idx:
            raise ValueError(f"Pair {pair} dates are not present in APS time series")
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        observed = np.angle(diff).astype(np.float32)
        dm = aps_stack_mm[date_to_idx[master]] / 1000.0
        ds = aps_stack_mm[date_to_idx[slave]] / 1000.0
        deformation_phase = phase_scale * (ds - dm)
        residual_stack.append(wrap_phase(observed - deformation_phase).astype(np.float32))
        coh_stack.append(cc.astype(np.float32))
    residual_stack = np.stack(residual_stack, axis=0)
    coh_stack = np.stack(coh_stack, axis=0)

    rows_out = []
    points = []
    for island_index, info in enumerate(islands_info.itertuples(index=False), start=1):
        island_id = int(info.island_id)
        if island_index == 1 or island_index % 50 == 0 or island_index == len(islands_info):
            print(f"processing island {island_index}/{len(islands_info)} id={island_id}", flush=True)
        keep = label == island_id
        if int(np.sum(keep)) < args.min_pixels:
            continue
        rr, cc_idx = np.nonzero(keep)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc_idx.min()), int(cc_idx.max()) + 1
        patch_keep = keep[r0:r1, c0:c1]
        if amp_disp is not None and args.max_amplitude_dispersion > 0:
            da_patch = amp_disp[r0:r1, c0:c1]
            patch_keep = patch_keep & np.isfinite(da_patch) & (da_patch <= args.max_amplitude_dispersion)
        if int(np.sum(patch_keep)) < args.min_pixels:
            continue
        pix = np.nonzero(patch_keep.ravel())[0]
        n_pix = len(pix)
        phases = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
        cohs = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
        for k in range(len(pairs)):
            p = residual_stack[k, r0:r1, c0:c1]
            c = coh_stack[k, r0:r1, c0:c1]
            valid = patch_keep & np.isfinite(p) & np.isfinite(c) & (c >= args.min_coherence)
            if int(np.sum(valid)) < args.min_pixels:
                continue
            if args.unwrap_method == "temporal":
                valid_vec = valid.ravel()[pix]
                p_vec = p.ravel()[pix]
                coh_vec = c.ravel()[pix]
                phases[k, valid_vec] = p_vec[valid_vec]
                cohs[k, valid_vec] = coh_vec[valid_vec]
            else:
                try:
                    unw, _unwrap_status = unwrap_patch(p, valid, args.unwrap_method, wavelength)
                except Exception:
                    continue
                valid_vec = valid.ravel()[pix]
                unw_vec = unw.ravel()[pix]
                coh_vec = c.ravel()[pix]
                phases[k, valid_vec] = unw_vec[valid_vec]
                cohs[k, valid_vec] = coh_vec[valid_vec]
        if args.unwrap_method == "temporal":
            dem, pixel_rmse, n_valid = solve_wrapped_dem_residual_pixels(phases, cohs, a, args.min_pairs, args.temporal_unwrap_iterations)
        else:
            dem, pixel_rmse, n_valid = solve_dem_residual_pixels(phases, cohs, a, args.min_pairs)
        dem[(pixel_rmse > args.max_pixel_rmse_rad) | (n_valid < args.min_pairs)] = np.nan
        reference_vec = reference_height[r0:r1, c0:c1].ravel()[pix]
        amp_disp_vec = amp_disp[r0:r1, c0:c1].ravel()[pix] if amp_disp is not None else None
        island_height = summarize_height(dem, reference_vec, args.ground_dem_m, args.residual_sign, args.min_pixels, args.grubbs_alpha)
        primary_fid = int(info.primary_uid)
        uid = fid_to_uid.get(primary_fid)
        row_out = {
            "island_id": island_id,
            "primary_touying_fid": primary_fid,
            "uid": uid,
            "uid_count": int(info.uid_count),
            "pixel_count": int(info.pixel_count),
            "pixel_count_used": island_height["pixel_count_used"],
            "height_m": island_height["height_m"],
            "dem_error_p05_m": island_height["dem_error_p05_m"],
            "dem_error_median_m": island_height["dem_error_median_m"],
            "dem_error_p95_m": island_height["dem_error_p95_m"],
            "reference_elevation_median_m": island_height["reference_elevation_median_m"],
            "roof_elevation_p05_m": island_height["roof_elevation_p05_m"],
            "roof_elevation_median_m": island_height["roof_elevation_median_m"],
            "roof_elevation_p75_m": island_height["roof_elevation_p75_m"],
            "roof_elevation_p85_m": island_height["roof_elevation_p85_m"],
            "roof_elevation_p90_m": island_height["roof_elevation_p90_m"],
            "roof_elevation_p95_m": island_height["roof_elevation_p95_m"],
            "building_height_p05_m": island_height["building_height_p05_m"],
            "building_height_p25_m": island_height["building_height_p25_m"],
            "building_height_p50_m": island_height["building_height_p50_m"],
            "building_height_p75_m": island_height["building_height_p75_m"],
            "building_height_p85_m": island_height["building_height_p85_m"],
            "building_height_p90_m": island_height["building_height_p90_m"],
            "building_height_p95_m": island_height["building_height_p95_m"],
            "building_height_max_m": island_height["building_height_max_m"],
            "max_height_grubbs_n": island_height["max_height_grubbs_n"],
            "max_height_grubbs_g": island_height["max_height_grubbs_g"],
            "max_height_grubbs_gcrit": island_height["max_height_grubbs_gcrit"],
            "max_height_grubbs_p": island_height["max_height_grubbs_p"],
            "max_height_reject_outlier": island_height["max_height_reject_outlier"],
            "max_height_reliable": island_height["max_height_reliable"],
            "building_height_grubbs_top_m": island_height["building_height_grubbs_top_m"],
            "grubbs_top_removed_count": island_height["grubbs_top_removed_count"],
            "grubbs_top_remaining_count": island_height["grubbs_top_remaining_count"],
            "grubbs_top_last_g": island_height["grubbs_top_last_g"],
            "grubbs_top_last_gcrit": island_height["grubbs_top_last_gcrit"],
            "grubbs_top_last_p": island_height["grubbs_top_last_p"],
            "grubbs_top_reliable": island_height["grubbs_top_reliable"],
            "median_coherence": float(np.nanmedian(cohs)) if np.isfinite(cohs).any() else np.nan,
            "median_pixel_dem_rmse_rad": float(np.nanmedian(pixel_rmse)) if np.isfinite(pixel_rmse).any() else np.nan,
            "median_valid_pairs": float(np.nanmedian(n_valid[n_valid > 0])) if np.any(n_valid > 0) else np.nan,
            "median_amplitude_dispersion": float(np.nanmedian(amp_disp_vec)) if amp_disp_vec is not None and np.isfinite(amp_disp_vec).any() else np.nan,
            "method": "aps_sbas_deformation_removed_dem_residual_plus_dsm_minus_ground",
        }
        rows_out.append(row_out)

        fid_patch = fid_mask[r0:r1, c0:c1].ravel()[pix].astype(np.int64)
        for fid in sorted(int(v) for v in np.unique(fid_patch) if int(v) > 0):
            building_uid = fid_to_uid.get(fid)
            if building_uid is None:
                continue
            building_keep = fid_patch == fid
            building_summary = summarize_height(
                dem[building_keep],
                reference_vec[building_keep],
                args.ground_dem_m,
                args.residual_sign,
                args.min_pixels,
                args.grubbs_alpha,
            )
            if not np.isfinite(building_summary["height_m"]):
                continue
            building_coh = cohs[:, building_keep]
            building_rmse = pixel_rmse[building_keep]
            building_nvalid = n_valid[building_keep]
            building_da = amp_disp_vec[building_keep] if amp_disp_vec is not None else None
            points.append(
                {
                    "uid": int(building_uid),
                    "touying_fid": int(fid),
                    "island_id": island_id,
                    "island_uid_count": int(info.uid_count),
                    "height_m": building_summary["height_m"],
                    "pixel_count_used": building_summary["pixel_count_used"],
                    "dem_error_p05_m": building_summary["dem_error_p05_m"],
                    "dem_error_median_m": building_summary["dem_error_median_m"],
                    "dem_error_p95_m": building_summary["dem_error_p95_m"],
                    "reference_elevation_median_m": building_summary["reference_elevation_median_m"],
                    "roof_elevation_p05_m": building_summary["roof_elevation_p05_m"],
                    "roof_elevation_median_m": building_summary["roof_elevation_median_m"],
                    "roof_elevation_p75_m": building_summary["roof_elevation_p75_m"],
                    "roof_elevation_p85_m": building_summary["roof_elevation_p85_m"],
                    "roof_elevation_p90_m": building_summary["roof_elevation_p90_m"],
                    "roof_elevation_p95_m": building_summary["roof_elevation_p95_m"],
                    "building_height_p05_m": building_summary["building_height_p05_m"],
                    "building_height_p25_m": building_summary["building_height_p25_m"],
                    "building_height_p50_m": building_summary["building_height_p50_m"],
                    "building_height_p75_m": building_summary["building_height_p75_m"],
                    "building_height_p85_m": building_summary["building_height_p85_m"],
                    "building_height_p90_m": building_summary["building_height_p90_m"],
                    "building_height_p95_m": building_summary["building_height_p95_m"],
                    "building_height_max_m": building_summary["building_height_max_m"],
                    "max_height_grubbs_n": building_summary["max_height_grubbs_n"],
                    "max_height_grubbs_g": building_summary["max_height_grubbs_g"],
                    "max_height_grubbs_gcrit": building_summary["max_height_grubbs_gcrit"],
                    "max_height_grubbs_p": building_summary["max_height_grubbs_p"],
                    "max_height_reject_outlier": building_summary["max_height_reject_outlier"],
                    "max_height_reliable": building_summary["max_height_reliable"],
                    "building_height_grubbs_top_m": building_summary["building_height_grubbs_top_m"],
                    "grubbs_top_removed_count": building_summary["grubbs_top_removed_count"],
                    "grubbs_top_remaining_count": building_summary["grubbs_top_remaining_count"],
                    "grubbs_top_last_g": building_summary["grubbs_top_last_g"],
                    "grubbs_top_last_gcrit": building_summary["grubbs_top_last_gcrit"],
                    "grubbs_top_last_p": building_summary["grubbs_top_last_p"],
                    "grubbs_top_reliable": building_summary["grubbs_top_reliable"],
                    "coh_mean": float(np.nanmedian(building_coh)) if np.isfinite(building_coh).any() else np.nan,
                    "dem_residual_rmse_rad": float(np.nanmedian(building_rmse)) if np.isfinite(building_rmse).any() else np.nan,
                    "valid_pairs_median": float(np.nanmedian(building_nvalid[building_nvalid > 0])) if np.any(building_nvalid > 0) else np.nan,
                    "amplitude_dispersion": float(np.nanmedian(building_da)) if building_da is not None and np.isfinite(building_da).any() else "",
                    "method": "aps_sbas_deformation_removed_dem_residual_plus_dsm_minus_ground",
                }
            )

    out = pd.DataFrame(rows_out)
    pts = pd.DataFrame(points)
    Path(args.output_islands).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_points).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_islands, index=False)
    pts.to_csv(args.output_points, index=False)
    plot_qc(out, label, islands_info, Path(args.figure), Path(args.figure_svg) if args.figure_svg else None)
    summary = {
        "pairs": int(len(pairs)),
        "islands_total": int(len(islands_info)),
        "islands_processed": int(len(out)),
        "islands_with_height": int(out["height_m"].notna().sum()) if not out.empty else 0,
        "height_points": int(len(pts)),
        "height_point_buildings": int(pts["uid"].nunique()) if not pts.empty and "uid" in pts else 0,
        "aps_run_dir": args.aps_run_dir,
        "reference_height_rdc": args.reference_height_rdc,
        "ground_dem_m": args.ground_dem_m,
        "residual_sign": args.residual_sign,
        "unwrap_method": args.unwrap_method,
        "temporal_unwrap_iterations": args.temporal_unwrap_iterations,
        "min_coherence": args.min_coherence,
        "max_amplitude_dispersion": args.max_amplitude_dispersion,
        "min_pairs": args.min_pairs,
        "max_pixel_rmse_rad": args.max_pixel_rmse_rad,
        "grubbs_alpha": args.grubbs_alpha,
        "output_islands": args.output_islands,
        "output_points": args.output_points,
        "figure": args.figure,
        "figure_svg": args.figure_svg,
        "method": "APS-corrected SBAS deformation phase is removed from DSM differential interferograms; remaining unwrapped residual phase is regressed against Bperp to estimate DSM-relative height residual. No shp height field is used for fitting, filtering, or calibration.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
