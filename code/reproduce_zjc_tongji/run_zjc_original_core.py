#!/usr/bin/env python3
"""Re-run the Tongji high-rise experiment from the ZJC MATLAB core logic.

The numerical core is a direct, auditable translation of:
  zjc/my_study/unwrap_phase_matrix.m
  zjc/my_study/local_knn.m
  zjc/my_study/LGR_demerror_est.m

Only deterministic data-interface defects are corrected: background pixels are
not graph nodes, pair rows are joined by date-based paths, zero phase is not
NoData, and the number of valid pairs is explicit.  The original numerical
choices remain N_nearest=9, N_knn=4, Bisquare tune=4.685, 1 mm arc cutoff,
unweighted LGR and building height=max(DEM error)-min(DEM error).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.restoration import unwrap_phase

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TONGJI_CODE = PROJECT / "code" / "reproduce_tongji"
sys.path.insert(0, str(TONGJI_CODE))

from benchmark_paper_unwrap import paper_like_unwrap  # noqa: E402
from extract_gamma_differential_island_observations import read_fcomplex, read_float  # noqa: E402
from inventory_data import parse_gamma_par  # noqa: E402


def design_matrix(
    bperp_m: np.ndarray,
    dt_days: np.ndarray,
    wavelength_m: float,
    range_m: float,
    incidence_deg: float,
) -> np.ndarray:
    """Match the two columns built in ZJC LGR_demerror_est.m."""
    return np.column_stack(
        [
            4.0 * np.pi * bperp_m / (wavelength_m * range_m * np.sin(np.deg2rad(incidence_deg))),
            4.0 * np.pi * dt_days / wavelength_m,
        ]
    )


def solve_zjc_lgr(phases: np.ndarray, matrix: np.ndarray, min_pairs: int) -> dict[str, np.ndarray]:
    """Unweighted per-pixel LGR with the output sign used by the root ZJC code."""
    n_pix = phases.shape[1]
    dem_error = np.full(n_pix, np.nan, dtype=np.float32)
    rate_m_year = np.full(n_pix, np.nan, dtype=np.float32)
    rmse_rad = np.full(n_pix, np.nan, dtype=np.float32)
    n_valid = np.zeros(n_pix, dtype=np.int16)
    for col in range(n_pix):
        valid = np.isfinite(phases[:, col])
        count = int(valid.sum())
        n_valid[col] = count
        if count < min_pairs:
            continue
        coef, *_ = np.linalg.lstsq(matrix[valid], phases[valid, col], rcond=None)
        residual = phases[valid, col] - matrix[valid] @ coef
        dem_error[col] = -float(coef[0])
        rate_m_year[col] = float(coef[1]) * 365.0
        rmse_rad[col] = float(np.sqrt(np.mean(residual**2)))
    return {
        "dem_error_m": dem_error,
        "rate_m_year": rate_m_year,
        "rmse_rad": rmse_rad,
        "n_valid": n_valid,
    }


def first_reference_shift(values: np.ndarray) -> np.ndarray:
    """Set each island/pair to one local zero, matching the ZJC graph gauge."""
    out = values.copy()
    finite = np.flatnonzero(np.isfinite(out))
    if finite.size:
        out -= out[finite[0]]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--intf-root", default="work/gamma_sbas/intf_triangular_dsm")
    p.add_argument("--island-label", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy")
    p.add_argument("--islands-csv", default="work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv")
    p.add_argument("--fid-map", default="results/tables/clean_equal_height_fid_uid_map.csv")
    p.add_argument("--amplitude-dispersion", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    p.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    p.add_argument("--rows", type=int, default=630)
    p.add_argument("--cols", type=int, default=900)
    p.add_argument("--floor-min-exclusive", type=float, default=10.0, help="Tongji analogue of the paper high-rise subset; >10 gives about 79 vector buildings.")
    p.add_argument("--min-coherence", type=float, default=0.75)
    p.add_argument("--max-amplitude-dispersion", type=float, default=0.40)
    p.add_argument("--min-pairs", type=int, default=12)
    p.add_argument("--min-building-pixels", type=int, default=15)
    p.add_argument("--work-dir", default="work/zjc_original_reproduction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT
    resolve = lambda value: root / value
    work_dir = resolve(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    pairs = pd.read_csv(resolve(args.pairs_csv)).copy()
    islands = pd.read_csv(resolve(args.islands_csv)).copy()
    fid_map = pd.read_csv(resolve(args.fid_map)).copy()
    fid_meta = fid_map.set_index("touying_fid")
    islands["floor_prior"] = islands["primary_uid"].map(fid_meta["floor_prior"])
    islands["height_prior_m"] = islands["primary_uid"].map(fid_meta["height_prior_m"])
    selected = islands.loc[islands["floor_prior"] > args.floor_min_exclusive].copy()
    if selected.empty:
        raise RuntimeError("No islands satisfy the high-rise selection")

    label = np.load(resolve(args.island_label)).astype(np.int32)
    da = np.load(resolve(args.amplitude_dispersion)).astype(np.float32)
    if label.shape != (args.rows, args.cols) or da.shape != label.shape:
        raise ValueError(f"Unexpected raster shapes: label={label.shape}, DA={da.shape}")
    selected_ids = set(selected["island_id"].astype(int))
    selected_mask = np.isin(label, list(selected_ids)) & np.isfinite(da) & (da <= args.max_amplitude_dispersion)
    flat_pixels = np.flatnonzero(selected_mask.ravel())
    flat_to_pos = np.full(label.size, -1, dtype=np.int64)
    flat_to_pos[flat_pixels] = np.arange(flat_pixels.size, dtype=np.int64)

    par = parse_gamma_par(resolve(args.reference_par))
    wavelength_m = 299792458.0 / float(par["radar_frequency"])
    matrix = design_matrix(
        pairs["bperp_m"].to_numpy(np.float64),
        pairs["dt_days"].to_numpy(np.float64),
        wavelength_m,
        float(par["center_range_slc"]),
        float(par["incidence_angle"]),
    )

    n_pairs = len(pairs)
    n_pix = flat_pixels.size
    wrapped = np.full((n_pairs, args.rows, args.cols), np.nan, dtype=np.float32)
    coherence = np.full_like(wrapped, np.nan)
    global_selected = np.full((n_pairs, n_pix), np.nan, dtype=np.float32)
    print(f"Loading and globally unwrapping {n_pairs} date-keyed pairs ...", flush=True)
    for pair_idx, row in enumerate(pairs.itertuples(index=False), start=0):
        pair = f"{int(row.master):08d}_{int(row.slave):08d}"
        pair_dir = resolve(args.intf_root) / pair
        diff_file = pair_dir / f"{pair}.diff"
        coh_file = pair_dir / f"{pair}.cc"
        if not diff_file.exists() or not coh_file.exists():
            raise FileNotFoundError(f"Missing date-keyed pair product for {pair}")
        diff = read_fcomplex(diff_file, args.rows, args.cols)
        coh = read_float(coh_file, args.rows, args.cols)
        phase = np.angle(diff).astype(np.float32)
        wrapped[pair_idx] = phase
        coherence[pair_idx] = coh
        global_unwrapped = unwrap_phase(phase).astype(np.float32)
        global_selected[pair_idx] = global_unwrapped.ravel()[flat_pixels]
        print(f"  pair {pair_idx + 1:02d}/{n_pairs}: {pair}", flush=True)

    local_selected = np.full((n_pairs, n_pix), np.nan, dtype=np.float32)
    global_gauged = np.full_like(global_selected, np.nan)
    unwrap_status: Counter[str] = Counter()
    print(f"Running ZJC island graph unwrap on {len(selected)} high-rise islands ...", flush=True)
    for island_seq, info in enumerate(selected.itertuples(index=False), start=1):
        island_id = int(info.island_id)
        island_keep = (label == island_id) & selected_mask
        rr, cc = np.nonzero(island_keep)
        if rr.size == 0:
            continue
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        patch_keep = island_keep[r0:r1, c0:c1]
        patch_flat = np.flatnonzero(patch_keep.ravel())
        global_flat = (rr * args.cols + cc).astype(np.int64)
        positions = flat_to_pos[global_flat]
        if np.any(positions < 0):
            raise RuntimeError(f"Pixel index mapping failed for island {island_id}")
        for pair_idx in range(n_pairs):
            phase_patch = wrapped[pair_idx, r0:r1, c0:c1]
            coh_patch = coherence[pair_idx, r0:r1, c0:c1]
            valid = patch_keep & np.isfinite(coh_patch) & (coh_patch >= args.min_coherence)
            if int(valid.sum()) < args.min_building_pixels:
                unwrap_status["too_few_valid"] += 1
                continue
            try:
                local_unwrapped, info_unwrap = paper_like_unwrap(phase_patch, valid, wavelength_m)
                status = str(info_unwrap.get("status", "unknown"))
            except Exception as exc:  # keep the run auditable instead of silently stopping all islands
                status = f"exception:{type(exc).__name__}"
                unwrap_status[status] += 1
                continue
            unwrap_status[status] += 1
            local_vec = local_unwrapped.ravel()[patch_flat]
            valid_vec = valid.ravel()[patch_flat]
            local_vec[~valid_vec] = np.nan
            local_selected[pair_idx, positions] = first_reference_shift(local_vec)
            global_vec = global_selected[pair_idx, positions].copy()
            global_vec[~valid_vec] = np.nan
            global_gauged[pair_idx, positions] = first_reference_shift(global_vec)
        if island_seq == 1 or island_seq % 10 == 0 or island_seq == len(selected):
            print(f"  island {island_seq:02d}/{len(selected)}: id={island_id}, pixels={len(positions)}", flush=True)

    local_lgr = solve_zjc_lgr(local_selected, matrix, args.min_pairs)
    global_lgr = solve_zjc_lgr(global_gauged, matrix, args.min_pairs)
    rows_out: list[dict[str, object]] = []
    local_height_map = np.full(label.shape, np.nan, dtype=np.float32)
    global_height_map = np.full(label.shape, np.nan, dtype=np.float32)
    local_profile_map = np.full(label.shape, np.nan, dtype=np.float32)
    global_profile_map = np.full(label.shape, np.nan, dtype=np.float32)

    for info in selected.itertuples(index=False):
        island_id = int(info.island_id)
        full_idx = np.flatnonzero(((label == island_id) & selected_mask).ravel())
        positions = flat_to_pos[full_idx]
        local_dem = local_lgr["dem_error_m"][positions]
        global_dem = global_lgr["dem_error_m"][positions]
        local_valid = np.isfinite(local_dem)
        global_valid = np.isfinite(global_dem)
        local_height = float(np.nanmax(local_dem) - np.nanmin(local_dem)) if int(local_valid.sum()) >= args.min_building_pixels else np.nan
        global_height = float(np.nanmax(global_dem) - np.nanmin(global_dem)) if int(global_valid.sum()) >= args.min_building_pixels else np.nan
        if np.isfinite(local_height):
            local_height_map.ravel()[full_idx] = local_height
            local_profile_map.ravel()[full_idx[local_valid]] = local_dem[local_valid] - np.nanmin(local_dem)
        if np.isfinite(global_height):
            global_height_map.ravel()[full_idx] = global_height
            global_profile_map.ravel()[full_idx[global_valid]] = global_dem[global_valid] - np.nanmin(global_dem)
        rows_out.append(
            {
                "island_id": island_id,
                "touying_fid": int(info.primary_uid),
                "floor_prior": float(info.floor_prior),
                "vector_height_prior_m": float(info.height_prior_m),
                "mask_pixels": int(len(full_idx)),
                "zjc_valid_pixels": int(local_valid.sum()),
                "global_valid_pixels": int(global_valid.sum()),
                "zjc_height_range_m": local_height,
                "global_height_range_m": global_height,
                "zjc_median_lgr_rmse_rad": float(np.nanmedian(local_lgr["rmse_rad"][positions])) if local_valid.any() else np.nan,
                "global_median_lgr_rmse_rad": float(np.nanmedian(global_lgr["rmse_rad"][positions])) if global_valid.any() else np.nan,
                "zjc_median_valid_pairs": float(np.nanmedian(local_lgr["n_valid"][positions][local_valid])) if local_valid.any() else np.nan,
                "global_median_valid_pairs": float(np.nanmedian(global_lgr["n_valid"][positions][global_valid])) if global_valid.any() else np.nan,
            }
        )

    island_results = pd.DataFrame(rows_out)
    building_results = (
        island_results.sort_values("zjc_height_range_m", ascending=False)
        .groupby("touying_fid", as_index=False)
        .first()
    )
    island_results.to_csv(work_dir / "zjc_island_results.csv", index=False)
    building_results.to_csv(work_dir / "zjc_building_results.csv", index=False)
    np.savez_compressed(
        work_dir / "zjc_reproduction_arrays.npz",
        local_height_map=local_height_map,
        global_height_map=global_height_map,
        local_profile_map=local_profile_map,
        global_profile_map=global_profile_map,
        selected_mask=selected_mask,
        label=label,
    )
    elapsed = time.time() - started
    summary = {
        "algorithm_basis": [
            "zjc/my_study/unwrap_phase_matrix.m",
            "zjc/my_study/local_knn.m",
            "zjc/my_study/LGR_demerror_est.m",
        ],
        "interface_corrections": [
            "background pixels excluded from graph nodes",
            "date-keyed pair-to-file matching",
            "zero phase retained as valid numeric observation",
            "explicit min-pairs threshold",
        ],
        "parameters": {
            "N_nearest": 9,
            "N_knn": 4,
            "bisquare_tune": 4.685,
            "arc_accuracy_mm": 1.0,
            "floor_min_exclusive": args.floor_min_exclusive,
            "min_coherence": args.min_coherence,
            "max_amplitude_dispersion": args.max_amplitude_dispersion,
            "min_pairs": args.min_pairs,
            "min_building_pixels": args.min_building_pixels,
            "lgr_weighting": "unweighted, matching ZJC root implementation",
            "height_statistic": "max(dem_error)-min(dem_error), matching ZJC scripts",
        },
        "pairs": int(n_pairs),
        "selected_islands": int(len(selected)),
        "selected_buildings": int(selected["primary_uid"].nunique()),
        "selected_mask_pixels_after_DA": int(n_pix),
        "zjc_solved_islands": int(island_results["zjc_height_range_m"].notna().sum()),
        "zjc_solved_buildings": int(building_results["zjc_height_range_m"].notna().sum()),
        "global_solved_islands": int(island_results["global_height_range_m"].notna().sum()),
        "unwrap_status_counts": dict(sorted(unwrap_status.items())),
        "height_prior_used_as_fill": False,
        "height_prior_policy": "selection metadata and projection geometry only; never used to fill or replace ZJC/global inversion heights",
        "elapsed_seconds": elapsed,
        "note": "The vector height/floor fields select the high-rise analogue subset and are used later only for a clearly labelled consistency plot; they do not enter phase unwrapping or LGR height inversion.",
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
