#!/usr/bin/env python3
"""Diagnose LGR-stage no-solution failures for the current audited mask.

This script is documentation-local on purpose: it does not change the active
height workflow, and it only writes audit outputs under docs/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tongji")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CODE = BASE / "code/reproduce_tongji"
sys.path.insert(0, str(CODE))

from estimate_pixel_lgr_building_heights import design, solve_pixels, unwrap_patch  # noqa: E402
from extract_gamma_differential_island_observations import read_fcomplex, read_float  # noqa: E402
from inventory_data import parse_gamma_par  # noqa: E402


def load_stacks(pairs: pd.DataFrame, intf_root: Path, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    phase_stack = []
    coh_stack = []
    for row in pairs.itertuples(index=False):
        pair = f"{row.master}_{row.slave}"
        pair_dir = intf_root / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", rows, cols)
        cc = read_float(pair_dir / f"{pair}.cc", rows, cols)
        phase_stack.append(np.angle(diff).astype(np.float32))
        coh_stack.append(cc.astype(np.float32))
    return np.stack(phase_stack, axis=0), np.stack(coh_stack, axis=0)


def classify_building(row: pd.Series, min_pixels: int) -> str:
    if row["max_final_valid_pixels"] >= min_pixels:
        return "has_enough_lgr_pixels_but_still_no_building_height"
    if row["max_rmse_pass_pixels"] < min_pixels:
        if row["max_bperp_pass_pixels"] < min_pixels:
            if row["max_valid_pairs_pass_pixels"] < min_pixels:
                return "valid_pairs_lt_min_after_unwrap"
            return "bperp_span_lt_min"
        return "rmse_gt_max"
    return "height_summary_or_reference_filter_lt_min"


def main() -> None:
    rows = 630
    cols = 900
    min_pixels = 20
    min_pairs = 12
    min_coherence = 0.75
    max_amplitude_dispersion = 0.40
    max_pixel_rmse_rad = 1.25
    min_bperp_span_m = 120.0
    unwrap_method = "paper"

    docs = BASE / "docs"
    no_csv = docs / "no_solution_failure_audit_20260707.csv"
    islands_csv = BASE / "work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv"
    label_npy = BASE / "work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy"
    amp_npy = BASE / "work/mli/amplitude_dispersion_crop_bmp.npy"
    pairs_csv = BASE / "work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv"
    intf_root = BASE / "work/gamma_sbas/intf_triangular_dsm"
    par_path = BASE / "data/tongji_rslc/20200708.rslc.par"

    no = pd.read_csv(no_csv)
    target_clean_ids = set(
        no.loc[
            no["failure_class_available_evidence"].eq("lgr_pixel_filters_left_too_few_pixels"),
            "clean_id",
        ]
        .dropna()
        .astype(int)
        .tolist()
    )
    islands = pd.read_csv(islands_csv)
    islands = islands[islands["primary_uid"].astype(int).isin(target_clean_ids)].copy()
    islands = islands.sort_values(["primary_uid", "island_id"]).reset_index(drop=True)

    pairs = pd.read_csv(pairs_csv)
    label = np.load(label_npy)
    amp = np.load(amp_npy).astype(np.float32)
    par = parse_gamma_par(par_path)
    wavelength = 299792458.0 / float(par["radar_frequency"])
    a = design(
        pairs["bperp_m"].to_numpy(dtype=np.float64),
        pairs["dt_days"].to_numpy(dtype=np.float64),
        wavelength,
        float(par["center_range_slc"]),
        float(par["incidence_angle"]),
    )
    bperp = pairs["bperp_m"].to_numpy(dtype=np.float64)

    print(f"loading {len(pairs)} interferogram pairs", flush=True)
    phase_stack, coh_stack = load_stacks(pairs, intf_root, rows, cols)

    out_rows: list[dict[str, float | int | str]] = []
    for idx, info in enumerate(islands.itertuples(index=False), start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(islands):
            print(f"diagnosing island {idx}/{len(islands)} id={int(info.island_id)}", flush=True)

        island_id = int(info.island_id)
        clean_id = int(info.primary_uid)
        keep = label == island_id
        raw_pixels = int(np.sum(keep))
        row = {
            "clean_id": clean_id,
            "island_id": island_id,
            "raw_mask_pixels": raw_pixels,
            "da_pass_pixels": 0,
            "pair_attempts": 0,
            "pair_unwrap_success": 0,
            "pair_low_valid_pixels": 0,
            "pair_unwrap_exceptions": 0,
            "pixels_with_any_unwrapped_obs": 0,
            "valid_pairs_pass_pixels": 0,
            "bperp_pass_pixels": 0,
            "rmse_pass_pixels": 0,
            "final_valid_pixels": 0,
            "median_valid_pairs_all_da_pixels": np.nan,
            "median_bperp_span_valid_pairs": np.nan,
            "median_rmse_bperp_pass": np.nan,
        }
        if raw_pixels < min_pixels:
            out_rows.append(row)
            continue

        rr, cc_idx = np.nonzero(keep)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc_idx.min()), int(cc_idx.max()) + 1
        patch_keep = keep[r0:r1, c0:c1]
        da_patch = amp[r0:r1, c0:c1]
        patch_keep = patch_keep & np.isfinite(da_patch) & (da_patch <= max_amplitude_dispersion)
        row["da_pass_pixels"] = int(np.sum(patch_keep))
        if row["da_pass_pixels"] < min_pixels:
            out_rows.append(row)
            continue

        pix = np.nonzero(patch_keep.ravel())[0]
        n_pix = len(pix)
        phases = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
        cohs = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)

        for k in range(len(pairs)):
            p = phase_stack[k, r0:r1, c0:c1]
            c = coh_stack[k, r0:r1, c0:c1]
            valid = patch_keep & np.isfinite(p) & np.isfinite(c) & (c >= min_coherence)
            if int(np.sum(valid)) < min_pixels:
                row["pair_low_valid_pixels"] += 1
                continue
            row["pair_attempts"] += 1
            try:
                unw, _ = unwrap_patch(p, valid, unwrap_method, wavelength)
            except Exception:
                row["pair_unwrap_exceptions"] += 1
                continue
            row["pair_unwrap_success"] += 1
            valid_vec = valid.ravel()[pix]
            unw_vec = unw.ravel()[pix]
            coh_vec = c.ravel()[pix]
            phases[k, valid_vec] = unw_vec[valid_vec]
            cohs[k, valid_vec] = coh_vec[valid_vec]

        dem, _rate, pixel_rmse, n_valid, bperp_span = solve_pixels(phases, cohs, a, bperp, min_pairs)
        any_obs = n_valid > 0
        valid_pairs_pass = n_valid >= min_pairs
        bperp_pass = valid_pairs_pass & np.isfinite(bperp_span) & (bperp_span >= min_bperp_span_m)
        rmse_pass = bperp_pass & np.isfinite(pixel_rmse) & (pixel_rmse <= max_pixel_rmse_rad)
        final = rmse_pass & np.isfinite(dem)

        row["pixels_with_any_unwrapped_obs"] = int(np.sum(any_obs))
        row["valid_pairs_pass_pixels"] = int(np.sum(valid_pairs_pass))
        row["bperp_pass_pixels"] = int(np.sum(bperp_pass))
        row["rmse_pass_pixels"] = int(np.sum(rmse_pass))
        row["final_valid_pixels"] = int(np.sum(final))
        if np.any(any_obs):
            row["median_valid_pairs_all_da_pixels"] = float(np.median(n_valid[any_obs]))
        if np.any(valid_pairs_pass & np.isfinite(bperp_span)):
            row["median_bperp_span_valid_pairs"] = float(np.median(bperp_span[valid_pairs_pass & np.isfinite(bperp_span)]))
        if np.any(bperp_pass & np.isfinite(pixel_rmse)):
            row["median_rmse_bperp_pass"] = float(np.median(pixel_rmse[bperp_pass & np.isfinite(pixel_rmse)]))
        out_rows.append(row)

    island_diag = pd.DataFrame(out_rows)
    island_out = docs / "lgr_failure_gate_diagnostics_islands_20260707.csv"
    island_diag.to_csv(island_out, index=False)

    building = (
        island_diag.groupby("clean_id")
        .agg(
            island_count=("island_id", "count"),
            raw_mask_pixels=("raw_mask_pixels", "sum"),
            da_pass_pixels=("da_pass_pixels", "sum"),
            max_da_pass_pixels=("da_pass_pixels", "max"),
            max_pair_unwrap_success=("pair_unwrap_success", "max"),
            max_pixels_with_any_unwrapped_obs=("pixels_with_any_unwrapped_obs", "max"),
            max_valid_pairs_pass_pixels=("valid_pairs_pass_pixels", "max"),
            max_bperp_pass_pixels=("bperp_pass_pixels", "max"),
            max_rmse_pass_pixels=("rmse_pass_pixels", "max"),
            max_final_valid_pixels=("final_valid_pixels", "max"),
            median_valid_pairs_all_da_pixels=("median_valid_pairs_all_da_pixels", "median"),
            median_bperp_span_valid_pairs=("median_bperp_span_valid_pairs", "median"),
            median_rmse_bperp_pass=("median_rmse_bperp_pass", "median"),
            source_islands=("island_id", lambda s: ",".join(str(int(v)) for v in sorted(s))),
        )
        .reset_index()
    )
    building["lgr_gate_failure_class"] = building.apply(lambda row: classify_building(row, min_pixels), axis=1)
    building_out = docs / "lgr_failure_gate_diagnostics_buildings_20260707.csv"
    building.to_csv(building_out, index=False)

    summary = {
        "date": "2026-07-07",
        "target_failure_class": "lgr_pixel_filters_left_too_few_pixels",
        "target_buildings": int(len(target_clean_ids)),
        "target_islands": int(len(islands)),
        "building_failure_counts": {
            str(k): int(v)
            for k, v in building["lgr_gate_failure_class"].value_counts().sort_index().items()
        },
        "parameters": {
            "min_pixels": min_pixels,
            "min_pairs": min_pairs,
            "min_coherence": min_coherence,
            "max_amplitude_dispersion": max_amplitude_dispersion,
            "max_pixel_rmse_rad": max_pixel_rmse_rad,
            "min_bperp_span_m": min_bperp_span_m,
            "unwrap_method": unwrap_method,
            "pair_count": int(len(pairs)),
        },
        "height_field_use": "not_read_for_fitting_filtering_calibration_selection_or_qc",
        "outputs": {
            "island_csv": str(island_out.relative_to(BASE)),
            "building_csv": str(building_out.relative_to(BASE)),
            "summary_json": "docs/lgr_failure_gate_diagnostics_20260707_summary.json",
        },
    }
    summary_out = docs / "lgr_failure_gate_diagnostics_20260707_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
