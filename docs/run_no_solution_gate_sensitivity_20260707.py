#!/usr/bin/env python3
"""Gate-level sensitivity pretest for current no-solution buildings.

This is a diagnostic pretest only. It does not write any active height product;
it estimates which no-solution buildings can reach at least 20 LGR pixels under
small gate changes before running a full product rebuild.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Variant:
    name: str
    target_class: str
    min_pairs: int = 12
    max_amplitude_dispersion: float = 0.40
    max_pixel_rmse_rad: float = 1.25
    min_coherence: float = 0.75
    min_bperp_span_m: float = 120.0


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


def diagnose_island(
    info: pd.Series,
    label: np.ndarray,
    amp: np.ndarray,
    phase_stack: np.ndarray,
    coh_stack: np.ndarray,
    pairs: pd.DataFrame,
    design_matrix: np.ndarray,
    bperp: np.ndarray,
    wavelength: float,
    variant: Variant,
    min_pixels: int,
) -> dict[str, float | int | str]:
    island_id = int(info["island_id"])
    clean_id = int(info["primary_uid"])
    keep = label == island_id
    raw_pixels = int(np.sum(keep))
    row: dict[str, float | int | str] = {
        "variant": variant.name,
        "target_class": variant.target_class,
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
        "reaches_min_pixels": False,
    }
    if raw_pixels < min_pixels:
        return row

    rr, cc_idx = np.nonzero(keep)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc_idx.min()), int(cc_idx.max()) + 1
    patch_keep = keep[r0:r1, c0:c1]
    da_patch = amp[r0:r1, c0:c1]
    patch_keep = patch_keep & np.isfinite(da_patch) & (da_patch <= variant.max_amplitude_dispersion)
    row["da_pass_pixels"] = int(np.sum(patch_keep))
    if int(row["da_pass_pixels"]) < min_pixels:
        return row

    pix = np.nonzero(patch_keep.ravel())[0]
    n_pix = len(pix)
    phases = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
    cohs = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
    for k in range(len(pairs)):
        p = phase_stack[k, r0:r1, c0:c1]
        c = coh_stack[k, r0:r1, c0:c1]
        valid = patch_keep & np.isfinite(p) & np.isfinite(c) & (c >= variant.min_coherence)
        if int(np.sum(valid)) < min_pixels:
            row["pair_low_valid_pixels"] = int(row["pair_low_valid_pixels"]) + 1
            continue
        row["pair_attempts"] = int(row["pair_attempts"]) + 1
        try:
            unw, _ = unwrap_patch(p, valid, "paper", wavelength)
        except Exception:
            row["pair_unwrap_exceptions"] = int(row["pair_unwrap_exceptions"]) + 1
            continue
        row["pair_unwrap_success"] = int(row["pair_unwrap_success"]) + 1
        valid_vec = valid.ravel()[pix]
        unw_vec = unw.ravel()[pix]
        coh_vec = c.ravel()[pix]
        phases[k, valid_vec] = unw_vec[valid_vec]
        cohs[k, valid_vec] = coh_vec[valid_vec]

    dem, _rate, pixel_rmse, n_valid, bperp_span = solve_pixels(
        phases,
        cohs,
        design_matrix,
        bperp,
        variant.min_pairs,
    )
    any_obs = n_valid > 0
    valid_pairs_pass = n_valid >= variant.min_pairs
    bperp_pass = valid_pairs_pass & np.isfinite(bperp_span) & (bperp_span >= variant.min_bperp_span_m)
    rmse_pass = bperp_pass & np.isfinite(pixel_rmse) & (pixel_rmse <= variant.max_pixel_rmse_rad)
    final = rmse_pass & np.isfinite(dem)
    row["pixels_with_any_unwrapped_obs"] = int(np.sum(any_obs))
    row["valid_pairs_pass_pixels"] = int(np.sum(valid_pairs_pass))
    row["bperp_pass_pixels"] = int(np.sum(bperp_pass))
    row["rmse_pass_pixels"] = int(np.sum(rmse_pass))
    row["final_valid_pixels"] = int(np.sum(final))
    row["reaches_min_pixels"] = bool(np.sum(final) >= min_pixels)
    if np.any(any_obs):
        row["median_valid_pairs_all_da_pixels"] = float(np.median(n_valid[any_obs]))
    if np.any(valid_pairs_pass & np.isfinite(bperp_span)):
        row["median_bperp_span_valid_pairs"] = float(np.median(bperp_span[valid_pairs_pass & np.isfinite(bperp_span)]))
    if np.any(bperp_pass & np.isfinite(pixel_rmse)):
        row["median_rmse_bperp_pass"] = float(np.median(pixel_rmse[bperp_pass & np.isfinite(pixel_rmse)]))
    return row


def main() -> None:
    rows = 630
    cols = 900
    min_pixels = 20
    docs = BASE / "docs"
    no = pd.read_csv(docs / "no_solution_failure_audit_20260707.csv")
    islands = pd.read_csv(BASE / "work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv")
    label = np.load(BASE / "work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy")
    amp = np.load(BASE / "work/mli/amplitude_dispersion_crop_bmp.npy").astype(np.float32)
    pairs = pd.read_csv(BASE / "work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    par = parse_gamma_par(BASE / "data/tongji_rslc/20200708.rslc.par")
    wavelength = 299792458.0 / float(par["radar_frequency"])
    design_matrix = design(
        pairs["bperp_m"].to_numpy(dtype=np.float64),
        pairs["dt_days"].to_numpy(dtype=np.float64),
        wavelength,
        float(par["center_range_slc"]),
        float(par["incidence_angle"]),
    )
    bperp = pairs["bperp_m"].to_numpy(dtype=np.float64)

    variants = [
        Variant("min_pairs10", "lgr_valid_pairs_lt_min_after_unwrap", min_pairs=10),
        Variant("rmse150", "lgr_rmse_gt_max", max_pixel_rmse_rad=1.50),
        Variant("da045", "amplitude_dispersion_pixels_too_few", max_amplitude_dispersion=0.45),
    ]

    print(f"loading {len(pairs)} interferogram pairs", flush=True)
    phase_stack, coh_stack = load_stacks(pairs, BASE / "work/gamma_sbas/intf_triangular_dsm", rows, cols)

    all_rows: list[dict[str, float | int | str]] = []
    for variant in variants:
        target_ids = set(
            no.loc[no["refined_failure_class"].eq(variant.target_class), "clean_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
        target_islands = islands[islands["primary_uid"].astype(int).isin(target_ids)].copy()
        target_islands = target_islands.sort_values(["primary_uid", "island_id"]).reset_index(drop=True)
        print(f"variant {variant.name}: {len(target_ids)} buildings, {len(target_islands)} islands", flush=True)
        for idx, info in target_islands.iterrows():
            if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(target_islands):
                print(f"  {variant.name} island {idx + 1}/{len(target_islands)}", flush=True)
            all_rows.append(
                diagnose_island(
                    info,
                    label,
                    amp,
                    phase_stack,
                    coh_stack,
                    pairs,
                    design_matrix,
                    bperp,
                    wavelength,
                    variant,
                    min_pixels,
                )
            )

    island_diag = pd.DataFrame(all_rows)
    island_out = docs / "no_solution_gate_sensitivity_islands_20260707.csv"
    island_diag.to_csv(island_out, index=False)
    building = (
        island_diag.groupby(["variant", "target_class", "clean_id"])
        .agg(
            island_count=("island_id", "count"),
            max_raw_mask_pixels=("raw_mask_pixels", "max"),
            max_da_pass_pixels=("da_pass_pixels", "max"),
            max_valid_pairs_pass_pixels=("valid_pairs_pass_pixels", "max"),
            max_bperp_pass_pixels=("bperp_pass_pixels", "max"),
            max_rmse_pass_pixels=("rmse_pass_pixels", "max"),
            max_final_valid_pixels=("final_valid_pixels", "max"),
            reaches_min_pixels=("reaches_min_pixels", "max"),
            median_valid_pairs_all_da_pixels=("median_valid_pairs_all_da_pixels", "median"),
            median_bperp_span_valid_pairs=("median_bperp_span_valid_pairs", "median"),
            median_rmse_bperp_pass=("median_rmse_bperp_pass", "median"),
            source_islands=("island_id", lambda s: ",".join(str(int(v)) for v in sorted(s))),
        )
        .reset_index()
    )
    building_out = docs / "no_solution_gate_sensitivity_buildings_20260707.csv"
    building.to_csv(building_out, index=False)

    summary = {
        "date": "2026-07-07",
        "method": "Gate-level sensitivity pretest. Counts buildings that can reach at least 20 retained LGR pixels under one small gate change; this is not a rebuilt height product.",
        "height_field_use": "not_read_for_fitting_filtering_calibration_selection_or_qc",
        "variant_counts": {},
        "outputs": {
            "building_csv": str(building_out.relative_to(BASE)),
            "island_csv": str(island_out.relative_to(BASE)),
            "summary_json": "docs/no_solution_gate_sensitivity_20260707_summary.json",
        },
    }
    for variant, grp in building.groupby("variant"):
        summary["variant_counts"][str(variant)] = {
            "target_buildings": int(len(grp)),
            "reaches_min_pixels": int(grp["reaches_min_pixels"].sum()),
            "median_final_valid_pixels": float(grp["max_final_valid_pixels"].median()),
            "max_final_valid_pixels": int(grp["max_final_valid_pixels"].max()),
        }
    summary_out = docs / "no_solution_gate_sensitivity_20260707_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
