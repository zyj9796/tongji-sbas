#!/usr/bin/env python3
"""Dense ZJC building-island unwrap followed by GAMMA ``mb_pt`` SBAS.

This reproduces the original-code logic that building height is the internal
elevation span of a complete projected layover island.  Dense island pixels are
used for local unwrapping; DA/coherence screening is applied only after GAMMA
SBAS.  The vector height is unavailable to inversion and is merged afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from skimage.draw import polygon as draw_polygon

from benchmark_paper_unwrap import paper_like_unwrap


ROWS, COLS = 630, 900


def env() -> dict[str, str]:
    result = os.environ.copy()
    result["PATH"] = "/usr/local/GAMMA/IPTA/bin:/usr/local/GAMMA/IPTA/scripts:" + result.get("PATH", "")
    compat = Path("/tmp/gamma_gdal_compat")
    compat.mkdir(parents=True, exist_ok=True)
    gdal_compat = compat / "libgdal.so.26"
    if not gdal_compat.exists():
        gdal_compat.symlink_to("/lib/libgdal.so.30")
    result["LD_LIBRARY_PATH"] = str(compat)
    return result


def rasterize(geom) -> np.ndarray:
    out = np.zeros((ROWS, COLS), dtype=bool)
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for part in parts:
        xy = np.asarray(part.exterior.coords)
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=out.shape)
        out[rr, cc] = True
    return out


def run(cmd: list[str], log: Path) -> None:
    proc = subprocess.run(cmd, env=env(), text=True, capture_output=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n" + proc.stdout + proc.stderr + "\n")
    if proc.returncode:
        raise RuntimeError(proc.stdout + proc.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-id", type=int, action="append", required=True)
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--source-dir", default="work/gamma_native_ipta_sbas")
    p.add_argument("--projection", default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    p.add_argument("--da", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    p.add_argument("--min-unwrap-coherence", type=float, default=0.20)
    p.add_argument("--min-final-coherence", type=float, default=0.75)
    p.add_argument("--max-da", type=float, default=0.40)
    p.add_argument("--gamma", type=float, default=100.0)
    p.add_argument("--max-dt-days", type=float, default=None)
    p.add_argument("--min-abs-bperp", type=float, default=None)
    p.add_argument("--max-abs-bperp", type=float, default=None)
    p.add_argument(
        "--original-rectangle-unwrap", action="store_true",
        help="Match the MATLAB implementation: include zeroed pixels outside the island but inside its bounding box",
    )
    p.add_argument("--out-dir", default="work/zjc_dense_island_gamma_sbas_pilot")
    p.add_argument("--result", default="results/tables/zjc_dense_island_gamma_sbas_pilot.csv")
    p.add_argument("--summary", default="results/metadata/zjc_dense_island_gamma_sbas_pilot_summary.json")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "gamma.log"
    log.write_text("", encoding="utf-8")
    pairs = pd.read_csv(args.pairs_csv, dtype={"master": str, "slave": str}).drop_duplicates(["master", "slave"])
    if args.max_dt_days is not None:
        pairs = pairs[pairs["dt_days"] <= args.max_dt_days]
    if args.min_abs_bperp is not None:
        pairs = pairs[pairs["bperp_m"].abs() >= args.min_abs_bperp]
    if args.max_abs_bperp is not None:
        pairs = pairs[pairs["bperp_m"].abs() <= args.max_abs_bperp]
    pairs = pairs.reset_index(drop=True)
    network_dates = sorted(set(pairs["master"]) | set(pairs["slave"]))
    date_to_record = {date: i + 1 for i, date in enumerate(network_dates)}
    network_slc_par_tab = out / "network_slc_par.tab"
    network_slc_par_tab.write_text(
        "\n".join(f"data/tongji_rslc/{date}.rslc.par" for date in network_dates) + "\n",
        encoding="utf-8",
    )
    network_pslc_par = out / "network.pSLC_par"
    if network_pslc_par.exists():
        network_pslc_par.unlink()
    run(["SLC_par2pt", str(network_slc_par_tab), str(network_pslc_par), "-"], log)
    projection = gpd.read_file(args.projection)
    buildings = gpd.read_file(args.buildings)
    da = np.load(args.da)
    frequency = None
    center_range = None
    incidence_deg = None
    for line in Path(args.reference_par).read_text(errors="replace").splitlines():
        if line.startswith("radar_frequency:"):
            frequency = float(line.split()[1])
        elif line.startswith("center_range_slc:"):
            center_range = float(line.split()[1])
        elif line.startswith("incidence_angle:"):
            incidence_deg = float(line.split()[1])
    wavelength = 299792458.0 / float(frequency)
    direct_design = np.column_stack([
        4.0 * np.pi * pairs["bperp_m"].to_numpy(float)
        / (wavelength * float(center_range) * np.sin(np.deg2rad(float(incidence_deg)))),
        4.0 * np.pi * pairs["dt_days"].to_numpy(float) / wavelength,
    ])
    records = []

    for clean_id in args.clean_id:
        subset = projection[(projection.clean_id == clean_id) & projection.surface.eq("layover")]
        if subset.empty:
            records.append({"clean_id": clean_id, "status": "missing_projection"})
            continue
        mask = rasterize(subset.geometry.iloc[0])
        yy, xx = np.nonzero(mask)
        if len(xx) < 20:
            records.append({"clean_id": clean_id, "status": "too_small"})
            continue
        r0, r1, c0, c1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        patch_mask = mask[r0:r1, c0:c1]
        local_y, local_x = np.nonzero(patch_mask)
        phases, coherences, statuses = [], [], []
        for pair in pairs.itertuples(index=False):
            name = f"{pair.master}_{pair.slave}"
            root = Path(args.source_dir) / "interferograms" / name
            diff = np.fromfile(root / f"{name}.adf.diff", dtype=">c8").reshape(ROWS, COLS)[r0:r1, c0:c1]
            coh = np.fromfile(root / f"{name}.cc", dtype=">f4").reshape(ROWS, COLS)[r0:r1, c0:c1]
            phase_patch = np.angle(diff)
            if args.original_rectangle_unwrap:
                phase_patch = phase_patch.copy()
                phase_patch[~patch_mask] = 0.0
                valid = np.ones_like(patch_mask, dtype=bool)
            else:
                valid = patch_mask & np.isfinite(coh) & (coh >= args.min_unwrap_coherence)
            unw, info = paper_like_unwrap(
                phase_patch,
                valid,
                wavelength,
                coherence_patch=coh,
                amplitude_dispersion_patch=da[r0:r1, c0:c1],
            )
            phases.append(unw[local_y, local_x])
            coherences.append(coh[local_y, local_x])
            statuses.append(str(info.get("status", "unknown")))
        phase = np.stack(phases)
        coherence = np.stack(coherences)
        common = np.all(np.isfinite(phase), axis=0)
        if int(common.sum()) < 10:
            records.append({"clean_id": clean_id, "status": "too_few_common_unwrapped", "common_points": int(common.sum())})
            continue
        phase = phase[:, common]
        coherence = coherence[:, common]
        gx = (local_x[common] + c0).astype(np.int32)
        gy = (local_y[common] + r0).astype(np.int32)
        # Fix the arbitrary island phase constant to one common point per pair,
        # while retaining its observed wrapped phase.  GAMMA uses exact 0.0 as
        # NULL, so setting the reference observation to zero invalidates it.
        reference_wrapped = []
        for pair in pairs.itertuples(index=False):
            name = f"{pair.master}_{pair.slave}"
            diff = np.fromfile(
                Path(args.source_dir) / "interferograms" / name / f"{name}.adf.diff", dtype=">c8"
            ).reshape(ROWS, COLS)
            reference_wrapped.append(float(np.angle(diff[gy[0], gx[0]])))
        phase = phase - phase[:, [0]] + np.asarray(reference_wrapped)[:, None]
        direct_coef = np.linalg.pinv(direct_design) @ phase
        direct_height = -direct_coef[0]
        with tempfile.TemporaryDirectory(prefix=f"dense_{clean_id}_", dir=out) as td:
            td = Path(td)
            plist = td / "plist"
            np.column_stack([gx, gy]).astype(">i4").tofile(plist)
            pmask = td / "pmask"
            np.ones(len(gx), dtype=np.uint8).tofile(pmask)
            punw = td / "punw"
            phase.astype(">f4").tofile(punw)
            local_itab = td / "pairs.itab"
            local_itab.write_text(
                "\n".join(
                    f"{date_to_record[row.master]} {date_to_record[row.slave]} {i + 1} 1"
                    for i, row in enumerate(pairs.itertuples(index=False))
                ) + "\n",
                encoding="utf-8",
            )
            itab_ts, pdiff_ts, pdiff_sim, psigma_ts = (td / n for n in ["itab_ts", "pdiff_ts", "pdiff_sim", "psigma_ts"])
            phgt, prate, pconst, psigma = (td / n for n in ["phgt", "prate", "pconst", "psigma"])
            run([
                "mb_pt", str(plist), str(pmask), str(network_pslc_par),
                str(local_itab), str(punw), "0", "-", str(itab_ts),
                str(pdiff_ts), str(pdiff_sim), str(psigma_ts), "1", str(phgt), str(args.gamma),
                str(prate), str(pconst), str(psigma), str(Path(args.reference_par)),
            ], log)
            height = np.fromfile(phgt, dtype=">f4")[:len(gx)].astype(float)
            sigma = np.fromfile(psigma, dtype=">f4")[:len(gx)].astype(float)
        mean_coh = np.nanmean(coherence, axis=0)
        final = (
            np.isfinite(height) & np.isfinite(sigma) & (sigma > 0) & (sigma <= 1.2)
            & (mean_coh >= args.min_final_coherence) & (da[gy, gx] <= args.max_da)
        )
        vals = height[final]
        direct_vals = direct_height[final]
        if len(vals) < 10:
            records.append({"clean_id": clean_id, "status": "too_few_final_points", "final_points": int(len(vals))})
            continue
        p01, p05, p95, p99 = np.nanpercentile(vals, [1, 5, 95, 99])
        records.append({
            "clean_id": clean_id,
            "status": "solved",
            "island_pixels": int(mask.sum()),
            "common_points": int(common.sum()),
            "final_points": int(final.sum()),
            "height_span_maxmin_m": float(np.nanmax(vals) - np.nanmin(vals)),
            "height_span_p01_p99_m": float(p99 - p01),
            "height_span_p05_p95_m": float(p95 - p05),
            "original_lgr_direct_span_maxmin_m": float(np.nanmax(direct_vals) - np.nanmin(direct_vals)),
            "original_lgr_direct_span_p05_p95_m": float(
                np.nanpercentile(direct_vals, 95) - np.nanpercentile(direct_vals, 5)
            ),
            "median_phase_sigma_rad": float(np.nanmedian(sigma[final])),
            "unwrap_ok_pairs": int(sum(s == "ok" for s in statuses)),
        })

    result = pd.DataFrame(records)
    prior = buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"})
    result = result.merge(prior, on="clean_id", how="left")
    result["difference_to_prior_m"] = result.get("height_span_p05_p95_m", np.nan) - result["prior_height_m"]
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_path, index=False)
    summary = {
        "method": "dense full layover island ZJC local robust unwrap -> GAMMA mb_pt -> post-SBAS DA/coherence -> robust internal height span",
        "prior_policy": "projection geometry only before inversion; vector height merged after solve for comparison; never fitted or filled",
        "pairs": len(pairs),
        "results": records,
        "result_csv": str(result_path),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
