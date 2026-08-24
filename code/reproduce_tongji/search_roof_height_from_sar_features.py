#!/usr/bin/env python3
"""Estimate an independent roof-position height interval from SAR features.

The building height attribute is never read.  A GAMMA R-D bottom polygon and
its displacement vector per metre define a 0..Hmax roof sweep.  Reference-scene
amplitude/edge contrast, a frozen multi-temporal validation edge image, and the
paper-quality InSAR mask select a roof position.  The result is an ambiguity
initializer only; final height must still be re-estimated by GAMMA mb_pt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import box

from refine_gamma_projection_with_sar_features import feature_strength, load_temporal_split, sample_image, sampled_boundary
from refine_projection_local_window_search import paper_quality_mask, raster_points, sar_display, score_components


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rd-projection", required=True)
    p.add_argument("--amplitude", default="work/mli/20200708_rslc_amplitude.npy")
    p.add_argument("--bmp-dir", default="data/tongji_rslc")
    p.add_argument("--pairs-csv", required=True)
    p.add_argument("--interferogram-dir", required=True)
    p.add_argument("--amplitude-dispersion", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    p.add_argument("--height-min", type=float, default=2.0)
    p.add_argument("--height-max", type=float, default=120.0)
    p.add_argument("--height-step", type=float, default=2.0)
    p.add_argument("--minimum-quality-points", type=int, default=4)
    p.add_argument("--minimum-quality-fraction", type=float, default=0.05)
    p.add_argument("--maximum-height-split-difference", type=float, default=8.0)
    p.add_argument("--maximum-shift-split-difference", type=int, default=2)
    p.add_argument("--minimum-height-margin", type=float, default=0.004)
    p.add_argument("--limit-buildings", type=int, default=0, help="0 searches all buildings")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()

    amplitude = np.load(args.amplitude).astype(np.float32)
    shape = amplitude.shape
    display = sar_display(amplitude)
    reference_feature = feature_strength(amplitude)
    _train, validation_amp, _tn, validation_names = load_temporal_split(Path(args.bmp_dir), shape)
    validation_feature = feature_strength(validation_amp)
    args.paper_quality_gate = True
    quality = paper_quality_mask(args, shape)
    if quality is None:
        raise RuntimeError("paper quality mask was not built")

    rd = gpd.read_file(args.rd_projection).set_crs(None, allow_override=True)
    footprint = box(0, 0, shape[1] - 1, shape[0] - 1)
    heights = np.arange(args.height_min, args.height_max + 0.5 * args.height_step, args.height_step)
    rows: list[dict[str, object]] = []
    features: list[dict[str, object]] = []

    for ordinal, (clean_id, group) in enumerate(rd.groupby("clean_id"), start=1):
        if args.limit_buildings and ordinal > args.limit_buildings:
            break
        bottom_rows = group[group["surface"].eq("bottom")]
        roof_rows = group[group["surface"].eq("roof")]
        if bottom_rows.empty or roof_rows.empty:
            continue
        bottom = bottom_rows.geometry.iloc[0]
        initial_roof = roof_rows.geometry.iloc[0]
        calibration_height = float(group["height_prior_m"].iloc[0])
        if bottom.is_empty or initial_roof.is_empty or calibration_height <= 0 or not bottom.buffer(100).intersects(footprint):
            continue
        dx = float((initial_roof.centroid.x - bottom.centroid.x) / calibration_height)
        dy = float((initial_roof.centroid.y - bottom.centroid.y) / calibration_height)
        boundary = sampled_boundary(bottom, spacing=1.0)
        interior = raster_points(bottom, shape, maximum=300)
        if len(boundary) < 8 or len(interior) < 4:
            continue
        minx, miny, maxx, maxy = bottom.bounds
        shift_limit = int(np.clip(math.ceil(1.5 + 0.045 * max(maxx - minx, maxy - miny)), 2, 5))
        candidates: list[dict[str, float | int]] = []
        for height in heights:
            geometric = np.asarray([dx * height, dy * height], dtype=float)
            for dr in range(-shift_limit, shift_limit + 1):
                for dc in range(-shift_limit, shift_limit + 1):
                    total = geometric + np.asarray([dc, dr], dtype=float)
                    primary, edge, contrast = score_components(
                        reference_feature, display, boundary, interior, int(round(total[1])), int(round(total[0]))
                    )
                    validation = float(np.mean(sample_image(validation_feature, boundary + total)))
                    sample_xy = interior + total
                    cc = np.rint(sample_xy[:, 0]).astype(int)
                    rr = np.rint(sample_xy[:, 1]).astype(int)
                    valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
                    quality_count = int(quality[rr[valid], cc[valid]].sum()) if np.any(valid) else 0
                    quality_fraction = quality_count / max(int(valid.sum()), 1)
                    objective = (
                        0.52 * primary
                        + 0.28 * validation
                        + 0.20 * quality_fraction
                        - 0.00035 * (dr * dr + dc * dc)
                    )
                    candidates.append({
                        "height": float(height), "dr": dr, "dc": dc,
                        "primary": primary, "edge": edge, "contrast": contrast,
                        "validation": validation, "quality_count": quality_count,
                        "quality_fraction": quality_fraction, "objective": objective,
                    })
        best = max(candidates, key=lambda item: float(item["objective"]))
        ref_best = max(candidates, key=lambda item: float(item["primary"]) + 0.15 * float(item["quality_fraction"]))
        val_best = max(candidates, key=lambda item: float(item["validation"]) + 0.15 * float(item["quality_fraction"]))
        distant = [item for item in candidates if abs(float(item["height"]) - float(best["height"])) >= 10.0]
        margin = float(best["objective"] - max(float(item["objective"]) for item in distant)) if distant else 0.0
        height_split = abs(float(ref_best["height"]) - float(val_best["height"]))
        shift_split = max(abs(int(ref_best["dr"]) - int(val_best["dr"])), abs(int(ref_best["dc"]) - int(val_best["dc"])))
        boundary_hit = bool(float(best["height"]) in (float(heights[0]), float(heights[-1])))
        accepted = bool(
            int(best["quality_count"]) >= args.minimum_quality_points
            and float(best["quality_fraction"]) >= args.minimum_quality_fraction
            and height_split <= args.maximum_height_split_difference
            and shift_split <= args.maximum_shift_split_difference
            and margin >= args.minimum_height_margin
            and not boundary_hit
        )
        selected_height = float(best["height"])
        roof = affinity.translate(bottom, xoff=dx * selected_height + int(best["dc"]), yoff=dy * selected_height + int(best["dr"]))
        row = {
            "clean_id": int(clean_id), "accepted": accepted,
            "sar_geometry_height_m": selected_height,
            "local_row_shift": int(best["dr"]), "local_col_shift": int(best["dc"]),
            "height_split_difference_m": float(height_split), "shift_split_difference_px": int(shift_split),
            "height_objective_margin": margin, "paper_quality_points": int(best["quality_count"]),
            "paper_quality_fraction": float(best["quality_fraction"]), "objective": float(best["objective"]),
            "reference_score": float(best["primary"]), "validation_score": float(best["validation"]),
            "rd_range_displacement_per_m": dx, "rd_azimuth_displacement_per_m": dy,
            "height_attribute_read_for_search": False,
        }
        rows.append(row)
        if accepted:
            props = bottom_rows.iloc[0].drop(labels="geometry").to_dict()
            props.update(row)
            props["surface"] = "roof"
            props["height_prior_m"] = selected_height
            props["height_role"] = "SAR_geometry_ambiguity_initialization_only_not_final_or_fill"
            features.append({**props, "geometry": roof})
        if ordinal % 100 == 0:
            print(f"SAR屋顶高度搜索：{ordinal}/{rd.clean_id.nunique()}", flush=True)

    metrics = pd.DataFrame(rows)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    output = (
        gpd.GeoDataFrame(features, geometry="geometry", crs=None)
        if features
        else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=None)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_file(args.output, driver="GeoJSON")
    accepted = metrics[metrics["accepted"]]
    payload = {
        "method": "prior-independent GAMMA R-D height sweep scored by reference amplitude, frozen temporal validation, and paper-quality InSAR evidence",
        "height_attribute_used": False,
        "height_search_m": [args.height_min, args.height_max, args.height_step],
        "searched_buildings": int(len(metrics)), "accepted_buildings": int(len(accepted)),
        "accepted_height_median_m": float(accepted["sar_geometry_height_m"].median()) if len(accepted) else None,
        "accepted_height_p05_p95_m": [float(accepted["sar_geometry_height_m"].quantile(.05)), float(accepted["sar_geometry_height_m"].quantile(.95))] if len(accepted) else None,
        "validation_images": validation_names, "output": args.output, "metrics": args.metrics,
        "final_height_role": "none; output is only an independent position/cycle initializer for subsequent GAMMA mb_pt",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
