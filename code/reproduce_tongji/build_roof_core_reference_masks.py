#!/usr/bin/env python3
"""Build adaptive roof-core and stable-ground reference masks for roof SBAS.

The building ``height`` attribute is deliberately ignored.  Roof geometry comes
from an already projected GAMMA product; reference pixels are selected only from
non-building SAR pixels using DSM ground elevation, amplitude dispersion, and
interferometric coherence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.draw import polygon as draw_polygon

from extract_gamma_differential_island_observations import read_float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection", default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson")
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--intf-root", default="work/gamma_sbas/intf_triangular_dsm")
    p.add_argument("--reference-height-rdc", default="work/gamma_sbas/dem/20200708_dsm_rdc.hgt")
    p.add_argument("--amplitude-dispersion", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    p.add_argument("--rows", type=int, default=630)
    p.add_argument("--cols", type=int, default=900)
    p.add_argument("--ground-height-m", type=float, default=4.0)
    p.add_argument("--ground-tolerance-m", type=float, default=1.0)
    p.add_argument("--roof-erosion-pixels", type=int, default=1)
    p.add_argument("--minimum-core-pixels", type=int, default=8)
    p.add_argument("--minimum-core-fraction", type=float, default=0.35)
    p.add_argument("--minimum-island-pixels", type=int, default=8)
    p.add_argument("--minimum-reliable-pixels-per-island", type=int, default=8)
    p.add_argument("--minimum-reliable-fraction-per-island", type=float, default=0.15)
    p.add_argument("--roof-max-da", type=float, default=0.40)
    p.add_argument("--roof-pair-coherence-threshold", type=float, default=0.55)
    p.add_argument("--roof-min-coherence", type=float, default=0.65)
    p.add_argument("--roof-min-pairs", type=int, default=12)
    p.add_argument("--reference-buffer-pixels", type=int, default=4)
    p.add_argument("--reference-max-da", type=float, default=0.40)
    p.add_argument("--reference-min-coherence", type=float, default=0.75)
    p.add_argument("--reference-min-pairs", type=int, default=12)
    p.add_argument("--out-dir", default="work/roof_sbas_optimized")
    return p.parse_args()


def rasterize_geometry(geom, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for part in parts:
        if part.is_empty:
            continue
        xy = np.asarray(part.exterior.coords)
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=shape)
        mask[rr, cc] = True
        for ring in part.interiors:
            hole = np.asarray(ring.coords)
            rr, cc = draw_polygon(hole[:, 1], hole[:, 0], shape=shape)
            mask[rr, cc] = False
    return mask


def main() -> None:
    args = parse_args()
    shape = (args.rows, args.cols)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projection = gpd.read_file(args.projection)
    roofs = projection[projection["surface"] == "roof"].copy()
    supports = projection[projection["surface"] == "layover"].copy()
    roof_only_projection = supports.empty
    support_geometries = roofs.geometry if roof_only_projection else supports.geometry
    owner = np.zeros(shape, dtype=np.int32)
    conflicts = np.zeros(shape, dtype=bool)
    full_roof_union = np.zeros(shape, dtype=bool)
    support_coverage = np.zeros(shape, dtype=np.uint16)
    audit_rows: list[dict[str, object]] = []

    # A roof pixel is independent only when it belongs to one projected
    # building-body support. This removes roof/layover contamination from
    # neighbouring buildings, not merely direct roof/roof overlap.
    for geom in support_geometries:
        support_coverage += rasterize_geometry(geom, shape).astype(np.uint16)

    for row in roofs.sort_values("clean_id").itertuples(index=False):
        clean_id = int(row.clean_id)
        full = rasterize_geometry(row.geometry, shape)
        full_count = int(full.sum())
        if full_count == 0:
            audit_rows.append({"clean_id": clean_id, "roof_pixels": 0, "core_pixels": 0, "core_mode": "outside_scene"})
            continue
        eroded = ndimage.binary_erosion(full, iterations=max(0, args.roof_erosion_pixels))
        required = max(args.minimum_core_pixels, int(np.ceil(args.minimum_core_fraction * full_count)))
        if int(eroded.sum()) >= required:
            core = eroded
            mode = "eroded_core"
        else:
            core = np.zeros_like(full)
            mode = "rejected_no_reliable_eroded_core"
        overlap = core & (owner > 0) & (owner != clean_id)
        conflicts |= overlap
        owner[core & (owner == 0)] = clean_id
        full_roof_union |= full
        audit_rows.append(
            {
                "clean_id": clean_id,
                "roof_pixels": full_count,
                "core_pixels_before_conflict_removal": int(core.sum()),
                "overlap_pixels": int(overlap.sum()),
                "core_mode": mode,
            }
        )
    owner[conflicts] = 0
    cross_building_support_conflicts = (owner > 0) & (support_coverage > 1)
    owner[cross_building_support_conflicts] = 0

    building_support = support_coverage > 0
    excluded = ndimage.binary_dilation(building_support | full_roof_union, iterations=args.reference_buffer_pixels)
    dsm = read_float(Path(args.reference_height_rdc), args.rows, args.cols)
    da = np.load(args.amplitude_dispersion).astype(np.float32)
    pairs = pd.read_csv(args.pairs_csv)
    coh_count = np.zeros(shape, dtype=np.int16)
    coh_valid_count = np.zeros(shape, dtype=np.int16)
    coh_sum = np.zeros(shape, dtype=np.float32)
    roof_coh_count = np.zeros(shape, dtype=np.int16)
    for pair_row in pairs.itertuples(index=False):
        pair = f"{pair_row.master}_{pair_row.slave}"
        coh = read_float(Path(args.intf_root) / pair / f"{pair}.cc", args.rows, args.cols)
        finite = np.isfinite(coh)
        reference_valid = finite & (coh >= args.reference_min_coherence)
        coh_count += reference_valid.astype(np.int16)
        coh_valid_count += finite.astype(np.int16)
        coh_sum += np.where(finite, coh, 0.0).astype(np.float32)
        roof_valid = finite & (coh >= args.roof_pair_coherence_threshold)
        roof_coh_count += roof_valid.astype(np.int16)
    coh_mean = np.divide(
        coh_sum,
        coh_valid_count,
        out=np.full(shape, np.nan, dtype=np.float32),
        where=coh_valid_count > 0,
    )
    roof_coh_mean = coh_mean
    roof_quality = (
        np.isfinite(da)
        & (da <= args.roof_max_da)
        & np.isfinite(roof_coh_mean)
        & (roof_coh_mean >= args.roof_min_coherence)
        & (roof_coh_count >= args.roof_min_pairs)
    )
    roof_core_pixels_before_reliability_gate = int((owner > 0).sum())

    structure = np.ones((3, 3), dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.int32)
    reliable_roof_points = np.zeros(shape, dtype=bool)
    island_rows: list[dict[str, int | float]] = []
    next_island = 1
    small_component_pixels_removed = 0
    reliability_rejected_component_pixels = 0
    for clean_id in sorted(int(v) for v in np.unique(owner) if int(v) > 0):
        components, count = ndimage.label(owner == clean_id, structure=structure)
        for component in range(1, count + 1):
            rr, cc = np.nonzero(components == component)
            if rr.size < args.minimum_island_pixels:
                small_component_pixels_removed += int(rr.size)
                owner[rr, cc] = 0
                continue
            reliable = roof_quality[rr, cc]
            reliable_count = int(reliable.sum())
            reliable_fraction = reliable_count / int(rr.size)
            if (
                reliable_count < args.minimum_reliable_pixels_per_island
                or reliable_fraction < args.minimum_reliable_fraction_per_island
            ):
                reliability_rejected_component_pixels += int(rr.size)
                owner[rr, cc] = 0
                continue
            labels[rr, cc] = next_island
            reliable_roof_points[rr[reliable], cc[reliable]] = True
            island_rows.append(
                {
                    "island_id": next_island,
                    "primary_uid": clean_id,
                    "uid_count": 1,
                    "pixel_count": int(rr.size),
                    "reliable_pixel_count": reliable_count,
                    "reliable_pixel_fraction": reliable_fraction,
                    "mean_coherence": float(np.nanmean(roof_coh_mean[rr, cc])),
                    "median_amplitude_dispersion": float(np.nanmedian(da[rr, cc])),
                    "row_min": int(rr.min()),
                    "row_max": int(rr.max()),
                    "col_min": int(cc.min()),
                    "col_max": int(cc.max()),
                }
            )
            next_island += 1

    final_counts = pd.Series(owner[owner > 0]).value_counts().to_dict()
    for row in audit_rows:
        row["final_reliable_island_pixels"] = int(final_counts.get(int(row["clean_id"]), 0))
    reference = (
        ~excluded
        & np.isfinite(dsm)
        & (np.abs(dsm - args.ground_height_m) <= args.ground_tolerance_m)
        & np.isfinite(da)
        & (da <= args.reference_max_da)
        & (coh_count >= args.reference_min_pairs)
        & (coh_mean >= args.reference_min_coherence)
    )

    np.save(out_dir / "roof_core_clean_id_mask.npy", owner)
    np.save(out_dir / "roof_core_island_label.npy", labels)
    np.save(out_dir / "full_corrected_roof_union.npy", full_roof_union)
    np.save(out_dir / "roof_quality_mask.npy", roof_quality)
    np.save(out_dir / "reliable_roof_points_mask.npy", reliable_roof_points)
    np.save(out_dir / "cross_building_support_conflict_mask.npy", cross_building_support_conflicts)
    np.save(out_dir / "stable_ground_reference_mask.npy", reference)
    pd.DataFrame(island_rows).to_csv(out_dir / "roof_core_islands.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "roof_core_mask_audit.csv", index=False)
    summary = {
        "method": "SAR-feature-refined GAMMA roof polygons; one-pixel eroded continuous core; remove roof/roof and cross-building layover conflicts; retain each connected island only when it contains enough coherent low-DA evidence pixels",
        "height_attribute_read_or_used": False,
        "projection": args.projection,
        "roof_only_projection": bool(roof_only_projection),
        "projected_roofs": int(len(roofs)),
        "roof_buildings_in_scene": int(np.unique(owner[owner > 0]).size),
        "roof_core_pixels": int((owner > 0).sum()),
        "roof_conflict_pixels_removed": int(conflicts.sum()),
        "cross_building_support_conflict_pixels_removed": int(cross_building_support_conflicts.sum()),
        "roof_core_pixels_before_reliability_gate": roof_core_pixels_before_reliability_gate,
        "reliable_roof_evidence_pixels": int(reliable_roof_points.sum()),
        "reliability_rejected_component_pixels": int(reliability_rejected_component_pixels),
        "small_component_pixels_removed": int(small_component_pixels_removed),
        "roof_islands": int(len(island_rows)),
        "stable_ground_reference_pixels": int(reference.sum()),
        "roof_island_rules": {
            "erosion_pixels": args.roof_erosion_pixels,
            "small_roof_full_boundary_fallback": False,
            "exclude_cross_building_layover_conflicts": not roof_only_projection,
            "roof_only_conflict_basis": "overlapping roof coverage" if roof_only_projection else None,
            "maximum_amplitude_dispersion": args.roof_max_da,
            "pair_coherence_threshold": args.roof_pair_coherence_threshold,
            "minimum_mean_coherence": args.roof_min_coherence,
            "minimum_coherent_pairs": args.roof_min_pairs,
            "minimum_connected_pixels": args.minimum_island_pixels,
            "minimum_reliable_pixels_per_island": args.minimum_reliable_pixels_per_island,
            "minimum_reliable_fraction_per_island": args.minimum_reliable_fraction_per_island,
        },
        "reference_rules": {
            "building_support_buffer_pixels": args.reference_buffer_pixels,
            "ground_height_m": args.ground_height_m,
            "ground_tolerance_m": args.ground_tolerance_m,
            "maximum_amplitude_dispersion": args.reference_max_da,
            "minimum_coherence": args.reference_min_coherence,
            "minimum_pairs": args.reference_min_pairs,
        },
    }
    (out_dir / "mask_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
