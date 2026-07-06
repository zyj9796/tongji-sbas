#!/usr/bin/env python3
"""Build refined SAR building masks from roof and facade projections.

The input projection is expected to contain matched `bottom` and `roof`
polygons in SAR image coordinates. For each building, this script constructs
roof and facade polygons, rasterizes them, refines the candidate mask using
local SAR amplitude, and resolves overlapping pixels by near-range LOS order.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Polygon as MplPolygon
from scipy import ndimage
from skimage.draw import polygon as draw_polygon
from skimage.morphology import disk


def stretch(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [2, 98])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def xy_from_feature(feat: dict[str, Any]) -> np.ndarray | None:
    coords = feat.get("geometry", {}).get("coordinates", [])
    if not coords:
        return None
    xy = np.asarray(coords[0], dtype=np.float64)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
        return None
    return xy


def load_roof_bottom(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_fid: dict[int, dict[str, Any]] = defaultdict(dict)
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        surface = str(props.get("surface", ""))
        if surface not in {"bottom", "roof"}:
            continue
        fid = int(float(props.get("fid", props.get("source_fid", props.get("uid", -1)))))
        if fid < 0:
            continue
        xy = xy_from_feature(feat)
        if xy is None:
            continue
        by_fid[fid][surface] = xy
        by_fid[fid].setdefault("props", props)
    return {fid: item for fid, item in by_fid.items() if "bottom" in item and "roof" in item}


def ring_match(bottom: np.ndarray, roof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(bottom), len(roof))
    return bottom[:n], roof[:n]


def face_polygons(bottom: np.ndarray, roof: np.ndarray) -> list[tuple[str, np.ndarray]]:
    bottom, roof = ring_match(bottom, roof)
    faces: list[tuple[str, np.ndarray]] = [("roof", roof)]
    n = len(bottom)
    for i in range(n):
        j = (i + 1) % n
        face = np.vstack([bottom[i], bottom[j], roof[j], roof[i]])
        faces.append(("facade", face))
    # The layover envelope is useful for diagnostics and for keeping side-wall
    # scattering between the ground and roof projections.
    faces.append(("layover_envelope", np.vstack([bottom, roof[::-1]])))
    return faces


def rasterize_polygon(xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=shape)
    if rr.size:
        mask[rr, cc] = True
    return mask


def local_refine(mask0: np.ndarray, amp: np.ndarray, buffer_px: int, min_pixels: int) -> tuple[np.ndarray, dict[str, float | int]]:
    if int(mask0.sum()) == 0:
        return mask0.copy(), {"threshold": np.nan, "candidate_pixels": 0, "refined_pixels": 0}
    selem = disk(max(1, buffer_px))
    candidate = ndimage.binary_dilation(mask0, structure=selem)
    vals_candidate = amp[candidate & np.isfinite(amp)]
    vals_mask0 = amp[mask0 & np.isfinite(amp)]
    if vals_candidate.size == 0 or vals_mask0.size == 0:
        return mask0.copy(), {"threshold": np.nan, "candidate_pixels": int(candidate.sum()), "refined_pixels": int(mask0.sum())}
    threshold = max(float(np.percentile(vals_candidate, 70)), float(np.percentile(vals_mask0, 55)))
    refined = candidate & (amp >= threshold)
    refined |= mask0 & (amp >= float(np.percentile(vals_mask0, 50)))
    refined = ndimage.binary_closing(refined, structure=np.ones((3, 3), dtype=bool))
    refined &= candidate
    labeled, nlab = ndimage.label(refined, structure=np.ones((3, 3), dtype=np.int8))
    if nlab > 1:
        keep_labels = []
        for lab in range(1, nlab + 1):
            comp = labeled == lab
            if np.any(comp & mask0) and int(comp.sum()) >= max(3, min_pixels // 6):
                keep_labels.append(lab)
        if keep_labels:
            refined = np.isin(labeled, keep_labels)
    if int(refined.sum()) < min_pixels:
        fallback = mask0 & (amp >= float(np.percentile(vals_mask0, 40)))
        refined = fallback if int(fallback.sum()) >= max(3, min_pixels // 4) else mask0.copy()
    return refined.astype(bool), {
        "threshold": float(threshold),
        "candidate_pixels": int(candidate.sum()),
        "refined_pixels": int(refined.sum()),
    }


def assign_los(owner: np.ndarray, priority: np.ndarray, fid: int, mask: np.ndarray, near_col: float) -> int:
    update = mask & ((owner == 0) | (near_col < priority))
    changed = int(update.sum())
    owner[update] = fid
    priority[update] = near_col
    return changed


def extract_value_islands(fid_mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[dict[str, int]]]:
    island = np.zeros(fid_mask.shape, dtype=np.int32)
    rows_out: list[dict[str, int]] = []
    next_id = 1
    for fid in sorted(int(v) for v in np.unique(fid_mask) if int(v) > 0):
        labeled, nlab = ndimage.label(fid_mask == fid, structure=np.ones((3, 3), dtype=np.int8))
        for lab in range(1, nlab + 1):
            rr, cc = np.nonzero(labeled == lab)
            if rr.size < min_pixels:
                continue
            island[rr, cc] = next_id
            rows_out.append(
                {
                    "island_id": next_id,
                    "primary_fid": fid,
                    "uid_count": 1,
                    "pixel_count": int(rr.size),
                    "row_min": int(rr.min()),
                    "row_max": int(rr.max()),
                    "col_min": int(cc.min()),
                    "col_max": int(cc.max()),
                }
            )
            next_id += 1
    return island, rows_out


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_surface_geojson(path: Path, surface_features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": "tongji_surface_projection_roof_facade",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "features": surface_features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_results(
    amp: np.ndarray,
    surface_features: list[dict[str, Any]],
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    conflict_count: np.ndarray,
    out_stem: Path,
    max_draw: int,
) -> dict[str, str]:
    bg = stretch(amp)
    extent_rr, extent_cc = np.nonzero(initial_mask > 0)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), dpi=300)
    for ax in axes.ravel():
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
        ax.set_axis_off()
    axes[0, 0].set_title("Projected roof and facade faces")
    drawn = 0
    for feat in surface_features:
        if max_draw > 0 and drawn >= max_draw:
            break
        surface = feat["properties"]["surface"]
        if surface == "layover_envelope":
            continue
        xy = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        color = "#ffb000" if surface == "roof" else "#00d4ff"
        axes[0, 0].add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor=color, linewidth=0.18, alpha=0.55))
        drawn += 1
    axes[0, 0].legend(
        handles=[
            Patch(facecolor="none", edgecolor="#ffb000", label="roof"),
            Patch(facecolor="none", edgecolor="#00d4ff", label="facade"),
        ],
        loc="lower right",
        fontsize=7,
        frameon=True,
    )

    axes[0, 1].set_title("Initial roof+facade mask")
    axes[0, 1].imshow(np.ma.masked_where(initial_mask == 0, initial_mask), cmap="turbo", alpha=0.52, interpolation="nearest")
    axes[1, 0].set_title("Refined amplitude-constrained mask")
    axes[1, 0].imshow(np.ma.masked_where(refined_mask == 0, refined_mask), cmap="turbo", alpha=0.56, interpolation="nearest")
    axes[1, 1].set_title("Overlap count before LOS assignment")
    axes[1, 1].imshow(np.ma.masked_where(conflict_count <= 1, conflict_count), cmap="magma", alpha=0.68, interpolation="nearest")

    if extent_rr.size:
        r0 = max(0, int(extent_rr.min()) - 60)
        r1 = min(amp.shape[0], int(extent_rr.max()) + 60)
        c0 = max(0, int(extent_cc.min()) - 60)
        c1 = min(amp.shape[1], int(extent_cc.max()) + 60)
        for ax in axes.ravel():
            ax.set_xlim(c0, c1)
            ax.set_ylim(r1, r0)
    fig.suptitle("Tongji SAR building projection mask refinement: roof + facade surfaces", fontsize=11)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {"png": f"{out_stem}.png", "svg": f"{out_stem}.svg"}
    fig.savefig(outputs["png"], dpi=360, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", default="tmp/touying_roof_workflow/results/blue_aligned/20200708_full_area_projection_sar_col_row_brightness_optimized.geojson")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--buffer-px", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=8)
    parser.add_argument("--max-buildings", type=int, default=0)
    parser.add_argument("--surface-geojson", default="work/projection/20200708_surface_projection_roof_facade.geojson")
    parser.add_argument("--initial-mask", default="work/masks/building_fid_mask_surface_initial.npy")
    parser.add_argument("--refined-mask", default="work/masks/building_fid_mask_surface_refined.npy")
    parser.add_argument("--conflict-count", default="work/masks/building_surface_initial_conflict_count.npy")
    parser.add_argument("--island-label", default="work/masks/island_label_surface_refined.npy")
    parser.add_argument("--islands-csv", default="work/masks/islands_surface_refined.csv")
    parser.add_argument("--metrics-csv", default="work/projection/20200708_surface_projection_refined_metrics.csv")
    parser.add_argument("--summary", default="results/metadata/surface_projection_refined_summary.json")
    parser.add_argument("--out-stem", default="results/pic_all/84_surface_projection_refined_masks")
    parser.add_argument("--max-draw", type=int, default=2500)
    args = parser.parse_args()

    amp = np.load(args.amp_npy).astype(np.float32)
    shape = (args.rows, args.cols)
    by_fid = load_roof_bottom(Path(args.projection))
    if args.max_buildings > 0:
        by_fid = dict(list(sorted(by_fid.items()))[: args.max_buildings])

    initial_owner = np.zeros(shape, dtype=np.int32)
    refined_owner = np.zeros(shape, dtype=np.int32)
    initial_priority = np.full(shape, np.inf, dtype=np.float32)
    refined_priority = np.full(shape, np.inf, dtype=np.float32)
    conflict_count = np.zeros(shape, dtype=np.uint16)
    surface_features: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    for fid, item in sorted(by_fid.items()):
        bottom, roof = ring_match(item["bottom"], item["roof"])
        faces = face_polygons(bottom, roof)
        roof_mask = np.zeros(shape, dtype=bool)
        facade_mask = np.zeros(shape, dtype=bool)
        envelope_mask = np.zeros(shape, dtype=bool)
        for surface, xy in faces:
            props = dict(item.get("props", {}))
            props.update({"fid": fid, "surface": surface})
            surface_features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Polygon", "coordinates": [xy.tolist() + [xy[0].tolist()]]},
                }
            )
            m = rasterize_polygon(xy, shape)
            if surface == "roof":
                roof_mask |= m
            elif surface == "facade":
                facade_mask |= m
            elif surface == "layover_envelope":
                envelope_mask |= m
        mask0 = (roof_mask | facade_mask | envelope_mask)
        if int(mask0.sum()) == 0:
            continue
        conflict_count[mask0] += 1
        near_col = float(np.nanmin(np.r_[bottom[:, 0], roof[:, 0]]))
        refined, refine_info = local_refine(mask0, amp, args.buffer_px, args.min_pixels)
        initial_changed = assign_los(initial_owner, initial_priority, fid, mask0, near_col)
        refined_changed = assign_los(refined_owner, refined_priority, fid, refined, near_col)
        vals0 = amp[mask0 & np.isfinite(amp)]
        vals_ref = amp[refined & np.isfinite(amp)]
        metrics.append(
            {
                "fid": fid,
                "height_m": float(item.get("props", {}).get("height_m", np.nan)),
                "roof_pixels": int(roof_mask.sum()),
                "facade_pixels": int(facade_mask.sum()),
                "envelope_pixels": int(envelope_mask.sum()),
                "initial_pixels": int(mask0.sum()),
                "refined_pixels_before_los": int(refined.sum()),
                "initial_pixels_after_los_update": initial_changed,
                "refined_pixels_after_los_update": refined_changed,
                "near_range_col": near_col,
                "amp_initial_mean": float(vals0.mean()) if vals0.size else np.nan,
                "amp_initial_p90": float(np.percentile(vals0, 90)) if vals0.size else np.nan,
                "amp_refined_mean": float(vals_ref.mean()) if vals_ref.size else np.nan,
                "amp_refined_p90": float(np.percentile(vals_ref, 90)) if vals_ref.size else np.nan,
                **refine_info,
            }
        )

    island_label, islands = extract_value_islands(refined_owner, args.min_pixels)
    for path, arr in [
        (args.initial_mask, initial_owner),
        (args.refined_mask, refined_owner),
        (args.conflict_count, conflict_count),
        (args.island_label, island_label),
    ]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
    write_surface_geojson(Path(args.surface_geojson), surface_features)
    save_csv(Path(args.metrics_csv), metrics)
    save_csv(Path(args.islands_csv), islands)
    outputs = plot_results(amp, surface_features, initial_owner, refined_owner, conflict_count, Path(args.out_stem), args.max_draw)

    summary = {
        "projection": args.projection,
        "projected_buildings": int(len(by_fid)),
        "surface_features": int(len(surface_features)),
        "initial_mask_pixels": int(np.sum(initial_owner > 0)),
        "refined_mask_pixels": int(np.sum(refined_owner > 0)),
        "overlap_pixels_before_los": int(np.sum(conflict_count > 1)),
        "max_overlap_count_before_los": int(conflict_count.max()) if conflict_count.size else 0,
        "refined_islands": int(len(islands)),
        "outputs": {
            "surface_geojson": args.surface_geojson,
            "initial_mask": args.initial_mask,
            "refined_mask": args.refined_mask,
            "conflict_count": args.conflict_count,
            "island_label": args.island_label,
            "islands_csv": args.islands_csv,
            "metrics_csv": args.metrics_csv,
            **outputs,
        },
        "method_note": "Roof and facade faces are rasterized from matched bottom/roof SAR-coordinate polygons. Amplitude refinement is constrained to a small initial-mask buffer. Overlapping pixels are assigned to the smaller range-column building as LOS foreground.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
