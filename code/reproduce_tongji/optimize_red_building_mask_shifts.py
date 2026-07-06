#!/usr/bin/env python3
"""Conservative local SAR-amplitude shift search for QC-review buildings.

This is a mask-level correction branch: it does not use the building height
attribute and it does not overwrite the original projected roof mask.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from scipy import ndimage

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


@dataclass(frozen=True)
class ShiftScore:
    clean_id: int
    row_shift: int
    col_shift: int
    score: float
    score_gain: float
    inside_mean: float
    ring_mean: float
    contrast: float
    contrast_gain: float
    boundary_edge_mean: float
    boundary_edge_gain: float
    overlap_other_frac: float
    kept_pixel_frac: float
    candidate_pixels: int
    accepted: bool
    reject_reason: str


def robust01(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32, copy=True)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    if not bool(finite.any()):
        return out
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def setup_chinese_font() -> FontProperties | None:
    if not Path(FONT_PATH).exists():
        return None
    fontManager.addfont(FONT_PATH)
    prop = FontProperties(fname=FONT_PATH)
    matplotlib.rcParams.update(
        {
            "font.family": prop.get_name(),
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return prop


def shift_bool(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    src_r0 = max(0, -dr)
    src_r1 = min(rows, rows - dr)
    dst_r0 = max(0, dr)
    dst_r1 = min(rows, rows + dr)
    src_c0 = max(0, -dc)
    src_c1 = min(cols, cols - dc)
    dst_c0 = max(0, dc)
    dst_c1 = min(cols, cols + dc)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return out
    out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def score_mask(candidate: np.ndarray, amp01: np.ndarray, edge01: np.ndarray, other_mask: np.ndarray, base_score: float) -> dict[str, float | int]:
    if int(candidate.sum()) == 0:
        return {
            "score": -math.inf,
            "score_gain": -math.inf,
            "inside_mean": math.nan,
            "ring_mean": math.nan,
            "contrast": math.nan,
            "contrast_gain": math.nan,
            "boundary_edge_mean": math.nan,
            "boundary_edge_gain": math.nan,
            "overlap_other_frac": 1.0,
            "kept_pixel_frac": 0.0,
            "candidate_pixels": 0,
        }
    structure = np.ones((3, 3), dtype=bool)
    dil = ndimage.binary_dilation(candidate, structure=structure, iterations=2)
    ero = ndimage.binary_erosion(candidate, structure=structure, iterations=1)
    ring = dil & ~candidate
    boundary = dil & ~ero
    inside_mean = float(np.nanmean(amp01[candidate]))
    ring_mean = float(np.nanmean(amp01[ring])) if bool(ring.any()) else inside_mean
    contrast = inside_mean - ring_mean
    boundary_edge_mean = float(np.nanmean(edge01[boundary])) if bool(boundary.any()) else 0.0
    overlap_other_frac = float(np.sum(candidate & other_mask) / max(int(candidate.sum()), 1))
    kept_pixel_frac = float(np.sum(candidate & ~other_mask) / max(int(candidate.sum()), 1))
    score = 0.95 * inside_mean + 0.55 * contrast + 0.75 * boundary_edge_mean - 1.8 * overlap_other_frac
    return {
        "score": float(score),
        "score_gain": float(score - base_score),
        "inside_mean": inside_mean,
        "ring_mean": ring_mean,
        "contrast": contrast,
        "contrast_gain": math.nan,
        "boundary_edge_mean": boundary_edge_mean,
        "boundary_edge_gain": math.nan,
        "overlap_other_frac": overlap_other_frac,
        "kept_pixel_frac": kept_pixel_frac,
        "candidate_pixels": int(candidate.sum()),
    }


def local_arrays(mask: np.ndarray, amp: np.ndarray, pad: int) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    rr, cc = np.nonzero(mask)
    r0 = max(0, int(rr.min()) - pad)
    r1 = min(mask.shape[0], int(rr.max()) + pad + 1)
    c0 = max(0, int(cc.min()) - pad)
    c1 = min(mask.shape[1], int(cc.max()) + pad + 1)
    local_amp = robust01(np.log1p(np.maximum(amp[r0:r1, c0:c1], 0.0)))
    gy, gx = np.gradient(local_amp)
    edge = robust01(np.hypot(gx, gy))
    return slice(r0, r1), slice(c0, c1), local_amp, edge


def best_shift_for_building(
    clean_id: int,
    uid_mask: np.ndarray,
    amp: np.ndarray,
    max_shift: int,
    min_pixels: int,
    min_score_gain: float,
    min_contrast_gain: float,
    min_edge_gain: float,
    max_overlap_frac: float,
    min_kept_frac: float,
) -> ShiftScore:
    original = uid_mask == clean_id
    if int(original.sum()) < min_pixels:
        return ShiftScore(clean_id, 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, int(original.sum()), False, "too_few_original_pixels")

    rs, cs, amp01, edge01 = local_arrays(original, amp, pad=max_shift + 8)
    local_original = original[rs, cs]
    local_other = (uid_mask[rs, cs] > 0) & (uid_mask[rs, cs] != clean_id)
    base = score_mask(local_original, amp01, edge01, local_other, 0.0)
    base_score = float(base["score"])
    base_contrast = float(base["contrast"])
    base_edge = float(base["boundary_edge_mean"])
    best: dict[str, float | int] | None = None
    best_dr = 0
    best_dc = 0
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            candidate = shift_bool(local_original, dr, dc)
            metrics = score_mask(candidate, amp01, edge01, local_other, base_score)
            dist_penalty = 0.012 * math.hypot(dr, dc)
            metrics["score"] = float(metrics["score"]) - dist_penalty
            metrics["score_gain"] = float(metrics["score"]) - base_score
            metrics["contrast_gain"] = float(metrics["contrast"]) - base_contrast
            metrics["boundary_edge_gain"] = float(metrics["boundary_edge_mean"]) - base_edge
            if best is None or float(metrics["score"]) > float(best["score"]):
                best = metrics
                best_dr = dr
                best_dc = dc
    assert best is not None

    reject_reason = ""
    accepted = True
    if best_dr == 0 and best_dc == 0:
        accepted = False
        reject_reason = "best_is_original"
    elif float(best["score_gain"]) < min_score_gain:
        accepted = False
        reject_reason = "score_gain_too_small"
    elif float(best["contrast_gain"]) < min_contrast_gain and float(best["boundary_edge_gain"]) < min_edge_gain:
        accepted = False
        reject_reason = "no_clear_contrast_or_edge_gain"
    elif float(best["overlap_other_frac"]) > max_overlap_frac:
        accepted = False
        reject_reason = "overlap_other_building_too_large"
    elif float(best["kept_pixel_frac"]) < min_kept_frac:
        accepted = False
        reject_reason = "too_many_pixels_lost_to_overlap"
    elif int(best["candidate_pixels"]) < min_pixels:
        accepted = False
        reject_reason = "too_few_candidate_pixels"
    if accepted:
        reject_reason = "accepted"

    return ShiftScore(
        clean_id=clean_id,
        row_shift=int(best_dr),
        col_shift=int(best_dc),
        score=float(best["score"]),
        score_gain=float(best["score_gain"]),
        inside_mean=float(best["inside_mean"]),
        ring_mean=float(best["ring_mean"]),
        contrast=float(best["contrast"]),
        contrast_gain=float(best["contrast_gain"]),
        boundary_edge_mean=float(best["boundary_edge_mean"]),
        boundary_edge_gain=float(best["boundary_edge_gain"]),
        overlap_other_frac=float(best["overlap_other_frac"]),
        kept_pixel_frac=float(best["kept_pixel_frac"]),
        candidate_pixels=int(best["candidate_pixels"]),
        accepted=bool(accepted),
        reject_reason=reject_reason,
    )


def apply_shifts(uid_mask: np.ndarray, scores: list[ShiftScore], min_pixels: int) -> np.ndarray:
    out = uid_mask.copy()
    accepted = [s for s in scores if s.accepted]
    accepted.sort(key=lambda s: s.score_gain, reverse=True)
    for s in accepted:
        original = uid_mask == s.clean_id
        if int(original.sum()) < min_pixels:
            continue
        shifted = shift_bool(original, s.row_shift, s.col_shift)
        shifted = shifted & ((out == 0) | (out == s.clean_id))
        if int(shifted.sum()) < min_pixels:
            continue
        out[out == s.clean_id] = 0
        out[shifted] = s.clean_id
    return out


def save_preview(original: np.ndarray, shifted: np.ndarray, amp: np.ndarray, scores: list[ShiftScore], out_png: Path) -> None:
    font = setup_chinese_font()
    changed = np.zeros(original.shape, dtype=np.uint8)
    changed[(original > 0) & (shifted > 0)] = 1
    changed[(original > 0) & (shifted == 0)] = 2
    changed[(original == 0) & (shifted > 0)] = 3
    accepted_ids = {s.clean_id for s in scores if s.accepted}
    accepted_mask = np.isin(shifted, list(accepted_ids)) if accepted_ids else np.zeros(shifted.shape, dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), dpi=220)
    amp_show = robust01(np.log1p(np.maximum(amp, 0.0)))
    axes[0].imshow(amp_show, cmap="gray", interpolation="nearest")
    axes[0].imshow(np.ma.masked_where(original <= 0, original % 251), cmap="turbo", alpha=0.55, interpolation="nearest")
    axes[0].set_title("原始 clean_id mask", fontproperties=font)
    axes[1].imshow(amp_show, cmap="gray", interpolation="nearest")
    axes[1].imshow(np.ma.masked_where(shifted <= 0, shifted % 251), cmap="turbo", alpha=0.55, interpolation="nearest")
    axes[1].imshow(np.ma.masked_where(~accepted_mask, accepted_mask), cmap=ListedColormap(["#d7191c"]), alpha=0.72, interpolation="nearest")
    axes[1].set_title("红线建筑保守位移后 mask", fontproperties=font)
    axes[2].imshow(changed, cmap=ListedColormap(["#ffffff", "#d9d9d9", "#d73027", "#1a9850"]), interpolation="nearest", vmin=0, vmax=3)
    axes[2].set_title("mask 变化：红=移出，绿=移入", fontproperties=font)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("投影偏差局部修正：仅对 QC 红线建筑搜索 SAR 幅度脊线位移", fontsize=13, fontproperties=font)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid-mask", default="work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--amplitude", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--red-manifest", default="results/diagnostics/cleanid_split_red_buildings/diagnostic_manifest.csv")
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--min-score-gain", type=float, default=0.035)
    parser.add_argument("--min-contrast-gain", type=float, default=0.012)
    parser.add_argument("--min-edge-gain", type=float, default=0.012)
    parser.add_argument("--max-overlap-frac", type=float, default=0.18)
    parser.add_argument("--min-kept-frac", type=float, default=0.78)
    parser.add_argument("--accept-clean-ids-file", default="", help="Optional CSV with clean_id values. If provided, only otherwise accepted shifts for these buildings are applied.")
    parser.add_argument("--out-mask", default="work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift.npy")
    parser.add_argument("--out-csv", default="work/projection/cleanid_split_red_building_mask_shift_metrics.csv")
    parser.add_argument("--preview", default="work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_preview.png")
    parser.add_argument("--summary", default="results/metadata/cleanid_split_red_building_mask_shift_summary.json")
    args = parser.parse_args()

    uid_mask = np.load(args.uid_mask).astype(np.int32)
    amp = np.load(args.amplitude).astype(np.float32)
    if amp.shape != uid_mask.shape:
        raise ValueError(f"amplitude shape {amp.shape} does not match uid mask {uid_mask.shape}")
    red = pd.read_csv(args.red_manifest)
    clean_ids = sorted(int(v) for v in red["clean_id"].dropna().unique())
    scores = [
        best_shift_for_building(
            clean_id,
            uid_mask,
            amp,
            args.max_shift,
            args.min_pixels,
            args.min_score_gain,
            args.min_contrast_gain,
            args.min_edge_gain,
            args.max_overlap_frac,
            args.min_kept_frac,
        )
        for clean_id in clean_ids
    ]
    whitelist: set[int] | None = None
    if args.accept_clean_ids_file:
        accept_df = pd.read_csv(args.accept_clean_ids_file)
        if "clean_id" not in accept_df.columns:
            raise ValueError("--accept-clean-ids-file must contain a clean_id column")
        whitelist = {int(v) for v in accept_df["clean_id"].dropna().astype(int)}
        scores = [
            s
            if (not s.accepted or s.clean_id in whitelist)
            else ShiftScore(
                clean_id=s.clean_id,
                row_shift=s.row_shift,
                col_shift=s.col_shift,
                score=s.score,
                score_gain=s.score_gain,
                inside_mean=s.inside_mean,
                ring_mean=s.ring_mean,
                contrast=s.contrast,
                contrast_gain=s.contrast_gain,
                boundary_edge_mean=s.boundary_edge_mean,
                boundary_edge_gain=s.boundary_edge_gain,
                overlap_other_frac=s.overlap_other_frac,
                kept_pixel_frac=s.kept_pixel_frac,
                candidate_pixels=s.candidate_pixels,
                accepted=False,
                reject_reason="not_in_audited_accept_list",
            )
            for s in scores
        ]
    shifted = apply_shifts(uid_mask, scores, args.min_pixels)

    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_mask, shifted.astype(np.int32))
    rows = [s.__dict__ for s in scores]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    save_preview(uid_mask, shifted, amp, scores, Path(args.preview))

    accepted = [s for s in scores if s.accepted]
    summary = {
        "method": "Conservative local SAR-amplitude/edge shift search for QC-review clean_id masks only. No shapefile height attribute is used.",
        "input_uid_mask": args.uid_mask,
        "input_amplitude": args.amplitude,
        "input_red_manifest": args.red_manifest,
        "red_buildings": int(len(clean_ids)),
        "accepted_shifts": int(len(accepted)),
        "rejected_shifts": int(len(scores) - len(accepted)),
        "max_shift_pixels": int(args.max_shift),
        "min_score_gain": float(args.min_score_gain),
        "min_contrast_gain": float(args.min_contrast_gain),
        "min_edge_gain": float(args.min_edge_gain),
        "max_overlap_frac": float(args.max_overlap_frac),
        "min_kept_frac": float(args.min_kept_frac),
        "accept_clean_ids_file": args.accept_clean_ids_file,
        "accept_clean_ids_count": None if whitelist is None else int(len(whitelist)),
        "changed_pixels": int(np.sum(uid_mask != shifted)),
        "accepted_clean_ids": [int(s.clean_id) for s in accepted],
        "reject_reason_counts": {str(k): int(v) for k, v in pd.Series([s.reject_reason for s in scores]).value_counts().sort_index().items()},
        "outputs": {
            "mask": args.out_mask,
            "csv": args.out_csv,
            "preview": args.preview,
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
