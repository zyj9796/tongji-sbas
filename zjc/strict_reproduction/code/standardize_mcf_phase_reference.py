#!/usr/bin/env python3
"""Reset all pairwise MCF phases to one fixed far-range reference per building."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unwrapped-root", type=Path, required=True)
    parser.add_argument("--quality-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    quality = np.load(args.quality_metrics)
    row = quality["row"].astype(np.int32)
    col = quality["col"].astype(np.int32)
    uid = quality["building_uid"].astype(np.int32)
    da = quality["amplitude_dispersion"].astype(np.float64)
    mean_cc = quality["mean_coherence"].astype(np.float64)
    point_quality = mean_cc * (0.05 + 0.95 * np.clip(1.0 - da / 0.4, 0.0, 1.0))
    fixed_root: dict[int, int] = {}
    for building in np.unique(uid):
        member = np.flatnonzero(uid == building)
        far_threshold = np.quantile(col[member], 0.95)
        candidates = member[col[member] >= far_threshold]
        fixed_root[int(building)] = int(candidates[np.argmax(point_quality[candidates])])

    args.output_root.mkdir(parents=True, exist_ok=True)
    pair_paths = sorted(args.unwrapped_root.glob("*.npz"))
    finite_counts = []
    maximum_root_absolute = []
    for sequence, path in enumerate(pair_paths, start=1):
        source = np.load(path)
        phase = source["unwrapped_phase_far_ground_zero_rad"].astype(np.float64)
        fixed = np.full_like(phase, np.nan)
        root_values = []
        for building, root in fixed_root.items():
            member = uid == building
            if np.isfinite(phase[root]):
                fixed[member] = phase[member] - phase[root]
                root_values.append(abs(fixed[root]))
        output = args.output_root / path.name
        np.savez_compressed(
            output, row=row, col=col, building_uid=uid,
            unwrapped_phase_fixed_far_ground_zero_rad=fixed.astype(np.float32),
            unwrapped_phase_original_first_pixel_zero_rad=source[
                "unwrapped_phase_original_first_pixel_zero_rad"
            ],
        )
        finite_counts.append(int(np.isfinite(fixed).sum()))
        maximum_root_absolute.append(float(max(root_values, default=np.nan)))
        print(f"[{sequence}/{len(pair_paths)}] {path.stem}", flush=True)

    result = {
        "pair_count": len(pair_paths),
        "building_count": len(fixed_root),
        "finite_pixel_count_unique": sorted(set(finite_counts)),
        "maximum_absolute_fixed_root_phase_rad": float(np.nanmax(maximum_root_absolute)),
        "fixed_reference_policy": "one point per building; far-range 5%; maximum mean_coherence*(0.05+0.95*amplitude_quality)",
        "floor_used": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
