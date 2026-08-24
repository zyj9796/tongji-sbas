#!/usr/bin/env python3
"""Audit temporal triangle closure before and after local phase referencing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wrap(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def quantiles(value: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(value, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-pair-root", type=Path, required=True)
    parser.add_argument("--referenced-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--quality-metrics", type=Path)
    args = parser.parse_args()

    paths = sorted(args.building_pair_root.glob("*.npz"))
    names = [path.stem for path in paths]
    pairs = [tuple(name.split("_")) for name in names]
    index = {pair: i for i, pair in enumerate(pairs)}
    dates = sorted({date for pair in pairs for date in pair})
    triangles: list[tuple[int, int, int]] = []
    for ia, first in enumerate(dates):
        for ib in range(ia + 1, len(dates)):
            second = dates[ib]
            for ic in range(ib + 1, len(dates)):
                third = dates[ic]
                if (first, second) in index and (second, third) in index and (first, third) in index:
                    triangles.append((index[(first, second)], index[(second, third)], index[(first, third)]))

    original = np.stack([np.load(path)["filtered_wrapped_phase_rad"] for path in paths]).astype(np.float64)
    referenced_paths = [args.referenced_root / f"{name}.npz" for name in names]
    referenced = np.stack(
        [np.load(path)["ground_referenced_filtered_wrapped_phase_rad"] for path in referenced_paths]
    ).astype(np.float64)
    local_reference = np.stack(
        [np.load(path)["local_ground_reference_phase_rad"] for path in referenced_paths]
    ).astype(np.float64)
    original_closure = np.stack([wrap(original[a] + original[b] - original[c]) for a, b, c in triangles])
    reference_closure = np.stack(
        [wrap(local_reference[a] + local_reference[b] - local_reference[c]) for a, b, c in triangles]
    )
    referenced_closure = np.stack(
        [wrap(referenced[a] + referenced[b] - referenced[c]) for a, b, c in triangles]
    )
    selected = np.ones(original.shape[1], dtype=bool)
    if args.quality_metrics:
        selected = np.load(args.quality_metrics)["paper_quality_selected"].astype(bool)
    original_closure = original_closure[:, selected]
    reference_closure = reference_closure[:, selected]
    referenced_closure = referenced_closure[:, selected]
    result = {
        "temporal_triangle_count": len(triangles),
        "pixel_count": int(selected.sum()),
        "selection": "paper_quality_selected" if args.quality_metrics else "all building-island pixels",
        "per_pixel_closure_rms_rad_quantiles": {
            "original": quantiles(np.sqrt(np.mean(original_closure**2, axis=0))),
            "local_reference": quantiles(np.sqrt(np.mean(reference_closure**2, axis=0))),
            "after_reference": quantiles(np.sqrt(np.mean(referenced_closure**2, axis=0))),
        },
        "all_triangle_pixel_absolute_closure_rad_quantiles": {
            "original": quantiles(np.abs(original_closure)),
            "local_reference": quantiles(np.abs(reference_closure)),
            "after_reference": quantiles(np.abs(referenced_closure)),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
