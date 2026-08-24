#!/usr/bin/env python3
"""Remap coordinate-valued NPZ products onto another point/UID hypothesis set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WIDTH = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-points", type=Path, required=True)
    parser.add_argument("--target-points", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("pair-observations", "sensitivity"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.source_points)
    target = np.load(args.target_points)
    source_flat = source["row"].astype(np.int64) * WIDTH + source["col"].astype(np.int64)
    order = np.argsort(source_flat)
    sorted_flat = source_flat[order]
    if np.any(np.diff(sorted_flat) == 0):
        raise RuntimeError("source point set must contain unique radar coordinates")
    target_flat = target["row"].astype(np.int64) * WIDTH + target["col"].astype(np.int64)
    location = np.searchsorted(sorted_flat, target_flat)
    if np.any(location == len(sorted_flat)) or not np.array_equal(sorted_flat[location], target_flat):
        raise RuntimeError("target contains a coordinate absent from source products")
    take = order[location]
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.source_root.glob("*.npz"))
    for sequence, path in enumerate(paths, start=1):
        data = np.load(path)
        if args.kind == "pair-observations":
            np.savez_compressed(
                args.output_root / path.name,
                row=target["row"], col=target["col"], label=target["label"],
                building_uid=target["building_uid"], floor_audit_only=target["floor"],
                wrapped_phase_rad=data["wrapped_phase_rad"][take],
                filtered_wrapped_phase_rad=data["filtered_wrapped_phase_rad"][take],
                coherence=data["coherence"][take],
            )
        else:
            np.savez_compressed(
                args.output_root / path.name,
                row=target["row"], col=target["col"],
                phase_sensitivity_rad_per_m=data["phase_sensitivity_rad_per_m"][take],
            )
        print(f"[{sequence}/{len(paths)}] {path.stem}", flush=True)
    result = {
        "kind": args.kind, "file_count": len(paths),
        "source_unique_coordinate_count": int(len(source_flat)),
        "target_hypothesis_point_count": int(len(target_flat)),
        "target_unique_coordinate_count": int(np.unique(target_flat).size),
        "coordinate_coverage_complete": True,
        "numeric_values_changed": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
