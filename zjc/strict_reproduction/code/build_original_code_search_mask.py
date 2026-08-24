#!/usr/bin/env python3
"""Reproduce the original MATLAB Floor-assisted radar search-mask expansion.

Floor controls only where observations are searched.  It is retained as an
audit attribute and is never converted to, copied into, or used to correct an
InSAR height estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation


WIDTH = 10_000
LINES = 7_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-island-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum-floor", type=int, default=30)
    parser.add_argument("--global-shift-down", type=int, default=10)
    parser.add_argument("--global-shift-right", type=int, default=35)
    parser.add_argument(
        "--overlap-policy", choices=("literal-overwrite", "preserve-per-building"),
        default="literal-overwrite",
    )
    args = parser.parse_args()

    source = np.load(args.ground_island_points)
    source_row = source["row"].astype(np.int32)
    source_col = source["col"].astype(np.int32)
    source_uid = source["building_uid"].astype(np.int32)
    source_floor = source["floor"].astype(np.int16)
    uids = np.unique(source_uid[source_floor > args.minimum_floor])
    uid_floor = {int(uid): int(np.median(source_floor[source_uid == uid])) for uid in uids}

    # This assignment order intentionally follows the MATLAB script: buildings
    # are sorted high-to-low, so later lower buildings overwrite overlaps.
    owner: dict[int, tuple[int, int]] = {}
    preserved_flat: list[np.ndarray] = []
    preserved_uid: list[np.ndarray] = []
    preserved_floor: list[np.ndarray] = []
    expansion_parameters: dict[int, dict[str, int]] = {}
    for uid in sorted(uids, key=lambda item: (-uid_floor[int(item)], int(item))):
        floor = uid_floor[int(uid)]
        member = source_uid == uid
        rows = source_row[member]
        cols = source_col[member]
        left = round(1.9 * floor) + round(50 * floor / 70)
        right = round(20 * floor / 70)
        up = round(10 * floor / 70)
        down = round(10 * floor / 70)
        r0, r1 = max(int(rows.min()) - up, 0), min(int(rows.max()) + down + 1, LINES)
        c0, c1 = max(int(cols.min()) - left, 0), min(int(cols.max()) + right + 1, WIDTH)
        seed = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        seed[rows - r0, cols - c0] = True
        structure = np.ones((up + down + 1, left + right + 1), dtype=bool)
        expanded = binary_dilation(seed, structure=structure)
        rr, cc = np.nonzero(expanded)
        rr = rr + r0 + args.global_shift_down
        cc = cc + c0 + args.global_shift_right
        inside = (rr >= 0) & (rr < LINES) & (cc >= 0) & (cc < WIDTH)
        current_flat = rr[inside].astype(np.int64) * WIDTH + cc[inside].astype(np.int64)
        preserved_flat.append(current_flat)
        preserved_uid.append(np.full(len(current_flat), int(uid), dtype=np.int32))
        preserved_floor.append(np.full(len(current_flat), floor, dtype=np.int16))
        for row, col in zip(rr[inside], cc[inside], strict=True):
            owner[int(row) * WIDTH + int(col)] = (int(uid), floor)
        expansion_parameters[int(uid)] = {
            "floor_audit_only": floor, "left": left, "right": right, "up": up, "down": down
        }

    if args.overlap_policy == "preserve-per-building":
        flat = np.concatenate(preserved_flat)
        uid = np.concatenate(preserved_uid)
        floor = np.concatenate(preserved_floor)
        order = np.lexsort((flat, uid))
        flat, uid, floor = flat[order], uid[order], floor[order]
    else:
        flat = np.array(sorted(owner), dtype=np.int64)
        uid = np.array([owner[int(index)][0] for index in flat], dtype=np.int32)
        floor = np.array([owner[int(index)][1] for index in flat], dtype=np.int16)
    row = (flat // WIDTH).astype(np.int32)
    col = (flat % WIDTH).astype(np.int32)
    # Each building UID is an independent island.  Compact labels make the
    # downstream per-island arrays deterministic.
    uid_order = np.unique(uid)
    label_lookup = {int(value): index + 1 for index, value in enumerate(uid_order)}
    label = np.array([label_lookup[int(value)] for value in uid], dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, row=row, col=col, label=label, building_uid=uid, floor=floor
    )
    summary = {
        "method": "literal original MATLAB directional expansion and global shift",
        "source_building_uid_count": int(len(uids)),
        "expanded_pixel_count": int(len(row)),
        "output_building_uid_count": int(len(uid_order)),
        "minimum_floor_exclusive": args.minimum_floor,
        "global_shift_down_px": args.global_shift_down,
        "global_shift_right_px": args.global_shift_right,
        "overlap_policy": (
            "independent per-building hypotheses retained, including duplicated SAR coordinates"
            if args.overlap_policy == "preserve-per-building"
            else "literal MATLAB assignment order: descending Floor, later lower building overwrites"
        ),
        "floor_use": "search-mask geometry only; never used as an inversion observation, bound, fill, or correction",
        "per_building_expansion": expansion_parameters,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_building_expansion"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
