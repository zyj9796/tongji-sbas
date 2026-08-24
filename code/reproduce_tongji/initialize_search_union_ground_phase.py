#!/usr/bin/env python3
"""Initialize a large roof-search point stack from a solved stable-ground stack.

Search-roof phases remain wrapped because the building-height model unwraps them
later. Only stable-ground points copy the GAMMA ``multi_def_pt`` unwrapped phase
and validity mask from a previously solved stack with the same pair ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ipta", required=True)
    parser.add_argument("--ground-source-ipta", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    target = Path(args.target_ipta)
    source = Path(args.ground_source_ipta)
    target_meta = pd.read_csv(target / "point_metadata.csv").sort_values("point_index")
    source_meta = pd.read_csv(source / "point_metadata.csv").sort_values("point_index")
    npairs = len(pd.read_csv(args.pairs).drop_duplicates(["master", "slave"]))
    ntarget, nsource = len(target_meta), len(source_meta)

    target_diff = np.memmap(target / "pdiff", dtype=">c8", mode="r", shape=(npairs, ntarget))
    target_unw = np.angle(np.asarray(target_diff)).astype(">f4")
    target_mask = np.ones(ntarget, dtype=np.uint8)

    source_unw = np.memmap(source / "punw", dtype=">f4", mode="r", shape=(npairs, nsource))
    source_mask = np.fromfile(source / "pmask", dtype=np.uint8)[:nsource] > 0
    source_ground = source_meta[source_meta["point_class"].eq("ground")].copy()
    lookup = {
        (int(row.range_pixel), int(row.azimuth_pixel)): int(row.point_index)
        for row in source_ground.itertuples(index=False)
    }
    target_ground = target_meta[target_meta["point_class"].eq("ground")]
    copied = 0
    missing = 0
    for row in target_ground.itertuples(index=False):
        target_index = int(row.point_index)
        source_index = lookup.get((int(row.range_pixel), int(row.azimuth_pixel)))
        if source_index is None or not source_mask[source_index]:
            target_mask[target_index] = 0
            missing += 1
            continue
        target_unw[:, target_index] = source_unw[:, source_index]
        copied += 1
    target_unw.tofile(target / "punw")
    target_mask.tofile(target / "pmask")

    payload = {
        "method": "copy GAMMA-unwrapped stable ground; retain wrapped roof-search phases for model unwrapping",
        "pairs": npairs,
        "target_points": ntarget,
        "target_ground_points": int(len(target_ground)),
        "ground_points_copied": copied,
        "ground_points_missing_or_invalid": missing,
        "roof_candidate_points": int((target_meta["point_class"] == "roof").sum()),
        "height_prior_used": False,
        "target_punw": str(target / "punw"),
        "target_pmask": str(target / "pmask"),
    }
    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
