#!/usr/bin/env python3
"""Split roof islands by clean building ID and local connected components."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


def save_preview(uid_mask: np.ndarray, island_label: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=220)
    uid_show = np.ma.masked_where(uid_mask <= 0, uid_mask % 251)
    island_show = np.ma.masked_where(island_label <= 0, island_label % 251)
    axes[0].imshow(uid_show, cmap="turbo", interpolation="nearest")
    axes[0].set_title("clean_id roof mask")
    axes[0].set_axis_off()
    axes[1].imshow(island_show, cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title("clean_id-split islands")
    axes[1].set_axis_off()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid-mask", default="work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--old-island-label", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--old-islands-csv", default="work/masks/islands_clean_equal_height_roof_only_full_area_128.csv")
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=8)
    parser.add_argument("--out-island-label", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_split.npy")
    parser.add_argument("--out-islands-csv", default="work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_split.csv")
    parser.add_argument("--preview", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_split.png")
    parser.add_argument("--summary", default="results/metadata/island_extraction_clean_equal_height_roof_only_full_area_128_cleanid_split_summary.json")
    args = parser.parse_args()

    uid_mask = np.load(args.uid_mask).astype(np.int32)
    old_label = np.load(args.old_island_label).astype(np.int32)
    old_islands = pd.read_csv(args.old_islands_csv)
    old_uid_count = dict(zip(old_islands["island_id"].astype(int), old_islands["uid_count"].astype(int)))

    structure = np.ones((3, 3), dtype=np.int8) if args.connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    out_label = np.zeros(uid_mask.shape, dtype=np.int32)
    rows: list[dict] = []
    source_old_ids: list[int] = []
    next_id = 1
    for clean_id in sorted(int(v) for v in np.unique(uid_mask) if int(v) > 0):
        building_mask = uid_mask == clean_id
        labeled, n_comp = ndimage.label(building_mask, structure=structure)
        for comp in range(1, n_comp + 1):
            rr, cc = np.nonzero(labeled == comp)
            if rr.size < args.min_pixels:
                continue
            old_ids = [int(v) for v in np.unique(old_label[rr, cc]) if int(v) > 0]
            old_counter = Counter(int(v) for v in old_label[rr, cc] if int(v) > 0)
            source_old = old_counter.most_common(1)[0][0] if old_counter else 0
            out_label[rr, cc] = next_id
            rows.append(
                {
                    "island_id": next_id,
                    "component_id": int(comp),
                    "dbscan_label": 0,
                    "primary_uid": clean_id,
                    "uid_count": 1,
                    "pixel_count": int(rr.size),
                    "row_min": int(rr.min()),
                    "row_max": int(rr.max()),
                    "col_min": int(cc.min()),
                    "col_max": int(cc.max()),
                    "source_old_island": int(source_old),
                    "source_old_island_uid_count": int(old_uid_count.get(source_old, 0)),
                    "source_old_islands": ";".join(map(str, old_ids)),
                }
            )
            source_old_ids.append(source_old)
            next_id += 1

    out_label_path = Path(args.out_island_label)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_label_path, out_label)
    out_csv = Path(args.out_islands_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    save_preview(uid_mask, out_label, Path(args.preview))

    old_multi_ids = {int(r.island_id) for r in old_islands.itertuples(index=False) if int(r.uid_count) > 1}
    summary = {
        "method": "Split existing roof mask by clean_id first, then by connected components inside each clean_id. This prevents one island from containing multiple clean building IDs.",
        "input_uid_mask": args.uid_mask,
        "input_old_island_label": args.old_island_label,
        "min_pixels": args.min_pixels,
        "connectivity": args.connectivity,
        "old_islands": int(len(old_islands)),
        "old_multi_clean_id_islands": int(len(old_multi_ids)),
        "new_islands": int(len(rows)),
        "new_multi_clean_id_islands": 0,
        "new_islands_from_old_multi_clean_id_islands": int(sum(1 for v in source_old_ids if v in old_multi_ids)),
        "uid_mask_pixels": int(np.sum(uid_mask > 0)),
        "new_island_pixels": int(np.sum(out_label > 0)),
        "dropped_small_component_pixels": int(np.sum((uid_mask > 0) & (out_label <= 0))),
        "outputs": {
            "island_label": args.out_island_label,
            "islands_csv": args.out_islands_csv,
            "preview": args.preview,
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
