#!/usr/bin/env python3
"""Compare two building-height GeoJSON/CSV products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    import geopandas as gpd

    return pd.DataFrame(gpd.read_file(p).drop(columns="geometry"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--label-old", default="old")
    parser.add_argument("--label-new", default="new")
    parser.add_argument("--out-base", required=True)
    args = parser.parse_args()

    old = read_table(args.old)
    new = read_table(args.new)
    old = old[["clean_id", "height_insar_m", "height", "has_insar_height"]].rename(
        columns={"height_insar_m": "height_old", "has_insar_height": "has_old"}
    )
    new = new[["clean_id", "height_insar_m", "height", "has_insar_height"]].rename(
        columns={"height_insar_m": "height_new", "has_insar_height": "has_new"}
    )
    merged = old.merge(new[["clean_id", "height_new", "has_new"]], on="clean_id", how="outer")
    for col in ["height_old", "height_new", "height"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    common = merged[np.isfinite(merged["height_old"]) & np.isfinite(merged["height_new"])].copy()
    common["delta_new_old_m"] = common["height_new"] - common["height_old"]
    common["old_gap_diag_m"] = common["height_old"] - common["height"]
    common["new_gap_diag_m"] = common["height_new"] - common["height"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=220)
    axes[0, 0].hist(merged["height_old"].dropna(), bins=60, alpha=0.55, label=args.label_old, color="#4477aa")
    axes[0, 0].hist(merged["height_new"].dropna(), bins=60, alpha=0.55, label=args.label_new, color="#cc6677")
    axes[0, 0].set_title("Height distributions")
    axes[0, 0].set_xlabel("height, m")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(common["height_old"], common["height_new"], s=9, alpha=0.45, color="#117733")
    axes[0, 1].axline((0, 0), slope=1, color="black", lw=0.8)
    axes[0, 1].set_title("Common buildings")
    axes[0, 1].set_xlabel(f"{args.label_old}, m")
    axes[0, 1].set_ylabel(f"{args.label_new}, m")

    axes[1, 0].hist(common["delta_new_old_m"], bins=60, color="#aa4499", edgecolor="white")
    axes[1, 0].axvline(0, color="black", lw=0.8)
    axes[1, 0].set_title("New - old")
    axes[1, 0].set_xlabel("m")

    axes[1, 1].hist(common["old_gap_diag_m"].dropna(), bins=60, alpha=0.55, label=args.label_old, color="#4477aa")
    axes[1, 1].hist(common["new_gap_diag_m"].dropna(), bins=60, alpha=0.55, label=args.label_new, color="#cc6677")
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Diagnostic gap to shp height field")
    axes[1, 1].set_xlabel("InSAR - height field, m")
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    out = Path(args.out_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)

    payload = {
        "old": args.old,
        "new": args.new,
        "label_old": args.label_old,
        "label_new": args.label_new,
        "old_count": int(np.isfinite(merged["height_old"]).sum()),
        "new_count": int(np.isfinite(merged["height_new"]).sum()),
        "common_count": int(len(common)),
        "old_median_m": float(np.nanmedian(merged["height_old"])),
        "new_median_m": float(np.nanmedian(merged["height_new"])),
        "delta_new_old_median_m": float(common["delta_new_old_m"].median()) if not common.empty else None,
        "delta_new_old_p05_m": float(common["delta_new_old_m"].quantile(0.05)) if not common.empty else None,
        "delta_new_old_p95_m": float(common["delta_new_old_m"].quantile(0.95)) if not common.empty else None,
        "old_diag_mae_m": float(common["old_gap_diag_m"].abs().mean()) if not common.empty else None,
        "new_diag_mae_m": float(common["new_gap_diag_m"].abs().mean()) if not common.empty else None,
        "old_diag_median_abs_m": float(common["old_gap_diag_m"].abs().median()) if not common.empty else None,
        "new_diag_median_abs_m": float(common["new_gap_diag_m"].abs().median()) if not common.empty else None,
        "height_field_use": "diagnostic_only_not_used_for_fitting_filtering_or_selection",
        "outputs": {"png": str(out.with_suffix(".png")), "svg": str(out.with_suffix(".svg")), "json": str(out.with_suffix(".json"))},
    }
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
