#!/usr/bin/env python3
"""Create a compact visualization for the physical p95 height result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/geodata/tongji_building_height_physical_p95.geojson")
    parser.add_argument("--output", default="results/pic_all/30_physical_p95_height_result_map.png")
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    ins = gdf[gdf["height_source_physical"].eq("insar_p95")].copy()
    ok = ins[ins["height_qc_physical"].eq("ok_physical_p95")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=240)

    gdf.plot(ax=axes[0], color="#eeeeee", linewidth=0.03, edgecolor="#bbbbbb")
    ins.plot(ax=axes[0], column="height_physical_p95_m", cmap="viridis", linewidth=0.04, edgecolor="#333333", legend=True)
    axes[0].set_title("Physical p95 height")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    gdf.plot(ax=axes[1], color="#eeeeee", linewidth=0.03, edgecolor="#bbbbbb")
    palette = {
        "ok_physical_p95": "#228833",
        "prior_diagnostic_gap": "#cc6677",
        "invalid_negative_height": "#882255",
        "no_insar": "#bbbbbb",
    }
    for cls, sub in gdf.groupby("height_qc_physical"):
        sub.plot(ax=axes[1], color=palette.get(cls, "#999999"), linewidth=0.04, edgecolor="#333333", label=cls)
    axes[1].set_title("Physical QC class")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend(frameon=False, fontsize=7, loc="best")

    bins = np.linspace(0, max(60.0, float(np.nanpercentile(ins["height_physical_p95_m"], 99))), 45)
    axes[2].hist(ins["height_physical_p95_m"].dropna(), bins=bins, color="#228833", alpha=0.72, label="all InSAR p95")
    axes[2].hist(ok["height_physical_p95_m"].dropna(), bins=bins, color="#66c2a5", alpha=0.68, label="QC ok")
    axes[2].set_title("Height distribution")
    axes[2].set_xlabel("height m")
    axes[2].set_ylabel("buildings")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
