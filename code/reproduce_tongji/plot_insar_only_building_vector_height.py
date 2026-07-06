#!/usr/bin/env python3
"""Plot InSAR-only building heights on vector footprints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def read_projected(path: Path, crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(crs)


def plot_height_map(ax: plt.Axes, gdf: gpd.GeoDataFrame, title: str, norm: Normalize, cmap: str) -> None:
    no = gdf[~gdf["insar_solution_available"]].copy()
    yes = gdf[gdf["insar_solution_available"]].copy()
    if not no.empty:
        no.plot(ax=ax, color="#eeeeee", edgecolor="#d0d0d0", linewidth=0.035)
    if not yes.empty:
        yes.plot(
            ax=ax,
            column="height_est_m",
            cmap=cmap,
            norm=norm,
            edgecolor="#222222",
            linewidth=0.055,
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=4)


def expanded_bounds(gdf: gpd.GeoDataFrame, pad: float = 60.0) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = gdf.total_bounds
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def save_all(fig: plt.Figure, stem: Path, dpi: int) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": f"{stem}.png",
        "svg": f"{stem}.svg",
        "pdf": f"{stem}.pdf",
    }
    fig.savefig(outputs["png"], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", default="results/geodata/tongji_building_height_insar_only_roof_all_source_range.geojson")
    parser.add_argument("--relaxed", default="results/geodata/tongji_building_height_insar_only_roof_all_coh05_source_range.geojson")
    parser.add_argument("--projected-crs", default="EPSG:32651")
    parser.add_argument("--out-stem", default="results/pic_all/62_insar_only_building_vector_height")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    setup_style()
    strict = read_projected(Path(args.strict), args.projected_crs)
    relaxed = read_projected(Path(args.relaxed), args.projected_crs)
    strict_yes = strict[strict["insar_solution_available"]].copy()
    relaxed_yes = relaxed[relaxed["insar_solution_available"]].copy()
    vmax = float(np.nanpercentile(relaxed_yes["height_est_m"], 98)) if not relaxed_yes.empty else 60.0
    vmax = max(30.0, min(vmax, 80.0))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = "viridis"

    fig = plt.figure(figsize=(12.5, 8.2), dpi=args.dpi)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.72], height_ratios=[1.0, 1.0], wspace=0.04, hspace=0.11)
    ax_relaxed = fig.add_subplot(gs[:, 0])
    ax_strict = fig.add_subplot(gs[:, 1])
    ax_zoom = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 2])

    plot_height_map(ax_relaxed, relaxed, "InSAR-only height on building footprints\ncoherence >= 0.5", norm, cmap)
    plot_height_map(ax_strict, strict, "InSAR-only height on building footprints\ncoherence >= 0.75", norm, cmap)

    if not relaxed_yes.empty:
        plot_height_map(ax_zoom, relaxed, "Local detail", norm, cmap)
        xmin, xmax, ymin, ymax = expanded_bounds(relaxed_yes, pad=45.0)
        ax_zoom.set_xlim(xmin, xmax)
        ax_zoom.set_ylim(ymin, ymax)

    ax_hist.hist(relaxed_yes["height_est_m"].dropna(), bins=28, color="#117733", alpha=0.82, edgecolor="white", label=f"coh>=0.5, n={len(relaxed_yes)}")
    ax_hist.hist(strict_yes["height_est_m"].dropna(), bins=18, color="#4477aa", alpha=0.58, edgecolor="white", label=f"coh>=0.75, n={len(strict_yes)}")
    ax_hist.set_xlabel("InSAR height estimate (m)")
    ax_hist.set_ylabel("Buildings")
    ax_hist.set_title("Height distribution", fontsize=9, pad=4)
    ax_hist.legend(loc="upper right", fontsize=7)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=[ax_relaxed, ax_strict, ax_zoom], fraction=0.028, pad=0.012)
    cbar.set_label("InSAR height estimate (m)")
    fig.legend(
        handles=[
            Patch(facecolor="#eeeeee", edgecolor="#d0d0d0", label="No InSAR solution"),
            Patch(facecolor=mpl.colormaps[cmap](0.68), edgecolor="#222222", label="Building with InSAR height"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.42, 0.012),
        fontsize=8,
    )
    fig.suptitle("Pure InSAR building-height estimates rendered on building vector footprints", fontsize=11, y=0.985)
    outputs = save_all(fig, Path(args.out_stem), args.dpi)
    plt.close(fig)

    summary = {
        "strict_input": args.strict,
        "relaxed_input": args.relaxed,
        "strict_insar_buildings": int(len(strict_yes)),
        "relaxed_insar_buildings": int(len(relaxed_yes)),
        "height_color_vmax_m": vmax,
        "outputs": outputs,
        "note": "Only InSAR-derived heights are colored. Buildings without an InSAR solution are grey; no DSM or shapefile fallback heights are drawn.",
    }
    summary_path = Path(f"{args.out_stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
