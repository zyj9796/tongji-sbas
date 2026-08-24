#!/usr/bin/env python3
"""Build Nature-style workflow figures and a multipage PDF report."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


BASE = Path("/home/u/geocoding/tongji_sbas")
OUT = BASE / "results/pic_all/nature_workflow_20260707"
SVG = OUT / "svg"
PNG = OUT / "png"
PDF = OUT / "tongji_insar_building_height_workflow_report_20260707.pdf"

COL = {
    "ink": "#1f2933",
    "muted": "#667085",
    "grid": "#d7dde4",
    "blue": "#3b6ea8",
    "cyan": "#46a6a6",
    "green": "#4f9d69",
    "orange": "#d9822b",
    "red": "#c2413b",
    "purple": "#7b61a8",
    "gray": "#b8c0cc",
    "light": "#f4f6f8",
}


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
            "axes.linewidth": 0.65,
            "axes.labelcolor": COL["ink"],
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_fig(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    SVG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    svg = SVG / f"{name}.svg"
    png = PNG / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    return svg, png


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom", ha="left")


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=8.5, weight="bold", color=COL["ink"], pad=5)


def wrap(text: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def robust_image(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    lo, hi = np.nanpercentile(vals, [2, 98])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def load_data() -> dict[str, object]:
    data: dict[str, object] = {}
    data["amp"] = np.load(BASE / "work/mli/mean_crop_bmp_amplitude.npy").astype(np.float32)
    data["da"] = np.load(BASE / "work/mli/amplitude_dispersion_crop_bmp.npy").astype(np.float32)
    data["fid_mask"] = np.load(BASE / "work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy")
    data["islands"] = np.load(BASE / "work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy")
    data["projection"] = gpd.read_file(BASE / "work/projection/20200708_clean_equal_height_roof_projection_sar.geojson")
    data["footprints"] = gpd.read_file(BASE / "data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.geojson")
    data["height"] = gpd.read_file(
        BASE
        / "results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_p95_floor_reestimated_insar_only.geojson"
    )
    data["topdown"] = pd.read_csv(
        BASE
        / "results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv"
    )
    data["p95floor"] = pd.read_csv(
        BASE
        / "results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_p95_floor_reestimated_insar_only.csv"
    )
    data["points"] = pd.read_csv(
        BASE / "work/height/height_points_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv"
    )
    data["island_lgr"] = pd.read_csv(
        BASE / "work/height/island_pixel_lgr_heights_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv"
    )
    data["shift"] = pd.read_csv(BASE / "work/projection/cleanid_split_red_building_mask_shift_metrics_audited.csv")
    data["baseline"] = pd.read_csv(BASE / "work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    data["review"] = pd.read_csv(BASE / "docs/review_building_audit_20260707.csv")
    data["nosol"] = pd.read_csv(BASE / "docs/no_solution_failure_audit_20260707.csv")
    data["gate"] = pd.read_csv(BASE / "docs/no_solution_gate_sensitivity_buildings_20260707.csv")
    data["minpairs_cmp"] = pd.read_csv(BASE / "docs/minpairs10_vs_current_topdown_comparison_20260707.csv")
    data["rmse_cmp"] = pd.read_csv(BASE / "docs/rmse150_vs_current_topdown_comparison_20260707.csv")
    data["delta"] = pd.read_csv(BASE / "results/tables/tongji_building_height_p95_floor_reestimate_vs_topdown_grubbs_diff.csv")
    data["variant_decision"] = pd.read_csv(BASE / "docs/threshold_variant_added_building_decision_table_20260707.csv")
    return data


def fig_01_pipeline(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_axis_off()
    steps = [
        ("Clean building\nfootprints", "1028 buildings"),
        ("SAR roof\nprojection", "1026 projected"),
        ("Audited roof\nmask", "16 shifts accepted"),
        ("Clean-ID\nislands", "662 islands"),
        ("Paper-like\nunwrapping", "local island phase"),
        ("SBAS/LGR\nresidual", "coh/DA/pairs/RMSE"),
        ("P95-floor\nroof top", "239 solved"),
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, ((name, note), x) in enumerate(zip(steps, xs)):
        y = 0.56 + 0.13 * math.sin(i / 1.5)
        rect = Rectangle((x - 0.062, y - 0.095), 0.124, 0.19, facecolor="white", edgecolor=COL["blue"], linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x, y + 0.025, name, ha="center", va="center", fontsize=7.3, weight="bold", color=COL["ink"])
        ax.text(x, y - 0.055, note, ha="center", va="center", fontsize=6.3, color=COL["muted"])
        if i < len(steps) - 1:
            x2 = xs[i + 1]
            y2 = 0.56 + 0.13 * math.sin((i + 1) / 1.5)
            ax.add_patch(FancyArrowPatch((x + 0.065, y), (x2 - 0.065, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=COL["muted"]))
    ax.text(0.02, 0.93, "a", fontsize=10, weight="bold")
    ax.text(0.06, 0.9, "Building-constrained InSAR height reconstruction workflow", fontsize=10, weight="bold", color=COL["ink"])
    ax.text(
        0.06,
        0.14,
        wrap(
            "The workflow keeps building footprints in the InSAR loop: footprints define roof-only radar masks, islands are split by clean_id, phase is unwrapped locally, DEM residuals are inverted by SBAS/LGR, and final roof-top heights are selected without using the vector height attribute.",
            120,
        ),
        fontsize=7,
        color=COL["muted"],
    )
    return fig, "01_workflow_overview", "Claim: the height product is a building-constrained InSAR chain, not a post-hoc vector fill."


def fig_02_projection(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    amp = robust_image(data["amp"])  # type: ignore[arg-type]
    projection: gpd.GeoDataFrame = data["projection"]  # type: ignore[assignment]
    footprints: gpd.GeoDataFrame = data["footprints"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    footprints.boundary.plot(ax=axes[0], color=COL["ink"], linewidth=0.25)
    axes[0].set_aspect("equal")
    title(axes[0], "Ground footprints")
    panel_label(axes[0], "a")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].tick_params(labelsize=6)
    axes[1].imshow(amp, cmap="gray", origin="upper")
    projection.boundary.plot(ax=axes[1], color=COL["cyan"], linewidth=0.25, alpha=0.8)
    axes[1].set_xlim(0, amp.shape[1])
    axes[1].set_ylim(amp.shape[0], 0)
    title(axes[1], "Projected roof polygons in SAR coordinates")
    panel_label(axes[1], "b")
    axes[1].set_xlabel("Range pixel")
    axes[1].set_ylabel("Azimuth pixel")
    axes[1].tick_params(labelsize=6)
    fig.suptitle("Projection transfers vector roof priors into radar geometry", y=1.02, fontsize=10, weight="bold")
    return fig, "02_projection_to_sar", "Ground footprints are mapped into radar coordinates before phase processing."


def fig_03_masks(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    amp = robust_image(data["amp"])  # type: ignore[arg-type]
    fid = data["fid_mask"]  # type: ignore[assignment]
    islands = data["islands"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    axes[0].imshow(amp, cmap="gray", origin="upper")
    title(axes[0], "Mean SAR amplitude")
    axes[1].imshow(amp, cmap="gray", origin="upper")
    axes[1].imshow(np.ma.masked_where(fid <= 0, fid > 0), cmap=ListedColormap([COL["cyan"]]), alpha=0.72, origin="upper")
    title(axes[1], "Audited roof mask")
    axes[2].imshow(np.ma.masked_where(islands <= 0, islands % 20), cmap="tab20", origin="upper", interpolation="nearest")
    title(axes[2], "Clean-ID split islands")
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + i))
        ax.set_axis_off()
    fig.suptitle("Mask construction isolates roof pixels and independent phase islands", y=1.02, fontsize=10, weight="bold")
    return fig, "03_mask_and_islands", "Roof-only masks and clean-ID islands constrain unwrapping to independent building targets."


def fig_04_redshift(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    shift: pd.DataFrame = data["shift"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    colors = np.where(shift["accepted"].astype(bool), COL["green"], COL["gray"])
    axes[0].scatter(shift["score_gain"], shift["contrast_gain"], c=colors, s=22, edgecolor="white", linewidth=0.35)
    axes[0].axvline(0, color=COL["grid"], lw=0.8)
    axes[0].axhline(0, color=COL["grid"], lw=0.8)
    axes[0].set_xlabel("SAR score gain")
    axes[0].set_ylabel("Contrast gain")
    title(axes[0], "Audited projection-shift candidates")
    panel_label(axes[0], "a")
    acc = shift[shift["accepted"].astype(bool)]
    rej = shift[~shift["accepted"].astype(bool)]
    axes[1].scatter(rej["col_shift"], -rej["row_shift"], s=18, color=COL["gray"], alpha=0.55, label="rejected")
    axes[1].scatter(acc["col_shift"], -acc["row_shift"], s=30, color=COL["green"], edgecolor="white", lw=0.4, label="accepted")
    axes[1].axhline(0, color=COL["grid"], lw=0.8)
    axes[1].axvline(0, color=COL["grid"], lw=0.8)
    axes[1].set_xlabel("Range shift (px)")
    axes[1].set_ylabel("Azimuth shift (px)")
    title(axes[1], "Accepted shifts are sparse and conservative")
    panel_label(axes[1], "b")
    axes[1].legend(loc="upper right", fontsize=6)
    fig.suptitle("Local projection correction is accepted only when internal SAR evidence improves", y=1.03, fontsize=10, weight="bold")
    return fig, "04_audited_projection_shift", "Projection correction is deliberately conservative: 16 of 81 candidates were accepted."


def fig_05_baseline(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    pairs: pd.DataFrame = data["baseline"]  # type: ignore[assignment]
    dates = sorted(set(pairs["master"].astype(str)) | set(pairs["slave"].astype(str)))
    pos = {d: i for i, d in enumerate(dates)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for row in pairs.itertuples(index=False):
        x0, x1 = pos[str(row.master)], pos[str(row.slave)]
        y0, y1 = float(row.bperp_m), float(row.bperp_m)
        axes[0].plot([x0, x1], [y0, y1], color=COL["blue"], alpha=0.55, lw=1.0)
        axes[0].scatter([x0, x1], [y0, y1], color=COL["ink"], s=7, zorder=3)
    axes[0].axhline(0, color=COL["grid"], lw=0.8)
    axes[0].set_xlabel("Acquisition order")
    axes[0].set_ylabel("Perpendicular baseline (m)")
    title(axes[0], "SBAS pair network")
    panel_label(axes[0], "a")
    axes[1].hist(pairs["dt_days"], bins=np.arange(0, 80, 10), color=COL["cyan"], edgecolor="white")
    axes[1].set_xlabel("Temporal baseline (days)")
    axes[1].set_ylabel("Pairs")
    title(axes[1], "Temporal support")
    panel_label(axes[1], "b")
    fig.suptitle("Baseline diversity supports DEM-residual estimation", y=1.03, fontsize=10, weight="bold")
    return fig, "05_sbas_baseline_network", "The SBAS network spans temporal and perpendicular baselines needed for LGR inversion."


def fig_06_lgr_quality(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    pts: pd.DataFrame = data["points"]  # type: ignore[assignment]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    panels = [
        ("coh_mean", "Coherence", COL["blue"], 0.75),
        ("amplitude_dispersion", "Amplitude dispersion", COL["orange"], 0.40),
        ("lgr_rmse_rad", "LGR RMSE (rad)", COL["red"], 1.25),
        ("valid_pairs_median", "Valid pairs", COL["green"], 12),
    ]
    for ax, (col, lab, color, gate) in zip(axes.ravel(), panels):
        vals = pd.to_numeric(pts[col], errors="coerce").dropna()
        ax.hist(vals, bins=24, color=color, alpha=0.82, edgecolor="white")
        ax.axvline(gate, color=COL["ink"], ls="--", lw=0.9)
        ax.set_xlabel(lab)
        ax.set_ylabel("Buildings")
        title(ax, f"{lab} after strict gating")
    for i, ax in enumerate(axes.ravel()):
        panel_label(ax, chr(ord("a") + i))
    fig.suptitle("LGR height points satisfy strict interferometric quality gates", y=1.02, fontsize=10, weight="bold")
    return fig, "06_lgr_quality_gates", "Solved height points have high coherence, low DA, low residual RMSE and sufficient valid pairs."


def fig_07_height_map(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    gdf: gpd.GeoDataFrame = data["height"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    solved = gdf[gdf["has_likely_top_height"].astype(bool)].copy()
    nosol = gdf[~gdf["has_likely_top_height"].astype(bool)].copy()
    if not nosol.empty:
        nosol.plot(ax=ax, color="#edf0f2", edgecolor="#ccd3da", linewidth=0.12)
    vmax = float(np.nanpercentile(solved["height_likely_top_m"], 95)) if not solved.empty else 50
    norm = Normalize(vmin=0, vmax=max(vmax, 20))
    solved.plot(ax=ax, column="height_likely_top_m", cmap="viridis", norm=norm, edgecolor="#26313c", linewidth=0.12)
    review = solved[solved["likely_top_reliability"].eq("review")]
    if not review.empty:
        review.boundary.plot(ax=ax, color=COL["red"], linewidth=0.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, cax=cax, label="Height (m)")
    ax.set_axis_off()
    title(ax, "P95-floor re-estimated building height")
    panel_label(ax, "a")
    fig.suptitle("Final InSAR-only height product retains unsolved buildings as empty", y=1.01, fontsize=10, weight="bold")
    return fig, "07_final_height_map", "The final map shows 239 solved buildings and keeps no-solution buildings unfilled."


def fig_08_top_reestimate(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    delta: pd.DataFrame = data["delta"]  # type: ignore[assignment]
    solved = delta.dropna(subset=["height_delta_m"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].hist(solved["height_delta_m"], bins=np.r_[np.linspace(-0.1, 0.1, 3), np.linspace(0.2, 65, 28)], color=COL["purple"], edgecolor="white")
    axes[0].set_xlabel("P95-floor minus top-down Grubbs (m)")
    axes[0].set_ylabel("Buildings")
    title(axes[0], "Re-estimation raises only a subset")
    panel_label(axes[0], "a")
    top = solved.sort_values("height_delta_m", ascending=False).head(15)
    axes[1].barh(top["clean_id"].astype(str), top["height_delta_m"], color=COL["purple"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Height increase (m)")
    axes[1].set_ylabel("clean_id")
    title(axes[1], "Largest low-bias corrections")
    panel_label(axes[1], "b")
    fig.suptitle("P95 floor mitigates low bias from over-removing upper roof pixels", y=1.03, fontsize=10, weight="bold")
    return fig, "08_p95_floor_reestimate", "Twenty-nine solved buildings are raised by the P95-floor top selection rule."


def fig_09_review_audit(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    rev: pd.DataFrame = data["review"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    counts = rev["primary_reason"].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color=COL["blue"])
    axes[0].set_xlabel("Buildings")
    title(axes[0], "Review cause classification")
    panel_label(axes[0], "a")
    axes[1].scatter(rev["grubbs_top_removed_count"], rev["top_tail_gap_m"], c=rev["lgr_rmse_rad"], cmap="magma", s=32, edgecolor="white", lw=0.35)
    axes[1].set_xlabel("Top pixels removed")
    axes[1].set_ylabel("Max - retained top (m)")
    title(axes[1], "Upper-tail instability")
    panel_label(axes[1], "b")
    fig.suptitle("Review buildings are dominated by unstable upper-tail height evidence", y=1.03, fontsize=10, weight="bold")
    return fig, "09_review_building_audit", "The 29 review buildings are not low-quality globally; they have unstable top-height tails."


def fig_10_nosolution(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    nosol: pd.DataFrame = data["nosol"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    counts = nosol["refined_failure_class"].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color=COL["orange"])
    axes[0].set_xlabel("Buildings")
    title(axes[0], "No-solution failure classes")
    panel_label(axes[0], "a")
    lgr = nosol[nosol["refined_failure_class"].str.startswith("lgr_", na=False)].copy()
    axes[1].scatter(lgr["max_valid_pairs_pass_pixels"], lgr["max_rmse_pass_pixels"], s=16, color=COL["orange"], alpha=0.65)
    axes[1].axhline(20, color=COL["ink"], ls="--", lw=0.8)
    axes[1].axvline(20, color=COL["ink"], ls="--", lw=0.8)
    axes[1].set_xlabel("Pixels passing min pairs")
    axes[1].set_ylabel("Pixels passing RMSE")
    title(axes[1], "LGR gate bottlenecks")
    panel_label(axes[1], "b")
    fig.suptitle("No-solution buildings split into projection, valid-pair and RMSE bottlenecks", y=1.03, fontsize=10, weight="bold")
    return fig, "10_no_solution_failure_modes", "Most unsolved buildings lack SAR roof islands or sufficient valid-pair support."


def fig_11_sensitivity(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    gate: pd.DataFrame = data["gate"]  # type: ignore[assignment]
    mp: pd.DataFrame = data["minpairs_cmp"]  # type: ignore[assignment]
    rm: pd.DataFrame = data["rmse_cmp"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))
    g = gate.groupby("variant")["reaches_min_pixels"].sum().reindex(["min_pairs10", "rmse150", "da045"])
    axes[0].bar(g.index, g.values, color=[COL["blue"], COL["orange"], COL["gray"]])
    axes[0].set_ylabel("Buildings reaching 20 pixels")
    axes[0].tick_params(axis="x", rotation=25)
    title(axes[0], "Gate pretest yield")
    panel_label(axes[0], "a")
    axes[1].bar(["current", "min_pairs10", "rmse150"], [239, int(mp["minpairs10_has"].sum()), int(rm["rmse150_has"].sum())], color=[COL["gray"], COL["blue"], COL["orange"]])
    axes[1].set_ylabel("Solved buildings")
    title(axes[1], "Product-level yield")
    panel_label(axes[1], "b")
    axes[2].bar(["current", "min_pairs10", "rmse150"], [29, int((mp["minpairs10_likely_top_reliability"] == "review").sum()), int((rm["rmse150_likely_top_reliability"] == "review").sum())], color=[COL["gray"], COL["blue"], COL["orange"]])
    axes[2].set_ylabel("Review buildings")
    title(axes[2], "Review burden")
    panel_label(axes[2], "c")
    fig.suptitle("Threshold relaxation improves coverage but changes review burden", y=1.03, fontsize=10, weight="bold")
    return fig, "11_threshold_sensitivity", "min_pairs=10 gives the highest coverage; RMSE=1.50 is smaller and model-fit sensitive."


def fig_12_decision(data: dict[str, object]) -> tuple[plt.Figure, str, str]:
    dec: pd.DataFrame = data["variant_decision"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    groups = [
        ("min_pairs10", int(dec["added_by_minpairs10"].sum())),
        ("rmse150", int(dec["added_by_rmse150"].sum())),
        ("overlap", int((dec["added_by_minpairs10"] & dec["added_by_rmse150"]).sum())),
    ]
    axes[0].bar([g[0] for g in groups], [g[1] for g in groups], color=[COL["blue"], COL["orange"], COL["green"]])
    axes[0].set_ylabel("Added buildings")
    title(axes[0], "Added-building overlap")
    panel_label(axes[0], "a")
    x = np.arange(len(dec))
    axes[1].scatter(dec["minpairs10_height_m"], dec["rmse150_height_m"], s=22, color=COL["purple"], alpha=0.75)
    axes[1].plot([0, 80], [0, 80], color=COL["grid"], ls="--", lw=0.8)
    axes[1].set_xlabel("min_pairs10 height (m)")
    axes[1].set_ylabel("RMSE=1.50 height (m)")
    title(axes[1], "Variant-added height agreement")
    panel_label(axes[1], "b")
    fig.suptitle("Strict product remains primary; threshold branches require targeted audit", y=1.03, fontsize=10, weight="bold")
    return fig, "12_threshold_variant_decision", "The higher-yield min_pairs=10 branch is a candidate, not a replacement for the strict product."


def add_report_page(pdf: PdfPages, fig_png: Path, title_text: str, caption: str, page_no: int) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(10, 1, height_ratios=[0.6, 0.3, 7.2, 0.4, 1.2, 0.3, 0.2, 0.2, 0.2, 0.2])
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_axis_off()
    ax_title.text(0.02, 0.5, title_text, fontsize=13, weight="bold", color=COL["ink"], va="center")
    ax_img = fig.add_subplot(gs[2])
    ax_img.set_axis_off()
    img = plt.imread(fig_png)
    ax_img.imshow(img)
    ax_cap = fig.add_subplot(gs[4])
    ax_cap.set_axis_off()
    ax_cap.text(0.02, 0.85, "Interpretation", fontsize=8.5, weight="bold", color=COL["ink"], va="top")
    ax_cap.text(0.02, 0.62, wrap(caption, 110), fontsize=7.5, color=COL["muted"], va="top", linespacing=1.35)
    fig.text(0.5, 0.025, f"{page_no}", ha="center", va="center", fontsize=7, color=COL["muted"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report(saved: list[tuple[str, Path, str]]) -> None:
    with PdfPages(PDF) as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        ax = cover.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.08, 0.78, "Building-height reconstruction from footprint-constrained InSAR", fontsize=19, weight="bold", color=COL["ink"], wrap=True)
        ax.text(0.08, 0.68, "Tongji SBAS/LGR workflow report", fontsize=12, color=COL["blue"], weight="bold")
        ax.text(
            0.08,
            0.56,
            wrap(
                "This report visualizes the full processing chain from building-footprint projection to SAR roof masks, island-local unwrapping, LGR quality gating, final P95-floor roof-top height estimation, failure-mode audits and threshold-sensitivity tests.",
                88,
            ),
            fontsize=9,
            color=COL["muted"],
            linespacing=1.45,
        )
        ax.text(0.08, 0.18, "All figures are generated from current project data. The building-vector height attribute is not used for fitting, filtering, calibration, selection or QC.", fontsize=8, color=COL["muted"], wrap=True)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        for idx, (title_text, png, caption) in enumerate(saved, start=2):
            add_report_page(pdf, png, title_text, caption, idx)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    makers = [
        fig_01_pipeline,
        fig_02_projection,
        fig_03_masks,
        fig_04_redshift,
        fig_05_baseline,
        fig_06_lgr_quality,
        fig_07_height_map,
        fig_08_top_reestimate,
        fig_09_review_audit,
        fig_10_nosolution,
        fig_11_sensitivity,
        fig_12_decision,
    ]
    saved: list[tuple[str, Path, str]] = []
    outputs = []
    for maker in makers:
        fig, name, caption = maker(data)
        svg, png = save_fig(fig, name)
        outputs.append({"name": name, "svg": str(svg.relative_to(BASE)), "png": str(png.relative_to(BASE)), "caption": caption})
        saved.append((name.replace("_", " ").title(), png, caption))
        plt.close(fig)
    build_report(saved)
    summary = {
        "date": "2026-07-07",
        "figure_count": len(outputs),
        "core_conclusion": "Footprint-constrained InSAR produces a strict 239-building height product; low-bias correction is handled at final top-selection, while remaining failures are dominated by projection and valid-pair support.",
        "backend": "python_matplotlib",
        "height_field_use": "not_read_for_fitting_filtering_calibration_selection_or_qc",
        "pdf": str(PDF.relative_to(BASE)),
        "figures": outputs,
    }
    (OUT / "nature_workflow_report_manifest_20260707.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
