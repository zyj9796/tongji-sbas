#!/usr/bin/env python3
"""Draw Figure 10 from refined GAMMA roof projection and independent islands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amplitude", default="work/mli/20200708_rslc_amplitude.npy")
    p.add_argument("--mask-dir", default="work/roof_islands_sar_refined")
    p.add_argument("--output", default="picall/10_建筑孤岛全景.svg")
    p.add_argument("--local-output", default="picall/11_建筑孤岛局部放大.svg")
    p.add_argument("--summary", default="results/metadata/figure10_optimized_building_islands_summary.json")
    return p.parse_args()


def sar_display(amplitude: np.ndarray) -> np.ndarray:
    transformed = np.power(np.maximum(amplitude, 0.0), 0.70)
    valid = np.isfinite(transformed)
    lo, hi = np.nanpercentile(transformed[valid], [1.0, 99.4])
    return np.clip((transformed - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)


def best_zoom(mask: np.ndarray, rows: int = 145, cols: int = 190) -> tuple[int, int, int, int]:
    """Return the fixed-size window containing the densest selected roof pixels."""
    density = ndimage.uniform_filter(mask.astype(np.float32), size=(rows, cols), mode="constant")
    row, col = np.unravel_index(int(np.argmax(density)), density.shape)
    r0 = int(np.clip(row - rows // 2, 0, mask.shape[0] - rows))
    c0 = int(np.clip(col - cols // 2, 0, mask.shape[1] - cols))
    return r0, r0 + rows, c0, c0 + cols


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    amplitude = np.load(args.amplitude).astype(np.float32)
    owner = np.load(mask_dir / "roof_core_clean_id_mask.npy").astype(np.int32)
    labels = np.load(mask_dir / "roof_core_island_label.npy").astype(np.int32)
    full_roofs = np.load(mask_dir / "full_corrected_roof_union.npy").astype(bool)
    conflicts = np.load(mask_dir / "cross_building_support_conflict_mask.npy").astype(bool)
    reliable = np.load(mask_dir / "reliable_roof_points_mask.npy").astype(bool)
    islands = pd.read_csv(mask_dir / "roof_core_islands.csv")
    mask_summary = json.loads((mask_dir / "mask_summary.json").read_text(encoding="utf-8"))
    if amplitude.shape != owner.shape or owner.shape != labels.shape or labels.shape != full_roofs.shape:
        raise ValueError("幅度影像、投影与孤岛栅格尺寸不一致")

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    display = sar_display(amplitude)
    ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax.contour(full_roofs.astype(np.uint8), levels=[0.5], colors=["#F28E2B"], linewidths=0.25, alpha=0.48)
    selected = owner > 0
    ax.contour(selected.astype(np.uint8), levels=[0.5], colors=["#00D4D8"], linewidths=0.58, alpha=0.96)
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title("投影校正后的独立可靠屋顶孤岛", pad=6, fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#F28E2B", lw=1.0, label="校正后的完整屋顶投影"),
            Line2D([0], [0], color="#00D4D8", lw=1.4, label="独立可靠屋顶孤岛"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        fontsize=7,
    )
    ax.text(
        0.012,
        0.985,
        (
            f"保留建筑：{mask_summary['roof_buildings_in_scene']}栋  "
            f"孤岛：{mask_summary['roof_islands']}个\n"
            f"孤岛像元：{mask_summary['roof_core_pixels']:,}  "
            f"可靠证据：{mask_summary['reliable_roof_evidence_pixels']:,}\n"
            f"跨楼冲突剔除：{mask_summary['cross_building_support_conflict_pixels_removed']:,}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "#111111", "edgecolor": "#D0D4D8", "alpha": 0.78, "pad": 2.5},
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "投影校正后的独立可靠屋顶孤岛"})
    plt.close(fig)

    r0, r1, c0, c1 = best_zoom(selected)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax.contour(full_roofs.astype(np.uint8), levels=[0.5], colors=["#F28E2B"], linewidths=0.42, alpha=0.62)
    ax.contour(selected.astype(np.uint8), levels=[0.5], colors=["#00D4D8"], linewidths=0.86, alpha=0.98)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title("独立可靠屋顶孤岛局部放大", pad=6, fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#F28E2B", lw=1.0, label="调整后的完整屋顶投影"),
            Line2D([0], [0], color="#00D4D8", lw=1.4, label="反演前可靠屋顶孤岛"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        fontsize=7,
    )
    ax.text(
        0.014,
        0.982,
        "屋顶腐蚀后剔除跨楼重叠；孤岛选择不读取最终高度解",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "#111111", "edgecolor": "#D0D4D8", "alpha": 0.78, "pad": 2.3},
    )
    local_output = Path(args.local_output)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(local_output, format="svg", metadata={"Title": "独立可靠屋顶孤岛局部放大"})
    plt.close(fig)

    summary = {
        "figures": [str(output), str(local_output)],
        "source_amplitude": args.amplitude,
        "source_projection": mask_summary["projection"],
        "island_selection_uses_final_height_solution": False,
        "height_used_as_matching_target_or_fill": False,
        "amplitude_display": "single 20200708 RSLC magnitude; global amplitude^0.70 and 1%-99.4% stretch",
        "selected_buildings": int(owner[owner > 0].size > 0 and np.unique(owner[owner > 0]).size),
        "selected_islands": int(labels.max()),
        "selected_pixels": int(selected.sum()),
        "reliable_evidence_pixels": int(reliable.sum()),
        "island_table_rows": int(len(islands)),
        "cross_building_conflict_pixels_removed": int(conflicts.sum()),
        "selection_rules": mask_summary["roof_island_rules"],
        "local_window_row_col": [int(r0), int(r1), int(c0), int(c1)],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
