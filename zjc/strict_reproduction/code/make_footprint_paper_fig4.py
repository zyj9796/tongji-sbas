#!/usr/bin/env python3
"""Reproduce Fig. 4: real building-support overlap and overlap exclusion.

Figure contract
---------------
Conclusion: intersecting radar-coordinate building supports are identified and
removed before independent unwrapping, leaving mutually disjoint phase domains.
Evidence: two real overlap cases, each shown as amplitude / original supports /
overlap-excluded supports.  Export: one Chinese SVG, editable labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


WIDTH, LINES = 10_000, 7_000
COLORS = ["#ef5b5b", "#45d46b", "#3f73d8"]

mpl.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.sans-serif": ["Noto Sans CJK SC"],
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "font.size": 7,
})


def read_case(points: np.lib.npyio.NpzFile, uids: tuple[int, ...], pad: int = 14):
    row = points["row"].astype(np.int32)
    col = points["col"].astype(np.int32)
    uid = points["building_uid"].astype(np.int64)
    member = np.isin(uid, uids)
    if not np.any(member):
        raise ValueError(f"no pixels for {uids}")
    r0, r1 = int(row[member].min()) - pad, int(row[member].max()) + pad + 1
    c0, c1 = int(col[member].min()) - pad, int(col[member].max()) + pad + 1
    masks = np.zeros((len(uids), r1-r0, c1-c0), dtype=bool)
    for i, building in enumerate(uids):
        m = uid == building
        masks[i, row[m]-r0, col[m]-c0] = True
    overlap = masks.sum(axis=0) > 1
    separated = masks & ~overlap[None, :, :]
    return (c0, c1, r0, r1), masks, overlap, separated


def enhance_amplitude(raw: np.memmap, crop: tuple[int, int, int, int]) -> np.ndarray:
    c0, c1, r0, r1 = crop
    z = np.log1p(np.asarray(raw[r0:r1, c0:c1], dtype=np.float32))
    ok = np.isfinite(z) & (z > 0)
    lo, hi = np.quantile(z[ok], (0.01, 0.995))
    return np.power(np.clip((z-lo)/(hi-lo), 0, 1), 0.72, dtype=np.float32)


def rgba_masks(masks: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    h, w = masks.shape[1:]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for mask, color in zip(masks, COLORS[:len(masks)]):
        rgb = mpl.colors.to_rgb(color)
        # Sequential alpha compositing makes overlaps visible before hatching.
        a = alpha * mask.astype(np.float32)
        rgba[..., :3] = rgba[..., :3] * (1-a[..., None]) + np.asarray(rgb) * a[..., None]
        rgba[..., 3] = np.maximum(rgba[..., 3], a)
    return rgba


def draw_case(axs, raw, points, uids, case_name):
    crop, masks, overlap, separated = read_case(points, uids)
    amp = enhance_amplitude(raw, crop)
    extent = (crop[0], crop[1], crop[3], crop[2])
    for ax in axs:
        ax.imshow(amp, cmap="gray", vmin=0, vmax=1, extent=extent,
                  interpolation="nearest", rasterized=True)
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[3], crop[2]); ax.set_aspect("auto"); ax.set_axis_off()
    axs[1].imshow(rgba_masks(masks), extent=extent, interpolation="nearest", rasterized=True)
    yy, xx = np.mgrid[crop[2]:crop[3], crop[0]:crop[1]]
    if overlap.any():
        axs[1].contourf(xx, yy, overlap.astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["////"], zorder=8)
        axs[1].contour(xx, yy, overlap.astype(float), levels=[0.5], colors="#111111",
                       linewidths=0.65, zorder=9)
    axs[2].imshow(rgba_masks(separated), extent=extent, interpolation="nearest", rasterized=True)
    for ax in axs:
        ax.set_aspect("auto")
    for row, label in enumerate(("原始雷达幅度", "投影支持域与重叠区", "剔除重叠后的独立支持域")):
        axs[row].text(0.018, 0.94, label, transform=axs[row].transAxes, ha="left", va="top",
                      color="white", fontsize=6.3,
                      bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.66, "pad": 1.1})
    axs[0].text(0.98, 0.94, case_name, transform=axs[0].transAxes, ha="right", va="top",
                color="white", fontsize=6.3,
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.66, "pad": 1.1})
    return int(overlap.sum()), [int(m.sum()) for m in masks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("zjc/strict_reproduction"))
    args = parser.parse_args()
    points = np.load(args.root / "work/islands/independent_expanded_search_points.npz")
    raw = np.memmap(args.root / "work/amplitude/20231007.mli", dtype=">f4", mode="r",
                    shape=(LINES, WIDTH))
    # Two real high-overlap groups selected only from duplicated radar pixels.
    # The left example has three partly overlapping supports; the right has
    # two.  Their overlap ratios (relative to the smallest support) are 0.57
    # and 0.27, close to the moderate partial overlaps illustrated in Fig. 4.
    cases = [((67652, 67653, 67654), "案例一"), ((94855, 97340), "案例二")]
    fig, axes = plt.subplots(3, 2, figsize=(18.3/2.54, 8.2/2.54),
                             gridspec_kw={"wspace": 0.018, "hspace": 0.025})
    audits = []
    for j, (uids, name) in enumerate(cases):
        overlap, counts = draw_case(axes[:, j], raw, points, uids, name)
        audits.append((uids, overlap, counts))
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.01, top=0.99)
    out = args.root / "results/footprint_constrained_paper/图4_建筑投影重叠掩膜示例.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"输出: {out}")
    for uids, overlap, counts in audits:
        print(f"建筑 {uids}: 支持域像元 {counts}, 重叠像元 {overlap}")


if __name__ == "__main__":
    main()
