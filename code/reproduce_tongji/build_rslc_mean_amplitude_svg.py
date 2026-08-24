#!/usr/bin/env python3
"""Build Figure 02 from one full-precision coregistered RSLC acquisition.

The reference acquisition is read directly as GAMMA SCOMPLEX. No temporal
averaging, BMP input, spatial filtering, resampling, multilooking,
interpolation, or local histogram equalization is used. A global amplitude
power transform matching the provided GAMMA-style reference display is applied
to the native 630 x 900 samples.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rslc-dir", default="data/tongji_rslc")
    p.add_argument("--reference-date", default="20200708")
    p.add_argument("--output-array", default="work/mli/20200708_rslc_amplitude.npy")
    p.add_argument("--summary", default="results/metadata/20200708_single_rslc_amplitude_summary.json")
    p.add_argument("--figure", default="picall/02_同济雷达单景幅度.svg")
    return p.parse_args()


def par_value(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(\S+)", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise ValueError(f"参数文件缺少 {key}: {path}")
    return match.group(1)


def read_scomplex(path: Path, rows: int, cols: int) -> np.ndarray:
    expected = rows * cols * 2
    raw = np.fromfile(path, dtype=">i2")
    if raw.size != expected:
        raise ValueError(f"RSLC尺寸不匹配: {path}, {raw.size} != {expected}")
    complex_pairs = raw.reshape(rows, cols, 2).astype(np.float32)
    return np.hypot(complex_pairs[..., 0], complex_pairs[..., 1]).astype(np.float32)


def display_image(amplitude: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    # The supplied reference BMP was traced back to its colocated full-precision
    # RSLC. Across unsaturated pixels its gray level is reproduced by a global
    # amplitude power of 0.70 (sub-gray-level fitting error), followed by the
    # same 1%-99.4% stretch used by the reference plotting script.
    transformed = np.power(np.maximum(amplitude, 0.0), 0.70)
    valid = np.isfinite(transformed)
    if not np.any(valid):
        raise ValueError("RSLC幅度没有有效像元")
    lo, hi = np.nanpercentile(transformed[valid], [1.0, 99.4])
    visual = np.clip((transformed - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0).astype(np.float32)
    return visual, {
        "amplitude_power": 0.70,
        "display_percentile_low": 1.0,
        "display_percentile_high": 99.4,
        "spatial_filter": "none",
    }


def draw(visual: np.ndarray, output: Path) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    ax.imshow(visual, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.annotate(
        "距离向",
        xy=(0.92, 0.95),
        xytext=(0.68, 0.95),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#2A9D8F", "lw": 1.4},
        color="#2A9D8F",
        va="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "同济雷达单景幅度（20200708）"})
    plt.close(fig)


def main() -> None:
    args = parse_args()
    directory = Path(args.rslc_dir)
    source_path = directory / f"{args.reference_date}.rslc"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    reference_par = directory / f"{args.reference_date}.rslc.par"
    rows = int(par_value(reference_par, "azimuth_lines"))
    cols = int(par_value(reference_par, "range_samples"))
    if par_value(reference_par, "image_format") != "SCOMPLEX":
        raise ValueError("参考RSLC不是SCOMPLEX")

    amplitude = read_scomplex(source_path, rows, cols)
    output_array = Path(args.output_array)
    output_array.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_array, amplitude)
    visual, display_parameters = display_image(amplitude)
    draw(visual, Path(args.figure))

    summary = {
        "method": "single full-precision SCOMPLEX acquisition -> complex magnitude -> global amplitude^0.70 display -> percentile stretch",
        "source": str(source_path),
        "source_date": args.reference_date,
        "source_count": 1,
        "shape_rows_cols": [rows, cols],
        "input_dtype": "big-endian signed int16 real/imaginary pairs",
        "temporal_aggregation": False,
        "spatial_filtering": False,
        "spatial_resampling": False,
        "multilooking": False,
        "bmp_or_8bit_input_used": False,
        "display_parameters": display_parameters,
        "output_array": args.output_array,
        "figure_svg": args.figure,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
