#!/usr/bin/env python3
"""Build a mean SAR amplitude proxy from available cropped BMP quicklooks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import numpy as np
from PIL import Image


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--output-png", default="work/mli/mean_crop_bmp_amplitude.png")
    parser.add_argument("--output-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--summary", default="results/metadata/mean_amplitude_summary.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    rslc_dir = Path(config["paths"]["rslc_dir"])
    bmps = sorted(rslc_dir.glob("*.crop.bmp"))
    if not bmps:
        raise FileNotFoundError(f"No *.crop.bmp files found in {rslc_dir}")

    acc = None
    shape = None
    used = []
    skipped = []
    for bmp in bmps:
        arr = read_gray(bmp)
        if shape is None:
            shape = arr.shape
            acc = np.zeros(shape, dtype=np.float64)
        if arr.shape != shape:
            skipped.append({"path": str(bmp), "shape": list(arr.shape), "reason": "shape_mismatch"})
            continue
        acc += arr
        used.append(str(bmp))
    if acc is None or not used:
        raise RuntimeError("No compatible BMP images were available")

    mean = (acc / len(used)).astype(np.float32)
    p2, p98 = np.percentile(mean, [2, 98])
    scaled = np.clip((mean - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)
    out_png = Path(args.output_png)
    out_npy = Path(args.output_npy)
    summary_path = Path(args.summary)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255).astype(np.uint8)).save(out_png)
    np.save(out_npy, mean)
    summary = {
        "input_dir": str(rslc_dir),
        "bmp_count": len(bmps),
        "used_count": len(used),
        "skipped_count": len(skipped),
        "shape_rows_cols": list(shape) if shape else None,
        "output_png": str(out_png),
        "output_npy": str(out_npy),
        "p2": float(p2),
        "p98": float(p98),
        "note": "This is a quicklook mean amplitude proxy from crop BMPs, not a calibrated GAMMA MLI.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
