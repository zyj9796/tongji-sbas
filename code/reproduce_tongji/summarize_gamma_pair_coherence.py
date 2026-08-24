#!/usr/bin/env python3
"""Summarize GAMMA ADF coherence for a pair table without using heights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--interferogram-root", required=True)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--lines", type=int, default=630)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs, dtype={"master": str, "slave": str})
    rows: list[dict[str, object]] = []
    expected = args.width * args.lines
    for pair in pairs.itertuples(index=False):
        master, slave = str(pair.master), str(pair.slave)
        name = f"{master}_{slave}"
        path = Path(args.interferogram_root) / name / f"{name}.cc"
        values = np.fromfile(path, dtype=">f4").astype(float)
        if values.size != expected:
            raise ValueError(f"{path}: {values.size} samples, expected {expected}")
        valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
        used = values[valid]
        rows.append(
            {
                "master": master,
                "slave": slave,
                "mean_cc": float(np.mean(used)),
                "median_cc": float(np.median(used)),
                "p25_cc": float(np.percentile(used, 25)),
                "p75_cc": float(np.percentile(used, 75)),
                "fraction_cc_ge_055": float(np.mean(used >= 0.55)),
                "valid_fraction": float(np.mean(valid)),
            }
        )
    output = pd.DataFrame(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.describe(include="all").to_string())


if __name__ == "__main__":
    main()
