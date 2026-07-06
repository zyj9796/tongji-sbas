#!/usr/bin/env python3
"""Run the GAMMA interferogram chain for all pairs in a baseline CSV."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--out-root", default="work/gamma_sbas/intf")
    parser.add_argument("--hgt-rdc", default="-")
    parser.add_argument("--limit", type=int, default=0, help="0 means all pairs")
    parser.add_argument("--make-diff", action="store_true")
    parser.add_argument("--summary", default="results/metadata/gamma_sbas_interferograms_summary.json")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_csv)
    if args.limit > 0:
        pairs = pairs.head(args.limit)
    items = []
    for row in pairs.itertuples(index=False):
        master = str(row.master)
        slave = str(row.slave)
        cmd = [
            ".venv/bin/python",
            "code/reproduce_tongji/run_gamma_pair_interferogram.py",
            "--master",
            master,
            "--slave",
            slave,
            "--out-root",
            args.out_root,
        ]
        if args.make_diff:
            cmd.append("--make-diff")
            cmd.extend(["--hgt-rdc", args.hgt_rdc])
        proc = subprocess.run(cmd, text=True, capture_output=True)
        item = {"master": master, "slave": slave, "returncode": proc.returncode}
        if proc.returncode != 0:
            item["stderr"] = proc.stderr[-2000:]
            items.append(item)
            break
        items.append(item)
    summary = {
        "pairs_requested": int(len(pairs)),
        "pairs_completed": int(sum(1 for item in items if item["returncode"] == 0)),
        "make_diff": bool(args.make_diff),
        "out_root": args.out_root,
        "hgt_rdc": args.hgt_rdc if args.make_diff else None,
        "items": items,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(item["returncode"] != 0 for item in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
