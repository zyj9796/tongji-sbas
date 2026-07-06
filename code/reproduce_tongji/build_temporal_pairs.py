#!/usr/bin/env python3
"""Build temporal candidate interferogram pairs from available RSLC dates.

The output is not the final SBAS network because perpendicular baselines still
must be computed from orbit/GAMMA products. It is the first candidate table that
applies the thesis temporal-baseline rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def available_dates(rslc_dir: Path) -> list[str]:
    return sorted(p.name.split(".")[0] for p in rslc_dir.glob("*.rslc.par"))


def build_pairs(dates: list[str], max_dt_days: int) -> list[dict[str, Any]]:
    out = []
    parsed = [(date, parse_date(date)) for date in dates]
    for i, (master, master_dt) in enumerate(parsed):
        for slave, slave_dt in parsed[i + 1 :]:
            dt_days = (slave_dt - master_dt).days
            if dt_days <= max_dt_days:
                out.append(
                    {
                        "master": master,
                        "slave": slave,
                        "dt_days": dt_days,
                        "bperp_m": "",
                        "height_ambiguity_m": "",
                        "status": "needs_bperp",
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["master", "slave", "dt_days", "bperp_m", "height_ambiguity_m", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--output", default="work/baselines/temporal_candidate_pairs.csv")
    parser.add_argument("--summary", default="results/metadata/temporal_candidate_pairs_summary.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    rslc_dir = Path(config["paths"]["rslc_dir"])
    max_dt_days = int(config["insar"]["max_temporal_baseline_days"])
    dates = available_dates(rslc_dir)
    rows = build_pairs(dates, max_dt_days)
    write_csv(Path(args.output), rows)
    summary = {
        "rslc_date_count": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "max_temporal_baseline_days": max_dt_days,
        "candidate_pair_count": len(rows),
        "output": args.output,
        "note": "Perpendicular baselines are not filled here; compute them with GAMMA/orbit products before final interferogram selection.",
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
