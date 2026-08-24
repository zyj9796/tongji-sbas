#!/usr/bin/env python3
"""Consolidate sharded GAMMA run summaries and enforce complete pair coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--sort-key", default="pair_index")
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise TypeError(f"summary is not a list: {path}")
        rows.extend(value)
    deduplicated = {}
    for row in rows:
        identity = row.get("pair_name")
        if not identity:
            raise KeyError("summary row has no pair_name")
        deduplicated[identity] = row
    rows = list(deduplicated.values())
    key = args.sort_key
    if key == "pair_name":
        rows.sort(key=lambda x: x["pair_name"])
    else:
        rows.sort(key=lambda x: (x.get(key, 10**9), x["pair_name"]))
    if len(rows) != args.expected:
        raise RuntimeError(f"expected {args.expected} unique pairs, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(rows)} unique pairs")


if __name__ == "__main__":
    main()
