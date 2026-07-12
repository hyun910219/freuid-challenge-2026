#!/usr/bin/env python3
"""Validate a FREUID submission.csv.

Checks:
  * header is exactly `id,label`
  * every label is a finite float
  * no duplicate ids
  * (optional, with --data) ids exactly match the images in the flat input dir
    (no missing, no extra)

Exit code 0 = valid, 1 = invalid. Pure stdlib — safe to run offline anywhere.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

SUPPORTED_EXTS = frozenset({".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


def discover_ids(data_dir: str) -> set:
    ids = set()
    for p in Path(data_dir).iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            ids.add(p.stem)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default="out/submission.csv", help="path to submission.csv")
    ap.add_argument("--data", default=None, help="flat input image dir to cross-check ids (optional)")
    args = ap.parse_args()

    sub = Path(args.submission)
    if not sub.is_file():
        print(f"FAIL: submission not found: {sub}")
        return 1

    errors: list[str] = []
    ids: list[str] = []

    with sub.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("FAIL: submission is empty (no header)")
            return 1
        if header != ["id", "label"]:
            print(f"FAIL: header must be exactly ['id','label'], got {header}")
            return 1

        for i, row in enumerate(reader, start=2):
            if len(row) != 2:
                errors.append(f"line {i}: expected 2 columns, got {len(row)}")
                continue
            rid, label = row
            ids.append(rid)
            try:
                v = float(label)
            except ValueError:
                errors.append(f"line {i}: label '{label}' is not a float")
                continue
            if not math.isfinite(v):
                errors.append(f"line {i}: label '{label}' is not finite")

    # duplicate ids
    seen, dups = set(), set()
    for rid in ids:
        if rid in seen:
            dups.add(rid)
        seen.add(rid)
    if dups:
        sample = sorted(dups)[:10]
        errors.append(f"{len(dups)} duplicate id(s), e.g. {sample}")

    # cross-check against input images
    if args.data:
        expected = discover_ids(args.data)
        got = set(ids)
        missing = expected - got
        extra = got - expected
        if missing:
            errors.append(f"{len(missing)} missing id(s), e.g. {sorted(missing)[:10]}")
        if extra:
            errors.append(f"{len(extra)} extra id(s), e.g. {sorted(extra)[:10]}")

    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        return 1

    msg = f"OK: {len(ids)} rows, header valid, all labels finite, no duplicates"
    if args.data:
        msg += ", ids exactly match /data"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
