#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


def scan_one(path: Path, chunk_rows: int = 2048) -> dict:
    X = np.load(path, mmap_mode="r")
    if X.ndim != 2:
        return {
            "path": str(path),
            "n_rows": None,
            "dim": None,
            "dtype": str(X.dtype),
            "nan_count": None,
            "inf_count": None,
            "bad_row_count": None,
            "first_bad_rows": "",
            "status": "INVALID_NDIM",
        }

    n, d = X.shape
    nan_count = 0
    inf_count = 0
    bad_rows = []

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = np.asarray(X[start:stop])

        isnan = np.isnan(block)
        isinf = np.isinf(block)

        nan_count += int(isnan.sum())
        inf_count += int(isinf.sum())

        row_bad = np.any(isnan | isinf, axis=1)
        if np.any(row_bad):
            idx = np.flatnonzero(row_bad) + start
            bad_rows.extend(idx.tolist())

    status = "FINITE" if (nan_count == 0 and inf_count == 0) else "INVALID_NONFINITE"

    return {
        "path": str(path),
        "n_rows": int(n),
        "dim": int(d),
        "dtype": str(X.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "bad_row_count": len(bad_rows),
        "first_bad_rows": ",".join(map(str, bad_rows[:50])),
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/audit/embedding_finiteness.csv"),
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted((repo / "embeddings").glob("tn*/*.npy"))
    if not files:
        raise FileNotFoundError("No embeddings/tn*/*.npy files found.")

    rows = []
    for i, path in enumerate(files, start=1):
        rel = path.relative_to(repo)
        print(f"[{i:02d}/{len(files)}] {rel}", flush=True)
        r = scan_one(path, chunk_rows=args.chunk_rows)
        r["path"] = rel.as_posix()
        rows.append(r)
        print(
            f"    {r['status']} | NaN={r['nan_count']} "
            f"| Inf={r['inf_count']} | bad_rows={r['bad_row_count']}",
            flush=True,
        )

    fields = [
        "path", "n_rows", "dim", "dtype",
        "nan_count", "inf_count", "bad_row_count",
        "first_bad_rows", "status",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["status"] != "FINITE"]

    print()
    print("=" * 72)
    print(f"Scanned : {len(rows)} embeddings")
    print(f"FINITE  : {len(rows) - len(bad)}")
    print(f"INVALID : {len(bad)}")
    print(f"Report  : {out}")
    if bad:
        print("Invalid configurations:")
        for r in bad:
            print(
                f"  - {r['path']} | NaN={r['nan_count']} "
                f"| Inf={r['inf_count']} | bad_rows={r['bad_row_count']} "
                f"| first={r['first_bad_rows']}"
            )
        raise SystemExit(2)

    print("ALL EMBEDDINGS FINITE")
    print("=" * 72)


if __name__ == "__main__":
    main()
