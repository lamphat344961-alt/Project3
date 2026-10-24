#!/usr/bin/env python
"""Freeze PAAC-VN dataset/config artifacts after tokenizer audit has selected N_CHARS."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(".."))
    ap.add_argument("--refresh-lock", action="store_true")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    data = root / "data_v1"
    pipeline = root / "pipeline"
    lock = root / "requirements-lock.txt"
    summary_path = data / "report" / "tokenizer_audit_summary.json"

    if args.refresh_lock:
        text = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lock.write_text(text, encoding="utf-8")

    required = [
        data / "benchmark_v1.parquet",
        data / "paired_en.parquet",
        pipeline / "models.yaml",
        data / "report" / "tokenizer_audit.csv",
        summary_path,
        lock,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Cannot freeze; missing:\n- " + "\n- ".join(missing))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_chars = summary.get("selected_n_chars")
    if not isinstance(n_chars, int) or n_chars <= 0:
        raise ValueError("tokenizer_audit_summary.json has no valid selected_n_chars")

    rel_and_hash = [(p.relative_to(root).as_posix(), sha256(p)) for p in required]
    now = datetime.now(timezone.utc).isoformat()

    lines = [
        "# PAAC-VN — FROZEN v1",
        "",
        f"- Frozen at (UTC): `{now}`",
        f"- TN1 `N_CHARS`: **{n_chars}**",
        "- After this point, do not modify benchmark datasets or model/tokenizer config during the main experiment.",
        "- Any later change requires a new dataset/config version and must not silently overwrite v1.",
        "",
        "## SHA-256",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for rel, digest in rel_and_hash:
        lines.append(f"| `{rel}` | `{digest}` |")

    frozen = data / "FROZEN.md"
    frozen.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {frozen}")
    print(f"N_CHARS={n_chars}")


if __name__ == "__main__":
    main()
