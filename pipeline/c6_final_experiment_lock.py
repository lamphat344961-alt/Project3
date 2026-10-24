#!/usr/bin/env python3
"""
PAAC-VN C6 — FINAL EXPERIMENT LOCK.

Reads the completed C5 audit and creates the immutable experiment-lock record.
It performs NO experiment, NO statistical test, and NO modification of prior artifacts.

C6 refuses to run unless C5 status is PASS_WITH_DOCUMENTED_LIMITATIONS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_C5_STATUS = "PASS_WITH_DOCUMENTED_LIMITATIONS"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args()
    root = args.repo.resolve()

    c5_dir = root / "results" / "final_closure" / "C5_final_integrity"
    c5_json = c5_dir / "C5_FINAL_INTEGRITY_AUDIT.json"
    c5_complete = c5_dir / "C5_COMPLETE.txt"
    if not c5_json.is_file() or not c5_complete.is_file():
        raise FileNotFoundError("C5 outputs are incomplete; C6 lock is not authorized.")

    c5 = json.loads(c5_json.read_text(encoding="utf-8-sig"))
    if c5.get("status") != REQUIRED_C5_STATUS:
        raise RuntimeError(
            f"C6 refused: C5 status is {c5.get('status')!r}, expected {REQUIRED_C5_STATUS!r}."
        )
    if c5.get("material_failures"):
        raise RuntimeError("C6 refused: C5 contains material integrity failures.")

    outdir = root / "results" / "final_closure" / "C6_FINAL_LOCK"
    if outdir.exists():
        raise FileExistsError(
            f"{outdir} already exists. The experiment lock is immutable and must not be overwritten."
        )
    outdir.mkdir(parents=True, exist_ok=False)

    created = datetime.now(timezone.utc).isoformat()

    # Build final artifact manifest from C5 sealed artifacts + C5 audit artifacts.
    rows = list(c5.get("sealed_artifacts", []))
    for p, role in [
        (c5_json, "c5_audit_json"),
        (c5_dir / "C5_FINAL_INTEGRITY_AUDIT.md", "c5_audit_report"),
        (c5_complete, "c5_completion_record"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)
        rows.append({
            "phase": "C5",
            "role": role,
            "path": str(p.relative_to(root)),
            "sha256": sha256(p),
            "size_bytes": p.stat().st_size,
        })

    # De-duplicate by path+hash.
    seen = set()
    unique = []
    for r in rows:
        key = (r["path"], r["sha256"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    manifest_csv = outdir / "final_artifact_manifest.csv"
    with manifest_csv.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["phase", "role", "path", "sha256", "size_bytes"]
        )
        w.writeheader()
        for r in sorted(unique, key=lambda x: (x["phase"], x["path"])):
            w.writerow({k: r[k] for k in w.fieldnames})

    limitations = c5.get("documented_limitations", [])
    lock_md = outdir / "FINAL_EXPERIMENT_LOCK.md"
    lines = [
        "# PAAC-VN — FINAL EXPERIMENT LOCK",
        "",
        f"- **Lock status:** `FINAL_EXPERIMENT_LOCKED`",
        f"- **Created (UTC):** `{created}`",
        f"- **C5 status:** `{c5.get('status')}`",
        f"- **C5 audit SHA-256:** `{sha256(c5_json)}`",
        f"- **Final artifact manifest SHA-256:** `{sha256(manifest_csv)}`",
        "",
        "## Scope of the lock",
        "",
        "The computational experiment for the submitted PAAC-VN study is closed.",
        "",
        "**No further dataset processing, embedding, clustering, evaluation, bootstrap, "
        "sensitivity analysis, error analysis, model comparison, threshold modification, "
        "or additional experiment will be executed for the submitted PAAC-VN study.**",
        "",
        "This lock also fixes the existing research questions, the main `k=49` design, "
        "the SESOI definition, the accepted Gate4/Gate5 results, and the descriptive role "
        "of C2/C3.",
        "",
        "## Permitted post-lock work",
        "",
        "Only non-computational manuscript work is permitted: writing, translation, formatting, "
        "citation management, table/figure placement, and faithful reporting of already sealed artifacts.",
        "",
        "No post-lock manuscript edit may silently create a new numerical result, recompute a metric, "
        "change a threshold, select a new winner, or reinterpret a descriptive analysis as confirmatory.",
        "",
        "## Formal reopening exception",
        "",
        "The lock may be reopened only if a **material implementation defect** is discovered that can "
        "plausibly change a main scientific conclusion. Reopening must be explicit and documented before "
        "any code is executed; it must identify the defect, affected artifacts, scope of rerun, and new "
        "version lineage. Convenience, aesthetics, reviewer curiosity, a better score, or a desire for "
        "additional analyses are not valid reopening reasons.",
        "",
        "## Documented limitations retained at lock",
        "",
    ]
    for item in limitations:
        lines.append(f"- **{item.get('id')}** — {item.get('statement')}")
    lines += [
        "",
        "## Provenance note",
        "",
        "The A1-C preregistration document was not recovered as a local sealed artifact before final lock. "
        "This is retained as a provenance limitation and is not retroactively reconstructed or represented "
        "as an original preregistration artifact.",
        "",
        "C2 was executed in the isolated `.venv_c2` environment. C4 was executed in the main `.venv` after "
        "the recorded NumPy drift, but C4 is restricted to read-only paper-ready export/reshape operations "
        "and performs no new experiment or inferential test.",
        "",
        "## Final state",
        "",
        "`PAAC-VN COMPUTATIONAL EXPERIMENT: CLOSED.`",
        "",
    ]
    lock_md.write_text("\n".join(lines), encoding="utf-8")

    lock_manifest = {
        "project": "PAAC-VN",
        "phase": "C6_FINAL_EXPERIMENT_LOCK",
        "created_at_utc": created,
        "status": "FINAL_EXPERIMENT_LOCKED",
        "c5_status": c5.get("status"),
        "c5_audit_sha256": sha256(c5_json),
        "final_artifact_manifest_sha256": sha256(manifest_csv),
        "final_experiment_lock_md_sha256": sha256(lock_md),
        "artifact_rows": len(unique),
        "reopen_only_for_material_implementation_defect": True,
    }
    lock_manifest_path = outdir / "c6_lock_manifest.json"
    lock_manifest_path.write_text(
        json.dumps(lock_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    complete = outdir / "C6_LOCKED.txt"
    complete.write_text(
        "PAAC-VN COMPUTATIONAL EXPERIMENT: CLOSED.\n"
        f"lock_manifest_sha256={sha256(lock_manifest_path)}\n"
        f"lock_document_sha256={sha256(lock_md)}\n"
        f"artifact_manifest_sha256={sha256(manifest_csv)}\n",
        encoding="utf-8",
    )

    print("C6 STATUS: FINAL_EXPERIMENT_LOCKED")
    print("Lock document:", lock_md)
    print("Artifact manifest:", manifest_csv)
    print("Lock manifest:", lock_manifest_path)
    print("PAAC-VN COMPUTATIONAL EXPERIMENT: CLOSED.")

if __name__ == "__main__":
    main()
