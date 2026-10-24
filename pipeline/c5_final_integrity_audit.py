#!/usr/bin/env python3
"""
PAAC-VN C5 — Final integrity audit.

Purpose:
- READ ONLY with respect to all pre-C5 canonical artifacts.
- Verify hashes, row counts, environment snapshots, and closure-chain consistency.
- Generate C5 audit outputs only.
- DO NOT run embedding, clustering, evaluation, bootstrap, sensitivity analysis,
  error analysis, model comparison, or any new inferential test.

Expected provenance:
- Main experimental/forensic artifacts C0/C1/C3 were produced in `.venv`
  before the recorded NumPy drift.
- C2 supplementary k-sweep was produced in isolated `.venv_c2`.
- C4 paper-ready export was produced in `.venv` after the recorded NumPy drift;
  C4 is an export/reshape layer only and must not be treated as a new experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT = "PAAC-VN"
C5_STATUS_PASS = "PASS_WITH_DOCUMENTED_LIMITATIONS"
ALLOWED_C2_EXTRAS = {
    "contourpy", "cycler", "fonttools", "kiwisolver",
    "matplotlib", "pillow", "pyparsing",
}
EXPECTED_C2_NUMPY = "1.26.4"
EXPECTED_MAIN_DRIFT_NUMPY = ("1.26.4", "2.5.3")

# Recorded chronology from the closure session.
# Windows LastWriteTime values are local (+07); JSON created_at_utc values are UTC.
RECORDED_MAIN_ENV_DRIFT_LOCAL = "2026-10-06T14:57:40+07:00"
RECORDED_MAIN_MATPLOTLIB_INSTALL_LOCAL = "2026-10-06T14:57:48+07:00"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_lines(path: Path) -> List[str]:
    b = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return b.decode(enc).splitlines()
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"Could not decode {path}")

def parse_freeze(path: Path) -> Dict[str, str]:
    out = {}
    for raw in read_lines(path):
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        k, v = line.split("==", 1)
        out[k.strip().lower()] = v.strip()
    return out

def compare_env(lock: Dict[str, str], env: Dict[str, str]):
    changed = [(k, lock[k], env[k]) for k in sorted(lock) if k in env and lock[k] != env[k]]
    missing = [k for k in sorted(lock) if k not in env]
    extra = sorted(set(env) - set(lock))
    return changed, missing, extra

def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        try:
            next(r)
        except StopIteration:
            return 0
        return sum(1 for _ in r)

def require_file(root: Path, rel: str, failures: List[str]) -> Path:
    p = root / rel
    if not p.is_file():
        failures.append(f"MISSING REQUIRED FILE: {rel}")
    return p

def check_hash(path: Path, expected: str, label: str, failures: List[str]):
    if not path.is_file():
        failures.append(f"MISSING FOR HASH CHECK: {label} -> {path}")
        return None
    got = sha256(path)
    if got != expected:
        failures.append(f"SHA256 MISMATCH: {label}: expected={expected}, got={got}")
    return got

def add_artifact(rows: List[dict], root: Path, path: Path, phase: str, role: str):
    if path.is_file():
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        rows.append({
            "phase": phase,
            "role": role,
            "path": rel,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args()
    root = args.repo.resolve()

    outdir = root / "results" / "final_closure" / "C5_final_integrity"
    if outdir.exists():
        raise FileExistsError(
            f"{outdir} already exists. C5 is immutable; do not overwrite. "
            "If a material implementation defect is discovered, formally reopen the closure."
        )
    outdir.mkdir(parents=True, exist_ok=False)

    failures: List[str] = []
    warnings: List[str] = []
    sealed: List[dict] = []

    # ------------------------------------------------------------------
    # 1) Environment snapshots
    # ------------------------------------------------------------------
    req_path = require_file(root, "requirements-lock.txt", failures)
    c2_env_path = require_file(root, "env_c2_freeze.txt", failures)
    main_env_path = require_file(root, "env_main_after_drift.txt", failures)

    env_audit = {}
    if req_path.is_file() and c2_env_path.is_file() and main_env_path.is_file():
        lock = parse_freeze(req_path)
        c2 = parse_freeze(c2_env_path)
        main_after = parse_freeze(main_env_path)

        c2_changed, c2_missing, c2_extra = compare_env(lock, c2)
        main_changed, main_missing, main_extra = compare_env(lock, main_after)

        env_audit = {
            "c2_environment": {
                "snapshot": "env_c2_freeze.txt",
                "changed_locked_packages": c2_changed,
                "missing_locked_packages": c2_missing,
                "extra_packages": c2_extra,
                "allowed_plotting_extras": sorted(ALLOWED_C2_EXTRAS),
                "numpy": c2.get("numpy"),
                "adjudication": None,
            },
            "main_environment_after_drift": {
                "snapshot": "env_main_after_drift.txt",
                "changed_locked_packages": main_changed,
                "missing_locked_packages": main_missing,
                "extra_packages": main_extra,
                "recorded_numpy_drift": {
                    "from": EXPECTED_MAIN_DRIFT_NUMPY[0],
                    "to": EXPECTED_MAIN_DRIFT_NUMPY[1],
                    "dist_info_last_write_local": RECORDED_MAIN_ENV_DRIFT_LOCAL,
                },
                "recorded_matplotlib_install_local": RECORDED_MAIN_MATPLOTLIB_INSTALL_LOCAL,
                "adjudication": None,
            },
            "execution_provenance": {
                "C0_C1_C3": ".venv before recorded NumPy 2.5.3 installation event",
                "C2": ".venv_c2 isolated supplementary environment",
                "C4": ".venv after recorded NumPy drift; export/reshape only, no new inferential test",
            },
        }

        if c2_changed or c2_missing:
            failures.append(
                f"C2 ENV LOCK VIOLATION: changed={c2_changed}, missing={c2_missing}"
            )
            env_audit["c2_environment"]["adjudication"] = "FAIL"
        elif set(c2_extra) - ALLOWED_C2_EXTRAS:
            failures.append(
                "C2 ENV HAS UNEXPECTED EXTRAS: "
                + repr(sorted(set(c2_extra) - ALLOWED_C2_EXTRAS))
            )
            env_audit["c2_environment"]["adjudication"] = "FAIL"
        elif c2.get("numpy") != EXPECTED_C2_NUMPY:
            failures.append(f"C2 NUMPY != {EXPECTED_C2_NUMPY}: {c2.get('numpy')}")
            env_audit["c2_environment"]["adjudication"] = "FAIL"
        else:
            env_audit["c2_environment"]["adjudication"] = "MATCHES_LOCK_PLUS_PLOTTING_EXTRAS"

        expected_main_changed = [("numpy", EXPECTED_MAIN_DRIFT_NUMPY[0], EXPECTED_MAIN_DRIFT_NUMPY[1])]
        if main_missing:
            failures.append(f"MAIN POST-DRIFT SNAPSHOT MISSING LOCKED PACKAGES: {main_missing}")
            env_audit["main_environment_after_drift"]["adjudication"] = "FAIL"
        elif main_changed != expected_main_changed:
            failures.append(
                "MAIN POST-DRIFT SNAPSHOT DIFFERS FROM RECORDED EVENT: "
                f"expected={expected_main_changed}, got={main_changed}"
            )
            env_audit["main_environment_after_drift"]["adjudication"] = "FAIL"
        else:
            env_audit["main_environment_after_drift"]["adjudication"] = (
                "RECORDED_NUMPY_DRIFT_CONFIRMED"
            )

        add_artifact(sealed, root, req_path, "ENV", "reference_lock")
        add_artifact(sealed, root, c2_env_path, "ENV", "c2_snapshot")
        add_artifact(sealed, root, main_env_path, "ENV", "main_post_drift_snapshot")

    # ------------------------------------------------------------------
    # 2) C0 — archive provenance and canonical artifact immutability
    # ------------------------------------------------------------------
    c0_path = require_file(
        root, "results/final_closure/C0_archive/c0_archive_manifest.json", failures
    )
    c0_complete = require_file(
        root, "results/final_closure/C0_archive/C0_COMPLETE.txt", failures
    )
    c0 = None
    if c0_path.is_file():
        c0 = read_json(c0_path)
        if c0.get("canonical_inputs_modified") is not False:
            failures.append("C0 canonical_inputs_modified is not false")
        for rec in c0.get("records", []):
            rel = rec["path"]
            p = root / rel
            if not p.is_file():
                failures.append(f"C0 CANONICAL ARTIFACT MISSING NOW: {rel}")
                continue
            got = sha256(p)
            if got != rec["sha256"]:
                failures.append(
                    f"C0 CANONICAL ARTIFACT CHANGED: {rel}: "
                    f"expected={rec['sha256']}, got={got}"
                )
            add_artifact(sealed, root, p, "C0", "canonical_or_recorded_source")
        add_artifact(sealed, root, c0_path, "C0", "manifest")
    if c0_complete.is_file():
        add_artifact(sealed, root, c0_complete, "C0", "completion_record")

    # ------------------------------------------------------------------
    # 3) C1 — forensic closure
    # ------------------------------------------------------------------
    c1_dir = root / "results" / "audit" / "final_forensic_closure"
    c1_paths = {
        "a1c": c1_dir / "a1c.json",
        "a2": c1_dir / "a2.json",
        "a3": c1_dir / "a3.json",
    }
    for k, p in c1_paths.items():
        if not p.is_file():
            failures.append(f"MISSING C1 OUTPUT: {p.relative_to(root)}")
    if all(p.is_file() for p in c1_paths.values()):
        a1c, a2, a3 = (read_json(c1_paths[k]) for k in ("a1c", "a2", "a3"))
        if a1c.get("status") != "A1-C CLOSED WITH SEMANTIC LIMITATION":
            failures.append(f"Unexpected A1-C status: {a1c.get('status')}")
        if a1c.get("official_results_action") != "RETAIN_FROZEN_RESULTS":
            failures.append(f"Unexpected A1-C official action: {a1c.get('official_results_action')}")
        if a1c.get("export_semantics") != "NOT_VERIFIED":
            failures.append(f"Unexpected A1-C export_semantics: {a1c.get('export_semantics')}")
        if a2.get("status") != "DIAGNOSTICS_RECORDED_NO_POSTHOC_THRESHOLD":
            failures.append(f"Unexpected A2 status: {a2.get('status')}")
        if a3.get("status") != "ALL_ROWS_RECOMPUTED_AND_COMPARED":
            failures.append(f"Unexpected A3 status: {a3.get('status')}")
        for name, obj in (("A1-C", a1c), ("A2", a2), ("A3", a3)):
            if obj.get("canonical_inputs_modified") is not False:
                failures.append(f"{name} canonical_inputs_modified is not false")
        for k, p in c1_paths.items():
            add_artifact(sealed, root, p, "C1", k)

    # ------------------------------------------------------------------
    # 4) C2 — supplementary saved-Z k-sweep
    # ------------------------------------------------------------------
    c2_manifest_path = require_file(
        root, "results/figures/k_sweep_tfidf_char_budget_manifest.json", failures
    )
    c2_csv = require_file(root, "results/figures/k_sweep_tfidf_char_budget.csv", failures)
    c2_png = require_file(root, "results/figures/k_sweep_tfidf_char_budget.png", failures)
    c2_script = require_file(root, "pipeline/k_sweep_saved_z.py", failures)
    c2_manifest = None
    if c2_manifest_path.is_file():
        c2_manifest = read_json(c2_manifest_path)
        if c2_manifest.get("selected") != "tn1__tfidf_lsa__vi__char_budget":
            failures.append(f"Unexpected C2 selected config: {c2_manifest.get('selected')}")
        rows = c2_manifest.get("reported_rows", [])
        ks = [r.get("requested_k") for r in rows]
        if ks != [30, 40, 49, 60, 80]:
            failures.append(f"Unexpected C2 k sweep: {ks}")
        if c2_csv.is_file() and csv_rows(c2_csv) != 5:
            failures.append(f"C2 CSV row count != 5: {csv_rows(c2_csv)}")
        add_artifact(sealed, root, c2_manifest_path, "C2", "manifest")
    for p, role in ((c2_csv, "k_sweep_csv"), (c2_png, "k_sweep_figure"),
                    (c2_script, "exact_executable")):
        if p.is_file():
            add_artifact(sealed, root, p, "C2", role)

    # ------------------------------------------------------------------
    # 5) C3 — descriptive error analysis
    # ------------------------------------------------------------------
    c3_manifest_path = require_file(
        root, "results/gate6/error_analysis/gate6_error_analysis_manifest.json", failures
    )
    c3 = None
    if c3_manifest_path.is_file():
        c3 = read_json(c3_manifest_path)
        if c3.get("design") != "descriptive_only":
            failures.append(f"C3 design is not descriptive_only: {c3.get('design')}")
        if c3.get("inference") != "NONE: no bootstrap, p-value, Holm, TOST or new genre-pair selection":
            failures.append(f"Unexpected C3 inference declaration: {c3.get('inference')}")
        if c3.get("gate6_cannot_change_main_conclusions") is not True:
            failures.append("C3 gate6_cannot_change_main_conclusions is not true")
        for key, info in c3.get("outputs", {}).items():
            p = root / info["path"]
            check_hash(p, info["sha256"], f"C3 {key}", failures)
            if p.is_file():
                n = csv_rows(p)
                if n != int(info["rows"]):
                    failures.append(
                        f"C3 ROW COUNT MISMATCH {key}: expected={info['rows']}, got={n}"
                    )
                add_artifact(sealed, root, p, "C3", key)
        add_artifact(sealed, root, c3_manifest_path, "C3", "manifest")

    # ------------------------------------------------------------------
    # 6) C4 — paper-ready export
    # ------------------------------------------------------------------
    c4_manifest_path = require_file(
        root, "results/final_closure/C4_paper_ready/c4_manifest.json", failures
    )
    c4 = None
    if c4_manifest_path.is_file():
        c4 = read_json(c4_manifest_path)
        if c4.get("status") != "COMPLETE":
            failures.append(f"C4 status != COMPLETE: {c4.get('status')}")
        if c4.get("canonical_inputs_modified") is not False:
            failures.append("C4 canonical_inputs_modified is not false")
        if c4.get("gate6_cannot_change_main_conclusions") is not True:
            failures.append("C4 gate6_cannot_change_main_conclusions is not true")
        method = c4.get("method", "")
        if "no new experiment or inferential test" not in method:
            warnings.append("C4 method string does not explicitly state no new experiment/inference")

        for item in c4.get("outputs", []):
            p = root / item["path"]
            check_hash(p, item["sha256"], "C4 output", failures)
            if p.is_file():
                add_artifact(sealed, root, p, "C4", "paper_ready_output")

        # Verify C4 source hashes against current source files.
        source_map = {
            "metrics_long": root / "results/evaluation/metrics_long.csv",
            "bootstrap_summary": root / "results/statistics/gate5/bootstrap_summary.csv",
            "gate5_manifest": root / "results/statistics/gate5/gate5_manifest.json",
            "statistical_audit": root / "results/statistics/gate5/statistical_audit.json",
            "k_sweep_csv": root / "results/figures/k_sweep_tfidf_char_budget.csv",
            "k_sweep_manifest": root / "results/figures/k_sweep_tfidf_char_budget_manifest.json",
            "c3_co_raw": root / "results/gate6/error_analysis/gate6_coassignment_6pairs.csv",
            "c3_co_summary": root / "results/gate6/error_analysis/gate6_coassignment_6pairs_summary.csv",
            "c3_map_raw": root / "results/gate6/error_analysis/gate6_mapped_confusion_6pairs.csv",
            "c3_map_summary": root / "results/gate6/error_analysis/gate6_mapped_confusion_6pairs_summary.csv",
            "c3_recon": root / "results/gate6/error_analysis/gate6_hungarian_reconciliation.csv",
            "c3_manifest": root / "results/gate6/error_analysis/gate6_error_analysis_manifest.json",
            "a1c": root / "results/audit/final_forensic_closure/a1c.json",
            "a2": root / "results/audit/final_forensic_closure/a2.json",
            "a3": root / "results/audit/final_forensic_closure/a3.json",
            "c0_manifest": root / "results/final_closure/C0_archive/c0_archive_manifest.json",
        }
        for key, expected in c4.get("source_sha256", {}).items():
            p = source_map.get(key)
            if p is None:
                warnings.append(f"No C5 path mapping for C4 source key: {key}")
                continue
            check_hash(p, expected, f"C4 source {key}", failures)
        add_artifact(sealed, root, c4_manifest_path, "C4", "manifest")

    # ------------------------------------------------------------------
    # 7) Supporting documents / provenance limitations
    # ------------------------------------------------------------------
    prereg_candidates = [
        root / "PAAC_VN_A1C_Preregistered_Checkpoint_Audit.md",
        root / "PAAC-VN_A1C_Preregistered_Checkpoint_Audit.md",
    ]
    prereg = next((p for p in prereg_candidates if p.is_file()), None)
    gate4_review_candidates = [
        root / "PAAC-VN_Gate4_Critical_Review_VI.md",
        root / "PAAC-VN_Gate4_Critical_Review.md",
    ]
    gate4_review = next((p for p in gate4_review_candidates if p.is_file()), None)

    provenance = {
        "a1c_preregistration_local_artifact": {
            "status": "PRESENT" if prereg else "NOT_RECOVERED_LOCALLY",
            "path": str(prereg.relative_to(root)) if prereg else None,
            "interpretation": (
                "Local preregistration artifact was available at final audit."
                if prereg else
                "The preregistration document was absent from the C0 archive and could not "
                "be recovered locally before final lock. Local preregistration provenance "
                "therefore cannot be independently verified from the sealed repository. "
                "This is a provenance limitation, not evidence of an implementation defect."
            )
        },
        "gate4_critical_review": {
            "status": "PRESENT" if gate4_review else "NOT_PRESENT_LOCALLY_AT_C5",
            "path": str(gate4_review.relative_to(root)) if gate4_review else None,
        }
    }
    if prereg:
        add_artifact(sealed, root, prereg, "SUPPORT", "a1c_preregistration")
    if gate4_review:
        add_artifact(sealed, root, gate4_review, "SUPPORT", "gate4_critical_review")
    else:
        warnings.append("Gate4 critical review not found in local repository at C5")

    # ------------------------------------------------------------------
    # 8) Documented limitations carried into final lock
    # ------------------------------------------------------------------
    limitations = [
        {
            "id": "L1_A1C_PREREG_PROVENANCE",
            "statement": (
                "PAAC_VN_A1C_Preregistered_Checkpoint_Audit.md was not present in the C0 "
                "archive and was not recovered locally before final lock; local preregistration "
                "provenance cannot be independently verified from the sealed repository."
            ),
            "material_to_main_results": False,
        },
        {
            "id": "L2_VIDEBERTA_EXPORT_SEMANTICS",
            "statement": (
                "ViDeBERTa runtime input embeddings match checkpoint `weight`, but effective "
                "pretrained discriminator/GDES export semantics remain NOT_VERIFIED. Frozen "
                "results are retained; no causal attribution to mean pooling is permitted."
            ),
            "material_to_main_results": False,
        },
        {
            "id": "L3_E5_NUMERICAL_PARITY",
            "statement": (
                "E5 TN2 EN native/token-budget embeddings differ only at machine-scale numerical "
                "magnitude despite identical effective max_len=512 and prefix. Downstream KMeans "
                "can amplify this perturbation. Differences must not be interpreted as a context-length effect."
            ),
            "material_to_main_results": False,
        },
        {
            "id": "L4_MAIN_ENV_DRIFT_AFTER_CLOSURE_ARTIFACTS",
            "statement": (
                "The main `.venv` later drifted from NumPy 1.26.4 to 2.5.3. C0/C1/C3 were created "
                "before the recorded drift event. C4 was executed afterward in `.venv`, but C4 is "
                "a read-only paper-ready export/reshape layer and performs no new inferential test."
            ),
            "material_to_main_results": False,
        },
        {
            "id": "L5_C2_ISOLATED_ENV",
            "statement": (
                "C2 supplementary k-sweep was executed in `.venv_c2`, which matches all locked "
                "package versions and contains only plotting-related extras. It uses saved linkage "
                "and does not refit embeddings, linkage, or the main k=49 clustering."
            ),
            "material_to_main_results": False,
        },
        {
            "id": "L6_HISTORICAL_EMBEDDING_HASH_GAP",
            "statement": (
                "Historical byte-level SHA-256 values of embedding NPY files at the exact time of "
                "Gate3 clustering were not recorded. Current NPY hashes reconcile with Gate4, but "
                "current fingerprints cannot prove byte identity at an earlier historical time."
            ),
            "material_to_main_results": False,
        },
    ]

    status = C5_STATUS_PASS if not failures else "FAIL_MATERIAL_INTEGRITY"
    created = datetime.now(timezone.utc).isoformat()

    # De-duplicate sealed artifact rows by (path, sha256), preserving first occurrence.
    seen = set()
    sealed_unique = []
    for r in sealed:
        key = (r["path"], r["sha256"])
        if key not in seen:
            seen.add(key)
            sealed_unique.append(r)

    report = {
        "project": PROJECT,
        "phase": "C5_FINAL_INTEGRITY_AUDIT",
        "created_at_utc": created,
        "status": status,
        "material_failures": failures,
        "warnings": warnings,
        "environment_audit": env_audit,
        "provenance": provenance,
        "documented_limitations": limitations,
        "sealed_artifact_count": len(sealed_unique),
        "sealed_artifacts": sealed_unique,
        "adjudication": (
            "All computational phases C0-C4 are accepted for final lock with documented "
            "provenance/scientific limitations. No rerun is authorized."
            if not failures else
            "Final experiment lock MUST NOT be created until material integrity failures are resolved "
            "through a formal reopening."
        ),
        "canonical_inputs_modified_by_c5": False,
    }

    json_path = outdir / "C5_FINAL_INTEGRITY_AUDIT.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# PAAC-VN — C5 Final Integrity Audit\n")
    md.append(f"- **Status:** `{status}`")
    md.append(f"- **Created (UTC):** `{created}`")
    md.append("- **C5 role:** integrity/provenance verification only; no new experiment or inferential test.")
    md.append(f"- **Sealed artifacts recorded:** {len(sealed_unique)}\n")

    if failures:
        md.append("## Material integrity failures\n")
        for x in failures:
            md.append(f"- {x}")
    else:
        md.append("## Adjudication\n")
        md.append(
            "**PASS_WITH_DOCUMENTED_LIMITATIONS.** C0-C4 may proceed to C6 final experiment lock. "
            "No rerun is required or authorized by this audit."
        )

    md.append("\n## Environment chronology\n")
    md.append(
        "- C0/C1/C3: executed in `.venv` before the recorded NumPy 2.5.3 installation event."
    )
    md.append(
        "- C2: executed in isolated `.venv_c2`; locked packages match `requirements-lock.txt`; "
        "only plotting-related extras are present."
    )
    md.append(
        "- C4: executed in `.venv` after the recorded NumPy drift. This is retained because C4 is "
        "a read-only paper-ready export/reshape layer and performs no new experiment/inferential test."
    )

    md.append("\n## Documented limitations carried to C6\n")
    for item in limitations:
        md.append(f"- **{item['id']}** — {item['statement']}")

    if warnings:
        md.append("\n## Non-material warnings\n")
        for x in warnings:
            md.append(f"- {x}")

    md.append("\n## C5 rule\n")
    md.append(
        "C5 does not authorize modification of datasets, embeddings, labels, metrics, bootstrap "
        "replicates, thresholds, SESOI, model selection, k, or conclusions."
    )

    md_path = outdir / "C5_FINAL_INTEGRITY_AUDIT.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Completion marker.
    complete = outdir / "C5_COMPLETE.txt"
    complete.write_text(
        f"C5 status={status}\n"
        f"audit_json_sha256={sha256(json_path)}\n"
        f"audit_md_sha256={sha256(md_path)}\n",
        encoding="utf-8",
    )

    print(f"C5 STATUS: {status}")
    print(f"C5 audit: {json_path}")
    print(f"C5 report: {md_path}")
    if failures:
        print("Material failures:")
        for x in failures:
            print(" -", x)
        sys.exit(2)

if __name__ == "__main__":
    main()
