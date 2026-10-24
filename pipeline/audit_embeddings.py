#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

FROZEN_MODELS_SHA256 = "bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6"
FROZEN_DATASETS = {
    "tn1": {
        "sha256": "0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b",
        "n_rows": 23883,
    },
    "tn2": {
        "sha256": "9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493",
        "n_rows": 13657,
    },
}

TN1_REPS = [
    "tfidf_lsa",
    "phobert_base_v2",
    "videberta_base",
    "sup_simcse_phobert",
    "bkai_vietnamese_biencoder",
    "multilingual_e5_large",
    "bge_m3",
    "qwen3_embedding_0_6b",
]
TN2_REPS = [
    "tfidf_lsa",
    "multilingual_e5_large",
    "bge_m3",
    "qwen3_embedding_0_6b",
]

FIELDS = [
    "experiment", "representation", "language", "truncation",
    "npy_exists", "rows_exists", "json_exists", "n_rows",
    "embedding_dim", "dataset_sha256", "models_yaml_sha256",
    "app_ids_sha256", "revision", "embedding_dtype", "l2_normalized",
    "status", "audit_note",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def app_ids_hash_from_parquet(path: Path) -> tuple[int, str, list[str]]:
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    names = pf.schema_arrow.names
    issues: list[str] = []
    for col in ["row_idx", "appId", "genreId"]:
        if col not in names:
            issues.append(f"rows.parquet missing {col} column")
    if issues:
        return n_rows, "", issues

    # Read only the two columns needed for row-order verification; do not materialize the dataset text.
    table = pq.read_table(path, columns=["row_idx", "appId"])
    row_idx = np.asarray(table.column("row_idx").to_pylist(), dtype=np.int64)
    if not np.array_equal(row_idx, np.arange(n_rows, dtype=np.int64)):
        issues.append("rows.parquet row_idx is not contiguous 0..n-1")
    app_ids = [str(x) for x in table.column("appId").to_pylist()]
    digest = hashlib.sha256("\n".join(app_ids).encode("utf-8")).hexdigest()
    return n_rows, digest, issues


def expected_configs() -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    for rid in TN1_REPS:
        for trunc in ["char_budget", "native"]:
            out.append(("tn1", rid, "vi", trunc))
    for rid in TN2_REPS:
        for lang in ["en", "vi"]:
            for trunc in ["token_budget", "native"]:
                out.append(("tn2", rid, lang, trunc))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PAAC-VN frozen embedding artifact audit")
    ap.add_argument("--repo", type=Path, default=Path("."))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    embeddings = repo / "embeddings"
    models_path = repo / "pipeline" / "models.yaml"
    out_dir = repo / "results" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not models_path.exists():
        raise FileNotFoundError(models_path)

    actual_models_sha = sha256_file(models_path)
    with models_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    reps = cfg["representations"]

    rows_out: list[dict] = []
    pair_hashes: dict[tuple[str, str], dict[str, str]] = {}

    # Verify canonical dataset bytes once, without loading parquet content.
    dataset_paths = {
        "tn1": repo / "data_v1" / "benchmark_v1.parquet",
        "tn2": repo / "data_v1" / "paired_en.parquet",
    }
    dataset_actual_hash: dict[str, str | None] = {}
    for exp, path in dataset_paths.items():
        dataset_actual_hash[exp] = sha256_file(path) if path.exists() else None

    for exp, rid, lang, trunc in expected_configs():
        base = embeddings / exp / f"{exp}__{rid}__{lang}__{trunc}"
        npy = base.with_suffix(".npy")
        rows = base.with_suffix(".rows.parquet")
        meta_path = base.with_suffix(".json")

        rec = {
            "experiment": exp,
            "representation": rid,
            "language": lang,
            "truncation": trunc,
            "npy_exists": npy.exists(),
            "rows_exists": rows.exists(),
            "json_exists": meta_path.exists(),
            "n_rows": "",
            "embedding_dim": "",
            "dataset_sha256": "",
            "models_yaml_sha256": "",
            "app_ids_sha256": "",
            "revision": "",
            "embedding_dtype": "",
            "l2_normalized": "",
            "status": "",
            "audit_note": "",
        }

        missing = [p.name for p in (npy, rows, meta_path) if not p.exists()]
        if missing:
            rec["status"] = "INCOMPLETE"
            rec["audit_note"] = "missing: " + ", ".join(missing)
            rows_out.append(rec)
            continue

        issues: list[str] = []
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            rec["status"] = "INVALID"
            rec["audit_note"] = f"invalid json: {e}"
            rows_out.append(rec)
            continue

        rec["n_rows"] = meta.get("n_rows", "")
        rec["embedding_dim"] = meta.get("embedding_dim", "")
        rec["dataset_sha256"] = meta.get("dataset_sha256", "")
        rec["models_yaml_sha256"] = meta.get("models_yaml_sha256", "")
        rec["app_ids_sha256"] = meta.get("app_ids_sha256", "")
        rec["revision"] = meta.get("revision", "")
        rec["embedding_dtype"] = meta.get("embedding_dtype", "")
        rec["l2_normalized"] = meta.get("l2_normalized", "")

        expected = FROZEN_DATASETS[exp]
        spec = reps[rid]

        meta_checks = {
            "experiment": (meta.get("experiment"), exp),
            "representation": (meta.get("representation"), rid),
            "language": (meta.get("language"), lang),
            "truncation_mode": (meta.get("truncation_mode"), trunc),
            "n_rows": (meta.get("n_rows"), expected["n_rows"]),
            "embedding_dim": (meta.get("embedding_dim"), int(spec["dim"])),
            "dataset_sha256": (meta.get("dataset_sha256"), expected["sha256"]),
            "models_yaml_sha256": (meta.get("models_yaml_sha256"), FROZEN_MODELS_SHA256),
            "revision": (meta.get("revision"), spec["revision"]),
            "embedding_dtype": (meta.get("embedding_dtype"), "float32"),
            "l2_normalized": (meta.get("l2_normalized"), True),
        }
        for key, (got, want) in meta_checks.items():
            if got != want:
                issues.append(f"meta {key}: got={got!r}, expected={want!r}")

        if actual_models_sha != FROZEN_MODELS_SHA256:
            issues.append(
                f"current models.yaml sha256={actual_models_sha}, frozen={FROZEN_MODELS_SHA256}"
            )
        if dataset_actual_hash[exp] is not None and dataset_actual_hash[exp] != expected["sha256"]:
            issues.append(
                f"current {exp} dataset sha256={dataset_actual_hash[exp]}, frozen={expected['sha256']}"
            )

        if exp == "tn1":
            if trunc == "char_budget" and meta.get("n_chars") != 950:
                issues.append(f"TN1 char_budget n_chars={meta.get('n_chars')!r}, expected=950")
            if trunc == "native" and meta.get("n_chars") is not None:
                issues.append(f"TN1 native n_chars={meta.get('n_chars')!r}, expected=None")
        else:
            if trunc == "token_budget" and meta.get("max_tokens_requested") != 512:
                issues.append(
                    f"TN2 token_budget max_tokens_requested={meta.get('max_tokens_requested')!r}, expected=512"
                )
            if trunc == "native" and meta.get("max_tokens_requested") is not None:
                issues.append(
                    f"TN2 native max_tokens_requested={meta.get('max_tokens_requested')!r}, expected=None"
                )

        # Verify row-sidecar count/schema and recompute app_ids hash from appId only.
        try:
            parquet_n_rows, parquet_app_hash, parquet_issues = app_ids_hash_from_parquet(rows)
            issues.extend(parquet_issues)
            if parquet_n_rows != expected["n_rows"]:
                issues.append(
                    f"rows.parquet n_rows={parquet_n_rows}, expected={expected['n_rows']}"
                )
            if parquet_app_hash and parquet_app_hash != meta.get("app_ids_sha256"):
                issues.append(
                    "rows.parquet appId hash does not match metadata app_ids_sha256"
                )
        except Exception as e:
            issues.append(f"rows.parquet verification failed: {e}")
            parquet_app_hash = ""

        # mmap: inspect header/shape/dtype without loading full vectors.
        try:
            X = np.load(npy, mmap_mode="r")
            if X.ndim != 2:
                issues.append(f"npy ndim={X.ndim}, expected=2")
            else:
                if X.shape[0] != expected["n_rows"]:
                    issues.append(f"npy rows={X.shape[0]}, expected={expected['n_rows']}")
                if X.shape[1] != int(spec["dim"]):
                    issues.append(f"npy dim={X.shape[1]}, expected={int(spec['dim'])}")
            if X.dtype != np.float32:
                issues.append(f"npy dtype={X.dtype}, expected=float32")
        except Exception as e:
            issues.append(f"npy mmap verification failed: {e}")

        if exp == "tn2" and parquet_app_hash:
            pair_hashes.setdefault((rid, trunc), {})[lang] = parquet_app_hash

        rec["status"] = "INVALID" if issues else "COMPLETE"
        rec["audit_note"] = " | ".join(issues)
        rows_out.append(rec)

    # Enforce TN2 EN/VI same appId order/hash for each representation+truncation.
    by_key = {
        (r["experiment"], r["representation"], r["language"], r["truncation"]): r
        for r in rows_out
    }
    for rid in TN2_REPS:
        for trunc in ["token_budget", "native"]:
            hashes = pair_hashes.get((rid, trunc), {})
            if set(hashes) == {"en", "vi"} and hashes["en"] != hashes["vi"]:
                for lang in ["en", "vi"]:
                    rec = by_key[("tn2", rid, lang, trunc)]
                    note = "TN2 EN/VI app_ids_sha256 mismatch"
                    rec["audit_note"] = (rec["audit_note"] + " | " + note).strip(" |")
                    rec["status"] = "INVALID"

    manifest = out_dir / "embedding_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)

    counts = {s: sum(r["status"] == s for r in rows_out) for s in ["COMPLETE", "INCOMPLETE", "INVALID", "NOT VERIFIED"]}
    verified = counts["COMPLETE"]

    report = out_dir / "embedding_audit.md"
    verdict = "32/32 VERIFIED COMPLETE" if verified == 32 else f"{verified}/32 VERIFIED COMPLETE"
    lines = [
        "# PAAC-VN — Embedding Artifact Audit",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "| Status | Count |",
        "|---|---:|",
        f"| COMPLETE | {counts['COMPLETE']} |",
        f"| INCOMPLETE | {counts['INCOMPLETE']} |",
        f"| INVALID | {counts['INVALID']} |",
        f"| NOT VERIFIED | {counts['NOT VERIFIED']} |",
        "",
        f"Current `models.yaml` SHA-256: `{actual_models_sha}`",
        f"Frozen `models.yaml` SHA-256: `{FROZEN_MODELS_SHA256}`",
        f"TN1 dataset actual SHA-256: `{dataset_actual_hash['tn1'] or 'NOT AVAILABLE'}`",
        f"TN2 dataset actual SHA-256: `{dataset_actual_hash['tn2'] or 'NOT AVAILABLE'}`",
        "",
    ]

    failures = [r for r in rows_out if r["status"] != "COMPLETE"]
    if failures:
        lines += [
            "## Configurations requiring attention",
            "",
            "| Configuration | Status | Reason |",
            "|---|---|---|",
        ]
        for r in failures:
            config = f"{r['experiment']}__{r['representation']}__{r['language']}__{r['truncation']}"
            reason = str(r["audit_note"]).replace("|", "\\|")
            lines.append(f"| `{config}` | {r['status']} | {reason} |")
        lines.append("")
    if verified == 32:
        lines += [
            "## Gate decision",
            "",
            "**PASS — embedding phase is VERIFIED COMPLETE. Do not return to embedding.**",
            "",
            "Next: verify `pipeline/cluster.py`, then smoke-test exactly one embedding configuration before full clustering.",
        ]
    else:
        lines += [
            "## Gate decision",
            "",
            "**FAIL/BLOCKED — do not start full clustering.**",
            "",
            "Only configurations marked INCOMPLETE/INVALID require targeted investigation. Do not rerun the full embedding grid.",
        ]

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(verdict)
    print(counts)
    print(manifest)
    print(report)
    if verified != 32:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
