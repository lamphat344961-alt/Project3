#!/usr/bin/env python
"""
PAAC-VN — audit official clustering artifacts.

Checks exactly the frozen 32 configurations from:
    results/audit/embedding_manifest.csv

For each configuration, verify:
- 5 clustering artifacts exist
- clustering metadata matches the current embedding metadata
- k = 49
- KMeans main: seeds 42..51, n_init=10, labels shape=(10,n), 49 clusters/seed
- KMeans diagnostic: seeds 42..51, n_init=1, labels shape=(10,n), 49 clusters/seed
- Agglomerative: labels shape=(n,), exactly 49 clusters
- Linkage: shape=(n-1,4), float64, finite
- protocol metadata: average linkage + cosine + maxclust
- hashes of all 5 clustering artifacts are recorded for provenance

Does not modify clustering results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


K = 49
SEEDS = np.arange(42, 52, dtype=np.int32)
MAIN_N_INIT = 10
DIAG_N_INIT = 1
EXPECTED_CONFIGS = 32

SUFFIXES = {
    "kmeans_main": "__kmeans_main.npz",
    "kmeans_diag": "__kmeans_diag.npz",
    "agglomerative": "__agglomerative.npy",
    "linkage": "__linkage.npy",
    "meta": "__cluster_meta.json",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scalar_int(x) -> int:
    return int(np.asarray(x).reshape(()))


def verify_kmeans(path: Path, n_rows: int, expected_n_init: int) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = {"seeds", "labels", "inertia", "n_iter", "n_init", "k"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{path.name}: missing NPZ keys {sorted(missing)}")

        seeds = np.asarray(z["seeds"])
        labels = np.asarray(z["labels"])
        inertia = np.asarray(z["inertia"])
        n_iter = np.asarray(z["n_iter"])
        n_init = scalar_int(z["n_init"])
        k = scalar_int(z["k"])

    if not np.array_equal(seeds.astype(np.int32, copy=False), SEEDS):
        raise ValueError(f"{path.name}: seeds != 42..51")
    if n_init != expected_n_init:
        raise ValueError(
            f"{path.name}: n_init={n_init}, expected={expected_n_init}"
        )
    if k != K:
        raise ValueError(f"{path.name}: k={k}, expected={K}")
    if labels.shape != (len(SEEDS), n_rows):
        raise ValueError(
            f"{path.name}: labels shape={labels.shape}, "
            f"expected={(len(SEEDS), n_rows)}"
        )
    if inertia.shape != (len(SEEDS),):
        raise ValueError(f"{path.name}: inertia shape={inertia.shape}")
    if n_iter.shape != (len(SEEDS),):
        raise ValueError(f"{path.name}: n_iter shape={n_iter.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{path.name}: labels dtype must be integer, got {labels.dtype}")
    if not np.isfinite(inertia).all():
        raise ValueError(f"{path.name}: inertia contains non-finite values")
    if not np.isfinite(n_iter).all():
        raise ValueError(f"{path.name}: n_iter contains non-finite values")
    if np.any(n_iter <= 0):
        raise ValueError(f"{path.name}: n_iter contains non-positive values")

    cluster_counts = []
    for i, seed in enumerate(SEEDS):
        row = labels[i]
        uniq = np.unique(row)
        if len(uniq) != K:
            raise ValueError(
                f"{path.name}: seed={int(seed)} has {len(uniq)} clusters, expected={K}"
            )
        if row.min() < 0 or row.max() >= K:
            raise ValueError(
                f"{path.name}: seed={int(seed)} labels outside 0..{K-1}"
            )
        cluster_counts.append(len(uniq))

    return {
        "labels_dtype": str(labels.dtype),
        "inertia_min": float(inertia.min()),
        "inertia_max": float(inertia.max()),
        "n_iter_min": int(n_iter.min()),
        "n_iter_max": int(n_iter.max()),
        "cluster_count_min": int(min(cluster_counts)),
        "cluster_count_max": int(max(cluster_counts)),
    }


def verify_agglomerative(path: Path, n_rows: int) -> dict:
    lab = np.load(path, mmap_mode="r")
    if lab.shape != (n_rows,):
        raise ValueError(
            f"{path.name}: shape={lab.shape}, expected={(n_rows,)}"
        )
    if not np.issubdtype(lab.dtype, np.integer):
        raise ValueError(f"{path.name}: dtype must be integer, got {lab.dtype}")
    if not np.isfinite(np.asarray(lab)).all():
        raise ValueError(f"{path.name}: contains non-finite values")

    uniq = np.unique(lab)
    if len(uniq) != K:
        raise ValueError(
            f"{path.name}: {len(uniq)} clusters, expected={K}"
        )

    # scipy.cluster.hierarchy.fcluster produces positive cluster IDs.
    if int(lab.min()) < 1:
        raise ValueError(f"{path.name}: agglomerative labels must be positive")

    return {
        "dtype": str(lab.dtype),
        "n_clusters": int(len(uniq)),
        "min_label": int(lab.min()),
        "max_label": int(lab.max()),
    }


def verify_linkage(path: Path, n_rows: int) -> dict:
    Z = np.load(path, mmap_mode="r")
    expected_shape = (n_rows - 1, 4)
    if Z.shape != expected_shape:
        raise ValueError(
            f"{path.name}: shape={Z.shape}, expected={expected_shape}"
        )
    if Z.dtype != np.float64:
        raise ValueError(f"{path.name}: dtype={Z.dtype}, expected=float64")
    if not np.isfinite(np.asarray(Z)).all():
        raise ValueError(f"{path.name}: contains non-finite values")
    if np.any(np.asarray(Z[:, 2]) < 0):
        raise ValueError(f"{path.name}: linkage distance contains negative values")

    return {
        "dtype": str(Z.dtype),
        "distance_min": float(np.min(Z[:, 2])),
        "distance_max": float(np.max(Z[:, 2])),
    }


def verify_meta(
    path: Path,
    exp: str,
    rid: str,
    lang: str,
    trunc: str,
    emb_meta: dict,
) -> dict:
    meta = load_json(path)

    cfg = meta.get("configuration", {})
    expected_cfg = {
        "experiment": exp,
        "representation": rid,
        "language": lang,
        "truncation_mode": trunc,
    }
    if cfg != expected_cfg:
        raise ValueError(
            f"{path.name}: configuration mismatch; "
            f"got={cfg}, expected={expected_cfg}"
        )

    src = meta.get("source_embedding", {})
    source_checks = {
        "n_rows": emb_meta.get("n_rows"),
        "embedding_dim": emb_meta.get("embedding_dim"),
        "dataset_sha256": emb_meta.get("dataset_sha256"),
        "models_yaml_sha256": emb_meta.get("models_yaml_sha256"),
        "app_ids_sha256": emb_meta.get("app_ids_sha256"),
        "revision": emb_meta.get("revision"),
        "embedding_dtype": emb_meta.get("embedding_dtype"),
        "l2_normalized": emb_meta.get("l2_normalized"),
    }
    for key, expected in source_checks.items():
        if src.get(key) != expected:
            raise ValueError(
                f"{path.name}: source_embedding.{key}="
                f"{src.get(key)!r}, expected={expected!r}"
            )

    p = meta.get("protocol", {})
    if p.get("k") != K:
        raise ValueError(f"{path.name}: protocol.k != {K}")

    km_main = p.get("kmeans_main", {})
    if km_main.get("n_init") != MAIN_N_INIT:
        raise ValueError(f"{path.name}: main n_init mismatch")
    if km_main.get("outer_seeds") != SEEDS.tolist():
        raise ValueError(f"{path.name}: main seeds mismatch")

    km_diag = p.get("kmeans_diagnostic", {})
    if km_diag.get("n_init") != DIAG_N_INIT:
        raise ValueError(f"{path.name}: diagnostic n_init mismatch")
    if km_diag.get("outer_seeds") != SEEDS.tolist():
        raise ValueError(f"{path.name}: diagnostic seeds mismatch")

    agg = p.get("agglomerative", {})
    expected_agg = {
        "method": "average",
        "distance": "cosine",
        "criterion": "maxclust",
        "condensed_dtype": "float64",
        "square_distance_matrix": False,
        "scipy_pdist_full_pipeline": False,
    }
    for key, expected in expected_agg.items():
        if agg.get(key) != expected:
            raise ValueError(
                f"{path.name}: agglomerative.{key}="
                f"{agg.get(key)!r}, expected={expected!r}"
            )

    block_size = agg.get("distance_block_size")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"{path.name}: invalid distance_block_size={block_size!r}")

    runtime = meta.get("runtime_seconds", {})
    required_runtime = {
        "kmeans_main",
        "kmeans_diagnostic",
        "distance_seconds",
        "linkage_seconds",
        "total",
    }
    missing_runtime = required_runtime - set(runtime)
    if missing_runtime:
        raise ValueError(
            f"{path.name}: missing runtime fields {sorted(missing_runtime)}"
        )
    if any(
        (not isinstance(runtime[k], (int, float))) or runtime[k] < 0
        for k in required_runtime
    ):
        raise ValueError(f"{path.name}: invalid runtime_seconds")

    return {
        "distance_block_size": int(block_size),
        "runtime_total_seconds": float(runtime["total"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/audit/embedding_manifest.csv"),
    )
    ap.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("results/labels"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/audit/clustering_manifest.csv"),
    )
    args = ap.parse_args()

    repo = args.repo.resolve()

    manifest_path = (
        args.manifest
        if args.manifest.is_absolute()
        else repo / args.manifest
    )
    labels_dir = (
        args.labels_dir
        if args.labels_dir.is_absolute()
        else repo / args.labels_dir
    )
    out_path = (
        args.out
        if args.out.is_absolute()
        else repo / args.out
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    manifest = pd.read_csv(manifest_path)
    required_cols = {
        "experiment",
        "representation",
        "language",
        "truncation",
        "status",
    }
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(
            f"Embedding manifest missing columns: {sorted(missing_cols)}"
        )

    complete = manifest[
        manifest["status"].astype(str).str.upper() == "COMPLETE"
    ].copy()

    if len(manifest) != EXPECTED_CONFIGS:
        raise ValueError(
            f"Expected {EXPECTED_CONFIGS} manifest rows, found {len(manifest)}"
        )
    if len(complete) != EXPECTED_CONFIGS:
        raise ValueError(
            f"Expected {EXPECTED_CONFIGS}/{EXPECTED_CONFIGS} COMPLETE embeddings, "
            f"found {len(complete)}"
        )

    records = []
    invalid_count = 0

    for idx, row in enumerate(complete.itertuples(index=False), start=1):
        exp = str(row.experiment)
        rid = str(row.representation)
        lang = str(row.language)
        trunc = str(row.truncation)

        stem = f"{exp}__{rid}__{lang}__{trunc}"
        print(f"[{idx:02d}/{EXPECTED_CONFIGS}] {stem}", flush=True)

        record = {
            "experiment": exp,
            "representation": rid,
            "language": lang,
            "truncation": trunc,
            "stem": stem,
            "status": "INVALID",
            "error": "",
        }

        try:
            paths = {
                key: labels_dir / f"{stem}{suffix}"
                for key, suffix in SUFFIXES.items()
            }

            missing = [p.name for p in paths.values() if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing clustering artifacts: {missing}"
                )

            emb_base = repo / "embeddings" / exp / stem
            emb_json = emb_base.with_suffix(".json")
            emb_npy = emb_base.with_suffix(".npy")
            emb_rows = emb_base.with_suffix(".rows.parquet")

            for p in (emb_json, emb_npy, emb_rows):
                if not p.exists():
                    raise FileNotFoundError(
                        f"Source embedding artifact missing: {p}"
                    )

            emb_meta = load_json(emb_json)
            n_rows = int(emb_meta["n_rows"])

            km_main = verify_kmeans(
                paths["kmeans_main"], n_rows, MAIN_N_INIT
            )
            km_diag = verify_kmeans(
                paths["kmeans_diag"], n_rows, DIAG_N_INIT
            )
            agg = verify_agglomerative(
                paths["agglomerative"], n_rows
            )
            link = verify_linkage(
                paths["linkage"], n_rows
            )
            meta = verify_meta(
                paths["meta"],
                exp, rid, lang, trunc,
                emb_meta,
            )

            record.update({
                "status": "COMPLETE",
                "n_rows": n_rows,
                "embedding_dim": int(emb_meta["embedding_dim"]),
                "app_ids_sha256": emb_meta.get("app_ids_sha256"),
                "dataset_sha256": emb_meta.get("dataset_sha256"),
                "models_yaml_sha256": emb_meta.get("models_yaml_sha256"),
                "k": K,
                "main_n_init": MAIN_N_INIT,
                "diag_n_init": DIAG_N_INIT,
                "main_n_iter_min": km_main["n_iter_min"],
                "main_n_iter_max": km_main["n_iter_max"],
                "diag_n_iter_min": km_diag["n_iter_min"],
                "diag_n_iter_max": km_diag["n_iter_max"],
                "agglomerative_n_clusters": agg["n_clusters"],
                "linkage_distance_min": link["distance_min"],
                "linkage_distance_max": link["distance_max"],
                "distance_block_size": meta["distance_block_size"],
                "runtime_total_seconds": meta["runtime_total_seconds"],
            })

            for key, path in paths.items():
                record[f"{key}_bytes"] = path.stat().st_size
                record[f"{key}_sha256"] = sha256(path)

            print("    COMPLETE", flush=True)

        except Exception as e:
            invalid_count += 1
            record["error"] = f"{type(e).__name__}: {e}"
            print(f"    INVALID: {record['error']}", flush=True)

        records.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    complete_count = int((out_df["status"] == "COMPLETE").sum())

    # Count only the five official clustering artifact suffixes.
    official_files = []
    for suffix in SUFFIXES.values():
        official_files.extend(labels_dir.glob(f"*{suffix}"))

    print()
    print("=" * 78)
    print(f"Configs audited : {len(out_df)}")
    print(f"COMPLETE        : {complete_count}")
    print(f"INVALID         : {invalid_count}")
    print(f"Official files  : {len(official_files)} / {EXPECTED_CONFIGS * len(SUFFIXES)}")
    print(f"Manifest        : {out_path}")

    if complete_count != EXPECTED_CONFIGS or invalid_count != 0:
        raise SystemExit(2)

    if len(official_files) != EXPECTED_CONFIGS * len(SUFFIXES):
        raise RuntimeError(
            f"Expected {EXPECTED_CONFIGS * len(SUFFIXES)} official clustering files, "
            f"found {len(official_files)}"
        )

    print("32/32 CLUSTERING VERIFIED COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
