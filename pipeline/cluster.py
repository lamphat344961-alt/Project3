#!/usr/bin/env python
"""
PAAC-VN clustering pipeline.

Frozen protocol:
- k = 49
- KMeans main: n_init=10, outer seeds 42..51
- KMeans diagnostic: n_init=1, outer seeds 42..51
- Agglomerative: average linkage, cosine distance, k=49, deterministic
- Cosine distances are computed blockwise into a SciPy condensed float64 buffer.
- No full (n, n) distance matrix and no scipy.spatial.distance.pdist in the full pipeline.

This script processes exactly ONE embedding configuration per invocation.
Use a separate output directory for smoke tests, e.g. results/smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

K = 49
SEEDS = list(range(42, 52))
MAIN_N_INIT = 10
DIAG_N_INIT = 1

VALID_TRUNC = {
    "tn1": {"char_budget", "native"},
    "tn2": {"token_budget", "native"},
}
VALID_LANG = {
    "tn1": {"vi"},
    "tn2": {"en", "vi"},
}


def format_seconds(seconds: float) -> str:
    """Human-readable runtime without changing any experimental result."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"


def log_time(label: str, seconds: float) -> None:
    print(f"[time] {label}: {format_seconds(seconds)}", flush=True)


def artifact_base(repo: Path, exp: str, rid: str, lang: str, trunc: str) -> Path:
    return repo / "embeddings" / exp / f"{exp}__{rid}__{lang}__{trunc}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_embedding_artifacts(
    npy_path: Path,
    rows_path: Path,
    json_path: Path,
    exp: str,
    rid: str,
    lang: str,
    trunc: str,
) -> tuple[np.ndarray, dict, int]:
    missing = [str(p) for p in (npy_path, rows_path, json_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing embedding artifact(s):\n" + "\n".join(missing))

    meta = load_json(json_path)

    expected_meta = {
        "experiment": exp,
        "representation": rid,
        "language": lang,
        "truncation_mode": trunc,
    }
    for key, expected in expected_meta.items():
        actual = meta.get(key)
        if actual != expected:
            raise ValueError(f"Metadata mismatch {key}: expected={expected!r}, got={actual!r}")

    if meta.get("embedding_dtype") != "float32":
        raise ValueError(f"Expected embedding_dtype='float32', got {meta.get('embedding_dtype')!r}")
    if meta.get("l2_normalized") is not True:
        raise ValueError("Embedding metadata does not confirm l2_normalized=true")

    X = np.load(npy_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Embedding array must be 2-D, got shape={X.shape}")
    if X.dtype != np.float32:
        raise ValueError(f"Embedding .npy dtype must be float32, got {X.dtype}")

    n_rows, dim = map(int, X.shape)
    if meta.get("n_rows") != n_rows:
        raise ValueError(f"n_rows mismatch: metadata={meta.get('n_rows')}, npy={n_rows}")
    if meta.get("embedding_dim") != dim:
        raise ValueError(f"embedding_dim mismatch: metadata={meta.get('embedding_dim')}, npy={dim}")

    pf = pq.ParquetFile(rows_path)
    if pf.metadata.num_rows != n_rows:
        raise ValueError(
            f"rows.parquet count mismatch: parquet={pf.metadata.num_rows}, npy={n_rows}"
        )

    names = set(pf.schema_arrow.names)
    required = {"row_idx", "appId"}
    if not required.issubset(names):
        raise ValueError(f"rows.parquet missing required columns: {sorted(required - names)}")

    table = pq.read_table(rows_path, columns=["row_idx", "appId"])
    row_idx = table["row_idx"].to_numpy(zero_copy_only=False)
    if not np.array_equal(row_idx, np.arange(n_rows, dtype=row_idx.dtype)):
        raise ValueError("rows.parquet row_idx is not exactly 0..n-1")

    app_ids = [str(x) for x in table["appId"].to_pylist()]
    app_hash = hashlib.sha256("\n".join(app_ids).encode("utf-8")).hexdigest()
    if meta.get("app_ids_sha256") != app_hash:
        raise ValueError(
            "app_ids_sha256 mismatch between metadata and rows.parquet; "
            "refuse to cluster misaligned rows."
        )

    return X, meta, n_rows


def run_kmeans_grid(X: np.ndarray, n_init: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.empty((len(SEEDS), X.shape[0]), dtype=np.int32)
    inertia = np.empty(len(SEEDS), dtype=np.float64)
    n_iter = np.empty(len(SEEDS), dtype=np.int32)

    for pos, seed in enumerate(SEEDS):
        print(f"[kmeans] n_init={n_init} seed={seed}", flush=True)
        t_seed = time.perf_counter()

        km = KMeans(
            n_clusters=K,
            n_init=n_init,
            random_state=seed,
        )
        pred = km.fit_predict(X)

        seed_seconds = time.perf_counter() - t_seed
        log_time(f"KMeans n_init={n_init} seed={seed}", seed_seconds)

        if len(np.unique(pred)) != K:
            raise RuntimeError(
                f"KMeans seed={seed}, n_init={n_init} produced "
                f"{len(np.unique(pred))} clusters; expected {K}."
            )
        labels[pos] = pred.astype(np.int32, copy=False)
        inertia[pos] = float(km.inertia_)
        n_iter[pos] = int(km.n_iter_)

    return labels, inertia, n_iter


def condensed_size(n: int) -> int:
    return n * (n - 1) // 2


def condensed_index_start(n: int, i: int, j: int) -> int:
    if not (0 <= i < j < n):
        raise ValueError((i, j, n))
    return n * i - i * (i + 1) // 2 + (j - i - 1)


def build_condensed_cosine(
    X: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Build cosine distances blockwise into a 1-D float64 SciPy condensed buffer.

    X is already L2-normalized by the frozen embedding pipeline, so
    cosine_distance(x_i, x_j) = 1 - dot(x_i, x_j).

    Temporary matrices are at most block_size x block_size; never n x n.
    """
    n = X.shape[0]
    m = condensed_size(n)
    gib = m * 8 / (1024 ** 3)
    print(
        f"[agglomerative] condensed pairs={m:,}; float64 buffer≈{gib:.2f} GiB",
        flush=True,
    )

    y = np.empty(m, dtype=np.float64)

    n_blocks = math.ceil(n / block_size)
    for bi, i0 in enumerate(range(0, n, block_size), start=1):
        i1 = min(i0 + block_size, n)
        Xi = np.asarray(X[i0:i1], dtype=np.float32)

        print(
            f"[distance] row-block {bi}/{n_blocks}: rows {i0}:{i1}",
            flush=True,
        )

        sim_diag = Xi @ Xi.T
        np.clip(sim_diag, -1.0, 1.0, out=sim_diag)

        for li, i in enumerate(range(i0, i1)):
            j_start = i + 1
            if j_start < i1:
                vals = 1.0 - sim_diag[li, (li + 1):]
                start = condensed_index_start(n, i, j_start)
                y[start:start + len(vals)] = vals.astype(np.float64, copy=False)

        del sim_diag

        for j0 in range(i1, n, block_size):
            j1 = min(j0 + block_size, n)
            Xj = np.asarray(X[j0:j1], dtype=np.float32)
            sim = Xi @ Xj.T
            np.clip(sim, -1.0, 1.0, out=sim)
            dist = 1.0 - sim

            width = j1 - j0
            for li, i in enumerate(range(i0, i1)):
                start = condensed_index_start(n, i, j0)
                y[start:start + width] = dist[li].astype(np.float64, copy=False)

            del Xj, sim, dist

    if not np.isfinite(y).all():
        bad = int(np.size(y) - np.isfinite(y).sum())
        raise RuntimeError(f"Condensed distance buffer contains {bad} non-finite values.")

    np.clip(y, 0.0, 2.0, out=y)
    return y


def run_agglomerative(
    X: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    t0 = time.perf_counter()
    y = build_condensed_cosine(X, block_size=block_size)
    distance_seconds = time.perf_counter() - t0
    log_time("Agglomerative cosine-distance construction", distance_seconds)

    print("[agglomerative] scipy linkage(method='average')", flush=True)
    t1 = time.perf_counter()
    Z = linkage(y, method="average")
    linkage_seconds = time.perf_counter() - t1
    log_time("Agglomerative average-linkage", linkage_seconds)

    del y

    lab = fcluster(Z, t=K, criterion="maxclust").astype(np.int32)
    n_clusters = len(np.unique(lab))
    if n_clusters != K:
        raise RuntimeError(
            f"Agglomerative fcluster produced {n_clusters} clusters; expected {K}."
        )

    return lab, Z, {
        "distance_seconds": distance_seconds,
        "linkage_seconds": linkage_seconds,
    }


def safe_output_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "kmeans_main": out_dir / f"{stem}__kmeans_main.npz",
        "kmeans_diag": out_dir / f"{stem}__kmeans_diag.npz",
        "agglomerative": out_dir / f"{stem}__agglomerative.npy",
        "linkage": out_dir / f"{stem}__linkage.npy",
        "meta": out_dir / f"{stem}__cluster_meta.json",
    }


def refuse_overwrite(paths: dict[str, Path], force: bool) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not force:
        msg = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing clustering artifact(s):\n"
            f"{msg}\nUse --force only when replacement is intentional."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--exp", choices=["tn1", "tn2"], required=True)
    ap.add_argument("--repr", required=True, dest="rid")
    ap.add_argument("--lang", choices=["vi", "en"], required=True)
    ap.add_argument(
        "--trunc",
        choices=["char_budget", "token_budget", "native"],
        required=True,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <repo>/results/labels. Use results/smoke for smoke tests.",
    )
    ap.add_argument(
        "--distance-block-size",
        type=int,
        default=512,
        help="BLAS block size for condensed cosine calculation; operational only.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow overwriting this configuration's output artifacts.",
    )
    args = ap.parse_args()

    repo = args.repo.resolve()

    if args.lang not in VALID_LANG[args.exp]:
        raise ValueError(f"{args.exp} does not allow language={args.lang!r}")
    if args.trunc not in VALID_TRUNC[args.exp]:
        raise ValueError(f"{args.exp} does not allow truncation={args.trunc!r}")
    if args.distance_block_size <= 0:
        raise ValueError("--distance-block-size must be > 0")

    base = artifact_base(repo, args.exp, args.rid, args.lang, args.trunc)
    npy_path = base.with_suffix(".npy")
    rows_path = base.with_suffix(".rows.parquet")
    json_path = base.with_suffix(".json")

    X, emb_meta, n_rows = validate_embedding_artifacts(
        npy_path=npy_path,
        rows_path=rows_path,
        json_path=json_path,
        exp=args.exp,
        rid=args.rid,
        lang=args.lang,
        trunc=args.trunc,
    )

    if n_rows < K:
        raise ValueError(f"n_rows={n_rows} < k={K}")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else repo / "results" / "labels"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.exp}__{args.rid}__{args.lang}__{args.trunc}"
    paths = safe_output_paths(out_dir, stem)
    refuse_overwrite(paths, force=args.force)

    print("=" * 72)
    print("PAAC-VN CLUSTERING")
    print(f"configuration : {stem}")
    print(f"shape         : {tuple(X.shape)}")
    print(f"k             : {K}")
    print(f"KMeans seeds  : {SEEDS[0]}..{SEEDS[-1]}")
    print(f"output        : {out_dir}")
    print("=" * 72, flush=True)

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    main_labels, main_inertia, main_n_iter = run_kmeans_grid(X, MAIN_N_INIT)
    main_seconds = time.perf_counter() - t0
    log_time("KMeans main total", main_seconds)

    np.savez_compressed(
        paths["kmeans_main"],
        seeds=np.asarray(SEEDS, dtype=np.int32),
        labels=main_labels,
        inertia=main_inertia,
        n_iter=main_n_iter,
        n_init=np.asarray(MAIN_N_INIT, dtype=np.int32),
        k=np.asarray(K, dtype=np.int32),
    )

    t0 = time.perf_counter()
    diag_labels, diag_inertia, diag_n_iter = run_kmeans_grid(X, DIAG_N_INIT)
    diag_seconds = time.perf_counter() - t0
    log_time("KMeans diagnostic total", diag_seconds)

    np.savez_compressed(
        paths["kmeans_diag"],
        seeds=np.asarray(SEEDS, dtype=np.int32),
        labels=diag_labels,
        inertia=diag_inertia,
        n_iter=diag_n_iter,
        n_init=np.asarray(DIAG_N_INIT, dtype=np.int32),
        k=np.asarray(K, dtype=np.int32),
    )

    ag_labels, Z, agg_times = run_agglomerative(
        X,
        block_size=args.distance_block_size,
    )

    np.save(paths["agglomerative"], ag_labels)
    np.save(paths["linkage"], Z.astype(np.float64, copy=False))

    total_seconds = time.perf_counter() - t_total
    log_time("TOTAL clustering runtime", total_seconds)

    meta = {
        "project": "PAAC-VN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "experiment": args.exp,
            "representation": args.rid,
            "language": args.lang,
            "truncation_mode": args.trunc,
        },
        "source_embedding": {
            "npy": str(npy_path.relative_to(repo).as_posix()),
            "rows": str(rows_path.relative_to(repo).as_posix()),
            "json": str(json_path.relative_to(repo).as_posix()),
            "n_rows": int(X.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "dataset_sha256": emb_meta.get("dataset_sha256"),
            "models_yaml_sha256": emb_meta.get("models_yaml_sha256"),
            "app_ids_sha256": emb_meta.get("app_ids_sha256"),
            "revision": emb_meta.get("revision"),
            "embedding_dtype": emb_meta.get("embedding_dtype"),
            "l2_normalized": emb_meta.get("l2_normalized"),
        },
        "protocol": {
            "k": K,
            "kmeans_main": {
                "n_init": MAIN_N_INIT,
                "outer_seeds": SEEDS,
            },
            "kmeans_diagnostic": {
                "n_init": DIAG_N_INIT,
                "outer_seeds": SEEDS,
            },
            "agglomerative": {
                "method": "average",
                "distance": "cosine",
                "criterion": "maxclust",
                "condensed_dtype": "float64",
                "distance_block_size": args.distance_block_size,
                "square_distance_matrix": False,
                "scipy_pdist_full_pipeline": False,
            },
        },
        "runtime_seconds": {
            "kmeans_main": main_seconds,
            "kmeans_diagnostic": diag_seconds,
            **agg_times,
            "total": total_seconds,
        },
    }
    paths["meta"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("CLUSTERING COMPLETE")
    print(f"KMeans main labels : {main_labels.shape}")
    print(f"KMeans diag labels : {diag_labels.shape}")
    print(f"Agglomerative      : {ag_labels.shape}")
    print(f"Linkage Z          : {Z.shape}")
    print(f"Artifacts          : {out_dir}")
    print(f"Total runtime      : {format_seconds(total_seconds)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
