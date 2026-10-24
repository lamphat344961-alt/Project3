#!/usr/bin/env python3
"""PAAC-VN Gate 4 forensic / TN2-A independent audit (READ-ONLY inputs).

Phases in frozen-plan order:
  --phase provenance: A0, inspect metadata provenance of all 32 artifacts
  --phase videberta : A1, inspect frozen ViDeBERTa checkpoint/tokenizer and geometry
  --phase videberta_consistency: post-A1 targeted label/geometry consistency
  --phase e5        : A2, numerical parity between TN2 E5 512 and native
  --phase tn2a      : A3, independently recompute EVERY TN2-A row from embeddings

Never edits canonical datasets, embeddings, cluster labels or evaluation outputs.
Writes diagnostic JSON into results/audit/forensic_gate4_v2/ by default,
and refuses to overwrite an existing report. This checks numerical consistency,
not historical embedding-byte identity or recovery-chain provenance.
A1 may use the ordinary Hugging Face model cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

TN1_N, TN2_N, K = 23883, 13657, 49
EXPECTED_DATA_HASH = {
    "tn1": "0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b",
    "tn2": "9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493",
}
MODELS_HASH = "bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6"
TN2_REPS = ("bge_m3", "multilingual_e5_large", "qwen3_embedding_0_6b", "tfidf_lsa")
TRUNCS = ("token_budget", "native")
METRICS_A = ("matched_cosine", "cosine_distance", "negative_cosine_median", "delta_cosine")


def digest(path: Path, chunksize: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(chunksize), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fd:
        return json.load(fd)


def assert_hash(path: Path, expected: str, desc: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {desc}: {path}")
    got = digest(path)
    if got != expected:
        raise RuntimeError(f"{desc}: SHA256 mismatch: {got} != {expected}")


def base_path(repo: Path, exp: str, rep: str, lang: str, trunc: str) -> Path:
    stem = "__".join([exp, rep, lang, trunc])
    return repo / "embeddings" / exp / stem


def load_emb(repo: Path, e_meta: dict, exp: str, rep: str,
             lang: str, tr: str) -> tuple[np.ndarray, dict, pd.DataFrame]:
    base = base_path(repo, exp, rep, lang, tr)
    key = base.name
    source = e_meta["sources"][key]
    fn = base.with_suffix(".npy")
    assert_hash(fn, source["sha256_npy_current"], key + " .npy")
    assert_hash(base.with_suffix(".json"), source["sha256_json"], key + " .json")
    meta = read_json(base.with_suffix(".json"))
    if meta["dataset_sha256"] != EXPECTED_DATA_HASH[exp] or meta["models_yaml_sha256"] != MODELS_HASH:
        raise RuntimeError(key + ": dataset/models hash mismatch")
    if (meta["experiment"], meta["representation"], meta["language"], meta["truncation_mode"]) != (exp, rep, lang, tr):
        raise RuntimeError(key + ": config metadata mismatch")
    X = np.load(fn, mmap_mode="r")
    n = TN1_N if exp == "tn1" else TN2_N
    if X.ndim != 2 or X.shape != (n, int(meta["embedding_dim"])) or X.dtype != np.float32:
        raise RuntimeError(key + ": embedding shape or dtype mismatch")
    rows = pd.read_parquet(base.with_suffix(".rows.parquet"), columns=["row_idx", "appId", "genreId"])
    if len(rows) != n or not np.array_equal(rows.row_idx.to_numpy(), np.arange(n)):
        raise RuntimeError(key + ": invalid row alignment")
    return X, meta, rows


def assert_same_rows(a: pd.DataFrame, b: pd.DataFrame, desc: str) -> None:
    for col in ("row_idx", "appId", "genreId"):
        if not a[col].equals(b[col]):
            raise RuntimeError(f"{desc}: mismatch for {col}")


def norm_finiteness(X: np.ndarray, chunk: int = 2048) -> dict:
    max_err, zero, nan, inf = 0.0, 0, 0, 0
    for s in range(0, len(X), chunk):
        block = np.asarray(X[s:s+chunk], dtype=np.float64)
        nan += int(np.isnan(block).sum())
        inf += int(np.isinf(block).sum())
        norms = np.linalg.norm(block, axis=1)
        zero += int(np.sum(norms == 0))
        if np.any(norms > 0):
            max_err = max(max_err, float(np.max(abs(norms[norms > 0] - 1))))
    if nan or inf or max_err > 1e-3:
        raise RuntimeError(f"Invalid embedding contents: nan={nan}, inf={inf}, norm_error={max_err}")
    return {"zero_vectors": zero, "max_nonzero_norm_error": max_err}


def quantiles(v: np.ndarray) -> dict:
    a = np.asarray(v, dtype=np.float64)
    return {"min": float(np.min(a)), "p01": float(np.percentile(a, 1)),
            "p05": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(np.max(a))}


def geometry_readonly(X: np.ndarray, genre: np.ndarray, seed: int = 42) -> dict:
    """Sample pairs solely as diagnostic; no new official evaluation metric."""
    n, d = X.shape
    totals = np.zeros(d, dtype=np.float64)
    sq_totals = np.zeros(d, dtype=np.float64)
    for s in range(0, n, 2048):
        block = X[s:s+2048].astype(np.float64)
        totals += block.sum(axis=0)
        sq_totals += (block * block).sum(axis=0)
    mu = totals / n
    var = np.maximum(0.0, sq_totals / n - mu * mu)

    rng = np.random.default_rng(seed)
    ii = rng.integers(0, n, size=120_000)
    jj = rng.integers(0, n, size=120_000)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]
    sim = np.empty(len(ii), dtype=np.float64)
    for s in range(0, len(ii), 4096):
        e = min(s + 4096, len(ii))
        a = X[ii[s:e]].astype(np.float64)
        b = X[jj[s:e]].astype(np.float64)
        sim[s:e] = np.einsum("ij,ij->i", a, b)
    same = genre[ii] == genre[jj]
    return {
        "feature_variance": quantiles(var),
        "near_constant_feature_dims_variance_lt_1e-10": int(np.sum(var < 1e-10)),
        "diagnostic_random_pair_count": int(len(ii)),
        "diagnostic_same_genre_pair_count": int(same.sum()),
        "all_pair_cosine": quantiles(sim),
        "same_genre_pair_cosine": quantiles(sim[same]) if same.any() else None,
        "different_genre_pair_cosine": quantiles(sim[~same]) if (~same).any() else None,
        "random_sampling_seed": seed,
        "warning": "Pair samples are forensic diagnostics, not a new benchmark or significance test.",
    }


def phase_provenance(repo: Path, e_meta: dict, modelcfg: dict) -> dict:
    """Metadata traceability audit. Does not claim code hashes retroactively."""
    import subprocess
    records, versions = [], {}
    observed = set(e_meta["sources"])
    expected_grid = {
        f"tn1__{rid}__vi__{tr}"
        for rid, spec in modelcfg["representations"].items()
        if bool(spec.get("tn1", False))
        for tr in ("char_budget", "native")
    } | {
        f"tn2__{rid}__{lang}__{tr}"
        for rid, spec in modelcfg["representations"].items()
        if bool(spec.get("tn2", False))
        for lang in ("en", "vi") for tr in ("token_budget", "native")
    }
    if len(observed) != 32 or observed != expected_grid:
        raise RuntimeError(f"Frozen embedding grid mismatch: found={len(observed)}, "
                           f"expected={len(expected_grid)}; "
                           f"missing={sorted(expected_grid-observed)}, "
                           f"extra={sorted(observed-expected_grid)}")
    for key in sorted(observed):
        exp, *rest = key.split("__")
        tr = rest[-1]
        lang = rest[-2]
        rep = "__".join(rest[:-2])
        base = base_path(repo, exp, rep, lang, tr)
        obj = read_json(base.with_suffix(".json"))
        assert_hash(base.with_suffix(".json"), e_meta["sources"][key]["sha256_json"], key + " .json")
        spec = modelcfg["representations"][rep]
        expected = {
            "experiment": exp, "representation": rep, "language": lang,
            "truncation_mode": tr,
            "dataset_sha256": EXPECTED_DATA_HASH[exp],
            "models_yaml_sha256": MODELS_HASH,
            "hf_name": spec["hf_name"], "revision": spec["revision"],
            "max_len_native": spec["max_len"], "pooling": spec["pooling"],
            "needs_word_segmentation": spec["needs_word_segmentation"],
            "clustering_prefix": spec["clustering_prefix"],
            "embedding_dim": spec["dim"],
            "n_rows": TN1_N if exp == "tn1" else TN2_N,
            "embedding_dtype": "float32",
            "l2_normalized": True,
            "n_chars": 950 if tr == "char_budget" else None,
            "max_tokens_requested": 512 if tr == "token_budget" else None,
        }
        for field, value in expected.items():
            if obj.get(field) != value:
                raise RuntimeError(f"{key}: {field} differs: {obj.get(field)!r} vs {value!r}")
        dtype = obj.get("model_compute_dtype")
        neural = rep != "tfidf_lsa"
        versions.setdefault(json.dumps(obj.get("package_versions", {}), sort_keys=True), []).append(key)
        records.append({
            "artifact": key, "neural": neural,
            "compute_dtype_recorded": dtype if dtype is not None else "NOT_RECORDED",
            "batch_size": obj.get("batch_size"),
            "model_revision": obj["revision"],
            "metadata_sha256": digest(base.with_suffix(".json")),
            "has_script_sha256": "script_sha256" in obj,
        })
    try:
        proc = subprocess.run(
            ["git", "log", "--format=%h %aI %s", "--follow", "--", "pipeline/embed.py"],
            cwd=repo, text=True, capture_output=True, timeout=15, check=False,
        )
        git_log = proc.stdout.splitlines()[:80] if proc.returncode == 0 else []
        git_note = proc.stderr[:500] if proc.returncode != 0 else "git history recorded, but not tied to each artifact"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        git_log, git_note = [], str(e)
    compute_groups = {}
    for row in records:
        k = ("neural" if row["neural"] else "tfidf") + ":" + row["compute_dtype_recorded"]
        compute_groups[k] = compute_groups.get(k, 0) + 1
    return {
        "phase": "A0_metadata_provenance",
        "status": "AUDITED_WITH_HISTORICAL_SCRIPT_LIMITATION",
        "count": len(records), "dtype_group_counts": compute_groups,
        "package_version_groups": [{"package_versions": json.loads(k), "count": len(keys), "artifacts": keys}
                                   for k, keys in versions.items()],
        "git_history": git_log, "git_note": git_note,
        "records": records,
        "limitation": "A current code hash or git history cannot prove which historical script generated each NPY."
    }


def videberta_checkpoint_and_tokenizer(repo: Path, spec: dict) -> dict:
    """Inspect exact frozen checkpoint loading and fast/slow UNK behavior; no embedding changes."""
    import gc
    import torch
    from pyvi import ViTokenizer
    from transformers import AutoModel, AutoTokenizer

    name, rev = spec["hf_name"], spec["revision"]
    loaded = AutoModel.from_pretrained(
        name, revision=rev, trust_remote_code=False,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Expected (model, loading_info) for output_loading_info=True")
    model, info = loaded
    all_missing = list(info.get("missing_keys", []))
    critical_missing = [key for key in all_missing
                        if ("embeddings." in key or "encoder." in key)]
    loading = {
        "hf_name": name, "revision": rev,
        "runtime_model_class": type(model).__name__,
        "dtype_first_parameter": str(next(model.parameters()).dtype),
        "missing_keys": all_missing,
        "missing_backbone_keys": critical_missing,
        "unexpected_keys": list(info.get("unexpected_keys", [])),
        "mismatched_keys": [str(k) for k in info.get("mismatched_keys", [])],
        "error_msgs": [str(v) for v in info.get("error_msgs", [])],
        "interpretation": "Inspect missing BACKBONE keys only; auxiliary head/pooler differences do not prove random encoder.",
    }
    del model
    gc.collect()
    print(f"[A1 checkpoint] model={loading['runtime_model_class']}; "
          f"missing={len(all_missing)}; missing_backbone={len(critical_missing)}; "
          f"unexpected={len(loading['unexpected_keys'])}", flush=True)

    fast = AutoTokenizer.from_pretrained(
        name, revision=rev, use_fast=True, trust_remote_code=False,
    )
    slow = None
    slow_error = None
    try:
        slow = AutoTokenizer.from_pretrained(
            name, revision=rev, use_fast=False, trust_remote_code=False,
        )
    except Exception as exc:
        slow_error = f"{type(exc).__name__}: {str(exc)[:600]}"
    if slow is not None and getattr(slow, "is_fast", False):
        slow_error = "use_fast=False resolved to a fast implementation; not a real fast-vs-slow comparison."
        slow = None
    ds = pd.read_parquet(repo / "data_v1/benchmark_v1.parquet", columns=["description_clean"])
    rng = np.random.default_rng(42026)
    candidates = np.setdiff1d(np.arange(len(ds)), np.array([639, 22342]))
    selected = sorted([639, 22342] + rng.choice(candidates, size=8, replace=False).tolist())
    cases = []
    for idx in selected:
        full = str(ds.description_clean.iloc[idx])
        for tr in ("char_budget", "native"):
            if tr == "char_budget" and len(full) > 950:
                prefix = full[:950]
                if full[950].isspace():
                    inp = prefix.rstrip()
                else:
                    import re
                    matches = list(re.finditer(r"\s+", prefix))
                    if not matches:
                        inp = prefix.rstrip()
                    else:
                        cut_at = matches[-1].start()
                        inp = (prefix[:cut_at].rstrip() if cut_at >= 20
                               else prefix.rstrip())
            else:
                inp = full.strip()
            segmented = ViTokenizer.tokenize(inp)
            one = {"row_idx": int(idx), "truncation": tr, "input_chars": len(inp),
                   "segmented_chars": len(segmented)}
            for label, tok in (("fast", fast), ("slow", slow)):
                if tok is None: continue
                ids = tok(segmented, add_special_tokens=True, truncation=True,
                          max_length=512)["input_ids"]
                unk_id = tok.unk_token_id
                unk = sum(t == unk_id for t in ids) if unk_id is not None else None
                one[label + "_tokens"] = len(ids)
                one[label + "_unk_count"] = unk
                one[label + "_unk_fraction"] = (unk / len(ids)) if unk is not None and ids else None
            cases.append(one)
    print(f"[A1 tokenizer] 10 app samples × 2 truncation modes; slow_available={slow is not None}", flush=True)
    return {"checkpoint_loading": loading, "tokenizer": {
        "fast_class": type(fast).__name__,
        "fast_is_fast": bool(fast.is_fast),
        "slow_class": type(slow).__name__ if slow is not None else None,
        "slow_unavailable_reason": slow_error,
        "samples": cases,
        "limitation": "10 samples and UNK alone do not prove pretraining tokenizer compatibility."
    }}


def phase_videberta(repo: Path, e_meta: dict, modelcfg: dict) -> dict:
    spec = modelcfg["representations"]["videberta_base"]
    expectation = {"max_len": 512, "pooling": "mean_attention_mask",
                   "needs_word_segmentation": True, "clustering_prefix": ""}
    for k, v in expectation.items():
        if spec[k] != v:
            raise RuntimeError(f"ViDeBERTa frozen {k} mismatch")
    assert_hash(repo / "data_v1/benchmark_v1.parquet", EXPECTED_DATA_HASH["tn1"], "TN1 dataset")
    ds = pd.read_parquet(repo / "data_v1/benchmark_v1.parquet", columns=["appId", "genreId"])
    if len(ds) != TN1_N: raise RuntimeError("TN1 wrong row count")
    results = {}
    for tr in ("char_budget", "native"):
        X, meta, rows = load_emb(repo, e_meta, "tn1", "videberta_base", "vi", tr)
        if (not rows.appId.equals(ds.appId) or not rows.genreId.equals(ds.genreId)):
            raise RuntimeError(f"ViDeBERTa {tr}: original dataset mismatch")
        if meta.get("model_compute_dtype") != "torch.float32":
            raise RuntimeError("ViDeBERTa FP32 repair provenance missing")
        if tr == "char_budget" and meta.get("n_chars") != 950:
            raise RuntimeError("TN1 char budget != 950")
        agg = np.load(repo / "results/labels" / f"tn1__videberta_base__vi__{tr}__agglomerative.npy", mmap_mode="r")
        lab, counts = np.unique(agg, return_counts=True)
        if len(lab) != K:
            raise RuntimeError("Agglomerative does not have 49 clusters")
        results[tr] = {
            "embedding_checks": norm_finiteness(X),
            "embedding_dim": int(X.shape[1]),
            "model_compute_dtype": meta.get("model_compute_dtype"),
            "preprocessing_metadata": {k: meta.get(k) for k in ["n_chars", "effective_max_tokens",
                                                             "truncated_count", "truncation_rate",
                                                             "token_length_mean_before_trunc", "token_length_p95_before_trunc"]},
            "geometry": geometry_readonly(X, ds.genreId.to_numpy()),
            "agglomerative_biggest_cluster_fraction": float(counts.max()/len(agg)),
            "agglomerative_singletons": int((counts == 1).sum()),
        }
        print(f"[A1] ViDeBERTa {tr}: norm/rows verified; biggest Agg cluster="
              f"{counts.max()/len(agg):.4%}; cosine diagnostics saved", flush=True)
    checkpoint_probe = videberta_checkpoint_and_tokenizer(repo, spec)
    return {"phase": "A1_ViDeBERTa", "status": "DIAGNOSTICS_RECORDED_NOT_CAUSAL_PROOF",
            "frozen_recipe": expectation, "results": results,
            **checkpoint_probe}


def chronological_diagnostic(embedding_meta: dict, cluster_meta: dict) -> dict:
    """Timestamp anomaly indicator only; NEVER a proof of historical byte identity."""
    emb = embedding_meta.get("created_at_utc")
    clu = cluster_meta.get("created_at_utc")
    out = {"embedding_created_at_utc": emb, "cluster_created_at_utc": clu,
           "timestamp_order": "NOT_VERIFIED"}
    try:
        if emb is not None and clu is not None:
            a, b = datetime.fromisoformat(emb), datetime.fromisoformat(clu)
            if a.utcoffset() is not None and b.utcoffset() is not None:
                out["timestamp_order"] = ("EMBEDDING_AFTER_CLUSTER_META_ANOMALY"
                    if a > b else "CHRONOLOGY_COMPATIBLE_NOT_PROOF")
    except (ValueError, TypeError):
        pass
    return out


def check_frozen_kmeans_labels(X: np.ndarray, path: Path) -> dict:
    """Read-only Lloyd fixed-point diagnostic for each of 10 stored seeds.

    Uses centroids computed from frozen labels and CURRENT vectors. A tolerance
    or fixed-point mismatch does not establish historical NPY byte identity.
    Does not rerun KMeans or generate scientific evaluation metrics.
    """
    with np.load(path, allow_pickle=False) as z:
        seeds = np.asarray(z["seeds"]).tolist()
        labs = np.asarray(z["labels"])
        inertia = np.asarray(z["inertia"], dtype=np.float64)
        n_iter = np.asarray(z["n_iter"], dtype=np.int64)
        k = int(np.asarray(z["k"]).item())
        n_init = int(np.asarray(z["n_init"]).item())
    if seeds != list(range(42, 52)) or k != K or labs.shape != (10, len(X)):
        raise RuntimeError(path.name + ": invalid frozen KMeans seeds/k/labels shape")
    expected_init = 1 if path.name.endswith("__kmeans_diag.npz") else 10
    if n_init != expected_init or inertia.shape != (10,) or n_iter.shape != (10,):
        raise RuntimeError(path.name + ": invalid n_init, inertia or n_iter")

    out = []
    n = len(X)
    for pos, seed in enumerate(seeds):
        y = labs[pos]
        ids, count = np.unique(y, return_counts=True)
        if len(ids) != K or not np.array_equal(ids, np.arange(K)):
            raise RuntimeError(path.name + f": unexpected label ids for seed {seed}")
        # Accumulate in float64 for a reference calculation. The original
        # sklearn fit uses input float32, so small inertia gaps are expected.
        centroids = np.empty((K, X.shape[1]), dtype=np.float64)
        for cid in range(K):
            centroids[cid] = np.mean(X[y == cid], axis=0, dtype=np.float64)
        norm_c = np.einsum("ij,ij->i", centroids, centroids)
        total_inertia = 0.0
        mismatches = 0
        examples = []
        mismatch_assigned_gap_max = 0.0
        for start in range(0, n, 1024):
            end = min(start + 1024, n)
            xb = np.asarray(X[start:end], dtype=np.float64)
            yi = y[start:end]
            distances = np.einsum("ij,ij->i", xb, xb)[:, None] + norm_c[None, :] - 2 * (xb @ centroids.T)
            nearest = np.argmin(distances, axis=1)
            errors = np.flatnonzero(nearest != yi)
            mismatches += len(errors)
            for local in errors[:max(0, 12 - len(examples))]:
                examples.append(int(start + local))
            assigned = distances[np.arange(len(xb)), yi]
            nearest_distance = distances[np.arange(len(xb)), nearest]
            if len(errors):
                mismatch_assigned_gap_max = max(mismatch_assigned_gap_max,
                    float(np.max(assigned[errors] - nearest_distance[errors])))
            # Inertia of STORED labels with recomputed means, not a new fit.
            total_inertia += float(np.sum(assigned, dtype=np.float64))
        delta = total_inertia - float(inertia[pos])
        out.append({
            "seed": int(seed), "n_init": n_init, "n_iter_recorded": int(n_iter[pos]),
            "n_iter_reached_default_max_iter_300": bool(n_iter[pos] >= 300),
            "n_points": n, "number_of_non_nearest_centroid_assignments": int(mismatches),
            "mismatch_fraction": float(mismatches / n),
            "mismatch_row_idx_examples": examples,
            "max_assigned_minus_nearest_squared_distance_for_mismatches": mismatch_assigned_gap_max,
            "inertia_recorded": float(inertia[pos]),
            "inertia_recomputed_from_stored_labels_and_current_X_float64": total_inertia,
            "inertia_delta_signed": delta, "inertia_delta_abs": abs(delta),
            "inertia_delta_relative_abs": abs(delta) / max(abs(float(inertia[pos])), 1e-30),
        })
    return {"artifact": path.name, "seed_diagnostics": out,
            "warning": "No arbitrary 99% cutoff; nonzero mismatches and inertia gaps require interpretation."
        }


def check_frozen_linkage(X: np.ndarray, zpath: Path, labelpath: Path) -> dict:
    """All recorded merge heights vs current-X average-cosine identity.

    The clustering script originally used BLOCKWISE FLOAT32 dot products,
    clipped cosine similarity to [-1,1], and distances to [0,2].
    Mean-identity computed here uses float64 aggregation without per-pair
    clipping; rounding/clipping differences are reported, not presumed defects.
    """
    from scipy.cluster.hierarchy import fcluster

    n, dim = X.shape
    z = np.load(zpath, mmap_mode="r")
    old = np.load(labelpath, mmap_mode="r")
    if z.shape != (n-1, 4) or z.dtype != np.float64 or old.shape != (n,):
        raise RuntimeError("ViDeBERTa linkage Z / labels incompatible with current X")
    if not np.isfinite(z).all():
        raise RuntimeError("Nonfinite linkage heights")

    z_for_fcluster = np.array(
    z, dtype=np.float64, order="C", copy=True
    )

    new = fcluster(
        z_for_fcluster, K, criterion="maxclust"
    ).astype(np.int32)
    # In original pipeline labels are fcluster(Z,...), not remapped cluster IDs.
    exact_partition_labels = bool(np.array_equal(new, old))
    # At most (n-1)*dim float64; no (n,n) matrix and no new linkage.
    means = np.empty((n-1, dim), dtype=np.float64)
    sizes = np.empty(2*n-1, dtype=np.int32)
    sizes[:n] = 1
    errors = np.empty(n-1, dtype=np.float64)
    highest = []
    for idx, row in enumerate(z):
        ai, bi = int(row[0]), int(row[1])
        if ai == bi or ai < 0 or bi < 0 or ai >= n+idx or bi >= n+idx:
            raise RuntimeError(f"Malformed linkage child at row {idx}")
        a = np.asarray(X[ai], dtype=np.float64) if ai < n else means[ai-n]
        b = np.asarray(X[bi], dtype=np.float64) if bi < n else means[bi-n]
        dist = 1.0 - float(np.dot(a, b))
        errors[idx] = float(row[2]) - dist
        na, nb = int(sizes[ai]), int(sizes[bi])
        sizes[n+idx] = na+nb
        means[idx] = (na * a + nb * b) / (na+nb)
    worst = np.argsort(np.abs(errors))[-10:][::-1]
    worst_records = [{"merge_index": int(i), "recorded_height": float(z[i,2]),
                      "recorded_minus_float64_mean_identity": float(errors[i])}
                     for i in worst]
    return {
        "label_recovered_exactly_by_fcluster": exact_partition_labels,
        "recovered_cluster_count": int(len(np.unique(new))),
        "stored_cluster_count": int(len(np.unique(old))),
        "merge_count_checked": n-1, "max_abs_merge_height_difference": float(np.max(np.abs(errors))),
        "abs_merge_height_error_quantiles": quantiles(np.abs(errors)),
        "merge_height_error_signed_mean": float(np.mean(errors)),
        "number_of_abs_errors_gt_1e-6_descriptive_only": int(np.count_nonzero(np.abs(errors) > 1e-6)),
        "worst_merge_height_rows": worst_records,
        "warning": ("Merge-height consistency does not check minimum competing merges at each step. "
                    "Float32 BLAS and clipping in the frozen algorithm can cause legitimate small errors. "
                    "No proof of historical NPY byte identity or merge-order invariance."),
    }


def phase_videberta_consistency(repo: Path, e_meta: dict) -> dict:
    """Target ONLY the two historical ViDeBERTa FP32 recovery configurations."""
    ds_path = repo / "data_v1/benchmark_v1.parquet"
    assert_hash(ds_path, EXPECTED_DATA_HASH["tn1"], "TN1 dataset")
    original = pd.read_parquet(ds_path, columns=["appId", "genreId"])
    manifest_path = repo / "results/audit/clustering_manifest.csv"
    assert_hash(manifest_path, e_meta["clustering_manifest_sha256"], "Gate 3 audited clustering manifest")
    manifest = pd.read_csv(manifest_path)
    result = {}
    for tr in ("char_budget", "native"):
        X, meta, rows = load_emb(repo, e_meta, "tn1", "videberta_base", "vi", tr)
        if not rows.appId.equals(original.appId) or not rows.genreId.equals(original.genreId):
            raise RuntimeError("ViDeBERTa row order differs from canonical dataset")
        if meta.get("model_compute_dtype") != "torch.float32":
            raise RuntimeError("ViDeBERTa FP32 provenance mismatch")
        norm_finiteness(X)
        stem = f"tn1__videberta_base__vi__{tr}"
        base = repo / "results" / "labels"
        evidence_rows = manifest[(manifest.experiment == "tn1") &
                                 (manifest.representation == "videberta_base") &
                                 (manifest.language == "vi") &
                                 (manifest.truncation == tr)]
        if len(evidence_rows) != 1 or evidence_rows.iloc[0]["status"] != "COMPLETE":
            raise RuntimeError(stem + ": not in the frozen Gate 3 manifest")
        audit_row = evidence_rows.iloc[0]
        for key, suffix in (("kmeans_main", "__kmeans_main.npz"),
                            ("kmeans_diag", "__kmeans_diag.npz"),
                            ("agglomerative", "__agglomerative.npy"),
                            ("linkage", "__linkage.npy"),
                            ("meta", "__cluster_meta.json")):
            assert_hash(base / (stem + suffix), str(audit_row[f"{key}_sha256"]),
                        stem + " Gate3 " + key)
        cm = read_json(base / (stem + "__cluster_meta.json"))
        for field in ("dataset_sha256", "models_yaml_sha256", "app_ids_sha256",
                      "revision", "n_rows", "embedding_dim"):
            if cm.get("source_embedding", {}).get(field) != meta.get(field):
                raise RuntimeError(stem + ": cluster/embedding metadata mismatch: " + field)
        timeline = chronological_diagnostic(meta, cm)
        result[tr] = {
            "npy_sha256_current_not_historic": digest(base_path(repo, "tn1", "videberta_base", "vi", tr).with_suffix(".npy")),
            "kmeans_main": check_frozen_kmeans_labels(X, base / (stem + "__kmeans_main.npz")),
            "kmeans_diagnostic": check_frozen_kmeans_labels(X, base / (stem + "__kmeans_diag.npz")),
            "agglomerative": check_frozen_linkage(X, base / (stem + "__linkage.npy"), base / (stem + "__agglomerative.npy")),
            "timestamp_evidence": timeline,
        }
        print(f"[A1+ consistency] {tr}: stored KMeans and linkage checked against current vectors", flush=True)
    return {
        "phase": "A1plus_ViDeBERTa_label_geometry_consistency",
        "status": "DIAGNOSTICS_RECORDED_INTERPRETATION_REQUIRED",
        "scope": "Only ViDeBERTa TN1 char_budget and native; no reclustering/refit.",
        "results": result,
        "limitation": "Cannot identify historical NPY bytes or guarantee merge-choice optimality using this diagnostic."
    }


def phase_e5(repo: Path, e_meta: dict, modelcfg: dict) -> dict:
    spec = modelcfg["representations"]["multilingual_e5_large"]
    if (spec["max_len"], spec["clustering_prefix"], spec["needs_word_segmentation"]) != (512, "query: ", False):
        raise RuntimeError("E5 frozen input recipe mismatch")
    assert_hash(repo / "data_v1/paired_en.parquet", EXPECTED_DATA_HASH["tn2"], "TN2 dataset")
    ds = pd.read_parquet(repo / "data_v1/paired_en.parquet", columns=["appId", "genreId"])
    results = {}
    for lang in ("en", "vi"):
        a, ma, ra = load_emb(repo, e_meta, "tn2", "multilingual_e5_large", lang, "token_budget")
        b, mb, rb = load_emb(repo, e_meta, "tn2", "multilingual_e5_large", lang, "native")
        assert_same_rows(ra, rb, "E5 " + lang)
        if not ra.appId.equals(ds.appId) or not ra.genreId.equals(ds.genreId):
            raise RuntimeError("E5 dataset app/genre alignment mismatch")
        norm_finiteness(a); norm_finiteness(b)
        if ma.get("max_tokens_requested") != 512 or int(ma["effective_max_tokens"]) != 512 or int(mb["effective_max_tokens"]) != 512:
            raise RuntimeError("E5 token/native context length not both 512")
        exact_rows, differing_elements, total_elements = 0, 0, 0
        max_abs = 0.0
        row_max_diff, row_cos = [], []
        for s in range(0, len(a), 1024):
            aa = np.asarray(a[s:s+1024]); bb = np.asarray(b[s:s+1024]); diff = np.abs(aa.astype(np.float64)-bb.astype(np.float64))
            local_max = np.max(diff, axis=1)
            exact_rows += int(np.sum(np.all(aa == bb, axis=1)))
            differing_elements += int(np.count_nonzero(aa != bb))
            total_elements += aa.size
            max_abs = max(max_abs, float(np.max(diff)))
            row_max_diff.append(local_max)
            row_cos.append(np.einsum('ij,ij->i', aa, bb, dtype=np.float64))
        row_max_diff = np.concatenate(row_max_diff)
        row_cos = np.clip(np.concatenate(row_cos), -1.0, 1.0)
        with np.load(repo / "results/labels" / f"tn2__multilingual_e5_large__{lang}__token_budget__kmeans_main.npz", allow_pickle=False) as za, \
             np.load(repo / "results/labels" / f"tn2__multilingual_e5_large__{lang}__native__kmeans_main.npz", allow_pickle=False) as zb:
            pa = za["labels"]; pb = zb["labels"]
            aris = [float(adjusted_rand_score(pa[i], pb[i])) for i in range(10)]
        results[lang] = {
            "exact_equal_rows": int(exact_rows), "total_rows": TN2_N,
            "different_elements": differing_elements, "total_elements": total_elements,
            "max_absolute_difference": max_abs, "row_max_abs_difference": quantiles(row_max_diff),
            "same_app_cosine_between_modes": quantiles(row_cos),
            "row_counts_max_abs_diff_gt_1e-7_1e-6_1e-5_1e-4_1e-3": {
                str(th): int(np.sum(row_max_diff > th)) for th in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            },
            "kmeans_matched_seed_ari_between_modes": aris,
            "warnings": "KMeans labels can change from tiny embedding changes; this is forensic, not a new benchmark.",
        }
        print(f"[A2] E5 {lang}: exact_rows={exact_rows}/{TN2_N}; "
              f"max_abs_diff={max_abs:.9g}; per-seed ARI={[round(v,4) for v in aris]}", flush=True)
    return {"phase": "A2_E5_TN2_context_parity", "status": "DIAGNOSTICS_RECORDED",
            "both_modes_max_tokens": 512, "results": results}


def independent_full_tn2a_from_vectors(en: np.ndarray, vi: np.ndarray,
                                       chunk: int = 64):
    """Independent reference: ALL non-matched VI vectors for EVERY EN app.

    Uses direct per-row median of a similarity row with the diagonal REMOVED,
    not the original evaluation's np.partition(..., diagonal=inf) algorithm.
    Normalizes the already validated input exactly once to account for fp32 L2
    roundoff, consistent with the frozen cosine definition.
    The GEMM is float32 as in official code; compared with numerical tolerance
    rather than demanding byte-identical floating-point BLAS execution.
    """
    if en.shape != vi.shape or en.ndim != 2:
        raise RuntimeError("Bad TN2 paired embedding shape")
    norm_finiteness(en)
    norm_finiteness(vi)
    n, dim = en.shape
    # Independent normalization; read-only original mmap, no inplace mutation.
    A = np.empty((n, dim), dtype=np.float32)
    B = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, 2048):
        stop = min(n, start + 2048)
        for source, target in ((en,A), (vi,B)):
            block = np.asarray(source[start:stop], dtype=np.float32)
            lengths = np.linalg.norm(block, axis=1, keepdims=True)
            target[start:stop] = block / np.maximum(lengths, np.float32(1e-30))
    B_t = np.asarray(B.T, order="C")
    matched = np.empty(n, dtype=np.float64)
    medians = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        similarities = A[start:stop] @ B_t
        np.clip(similarities, -1., 1., out=similarities)
        for local, row_idx in enumerate(range(start, stop)):
            # Use float64 dot for matched, matching the frozen metric formula.
            matched[row_idx] = float(np.clip(np.dot(
                A[row_idx].astype(np.float64), B[row_idx].astype(np.float64)), -1., 1.))
            # Independent direct median on all 13,656 nonmatching VI apps.
            negatives = np.concatenate((similarities[local, :row_idx],
                                        similarities[local, row_idx+1:]))
            if len(negatives) != n-1:
                raise RuntimeError("Incorrect TN2 negative count")
            medians[row_idx] = float(np.median(negatives.astype(np.float64)))
    return {
        "matched_cosine": matched,
        "cosine_distance": 1. - matched,
        "negative_cosine_median": medians,
        "delta_cosine": matched - medians,
    }


def phase_tn2a(repo: Path, e_meta: dict) -> dict:
    outdir = repo / "results/evaluation"
    f_pairs = outdir / "tn2_a_app_pairs.parquet"
    f_metrics = outdir / "metrics_long.csv"
    assert_hash(f_pairs, e_meta["tn2_a_app_pairs_sha256"], "Gate 4 TN2-A parquet")
    assert_hash(f_metrics, e_meta["metrics_long_sha256"], "Gate 4 metrics_long")
    assert_hash(repo / "data_v1/paired_en.parquet", EXPECTED_DATA_HASH["tn2"], "TN2 dataset")
    original = pd.read_parquet(repo / "data_v1/paired_en.parquet", columns=["appId", "genreId"])
    df = pd.read_parquet(f_pairs)
    metrics = pd.read_csv(f_metrics)
    expected_cols = {"experiment", "representation", "truncation", "row_idx", "appId", "genreId",
                     "matched_cosine", "cosine_distance", "negative_cosine_median", "delta_cosine"}
    if not expected_cols.issubset(df.columns):
        raise RuntimeError(f"TN2-A Parquet missing columns: {expected_cols - set(df.columns)}")
    expected = {(r, t) for r in TN2_REPS for t in TRUNCS}
    found = set(zip(df.representation, df.truncation))
    if found != expected or len(df) != 8 * TN2_N:
        raise RuntimeError(f"Wrong config grid or row count: {len(df)}")
    if set(df.experiment) != {"tn2"}:
        raise RuntimeError("Wrong experiment")
    if df.duplicated(["representation", "truncation", "row_idx"]).any():
        raise RuntimeError("Duplicate TN2-A (config, row_idx)")
    if len(original) != TN2_N or original.appId.duplicated().any() or original.appId.isna().any():
        raise RuntimeError("Canonical TN2 appIDs invalid")

    # Full recomputation, not an anchor sample. Values are compared on each
    # app, and failures are reported in JSON without modifying official data.
    tolerance = 2e-5  # numerical roundoff comparison; NOT a model-performance threshold
    per_config, any_failure = {}, False
    for rep, tr in sorted(expected):
        sub = df[(df.representation == rep) & (df.truncation == tr)].sort_values("row_idx")
        if len(sub) != TN2_N or not np.array_equal(sub.row_idx.to_numpy(), np.arange(TN2_N)):
            raise RuntimeError(f"TN2-A {rep}/{tr}: invalid row_idx")
        if not np.array_equal(sub.appId.to_numpy(), original.appId.to_numpy()) or \
           not np.array_equal(sub.genreId.to_numpy(), original.genreId.to_numpy()):
            raise RuntimeError(f"TN2-A {rep}/{tr}: not aligned to canonical dataset")
        en, _, er = load_emb(repo, e_meta, "tn2", rep, "en", tr)
        vi, _, vr = load_emb(repo, e_meta, "tn2", rep, "vi", tr)
        assert_same_rows(er, vr, "TN2-A EN/VI")
        if not np.array_equal(er.appId.to_numpy(), original.appId.to_numpy()) or \
           not np.array_equal(er.genreId.to_numpy(), original.genreId.to_numpy()):
            raise RuntimeError(f"TN2-A {rep}/{tr}: embedding rows misaligned")
        reference = independent_full_tn2a_from_vectors(en, vi)
        measured = {}
        config_failed = False
        for metric in METRICS_A:
            v = sub[metric].to_numpy(dtype=np.float64)
            if not np.isfinite(v).all():
                raise RuntimeError(f"TN2-A {rep}/{tr}/{metric}: NaN or Inf")
            ref = reference[metric]
            if metric in ("matched_cosine", "negative_cosine_median") and np.any(np.abs(v)>1+1e-6):
                raise RuntimeError(f"TN2-A {rep}/{tr}/{metric}: invalid cosine bounds")
            diff = np.abs(ref - v)
            bad = np.flatnonzero(diff > tolerance)
            if len(bad):
                config_failed = True
            measured[metric] = {
                "recomputed_rows": len(diff), "max_abs_error": float(np.max(diff)),
                "mean_abs_error": float(np.mean(diff)),
                "errors_over_1e-7": int(np.count_nonzero(diff > 1e-7)),
                "errors_over_1e-6": int(np.count_nonzero(diff > 1e-6)),
                "errors_over_1e-5": int(np.count_nonzero(diff > 1e-5)),
                "errors_over_2e-5": int(len(bad)),
                "mismatch_row_idx_examples": bad[:20].astype(int).tolist(),
            }
            # Original official summaries are audited separately from recomputation.
            sm = metrics[(metrics.section == "TN2_A") & (metrics.representation == rep) &
                         (metrics.truncation == tr) & (metrics.metric == metric)]
            if len(sm) != 2 or set(sm.pair_role) != {"mean", "median"}:
                raise RuntimeError(f"TN2-A metric_long summary missing for {rep}/{tr}/{metric}")
            for stat, observed in (("mean", float(np.mean(v))), ("median", float(np.median(v)))):
                value = float(sm.loc[sm.pair_role == stat, "value"].iloc[0])
                if not np.isfinite(value) or abs(value-observed) > 1e-8:
                    raise RuntimeError(f"TN2-A {rep}/{tr}/{metric}: metrics_long summary mismatch")
        # Verify internal algebraic relations on ALL existing rows.
        pair = sub.matched_cosine.to_numpy(dtype=np.float64)
        neg = sub.negative_cosine_median.to_numpy(dtype=np.float64)
        dist = sub.cosine_distance.to_numpy(dtype=np.float64)
        delt = sub.delta_cosine.to_numpy(dtype=np.float64)
        id_d = float(np.max(np.abs(dist - (1 - pair))))
        id_c = float(np.max(np.abs(delt - (pair - neg))))
        if id_d > 2e-7 or id_c > 2e-7:
            config_failed = True
        per_config[rep + "__" + tr] = {
            "n_apps": TN2_N, "all_nonmatched_negatives_per_app": TN2_N-1,
            "metrics": measured,
            "existing_row_distance_identity_max_abs_error": id_d,
            "existing_row_delta_identity_max_abs_error": id_c,
            "status": "DISCREPANCY" if config_failed else "CONSISTENT_WITH_TOLERANCE",
        }
        any_failure |= config_failed
        print(f"[A3] {rep}/{tr}: independently recomputed {TN2_N} rows; "
              f"max_negative_error={measured['negative_cosine_median']['max_abs_error']:.3g}; "
              f"status={per_config[rep + '__' + tr]['status']}", flush=True)
    return {
        "phase": "A3_TN2A_full_independent_row_recomputation",
        "status": "DISCREPANCY_REQUIRES_REVIEW" if any_failure else "NUMERICALLY_CONSISTENT",
        "configurations": len(per_config), "total_rows": len(df),
        "independently_recomputed_rows": len(df),
        "negative_sample_count_per_app": TN2_N-1,
        "numeric_tolerance_not_scientific_threshold": tolerance,
        "scope": ("Exact all-nonmatched per-app median recomputed for all 109256 stored rows; "
                  "no random anchor sampling, no official metric changes, no clustering."),
        "limitation": ("Numeric tolerance is not proof of identical BLAS execution or historic "
                       "embedding bytes. User must inspect discrepancies before sealing Gate 4."),
        "results": per_config,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--phase", choices=["provenance", "videberta", "videberta_consistency", "e5", "tn2a"], required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/audit/forensic_gate4_v2"))
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    report_path = out / (args.phase + ".json")
    if report_path.exists():
        raise FileExistsError(f"Refusing overwrite of forensic report: {report_path}")
    assert_hash(repo / "pipeline/models.yaml", MODELS_HASH, "frozen model configuration")
    e_meta = read_json(repo / "results/evaluation/evaluation_meta.json")
    if e_meta.get("status") != "COMPLETE" or e_meta.get("n_metrics_rows") != 30128:
        raise RuntimeError("Gate4 metadata not complete")
    modelcfg = yaml.safe_load((repo / "pipeline/models.yaml").read_text(encoding="utf-8"))
    if args.phase == "provenance":
        result = phase_provenance(repo, e_meta, modelcfg)
    elif args.phase == "videberta":
        result = phase_videberta(repo, e_meta, modelcfg)
    elif args.phase == "videberta_consistency":
        if not (out / "videberta.json").is_file():
            raise RuntimeError("A1 videberta.json required before targeted consistency diagnostic")
        result = phase_videberta_consistency(repo, e_meta)
    elif args.phase == "e5":
        result = phase_e5(repo, e_meta, modelcfg)
    else:
        result = phase_tn2a(repo, e_meta)
    result.update({"project": "PAAC-VN", "forensic_script_sha256": digest(Path(__file__).resolve()),
                   "source_evaluation_meta_sha256":
                   digest(repo / "results/evaluation/evaluation_meta.json"),
                   "no_official_inputs_modified": True})
    out.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".json.tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(report_path)
    print(f"{args.phase.upper()} FORENSIC REPORT WRITTEN: {report_path}")
    print("NOTE: Loading-info absence is not proof of full reproducibility or pretraining semantic validity.")

if __name__ == "__main__":
    main()
