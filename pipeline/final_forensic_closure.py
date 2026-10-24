#!/usr/bin/env python3
"""PAAC-VN Final Computational Closure — C1 forensic assurance.

Phases are independent and may run concurrently:
  --phase a1c : pre-registered ViDeBERTa checkpoint weight/_weight mapping audit
  --phase a2  : all-row multilingual-E5 token_budget/native numerical parity audit
  --phase a3  : all-row independent TN2-A recomputation from frozen embeddings

Canonical data/embeddings/labels/evaluation artifacts are READ-ONLY. Each phase
writes one new JSON report under results/audit/final_forensic_closure/ and refuses
overwrite. This script does not train, re-embed, re-cluster, or change metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TN1_N = 23883
TN2_N = 13657
MODEL_HASH = "bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6"
DATA_HASH = {
    "tn1": "0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b",
    "tn2": "9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493",
}
TN2_REPS = ("bge_m3", "multilingual_e5_large", "qwen3_embedding_0_6b", "tfidf_lsa")
TRUNCS = ("token_budget", "native")


def sha256(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def qstats(x: np.ndarray) -> dict:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {k: None for k in ("min", "p01", "p05", "median", "mean", "p95", "p99", "max")}
    return {
        "min": float(np.min(a)),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def row_hash(values) -> str:
    h = hashlib.sha256()
    for v in values:
        b = str(v).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def embedding_base(repo: Path, exp: str, rep: str, lang: str, tr: str) -> Path:
    return repo / "embeddings" / exp / f"{exp}__{rep}__{lang}__{tr}"


def load_eval_meta(repo: Path) -> dict:
    p = repo / "results/evaluation/evaluation_meta.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    m = read_json(p)
    if m.get("status") != "COMPLETE" or int(m.get("n_metrics_rows", -1)) != 30128:
        raise RuntimeError("Gate 4 evaluation metadata is not COMPLETE/30128")
    return m


def load_embedding(repo: Path, eval_meta: dict, exp: str, rep: str, lang: str, tr: str):
    base = embedding_base(repo, exp, rep, lang, tr)
    key = base.name
    src = eval_meta["sources"][key]
    npy = base.with_suffix(".npy")
    meta_p = base.with_suffix(".json")
    rows_p = base.with_suffix(".rows.parquet")
    if sha256(npy) != src["sha256_npy_current"]:
        raise RuntimeError(f"{key}: current NPY hash mismatch vs Gate4 metadata")
    if sha256(meta_p) != src["sha256_json"]:
        raise RuntimeError(f"{key}: JSON hash mismatch vs Gate4 metadata")
    meta = read_json(meta_p)
    if meta.get("dataset_sha256") != DATA_HASH[exp] or meta.get("models_yaml_sha256") != MODEL_HASH:
        raise RuntimeError(f"{key}: frozen dataset/model hash mismatch")
    X = np.load(npy, mmap_mode="r")
    n = TN1_N if exp == "tn1" else TN2_N
    if X.dtype != np.float32 or X.ndim != 2 or X.shape[0] != n:
        raise RuntimeError(f"{key}: shape/dtype mismatch: {X.shape}/{X.dtype}")
    rows = pd.read_parquet(rows_p, columns=["row_idx", "appId", "genreId"])
    if len(rows) != n or not np.array_equal(rows.row_idx.to_numpy(), np.arange(n)):
        raise RuntimeError(f"{key}: row index mismatch")
    return X, meta, rows


def finite_unit_copy(X: np.ndarray, chunk_rows: int = 2048) -> np.ndarray:
    """Independent roundoff normalization mirroring frozen evaluation semantics."""
    out = np.empty(X.shape, dtype=np.float32)
    max_nonzero_err = 0.0
    for s in range(0, len(X), chunk_rows):
        e = min(len(X), s + chunk_rows)
        z = np.asarray(X[s:e], dtype=np.float32)
        if not np.isfinite(z).all():
            raise RuntimeError(f"Nonfinite embedding rows {s}:{e}")
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        nz = norms[:, 0] > 0
        if nz.any():
            max_nonzero_err = max(max_nonzero_err,
                                  float(np.max(np.abs(norms[nz, 0] - 1.0))))
        out[s:e] = z / np.maximum(norms, np.float32(1e-30))
    if max_nonzero_err > 1e-3:
        raise RuntimeError(f"Frozen embedding norm validation failed: {max_nonzero_err}")
    return out

# ---------------- A1-C ----------------

def _snapshot_local(repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision,
                                      local_files_only=True)).resolve()
    except Exception as exc:
        raise FileNotFoundError(
            f"Frozen checkpoint not available in local HuggingFace cache: {repo_id}@{revision}; "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _state_file_inventory(snapshot: Path) -> tuple[list[dict], dict[str, tuple[Path, str]]]:
    """Return file inventory and tensor key -> (file, format) map without semantic assumptions."""
    inventory = []
    mapping: dict[str, tuple[Path, str]] = {}
    st_index = snapshot / "model.safetensors.index.json"
    pt_index = snapshot / "pytorch_model.bin.index.json"
    if st_index.is_file():
        idx = read_json(st_index)
        for key, rel in idx["weight_map"].items():
            mapping[key] = (snapshot / rel, "safetensors")
    elif pt_index.is_file():
        idx = read_json(pt_index)
        for key, rel in idx["weight_map"].items():
            mapping[key] = (snapshot / rel, "torch")
    else:
        st = snapshot / "model.safetensors"
        pt = snapshot / "pytorch_model.bin"
        if st.is_file():
            from safetensors import safe_open
            with safe_open(str(st), framework="pt", device="cpu") as f:
                for key in f.keys():
                    mapping[key] = (st, "safetensors")
        elif pt.is_file():
            import torch
            obj = torch.load(pt, map_location="cpu", weights_only=True)
            state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
            for key in state.keys():
                mapping[key] = (pt, "torch")
            del obj, state
        else:
            raise FileNotFoundError(f"No model.safetensors/pytorch_model.bin found in {snapshot}")
    for p in sorted(set(v[0] for v in mapping.values())):
        inventory.append({"relative_path": p.relative_to(snapshot).as_posix(),
                          "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    return inventory, mapping


def _read_tensor(key: str, mapping: dict[str, tuple[Path, str]]):
    import torch
    p, fmt = mapping[key]
    if fmt == "safetensors":
        from safetensors import safe_open
        with safe_open(str(p), framework="pt", device="cpu") as f:
            t = f.get_tensor(key)
    else:
        obj = torch.load(p, map_location="cpu", weights_only=True)
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        t = state[key]
    return t.detach().cpu()


def _find_suffix(keys, suffix: str) -> str | None:
    hits = [k for k in keys if k == suffix or k.endswith("." + suffix)]
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous checkpoint suffix {suffix}: {hits}")
    return hits[0] if hits else None


def tensor_diagnostics(arr: np.ndarray, sampled_rows: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    finite = bool(np.isfinite(a).all())
    norms = np.linalg.norm(a, axis=1)
    zero = norms == 0
    sample = a[sampled_rows]
    sn = np.linalg.norm(sample, axis=1, keepdims=True)
    unit = sample / np.maximum(sn, 1e-30)
    # All unique pair cosines within the same deterministic vocabulary-row sample.
    sim = unit @ unit.T
    tri = sim[np.triu_indices(len(sample), 1)]
    return {
        "shape": list(a.shape),
        "finite": finite,
        "dtype_after_numpy": str(arr.dtype),
        "row_l2_norm": qstats(norms),
        "global_std": float(np.std(a)),
        "all_zero_row_fraction": float(np.mean(zero)),
        "sampled_row_indices_sha256": hashlib.sha256(sampled_rows.astype(np.int64).tobytes()).hexdigest(),
        "sample_pair_cosine": qstats(tri),
    }


def compare_same_shape(a: np.ndarray, b: np.ndarray) -> dict:
    aa = np.asarray(a, dtype=np.float64); bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return {"same_shape": False, "shape_a": list(aa.shape), "shape_b": list(bb.shape)}
    d = aa - bb
    na = np.linalg.norm(aa, axis=1); nb = np.linalg.norm(bb, axis=1)
    denom = np.linalg.norm(aa)
    dot = np.einsum("ij,ij->i", aa, bb)
    cos = dot / np.maximum(na * nb, 1e-30)
    ratio = nb / np.maximum(na, 1e-30)
    return {
        "same_shape": True,
        "exact_equal": bool(np.array_equal(aa, bb)),
        "max_abs_diff": float(np.max(np.abs(d))),
        "mean_abs_diff": float(np.mean(np.abs(d))),
        "relative_frobenius_diff": float(np.linalg.norm(d) / max(denom, 1e-30)),
        "per_row_cosine": qstats(np.clip(cos, -1.0, 1.0)),
        "norm_ratio_b_over_a": qstats(ratio),
    }


def phase_a1c(repo: Path, modelcfg: dict, prereg_path: Path | None) -> dict:
    import torch
    import transformers
    import tokenizers
    import huggingface_hub
    try:
        import safetensors
        safetensors_version = safetensors.__version__
    except Exception:
        safetensors_version = None

    spec = modelcfg["representations"]["videberta_base"]
    name, rev = spec["hf_name"], spec["revision"]
    prereg = None
    if prereg_path is not None and prereg_path.is_file():
        prereg = {"path": str(prereg_path), "sha256": sha256(prereg_path)}

    env = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors_version,
    }
    try:
        snapshot = _snapshot_local(name, rev)
    except FileNotFoundError as exc:
        return {
            "phase": "A1-C_ViDeBERTa_checkpoint_mapping",
            "status": "A1-C NOT_VERIFIED",
            "pre_registered_disposition": "Required checkpoint unavailable locally -> insufficient evidence; retain limitation.",
            "checkpoint": {"hf_name": name, "revision": rev, "error": str(exc)},
            "environment": env, "preregistration": prereg,
            "official_results_action": "RETAIN_FROZEN_RESULTS_WITH_LIMITATION",
        }

    inventory, mapping = _state_file_inventory(snapshot)
    keys = sorted(mapping)
    weight_key = _find_suffix(keys, "deberta.embeddings.word_embeddings.weight")
    private_key = _find_suffix(keys, "deberta.embeddings.word_embeddings._weight")
    if weight_key is None:
        # Some checkpoints may omit the leading `deberta.` in stored key names.
        weight_key = _find_suffix(keys, "embeddings.word_embeddings.weight")
    if private_key is None:
        private_key = _find_suffix(keys, "embeddings.word_embeddings._weight")

    target = {}
    tensors = {}
    for label, key in (("weight", weight_key), ("_weight", private_key)):
        if key is not None:
            t = _read_tensor(key, mapping)
            tensors[label] = t
            target[label] = {"checkpoint_key": key, "stored_dtype": str(t.dtype),
                             "shape": list(t.shape),
                             "source_file": mapping[key][0].relative_to(snapshot).as_posix(),
                             "source_file_sha256": sha256(mapping[key][0])}
        else:
            target[label] = None

    vocab_rows = min([int(t.shape[0]) for t in tensors.values()]) if tensors else 0
    rng = np.random.default_rng(42)
    sample_n = min(512, vocab_rows)
    sampled_rows = np.sort(rng.choice(vocab_rows, size=sample_n, replace=False)) if sample_n else np.array([], dtype=np.int64)
    for label, t in tensors.items():
        target[label]["diagnostics"] = tensor_diagnostics(t.float().numpy(), sampled_rows)
    weight_private = None
    if "weight" in tensors and "_weight" in tensors:
        weight_private = compare_same_shape(tensors["weight"].numpy(), tensors["_weight"].numpy())

    # Same-class/same-settings runtime loader. `_weight` being unexpected/not a runtime
    # parameter is expected evidence to document, NOT a STOP condition.
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(
        str(snapshot), trust_remote_code=False, local_files_only=True,
        output_loading_info=True, torch_dtype=torch.float32,
    )
    runtime_model_class = type(model).__name__
    runtime = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    runtime_vs = {}
    for label, t in tensors.items():
        runtime_vs[label] = compare_same_shape(runtime, t.float().numpy())

    # Inspect ACTUALLY IMPORTED embedding forward implementation.
    emb_module = model.get_input_embeddings()
    parent = getattr(getattr(model, "deberta", None), "embeddings", None)
    forward_obj = parent.forward if parent is not None else emb_module.forward
    try:
        forward_src = inspect.getsource(forward_obj)
        forward_info = {
            "qualified_owner": f"{forward_obj.__module__}.{getattr(forward_obj, '__qualname__', type(parent).__name__)}",
            "source_sha256": sha256_text(forward_src),
            "references_literal__weight": "_weight" in forward_src,
            "references_word_embeddings": "word_embeddings" in forward_src,
            "source_excerpt": "\n".join([ln for ln in forward_src.splitlines()
                                           if "word_embeddings" in ln or "_weight" in ln][:40]),
        }
    except Exception as exc:
        forward_info = {"source_unavailable": f"{type(exc).__name__}: {exc}"}

    missing = [str(x) for x in info.get("missing_keys", [])]
    unexpected = [str(x) for x in info.get("unexpected_keys", [])]
    mismatched = [str(x) for x in info.get("mismatched_keys", [])]
    runtime_weight_exact = bool(runtime_vs.get("weight", {}).get("exact_equal", False))

    # Pre-registered disposition: no arbitrary threshold on weight/_weight relation.
    # `_weight` not entering the standard forward path is known and is NOT a STOP.
    if weight_key is None:
        disposition = "A1-C NOT_VERIFIED"
        action = "RETAIN_FROZEN_RESULTS_WITH_LIMITATION"
        rationale = "Checkpoint effective `weight` tensor could not be identified unambiguously."
    elif runtime_weight_exact and not missing and not mismatched:
        # Export/effective-GDES semantics are not automatically inferred from tensor names.
        disposition = "A1-C CLOSED WITH SEMANTIC LIMITATION"
        action = "RETAIN_FROZEN_RESULTS"
        rationale = ("Runtime input embedding matches checkpoint `weight`; loader mapping is verified. "
                     "Effective pretrained discriminator/GDES export semantics are NOT_VERIFIED by this numerical audit alone.")
    else:
        disposition = "A1-C INVESTIGATION REQUIRED"
        action = "DO_NOT_AUTONOMOUSLY_RERUN_OR_MODIFY_BENCHMARK"
        rationale = ("Runtime input embedding does not exactly reproduce checkpoint `weight`, or loader reports "
                     "missing/mismatched parameters. Independent implementation review required.")

    del model
    return {
        "phase": "A1-C_ViDeBERTa_checkpoint_mapping",
        "status": disposition,
        "pre_registered_disposition": disposition,
        "official_results_action": action,
        "rationale": rationale,
        "preregistration": prereg,
        "environment": env,
        "checkpoint": {
            "hf_name": name, "revision": rev, "local_snapshot": str(snapshot),
            "weight_files": inventory, "tensor_key_count": len(keys),
            "weight_key": weight_key, "_weight_key": private_key,
        },
        "target_tensors": target,
        "weight_minus__weight": weight_private,
        "runtime_loader": {
            "model_class": runtime_model_class,
            "runtime_input_embedding_shape": list(runtime.shape),
            "runtime_vs_checkpoint": runtime_vs,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "mismatched_keys": mismatched,
            "note": "Unexpected `_weight` / no forward use is documented evidence, not an automatic STOP condition.",
        },
        "imported_forward_implementation": forward_info,
        "export_semantics": "NOT_VERIFIED",
        "scientific_constraint": ("This audit cannot attribute ViDeBERTa AMI to mean pooling alone and does not establish "
                                  "that the released architecture universally underperforms."),
    }

# ---------------- A2 ----------------

def phase_a2(repo: Path, eval_meta: dict, modelcfg: dict) -> dict:
    from sklearn.metrics import adjusted_rand_score
    spec = modelcfg["representations"]["multilingual_e5_large"]
    if (int(spec["max_len"]), spec["clustering_prefix"], bool(spec["needs_word_segmentation"])) != (512, "query: ", False):
        raise RuntimeError("Frozen E5 recipe mismatch")
    ds = pd.read_parquet(repo / "data_v1/paired_en.parquet", columns=["appId", "genreId"])
    results = {}
    for lang in ("en", "vi"):
        a, ma, ra = load_embedding(repo, eval_meta, "tn2", "multilingual_e5_large", lang, "token_budget")
        b, mb, rb = load_embedding(repo, eval_meta, "tn2", "multilingual_e5_large", lang, "native")
        if not ra.equals(rb) or not ra.appId.equals(ds.appId) or not ra.genreId.equals(ds.genreId):
            raise RuntimeError(f"E5 {lang}: row/app/genre alignment mismatch")
        if int(ma.get("effective_max_tokens", -1)) != 512 or int(mb.get("effective_max_tokens", -1)) != 512:
            raise RuntimeError(f"E5 {lang}: both modes must have effective max 512")

        exact_rows = 0; diff_elements = 0; total_elements = 0
        max_abs = 0.0; abs_sum = 0.0
        row_max = []; row_cos = []; norm_diff = []; norm_ratio = []
        for s in range(0, TN2_N, 1024):
            aa = np.asarray(a[s:s+1024], dtype=np.float32)
            bb = np.asarray(b[s:s+1024], dtype=np.float32)
            if not np.isfinite(aa).all() or not np.isfinite(bb).all():
                raise RuntimeError(f"E5 {lang}: nonfinite embedding")
            d = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
            exact_rows += int(np.sum(np.all(aa == bb, axis=1)))
            diff_elements += int(np.count_nonzero(aa != bb)); total_elements += aa.size
            max_abs = max(max_abs, float(np.max(d))); abs_sum += float(np.sum(d))
            row_max.append(np.max(d, axis=1))
            na = np.linalg.norm(aa.astype(np.float64), axis=1)
            nb = np.linalg.norm(bb.astype(np.float64), axis=1)
            cos = np.einsum("ij,ij->i", aa.astype(np.float64), bb.astype(np.float64)) / np.maximum(na*nb, 1e-30)
            row_cos.append(np.clip(cos, -1.0, 1.0))
            norm_diff.append(nb-na); norm_ratio.append(nb/np.maximum(na, 1e-30))
        row_max = np.concatenate(row_max); row_cos = np.concatenate(row_cos)
        norm_diff = np.concatenate(norm_diff); norm_ratio = np.concatenate(norm_ratio)

        # Auxiliary label sensitivity only; not a new inferential endpoint.
        pa = repo / "results/labels" / f"tn2__multilingual_e5_large__{lang}__token_budget__kmeans_main.npz"
        pb = repo / "results/labels" / f"tn2__multilingual_e5_large__{lang}__native__kmeans_main.npz"
        with np.load(pa, allow_pickle=False) as za, np.load(pb, allow_pickle=False) as zb:
            if not np.array_equal(za["seeds"], zb["seeds"]):
                raise RuntimeError("E5 label seed mismatch")
            aris = [float(adjusted_rand_score(za["labels"][i], zb["labels"][i])) for i in range(10)]

        results[lang] = {
            "row_order_equal": True,
            "metadata": {
                "token_budget_effective_max_tokens": int(ma["effective_max_tokens"]),
                "native_effective_max_tokens": int(mb["effective_max_tokens"]),
                "token_budget_prefix": ma.get("prefix_applied", ma.get("clustering_prefix")),
                "native_prefix": mb.get("prefix_applied", mb.get("clustering_prefix")),
                "token_budget_npy_sha256": eval_meta["sources"][embedding_base(repo,"tn2","multilingual_e5_large",lang,"token_budget").name]["sha256_npy_current"],
                "native_npy_sha256": eval_meta["sources"][embedding_base(repo,"tn2","multilingual_e5_large",lang,"native").name]["sha256_npy_current"],
            },
            "exact_array_equal": bool(exact_rows == TN2_N and diff_elements == 0),
            "exact_equal_rows": int(exact_rows),
            "different_rows": int(TN2_N - exact_rows),
            "different_elements": int(diff_elements),
            "total_elements": int(total_elements),
            "max_abs_diff": float(max_abs),
            "mean_abs_diff": float(abs_sum / total_elements),
            "row_max_abs_diff": qstats(row_max),
            "rowwise_cosine_between_modes": qstats(row_cos),
            "row_norm_difference_native_minus_budget": qstats(norm_diff),
            "row_norm_ratio_native_over_budget": qstats(norm_ratio),
            "row_counts_over_abs_diff_thresholds_descriptive_only": {
                str(th): int(np.sum(row_max > th)) for th in (1e-7,1e-6,1e-5,1e-4,1e-3)
            },
            "kmeans_matched_seed_ari_between_modes_auxiliary": aris,
            "materiality_rule": "No post-hoc numeric PASS cutoff is applied; discrepancy is documented for adjudication.",
        }
    return {
        "phase": "A2_E5_numerical_parity",
        "status": "DIAGNOSTICS_RECORDED_NO_POSTHOC_THRESHOLD",
        "results": results,
        "scientific_constraint": "Do not attribute any E5 native/token difference to context capacity: both frozen modes use max_len=512.",
    }

# ---------------- A3 ----------------

def independent_negative_median(en_unit: np.ndarray, vi_unit: np.ndarray, chunk_rows: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Independent all-row implementation of TN2-A using exact all j!=i medians."""
    if en_unit.shape != vi_unit.shape or en_unit.ndim != 2:
        raise RuntimeError("TN2-A EN/VI shape mismatch")
    n = len(en_unit)
    matched = np.einsum("ij,ij->i", en_unit.astype(np.float64), vi_unit.astype(np.float64))
    matched = np.clip(matched, -1.0, 1.0)
    med = np.empty(n, dtype=np.float64)
    n_neg = n - 1
    lo = (n_neg - 1)//2; hi = n_neg//2
    vi_t = np.asarray(vi_unit.T, dtype=np.float32, order="C")
    for s in range(0, n, chunk_rows):
        e = min(n, s + chunk_rows)
        sim = np.asarray(en_unit[s:e] @ vi_t, dtype=np.float32)
        np.clip(sim, -1.0, 1.0, out=sim)
        sim[np.arange(e-s), np.arange(s,e)] = np.inf
        part = np.partition(sim, (lo,hi), axis=1)
        med[s:e] = (part[:,lo].astype(np.float64)+part[:,hi].astype(np.float64))*0.5
        print(f"[A3] rows {s}:{e}/{n}", flush=True)
    return matched, med, matched-med


def phase_a3(repo: Path, eval_meta: dict, chunk_rows: int) -> dict:
    pairs_p = repo / "results/evaluation/tn2_a_app_pairs.parquet"
    metrics_p = repo / "results/evaluation/metrics_long.csv"
    if sha256(pairs_p) != eval_meta["tn2_a_app_pairs_sha256"]:
        raise RuntimeError("Official TN2-A parquet hash mismatch")
    if sha256(metrics_p) != eval_meta["metrics_long_sha256"]:
        raise RuntimeError("Official metrics_long hash mismatch")
    ds = pd.read_parquet(repo / "data_v1/paired_en.parquet", columns=["appId","genreId"])
    off = pd.read_parquet(pairs_p)
    metrics = pd.read_csv(metrics_p)
    if len(off) != 8*TN2_N:
        raise RuntimeError(f"Expected {8*TN2_N} official TN2-A rows; found {len(off)}")
    results = {}
    total_checked = 0
    for rep in TN2_REPS:
        for tr in TRUNCS:
            key = f"{rep}__{tr}"
            sub = off[(off.representation==rep)&(off.truncation==tr)].sort_values("row_idx")
            if len(sub)!=TN2_N or not np.array_equal(sub.row_idx.to_numpy(), np.arange(TN2_N)):
                raise RuntimeError(f"{key}: official row grid mismatch")
            if not np.array_equal(sub.appId.astype(str).to_numpy(), ds.appId.astype(str).to_numpy()) or \
               not np.array_equal(sub.genreId.astype(str).to_numpy(), ds.genreId.astype(str).to_numpy()):
                raise RuntimeError(f"{key}: official app/genre alignment mismatch")
            en, _, er = load_embedding(repo, eval_meta, "tn2", rep, "en", tr)
            vi, _, vr = load_embedding(repo, eval_meta, "tn2", rep, "vi", tr)
            if not er.equals(vr) or not er.appId.astype(str).equals(ds.appId.astype(str)):
                raise RuntimeError(f"{key}: frozen embedding rows mismatch")
            en_u = finite_unit_copy(en); vi_u = finite_unit_copy(vi)
            matched, negative, delta = independent_negative_median(en_u, vi_u, chunk_rows)
            distance = 1.0-matched
            computed = {"matched_cosine":matched, "cosine_distance":distance,
                        "negative_cosine_median":negative, "delta_cosine":delta}
            comp_report = {}
            for metric, arr in computed.items():
                ref = sub[metric].to_numpy(dtype=np.float64)
                err = np.abs(arr-ref)
                comp_report[metric] = {
                    "max_abs_error": float(np.max(err)),
                    "mean_abs_error": float(np.mean(err)),
                    "p99_abs_error": float(np.percentile(err,99)),
                    "exact_equal_count_float64": int(np.sum(arr==ref)),
                    "all_rows_finite": bool(np.isfinite(arr).all()),
                }
                # Aggregate reconciliation against metrics_long; no arbitrary scientific threshold.
                msub = metrics[(metrics.section=="TN2_A") & (metrics.representation==rep) &
                               (metrics.truncation==tr) & (metrics.metric==metric)]
                agg = {}
                for stat, val in (("mean",float(np.mean(arr))), ("median",float(np.median(arr)))):
                    rr = msub[msub.pair_role==stat]
                    if len(rr)!=1:
                        raise RuntimeError(f"{key}/{metric}: missing unique Gate4 aggregate {stat}")
                    agg[stat] = {"recomputed":val, "official":float(rr.value.iloc[0]),
                                 "abs_error":abs(val-float(rr.value.iloc[0]))}
                comp_report[metric]["aggregate_reconciliation"] = agg
            total_checked += 4*TN2_N
            results[key] = {
                "n_apps": TN2_N,
                "per_app_recomputed": True,
                "all_negative_median_scope": "all j != i",
                "comparison": comp_report,
            }
            del en_u, vi_u, matched, negative, delta, distance
    return {
        "phase":"A3_TN2A_all_row_independent_recomputation",
        "status":"ALL_ROWS_RECOMPUTED_AND_COMPARED",
        "official_rows":int(len(off)),
        "scalar_values_recomputed":int(total_checked),
        "negative_chunk_rows":int(chunk_rows),
        "results":results,
        "adjudication_note":"Report numerical discrepancies directly; do not redefine TN2-A or rerun official evaluation based on aesthetics.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--phase", choices=["a1c","a2","a3"], required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/audit/final_forensic_closure"))
    ap.add_argument("--negative-chunk-rows", type=int, default=128,
                    help="Operational memory/speed control for A3 only; metric is unchanged.")
    ap.add_argument("--a1c-prereg", type=Path, default=Path("PAAC_VN_A1C_Preregistered_Checkpoint_Audit.md"))
    args = ap.parse_args()
    if args.negative_chunk_rows < 1:
        raise ValueError("--negative-chunk-rows must be >=1")
    repo = args.repo.resolve()
    if sha256(repo/"pipeline/models.yaml") != MODEL_HASH:
        raise RuntimeError("Frozen models.yaml SHA256 mismatch")
    if sha256(repo/"data_v1/paired_en.parquet") != DATA_HASH["tn2"]:
        raise RuntimeError("Frozen TN2 dataset SHA256 mismatch")
    modelcfg = yaml.safe_load((repo/"pipeline/models.yaml").read_text(encoding="utf-8"))
    eval_meta = load_eval_meta(repo)
    out = args.out_dir if args.out_dir.is_absolute() else repo/args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    report = out/f"{args.phase}.json"
    if report.exists():
        raise FileExistsError(f"Refusing overwrite: {report}")

    if args.phase=="a1c":
        pp = args.a1c_prereg if args.a1c_prereg.is_absolute() else repo/args.a1c_prereg
        result = phase_a1c(repo, modelcfg, pp)
    elif args.phase=="a2":
        result = phase_a2(repo, eval_meta, modelcfg)
    else:
        result = phase_a3(repo, eval_meta, args.negative_chunk_rows)

    result.update({
        "project":"PAAC-VN", "closure_phase":"C1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256":sha256(Path(__file__).resolve()),
        "evaluation_meta_sha256":sha256(repo/"results/evaluation/evaluation_meta.json"),
        "canonical_inputs_modified":False,
    })
    tmp = report.with_suffix(".json.tmp")
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(tmp,report)
    print(f"C1 {args.phase.upper()} REPORT: {report}")
    print(f"script_sha256={sha256(Path(__file__).resolve())}")

if __name__=="__main__":
    main()
