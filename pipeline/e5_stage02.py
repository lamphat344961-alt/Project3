"""Stage 02 E5 runner. Separate cells; never writes historical embeddings.

Run from the repository root: python -m pipeline.e5_stage02 preflight RUN_ID
Smoke uses the same encoder/runtime as full, with two fresh model loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pipeline import embed

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pipeline/e5_stage02_contract.json"
SOURCES = ("pipeline/e5_stage02.py", "pipeline/e5_stage02_contract.json",
           "pipeline/embed.py", "pipeline/models.yaml")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_contract(path=CONTRACT):
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(contract["schema_version"] == 1, "CONTRACT_VERSION")
    require(contract["reuse_native_t512"] is False, "REUSE_NOT_APPROVED")
    expected = {("tn1", "vi", "char_budget"), ("tn1", "vi", "native"),
                ("tn2", "en", "native"), ("tn2", "vi", "native"),
                ("tn2", "en", "token_budget"), ("tn2", "vi", "token_budget")}
    require(len(contract["cells"]) == 6 and
            {tuple(c) for c in contract["cells"]} == expected, "SIX_SEPARATE_CELLS_REQUIRED")
    require(contract["runtime"] == {
        "device": "cuda:0", "dtype": "float32", "batch_size": 8,
        "seed": 42, "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True, "tf32": False,
        "autocast": False, "batch_order": "token_length_then_original_index",
    }, "RUNTIME_CONTRACT_CHANGED")
    require(contract["smoke"] == {"n_apps": 500, "max_abs_difference_exclusive": 1e-5,
                                  "max_l2_norm_error": 1e-5}, "SMOKE_CONTRACT_CHANGED")
    return contract


def source_binding():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "sha256": {name: embed.sha256(ROOT / name) for name in SOURCES}}


def load_inputs(contract):
    frames = {}
    for experiment, expected in contract["datasets"].items():
        path = ROOT / expected["path"]
        require(embed.sha256(path) == expected["sha256"], f"DATASET_HASH:{experiment}")
        df = pd.read_parquet(path)
        require(len(df) == expected["n_rows"], f"DATASET_ROWS:{experiment}")
        require(df.appId.is_unique, f"DUPLICATE_IDS:{experiment}")
        require(embed.ordered_ids_sha256(df.appId) == expected["ordered_ids_sha256"],
                f"DATASET_ORDER:{experiment}")
        frames[experiment] = df
    require(embed.sha256(ROOT / "pipeline/models.yaml") == contract["models_yaml_sha256"],
            "MODEL_CONFIG_HASH")
    spec = embed.load_models(ROOT / "pipeline/models.yaml")["representations"]["multilingual_e5_large"]
    require(spec["hf_name"] == embed.E5_CHECKPOINT and spec["revision"] == embed.E5_REVISION,
            "CHECKPOINT_CHANGED")
    require(spec["pooling"] == "mean_attention_mask" and spec["clustering_prefix"] == "query: "
            and spec["max_len"] == 512 and not spec["needs_word_segmentation"], "RECIPE_CHANGED")
    return frames, spec


def runtime_info():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": embed.package_versions(), "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "cuda_devices": torch.cuda.device_count()}


def configure_runtime(runtime):
    # Set before any CUDA context/model operation; do not silently replace an existing choice.
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    require(existing in (None, runtime["cublas_workspace_config"]), "CUBLAS_CONFIG_CONFLICT")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = runtime["cublas_workspace_config"]
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE_NO_CPU_FALLBACK")
    random.seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    torch.manual_seed(runtime["seed"])
    torch.cuda.manual_seed_all(runtime["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    embed.configure_cuda_fp32(device_name="cuda:0", dtype_name="float32", batch_size=8)
    return {**runtime_info(), "gpu": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32": torch.backends.cuda.matmul.allow_tf32}


def encode_prepared(prepared, tokenizer, model, device, batch_size):
    """Length-sort for batching, then restore original dataset order explicitly."""
    lengths = embed.token_lengths(tokenizer, prepared)
    order = np.lexsort((np.arange(len(prepared)), lengths))
    result = np.empty((len(prepared), 1024), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            encoded = tokenizer([prepared[i] for i in indices], padding=True,
                                truncation=True, max_length=512, add_special_tokens=True,
                                return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=False):
                outputs = model(**encoded, return_dict=True)
                pooled = embed.pool_outputs(outputs, encoded["attention_mask"], "mean_attention_mask")
                vectors = F.normalize(pooled.float(), p=2, dim=1).cpu().numpy()
            require(vectors.shape == (len(indices), 1024), "EMBEDDING_SHAPE")
            result[indices] = vectors
    validate_vectors(result, len(prepared))
    digest = hashlib.sha256(json.dumps(prepared, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    return result, {"prepared_text_sha256": digest,
                    "batch_order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
                    "truncated_count": int((lengths > 512).sum())}


def validate_vectors(vectors, n_rows):
    require(vectors.shape == (n_rows, 1024), "OUTPUT_SHAPE")
    require(vectors.dtype == np.float32 and np.isfinite(vectors).all(), "OUTPUT_NONFINITE_OR_DTYPE")
    error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)))
    require(error <= 1e-5, "OUTPUT_NORM")
    return error


def compare_repeats(first, second):
    validate_vectors(first, len(first))
    validate_vectors(second, len(first))
    maximum = float(np.max(np.abs(first - second)))
    cosine = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=1) / (
        np.linalg.norm(first.astype(np.float64), axis=1) *
        np.linalg.norm(second.astype(np.float64), axis=1))
    return {"max_abs_difference": maximum, "min_pair_cosine": float(cosine.min()),
            "passed": maximum < 1e-5}


def resolve_assets(spec, *, allow_download):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo_id=spec["hf_name"], revision=spec["revision"],
                                      local_files_only=not allow_download))
    require(snapshot.name == spec["revision"], "CHECKPOINT_SNAPSHOT_REVISION")
    files = {p.relative_to(snapshot).as_posix(): embed.sha256(p)
             for p in sorted(snapshot.rglob("*")) if p.is_file()}
    require(files, "CHECKPOINT_SNAPSHOT_EMPTY")
    return snapshot, {"checkpoint": spec["hf_name"], "revision": spec["revision"], "files_sha256": files}


def run_cell(df, spec, cell, runtime, *, snapshot):
    experiment, language, condition = cell
    field = "description_clean" if language == "vi" else "description_original_clean"
    prepared = embed.prepare_neural_texts(df[field].tolist(), spec, condition,
                                         950 if condition == "char_budget" else None)
    require(bool(embed.raw_guard_mask(prepared).all()), "INPUT_ROW_FILTER_FORBIDDEN")
    tokenizer = embed.AutoTokenizer.from_pretrained(
        snapshot, use_fast=True, trust_remote_code=False, local_files_only=True)
    model = embed.AutoModel.from_pretrained(
        snapshot, torch_dtype=torch.float32, trust_remote_code=False, local_files_only=True)
    try:
        model.to(runtime["device"]).eval()
        require(all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point()),
                "MODEL_DTYPE")
        return encode_prepared(prepared, tokenizer, model, torch.device(runtime["device"]),
                               runtime["batch_size"])
    finally:
        del model
        torch.cuda.empty_cache()


def audit_full_paired_inputs(df, spec, *, snapshot):
    tokenizer = embed.AutoTokenizer.from_pretrained(
        snapshot, use_fast=True, trust_remote_code=False, local_files_only=True)
    results = {}
    for language, field in (("en", "description_original_clean"), ("vi", "description_clean")):
        texts = df[field].tolist()
        native = embed.prepare_neural_texts(texts, spec, "native", None)
        t512 = embed.prepare_neural_texts(texts, spec, "token_budget", None)
        results[language] = embed.audit_native_t512_token_equality(
            tokenizer=tokenizer, prepared_native=native, prepared_t512=t512, batch_size=8)
    return {"ordered_ids_sha256": embed.ordered_ids_sha256(df.appId),
            "languages": results, "reuse_authorized": False}


def safe_namespace(run_id):
    require(run_id and all(c.isalnum() or c in "_-" for c in run_id), "INVALID_RUN_ID")
    return ROOT / "results/e5_stage02" / run_id


def require_committed_sources(binding):
    for name, digest in binding["sha256"].items():
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == digest, f"UNCOMMITTED_SOURCE:{name}")


def validate_saved_output(base, expected_vectors, expected_ids):
    saved = np.load(base.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
    validate_vectors(saved, len(expected_ids))
    rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
    require(rows.appId.tolist() == list(expected_ids), "SAVED_ROW_ORDER_MISMATCH")
    require(rows.row_idx.tolist() == list(range(len(rows))), "SAVED_ROW_INDEX_MISMATCH")
    require(np.array_equal(saved, expected_vectors), "SAVED_VECTOR_BYTES_MISMATCH")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "smoke", "full"])
    parser.add_argument("run_id")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    namespace = safe_namespace(args.run_id)
    require(not namespace.exists(), "OUTPUT_NAMESPACE_EXISTS")
    contract = load_contract()
    frames, spec = load_inputs(contract)
    binding = source_binding()
    namespace.mkdir(parents=True, exist_ok=False)
    receipt = {"mode": args.mode, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_binding": binding, "runtime": runtime_info(), "cells": [],
               "reuse_native_t512": False}
    try:
        if args.mode == "preflight":
            receipt.update(status="READY_FOR_SMOKE" if torch.cuda.is_available() else "BLOCKED_RUNTIME",
                           blockers=[] if torch.cuda.is_available() else ["CUDA_UNAVAILABLE_NO_CPU_FALLBACK"],
                           dataset_checks="HASH_ROWS_ORDER_MATCH", planned_cells=contract["cells"])
        else:
            if args.mode == "full":
                require(contract["full_run_authorized"] is True, "FULL_RUN_PENDING_REVIEW")
                require_committed_sources(binding)
                require(args.smoke_receipt is not None, "SMOKE_RECEIPT_REQUIRED")
                smoke = json.loads(args.smoke_receipt.read_text(encoding="utf-8"))
                require(smoke["status"] == "SMOKE_PASS" and smoke["source_binding"] == binding,
                        "SMOKE_SOURCE_OR_STATUS_MISMATCH")
                require(len(smoke["cells"]) == 6 and all(c["comparison"]["passed"] for c in smoke["cells"]),
                        "SMOKE_CELLS_INCOMPLETE")
            receipt["runtime"] = configure_runtime(contract["runtime"])
            snapshot, receipt["model_assets"] = resolve_assets(spec, allow_download=args.allow_download)
            if args.mode == "full":
                require(smoke["runtime"] == receipt["runtime"], "SMOKE_RUNTIME_MISMATCH")
                require(smoke["model_assets"] == receipt["model_assets"], "SMOKE_ASSETS_MISMATCH")
            receipt["full_cohort_token_audit"] = audit_full_paired_inputs(
                frames["tn2"], spec, snapshot=snapshot)
            for cell in contract["cells"]:
                experiment, language, condition = cell
                df = frames[experiment]
                if args.mode == "smoke":
                    df = df.iloc[:contract["smoke"]["n_apps"]]
                first, details = run_cell(df, spec, cell, contract["runtime"], snapshot=snapshot)
                summary = {"cell": cell, "n_rows": len(df), **details}
                if args.mode == "smoke":
                    second, _ = run_cell(df, spec, cell, contract["runtime"], snapshot=snapshot)
                    summary["comparison"] = compare_repeats(first, second)
                    require(summary["comparison"]["passed"], f"SMOKE_REPEAT_FAILED:{cell}")
                base = namespace / ("__".join([experiment, "multilingual_e5_large", language, condition]))
                if args.mode == "smoke":
                    repeat_path = base.with_suffix(".repeat.npy")
                    with repeat_path.open("xb") as handle:
                        np.save(handle, second)
                    summary["repeat_npy_sha256"] = embed.sha256(repeat_path)
                else:
                    historical = ROOT / "embeddings" / experiment / (base.name + ".npy")
                    old = np.load(historical, mmap_mode="r", allow_pickle=False)
                    validate_vectors(old, len(df))
                    summary["historical_comparison"] = {
                        "historical_npy_sha256": embed.sha256(historical),
                        "max_abs_difference": float(np.max(np.abs(first - old))),
                        "mean_pair_cosine": float(np.mean(np.sum(
                            first.astype(np.float64) * old.astype(np.float64), axis=1))),
                        "interpretation": "observed_artifact_difference_not_historical_cause"}
                rows = df[["appId", "genreId"]].reset_index(drop=True)
                rows.insert(0, "row_idx", np.arange(len(rows)))
                summary["ordered_ids_sha256"] = embed.ordered_ids_sha256(rows.appId)
                embed.save_embedding(first, rows, base, create_only=True)
                embed.save_metadata(base, {**summary, "recipe": spec, "runtime": receipt["runtime"],
                                          "source_binding": binding,
                                          "dataset_sha256": contract["datasets"][experiment]["sha256"],
                                          "mode": args.mode, "reused": False}, create_only=True)
                validate_saved_output(base, first, rows.appId.tolist())
                summary["artifacts"] = {p.name: embed.sha256(p) for p in embed.output_artifact_paths(base)}
                receipt["cells"].append(summary)
                print(json.dumps({"completed_cell": cell, "mode": args.mode}), flush=True)
            receipt["status"] = "SMOKE_PASS" if args.mode == "smoke" else "FULL_COMPLETE"
    except Exception as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        embed.write_json_create_only(namespace / "execution_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(namespace / "execution_receipt.json")}))
    return 0 if receipt["status"] != "BLOCKED_RUNTIME" else 2


if __name__ == "__main__":
    raise SystemExit(main())
