#!/usr/bin/env python
"""
PAAC-VN embedding pipeline.

Implements the frozen three truncation modes:
  TN1 char_budget: clean text -> same N raw chars -> PyVi/prefix -> tokenizer
  TN2 token_budget: clean text -> PyVi/prefix -> tokenizer(max_length=512)
  native:           clean text -> PyVi/prefix -> tokenizer(max_length=model max_len)

Outputs:
  embeddings/{exp}__{repr}__{lang}__{trunc}.npy      float32, L2-normalized
  embeddings/{exp}__{repr}__{lang}__{trunc}.json     reproducibility metadata
  embeddings/{exp}__{repr}__{lang}__{trunc}.rows.parquet
  data_v1/report/truncation_rates.csv
  data_v1/report/empty_after_trunc.csv
"""
from __future__ import annotations
import time
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pyvi import ViTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

SEED = 42
MIN_LEN_AFTER_TRUNC = 20
TN1_FILE = "benchmark_v1.parquet"
TN2_FILE = "paired_en.parquet"
TN1_FIELD = "description_clean"
TN2_FIELDS = {"en": "description_original_clean", "vi": "description_clean"}
TN2_REPS = {"tfidf_lsa", "multilingual_e5_large", "bge_m3", "qwen3_embedding_0_6b"}
E5_PRODUCTION_TASK_ID = "V4-G1A-E5-PROD-001"
E5_PRODUCTION_RUN_ID = "V4-G1A-E5-PROD-001-KAGGLE-RUN-001"
E5_PRODUCTION_PLAN_SHA256 = "2b1b2b82037dfb0a5721a255bffbe008cd21fd238667b24a0b92449da0eff255"
E5_REPRESENTATION_ID = "multilingual_e5_large"
E5_CHECKPOINT = "intfloat/multilingual-e5-large"
E5_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
E5_CANONICAL_ORDERED_ID_SHA256 = "69a7363812332f2e99b424858a605c143705924670e450feb328082b3576bc64"
E5_PHASE_B_AUTHORIZATION_STATUS = "AUTHORIZED_FOR_KAGGLE_TN2_RECOVERY"

hf_logging.set_verbosity_error()


class ProductionGuardError(RuntimeError):
    """Fail-closed production guard with a stable machine-readable code."""

    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}:{detail}")


def require_production(condition: bool, code: str, detail: str | None = None) -> None:
    if not condition:
        raise ProductionGuardError(code, detail)


def write_json_create_only(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ProductionGuardError("OUTPUT_COLLISION", str(path)) from exc


def load_production_contract(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    require_production(contract.get("task_id") == E5_PRODUCTION_TASK_ID, "TASK_ID_MISMATCH")
    require_production(contract.get("immutable_plan_sha256") == E5_PRODUCTION_PLAN_SHA256, "PLAN_HASH_MISMATCH")
    require_production(contract.get("run_id") == E5_PRODUCTION_RUN_ID, "RUN_ID_MISMATCH")
    model = contract.get("model", {})
    require_production(model.get("representation_id") == E5_REPRESENTATION_ID, "REPRESENTATION_MISMATCH")
    require_production(model.get("checkpoint") == E5_CHECKPOINT, "CHECKPOINT_MISMATCH")
    require_production(model.get("revision") == E5_REVISION, "REVISION_MISMATCH")
    require_production(model.get("pooling") == "mean_attention_mask", "POOLING_MISMATCH")
    require_production(model.get("normalization") == "float32_l2", "NORMALIZATION_MISMATCH")
    runtime = contract.get("runtime", {})
    require_production(runtime.get("device") == "cuda:0", "DEVICE_CONTRACT_MISMATCH")
    require_production(runtime.get("dtype") == "float32", "DTYPE_CONTRACT_MISMATCH")
    require_production(runtime.get("batch_size") == 8, "BATCH_CONTRACT_MISMATCH")
    require_production(runtime.get("autocast") is False, "AUTOCAST_MUST_BE_DISABLED")
    require_production(runtime.get("cuda_matmul_tf32") is False, "TF32_CONTRACT_MISMATCH")
    require_production(runtime.get("cudnn_tf32") is False, "TF32_CONTRACT_MISMATCH")
    require_production(runtime.get("cpu_fallback") is False, "CPU_FALLBACK_FORBIDDEN")
    require_production(runtime.get("automatic_batch_reduction") is False, "BATCH_REDUCTION_FORBIDDEN")
    require_production(runtime.get("silent_oom_retry") is False, "OOM_RETRY_FORBIDDEN")
    units = contract.get("outputs", {}).get("units", [])
    require_production(len(units) == 4, "PRODUCTION_UNIT_COUNT_MISMATCH")
    require_production(len({unit.get("stem") for unit in units}) == 4, "PRODUCTION_UNIT_STEM_DUPLICATE")
    required_unit_fields = {
        "name", "stem", "experiment", "language", "truncation_mode",
        "n_chars", "max_tokens_requested", "shape", "ordered_id_sha256",
    }
    require_production(
        all(required_unit_fields.issubset(unit) for unit in units),
        "PRODUCTION_UNIT_CONTRACT_INCOMPLETE",
    )
    require_production(
        all(unit.get("experiment") == "tn2" for unit in units),
        "TN2_ONLY_SCOPE_REQUIRED",
    )
    expected_conditions = {
        ("en", "native", None),
        ("vi", "native", None),
        ("en", "token_budget", 512),
        ("vi", "token_budget", 512),
    }
    actual_conditions = {
        (unit.get("language"), unit.get("truncation_mode"), unit.get("max_tokens_requested"))
        for unit in units
    }
    require_production(actual_conditions == expected_conditions, "TN2_CONDITION_SET_MISMATCH")
    require_production(
        contract.get("outputs", {}).get("native_t512_reuse_authorized") is True,
        "T512_REUSE_NOT_AUTHORIZED",
    )
    require_production(
        contract.get("completion_status") == "TARGETED_TN2_RECOVERY_COMPLETE",
        "TARGETED_COMPLETION_STATUS_MISMATCH",
    )
    require_production(
        contract.get("historical_tn1", {}).get("status") == "PROVENANCE_INCOMPLETE",
        "TN1_PROVENANCE_STATUS_MISMATCH",
    )
    require_production(
        contract.get("historical_tn1", {}).get("new_inference_authorized") is False,
        "TN1_EXECUTION_FORBIDDEN",
    )
    dataset = contract.get("datasets", {}).get("tn2", {})
    for key in ("ordered_id_sha256", "runtime_ordered_id_sha256", "frozen_ordered_id_sha256", "audit_ordered_id_sha256"):
        require_production(dataset.get(key) == E5_CANONICAL_ORDERED_ID_SHA256, "CANONICAL_ORDERED_ID_BINDING_MISMATCH", key)
    require_production(dataset.get("ordered_id_binding_status") == "CANONICAL_MATCH", "ORDERED_ID_BINDING_STATUS_MISMATCH")
    require_production(
        all(unit.get("ordered_id_sha256") == E5_CANONICAL_ORDERED_ID_SHA256 for unit in units),
        "UNIT_ORDERED_ID_BINDING_MISMATCH",
    )
    paths = contract.get("paths", {})
    require_production(paths.get("project_root") == "/kaggle/working/PAAC-VN", "PROJECT_ROOT_MISMATCH")
    require_production(paths.get("python") == "/kaggle/working/paac-e5-venv/bin/python", "PYTHON_PATH_MISMATCH")
    require_production(paths.get("dataset_root") == "/kaggle/working/PAAC-VN/data_v1", "DATA_ROOT_MISMATCH")
    require_production(paths.get("model_cache") == "/kaggle/working/hf-cache", "MODEL_CACHE_MISMATCH")
    require_production(paths.get("output_root") == "/kaggle/working/PAAC_V4_E5_PRODUCTION", "OUTPUT_ROOT_MISMATCH")
    require_production(paths.get("namespace") == f"/kaggle/working/PAAC_V4_E5_PRODUCTION/{E5_PRODUCTION_RUN_ID}", "NAMESPACE_MISMATCH")
    require_production(contract.get("fresh_token_audit_required") is True, "FRESH_TOKEN_AUDIT_REQUIRED")
    authorization = contract.get("authorization", {})
    require_production(authorization.get("t512_reuse_authorized") is True, "T512_REUSE_NOT_AUTHORIZED")
    require_production(authorization.get("tn1_execution_authorized") is False, "TN1_EXECUTION_FORBIDDEN")
    require_production(authorization.get("full_e5_rerun_authorized") is False, "FULL_E5_RERUN_FORBIDDEN")
    return contract


def configure_cuda_fp32(
    *, device_name: str, dtype_name: str, batch_size: int, torch_module=torch
) -> dict:
    require_production(device_name == "cuda:0", "CUDA_DEVICE_REQUIRED", device_name)
    require_production(dtype_name == "float32", "FP32_REQUIRED", dtype_name)
    require_production(batch_size == 8, "BATCH_SIZE_MUST_BE_8", str(batch_size))
    require_production(bool(torch_module.cuda.is_available()), "CUDA_UNAVAILABLE")
    require_production(int(torch_module.cuda.device_count()) > 0, "CUDA_DEVICE_MISSING")
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("highest")
    require_production(torch_module.backends.cuda.matmul.allow_tf32 is False, "CUDA_MATMUL_TF32_NOT_DISABLED")
    require_production(torch_module.backends.cudnn.allow_tf32 is False, "CUDNN_TF32_NOT_DISABLED")
    return {
        "device": device_name,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "autocast": False,
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "float32_matmul_precision": (
            torch_module.get_float32_matmul_precision()
            if hasattr(torch_module, "get_float32_matmul_precision")
            else "UNAVAILABLE"
        ),
        "torch_version": str(getattr(torch_module, "__version__", "UNKNOWN")),
    }


def ordered_ids_sha256(app_ids: Iterable[object]) -> str:
    return hashlib.sha256("\n".join(map(str, app_ids)).encode("utf-8")).hexdigest()


def validate_production_dataset(
    df: pd.DataFrame, *, data_path: Path, expected: dict
) -> dict:
    require_production(sha256(data_path) == expected["sha256"], "DATASET_HASH_MISMATCH")
    require_production(len(df) == int(expected["row_count"]), "DATASET_ROW_COUNT_MISMATCH")
    require_production("appId" in df.columns, "DATASET_APP_ID_MISSING")
    ids = df["appId"].astype(str).tolist()
    require_production(len(ids) == len(set(ids)), "DUPLICATE_APP_ID")
    digest = ordered_ids_sha256(ids)
    runtime_digest = expected.get("runtime_ordered_id_sha256", expected.get("ordered_id_sha256"))
    require_production(digest == runtime_digest, "DATASET_ORDER_MISMATCH")
    frozen_digest = expected.get("frozen_ordered_id_sha256", expected.get("ordered_id_sha256"))
    discrepancy = expected.get("ordered_id_binding_status")
    require_production(runtime_digest == frozen_digest, "ORDERED_ID_BINDING_DISCREPANCY")
    require_production(discrepancy == "CANONICAL_MATCH", "ORDERED_ID_BINDING_STATUS_MISMATCH")
    return {
        "row_count": len(ids),
        "ordered_id_sha256": digest,
        "runtime_ordered_id_sha256": digest,
        "frozen_ordered_id_sha256": frozen_digest,
        "ordered_id_binding_status": discrepancy,
        "dataset_sha256": expected["sha256"],
    }


def validate_full_cohort_mask(keep: np.ndarray, expected_rows: int) -> None:
    require_production(len(keep) == expected_rows, "FULL_COHORT_ROW_COUNT_MISMATCH")
    removed = np.flatnonzero(~keep)
    require_production(len(removed) == 0, "FULL_COHORT_FILTERING_FORBIDDEN", str(removed[:20].tolist()))


def output_artifact_paths(base: Path) -> list[Path]:
    return [base.with_suffix(".npy"), base.with_suffix(".json"), base.with_suffix(".rows.parquet")]


def require_outputs_absent(bases: Iterable[Path]) -> None:
    collisions = [str(path) for base in bases for path in output_artifact_paths(base) if path.exists()]
    require_production(not collisions, "OUTPUT_COLLISION", ",".join(collisions[:20]))


def canonical_token_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_native_t512_token_equality(
    *, tokenizer, prepared_native: list[str], prepared_t512: list[str], batch_size: int = 8
) -> dict:
    require_production(prepared_native == prepared_t512, "NATIVE_T512_PREPARED_TEXT_MISMATCH")
    fingerprints: dict[str, object] = {}
    for label, texts in (("native", prepared_native), ("t512", prepared_t512)):
        aggregate = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        token_type_present: bool | None = None
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding=False,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            require_production("input_ids" in encoded, "TOKENIZER_INPUT_IDS_MISSING")
            require_production("attention_mask" in encoded, "TOKENIZER_ATTENTION_MASK_MISSING")
            current_present = "token_type_ids" in encoded
            token_type_present = current_present if token_type_present is None else token_type_present
            require_production(token_type_present == current_present, "TOKEN_TYPE_IDS_INCONSISTENT")
            aggregate["input_ids"].extend(encoded["input_ids"])
            aggregate["attention_mask"].extend(encoded["attention_mask"])
            if current_present:
                aggregate["token_type_ids"].extend(encoded["token_type_ids"])
        if not token_type_present:
            aggregate.pop("token_type_ids")
        fingerprints[label] = canonical_token_fingerprint(aggregate)
        fingerprints[f"{label}_payload"] = aggregate
        fingerprints[f"{label}_input_ids_sha256"] = canonical_token_fingerprint({"input_ids": aggregate["input_ids"]})
        fingerprints[f"{label}_attention_mask_sha256"] = canonical_token_fingerprint({"attention_mask": aggregate["attention_mask"]})
        fingerprints[f"{label}_token_type_ids_sha256"] = (
            canonical_token_fingerprint({"token_type_ids": aggregate["token_type_ids"]})
            if token_type_present else None
        )
        fingerprints[f"{label}_token_type_ids_present"] = bool(token_type_present)
    require_production(
        fingerprints["native_payload"] == fingerprints["t512_payload"],
        "NATIVE_T512_TOKEN_CONTRACT_MISMATCH",
    )
    return {
        "row_count": len(prepared_native),
        "max_length": 512,
        "tokenizer_batch_size": batch_size,
        "prepared_text_sha256": canonical_token_fingerprint({"texts": prepared_native}),
        "native_sha256": fingerprints["native"],
        "t512_sha256": fingerprints["t512"],
        "native_input_ids_sha256": fingerprints["native_input_ids_sha256"],
        "t512_input_ids_sha256": fingerprints["t512_input_ids_sha256"],
        "input_ids_exact_equal": fingerprints["native_input_ids_sha256"] == fingerprints["t512_input_ids_sha256"],
        "native_attention_mask_sha256": fingerprints["native_attention_mask_sha256"],
        "t512_attention_mask_sha256": fingerprints["t512_attention_mask_sha256"],
        "attention_mask_exact_equal": fingerprints["native_attention_mask_sha256"] == fingerprints["t512_attention_mask_sha256"],
        "token_type_ids_present": fingerprints["native_token_type_ids_present"],
        "native_token_type_ids_sha256": fingerprints["native_token_type_ids_sha256"],
        "t512_token_type_ids_sha256": fingerprints["t512_token_type_ids_sha256"],
        "token_type_ids_exact_equal": fingerprints["native_token_type_ids_sha256"] == fingerprints["t512_token_type_ids_sha256"],
        "exact_equal": True,
        "mismatch_count": 0,
    }


def load_and_validate_token_audit(path: Path, contract: dict) -> dict:
    require_production(path.is_file(), "TOKEN_AUDIT_RECEIPT_MISSING", str(path))
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionGuardError("TOKEN_AUDIT_RECEIPT_INVALID", str(path)) from exc
    require_production(receipt.get("task_id") == E5_PRODUCTION_TASK_ID, "TOKEN_AUDIT_TASK_MISMATCH")
    require_production(receipt.get("run_id") == E5_PRODUCTION_RUN_ID, "TOKEN_AUDIT_RUN_MISMATCH")
    require_production(receipt.get("checkpoint") == E5_CHECKPOINT, "TOKEN_AUDIT_CHECKPOINT_MISMATCH")
    require_production(receipt.get("revision") == E5_REVISION, "TOKEN_AUDIT_REVISION_MISMATCH")
    dataset = receipt.get("dataset")
    require_production(isinstance(dataset, dict), "TOKEN_AUDIT_DATASET_MISSING")
    expected_dataset = contract["datasets"]["tn2"]
    require_production(dataset.get("sha256") == expected_dataset["sha256"], "TOKEN_AUDIT_DATASET_HASH_MISMATCH")
    audit_order_digest = expected_dataset.get(
        "audit_ordered_id_sha256",
        expected_dataset.get("frozen_ordered_id_sha256", expected_dataset.get("ordered_id_sha256")),
    )
    require_production(dataset.get("ordered_id_sha256") == audit_order_digest, "TOKEN_AUDIT_ORDER_MISMATCH")
    inventory = receipt.get("asset_inventory")
    require_production(isinstance(inventory, dict), "TOKEN_AUDIT_ASSET_INVENTORY_MISSING")
    require_production(inventory.get("checkpoint") == E5_CHECKPOINT, "TOKEN_AUDIT_ASSET_CHECKPOINT_MISMATCH")
    require_production(inventory.get("revision") == E5_REVISION, "TOKEN_AUDIT_ASSET_REVISION_MISMATCH")
    require_production(isinstance(inventory.get("files"), list) and bool(inventory["files"]), "TOKEN_AUDIT_ASSET_INVENTORY_EMPTY")
    languages = receipt.get("languages")
    require_production(isinstance(languages, dict), "TOKEN_AUDIT_LANGUAGES_MISSING")
    require_production(set(languages) == {"en", "vi"}, "TOKEN_AUDIT_LANGUAGE_SET_MISMATCH")
    expected_rows = int(contract["datasets"]["tn2"]["row_count"])
    for language in ("en", "vi"):
        result = languages[language]
        require_production(isinstance(result, dict), "TOKEN_AUDIT_LANGUAGE_INVALID", language)
        require_production(result.get("row_count") == expected_rows, "TOKEN_AUDIT_ROW_COUNT_MISMATCH", language)
        require_production(result.get("max_length") == 512, "TOKEN_AUDIT_MAX_LENGTH_MISMATCH", language)
        require_production(result.get("tokenizer_batch_size") == 8, "TOKEN_AUDIT_BATCH_MISMATCH", language)
        require_production(result.get("exact_equal") is True, "TOKEN_AUDIT_NOT_EQUAL", language)
        require_production(result.get("mismatch_count") == 0, "TOKEN_AUDIT_MISMATCH_COUNT_NONZERO", language)
        native_hash = result.get("native_sha256")
        t512_hash = result.get("t512_sha256")
        valid_hash = lambda value: isinstance(value, str) and len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )
        require_production(valid_hash(native_hash), "TOKEN_AUDIT_FINGERPRINT_INVALID", language)
        require_production(valid_hash(t512_hash), "TOKEN_AUDIT_FINGERPRINT_INVALID", language)
        require_production(valid_hash(result.get("prepared_text_sha256")), "TOKEN_AUDIT_PREPARED_FINGERPRINT_INVALID", language)
        require_production(native_hash == t512_hash, "TOKEN_AUDIT_FINGERPRINT_MISMATCH", language)
        for component in ("input_ids", "attention_mask"):
            native_component = result.get(f"native_{component}_sha256")
            t512_component = result.get(f"t512_{component}_sha256")
            require_production(valid_hash(native_component), "TOKEN_AUDIT_COMPONENT_FINGERPRINT_INVALID", f"{language}:{component}")
            require_production(valid_hash(t512_component), "TOKEN_AUDIT_COMPONENT_FINGERPRINT_INVALID", f"{language}:{component}")
            require_production(native_component == t512_component, "TOKEN_AUDIT_COMPONENT_MISMATCH", f"{language}:{component}")
            require_production(result.get(f"{component}_exact_equal") is True, "TOKEN_AUDIT_COMPONENT_NOT_EQUAL", f"{language}:{component}")
        token_type_present = result.get("token_type_ids_present")
        require_production(isinstance(token_type_present, bool), "TOKEN_AUDIT_TOKEN_TYPE_PRESENCE_INVALID", language)
        for label in ("native_token_type_ids_sha256", "t512_token_type_ids_sha256"):
            value = result.get(label)
            require_production(valid_hash(value) if token_type_present else value is None, "TOKEN_AUDIT_COMPONENT_FINGERPRINT_INVALID", f"{language}:{label}")
        require_production(result.get("token_type_ids_exact_equal") is True, "TOKEN_AUDIT_COMPONENT_NOT_EQUAL", f"{language}:token_type_ids")
    return receipt


def expected_units_for_invocation(contract: dict, *, exp: str, trunc: str) -> list[dict]:
    return [
        unit for unit in contract["outputs"]["units"]
        if unit["experiment"] == exp and unit["truncation_mode"] == trunc
    ]


def validate_production_invocation(args, contract: dict, n_chars: int | None) -> list[dict]:
    units = expected_units_for_invocation(contract, exp=args.exp, trunc=args.trunc)
    require_production(bool(units), "UNAPPROVED_PRODUCTION_CONDITION")
    expected_languages = [unit["language"] for unit in units]
    require_production(args.lang == expected_languages, "PRODUCTION_LANGUAGE_SET_MISMATCH")
    expected_n_chars = units[0]["n_chars"]
    expected_max_tokens = units[0]["max_tokens_requested"]
    require_production(all(unit["n_chars"] == expected_n_chars for unit in units), "CONTRACT_N_CHARS_INCONSISTENT")
    require_production(all(unit["max_tokens_requested"] == expected_max_tokens for unit in units), "CONTRACT_MAX_TOKENS_INCONSISTENT")
    require_production(n_chars == expected_n_chars, "PRODUCTION_N_CHARS_MISMATCH")
    require_production(args.max_tokens == expected_max_tokens, "PRODUCTION_MAX_TOKENS_MISMATCH")
    return units


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    reps = cfg.get("representations", {})
    required = (
        "hf_name", "revision", "max_len", "pooling",
        "needs_word_segmentation", "query_prefix", "doc_prefix",
        "clustering_prefix", "dim", "license", "year", "params",
        "training_objective", "model_family",
    )
    for rid, spec in reps.items():
        for key in required:
            if key not in spec or spec[key] is None:
                raise ValueError(f"{rid}: required models.yaml field {key!r} is missing/null")
    return cfg


def cut_chars_at_word_boundary(text: str, n_chars: int) -> str:
    text = str(text)
    if len(text) <= n_chars:
        return text.strip()
    prefix = text[:n_chars]
    if text[n_chars].isspace():
        return prefix.rstrip()
    matches = list(re.finditer(r"\s+", prefix))
    if not matches:
        return prefix.rstrip()
    cut_at = matches[-1].start()
    return prefix[:cut_at].rstrip() if cut_at >= MIN_LEN_AFTER_TRUNC else prefix.rstrip()


def append_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def package_versions() -> dict:
    names = ["numpy", "pandas", "torch", "transformers", "scikit-learn", "pyvi", "pyyaml"]
    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def resolve_reps(specs: list[str], exp: str, reps: dict) -> list[str]:
    if specs == ["all"]:
        chosen = [rid for rid, s in reps.items() if bool(s.get(exp, False))]
    elif specs == ["mult"]:
        if exp != "tn2":
            raise ValueError("--repr mult is reserved for the 4 TN2 representations.")
        chosen = [rid for rid in reps if rid in TN2_REPS]
    else:
        chosen = []
        for item in specs:
            chosen.extend(x for x in item.split(",") if x)
    unknown = [x for x in chosen if x not in reps]
    if unknown:
        raise KeyError(f"Unknown representations: {unknown}")
    invalid = [x for x in chosen if not bool(reps[x].get(exp, False))]
    if invalid:
        raise ValueError(f"Representations not enabled for {exp}: {invalid}")
    return chosen


def raw_guard_mask(texts: list[str]) -> np.ndarray:
    return np.asarray([len(str(t).strip()) >= MIN_LEN_AFTER_TRUNC for t in texts], dtype=bool)


def prepare_neural_texts(texts: list[str], spec: dict, trunc: str, n_chars: int | None) -> list[str]:
    if trunc == "char_budget":
        if n_chars is None:
            raise ValueError("char_budget requires N_CHARS")
        base = [cut_chars_at_word_boundary(t, n_chars) for t in texts]
    else:
        base = [str(t).strip() for t in texts]

    if spec["needs_word_segmentation"]:
        base = [ViTokenizer.tokenize(t) for t in base]

    prefix = spec.get("clustering_prefix", "")
    return [prefix + t for t in base]


def token_lengths(tokenizer, texts: list[str], batch_size: int = 256) -> np.ndarray:
    vals: list[int] = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        vals.extend(len(x) for x in enc["input_ids"])
    return np.asarray(vals, dtype=np.int32)

def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Works for both left and right padding.
    if bool(torch.all(attention_mask[:, -1] == 1)):
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]


def pool_outputs(outputs, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean_attention_mask":
        return mean_pool(outputs.last_hidden_state, attention_mask)
    if mode == "cls":
        return outputs.last_hidden_state[:, 0]
    if mode == "last_token":
        return last_token_pool(outputs.last_hidden_state, attention_mask)
    if mode == "pooler_output":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("Model did not return pooler_output; do not silently change SimCSE pooling.")
        return pooled
    raise ValueError(f"Unsupported pooling={mode!r}")


def effective_max_len(spec: dict, trunc: str, max_tokens: int | None) -> int:
    if trunc == "token_budget":
        if max_tokens is None:
            raise ValueError("token_budget requires --max-tokens")
        return int(max_tokens)
    return int(spec["max_len"])


def save_embedding(
    X: np.ndarray,
    rows: pd.DataFrame,
    base: Path,
    create_only: bool = False,
) -> float:

    t_save_start = time.perf_counter()

    if X.dtype != np.float32:
        X = X.astype(np.float32)

    if len(X) != len(rows):
        raise ValueError(
            "Embedding row count does not match row sidecar."
        )

    npy_path = base.with_suffix(".npy")
    rows_path = base.with_suffix(".rows.parquet")
    if create_only:
        base.parent.mkdir(parents=True, exist_ok=True)
        try:
            with npy_path.open("xb") as handle:
                np.save(handle, X)
            with rows_path.open("xb") as handle:
                rows.to_parquet(handle, index=False)
        except FileExistsError as exc:
            raise ProductionGuardError("OUTPUT_COLLISION", str(exc.filename)) from exc
    else:
        np.save(npy_path, X)
        rows.to_parquet(rows_path, index=False)

    return time.perf_counter() - t_save_start


def save_metadata(
    base: Path,
    meta: dict,
    create_only: bool = False,
) -> None:
    path = base.with_suffix(".json")
    text = json.dumps(meta, ensure_ascii=False, indent=2)
    if create_only:
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except FileExistsError as exc:
            raise ProductionGuardError("OUTPUT_COLLISION", str(path)) from exc
    else:
        path.write_text(text, encoding="utf-8")


def embed_neural(
    rid: str,
    spec: dict,
    texts: list[str],
    trunc: str,
    n_chars: int | None,
    max_tokens: int | None,
    batch_size_override: int | None,
    production_contract: dict | None = None,
) -> tuple[np.ndarray, dict]:

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------
    t_preprocess_start = time.perf_counter()

    prepared = prepare_neural_texts(
        texts,
        spec,
        trunc,
        n_chars,
    )

    preprocess_seconds = time.perf_counter() - t_preprocess_start

    # --------------------------------------------------
    # LOAD TOKENIZER + MODEL
    # --------------------------------------------------
    t_load_start = time.perf_counter()

    cache_dir = None
    local_files_only = False
    if production_contract is not None:
        cache_dir = production_contract["paths"]["model_cache"]
        local_files_only = production_contract["_execution_authorization"]["receipt"]["asset_mode"] == "PREPOPULATED_OFFLINE_CACHE"
    tokenizer = AutoTokenizer.from_pretrained(
        spec["hf_name"],
        revision=spec["revision"],
        use_fast=bool(spec.get("tokenizer_use_fast", True)),
        trust_remote_code=False,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    if spec["pooling"] == "last_token":
        tokenizer.padding_side = "left"

    load_dtype = torch.float32 if (production_contract is not None or rid == "videberta_base") else "auto"

    model = AutoModel.from_pretrained(
        spec["hf_name"],
        revision=spec["revision"],
        trust_remote_code=False,
        torch_dtype=load_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    if production_contract is not None:
        runtime = production_contract["runtime"]
        configure_cuda_fp32(
            device_name=runtime["device"],
            dtype_name=runtime["dtype"],
            batch_size=int(runtime["batch_size"]),
        )
        device = torch.device(runtime["device"])
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model.to(device)
    model.eval()
    model_compute_dtype = str(next(model.parameters()).dtype)
    if production_contract is not None:
        require_production(model_compute_dtype == "torch.float32", "MODEL_DTYPE_MISMATCH", model_compute_dtype)

    sync_device(device)

    model_load_seconds = (
        time.perf_counter() - t_load_start
    )

    # --------------------------------------------------
    # TOKEN AUDIT
    # --------------------------------------------------
    max_len = effective_max_len(
        spec,
        trunc,
        max_tokens,
    )

    t_audit_start = time.perf_counter()

    lengths = token_lengths(
        tokenizer,
        prepared,
    )

    token_audit_seconds = (
        time.perf_counter() - t_audit_start
    )

    hit_count = int(np.sum(lengths > max_len))
    hit_rate = float(hit_count / len(lengths))

    # --------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------
    batch_size = (
        batch_size_override
        or int(spec.get("batch_size_hint", 16))
    )
    if production_contract is not None:
        require_production(batch_size == 8, "BATCH_SIZE_MUST_BE_8", str(batch_size))

    chunks: list[np.ndarray] = []

    sync_device(device)
    t_embed_start = time.perf_counter()

    with torch.inference_mode():
        for start in range(
            0,
            len(prepared),
            batch_size,
        ):
            batch = prepared[
                start:start + batch_size
            ]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            outputs = model(
                **enc,
                return_dict=True,
            )

            pooled = pool_outputs(
                outputs,
                enc["attention_mask"],
                spec["pooling"],
            )

            pooled = F.normalize(
                pooled.float(),
                p=2,
                dim=1,
            )

            chunks.append(
                pooled.cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

    sync_device(device)

    embedding_seconds = (
        time.perf_counter() - t_embed_start
    )

    samples_per_second = (
        len(prepared) / embedding_seconds
    )

    X = np.concatenate(
        chunks,
        axis=0,
    )

    if X.shape[1] != int(spec["dim"]):
        raise RuntimeError(
            f"{rid}: got dim={X.shape[1]}, "
            f"expected={spec['dim']}"

        )

    if not np.isfinite(X).all():
        bad_rows = np.flatnonzero(~np.isfinite(X).all(axis=1))
        raise RuntimeError(
            f"{rid}: embedding contains non-finite values; "
            f"bad_rows={bad_rows[:20].tolist()} "
            f"(total={len(bad_rows)})"
    )
    
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return X, {
        "effective_max_tokens": max_len,
        "model_compute_dtype": model_compute_dtype,
        "token_length_mean_before_trunc": float(
            lengths.mean()
        ),
        "token_length_p95_before_trunc": float(
            np.percentile(lengths, 95)
        ),
        "truncated_count": hit_count,
        "truncation_rate": hit_rate,
        "batch_size": batch_size,

        "preprocess_seconds": preprocess_seconds,
        "model_load_seconds": model_load_seconds,
        "token_audit_seconds": token_audit_seconds,
        "embedding_seconds": embedding_seconds,
        "samples_per_second": samples_per_second,
    }

def make_tfidf_vectorizer(spec: dict) -> TfidfVectorizer:
    v = spec["vectorizer"]
    return TfidfVectorizer(
        analyzer=v["analyzer"],
        ngram_range=tuple(v["ngram_range"]),
        min_df=v["min_df"],
        max_df=v["max_df"],
        sublinear_tf=v["sublinear_tf"],
        norm=v["norm"],
        lowercase=v["lowercase"],
        token_pattern=v["token_pattern"],
    )


def truncate_tfidf_tokens(texts: list[str], spec: dict, max_tokens: int) -> tuple[list[str], np.ndarray]:
    vec = make_tfidf_vectorizer(spec)
    pre = vec.build_preprocessor()
    tok = vec.build_tokenizer()
    out, lengths = [], []
    for text in texts:
        tokens = tok(pre(str(text)))
        lengths.append(len(tokens))
        out.append(" ".join(tokens[:max_tokens]))
    return out, np.asarray(lengths, dtype=np.int32)


def prepare_tfidf_texts(
    texts: list[str], spec: dict, trunc: str, n_chars: int | None, max_tokens: int | None
) -> tuple[list[str], np.ndarray | None]:
    if trunc == "char_budget":
        if n_chars is None:
            raise ValueError("char_budget requires N_CHARS")
        return [cut_chars_at_word_boundary(t, n_chars) for t in texts], None
    if trunc == "token_budget":
        if max_tokens is None:
            raise ValueError("token_budget requires --max-tokens")
        return truncate_tfidf_tokens(texts, spec, max_tokens)
    if trunc == "native":
        return [str(t).strip() for t in texts], None
    raise ValueError(trunc)


def fit_tfidf_svd(corpora: list[list[str]], spec: dict) -> list[np.ndarray]:
    """Fit one shared vectorizer+SVD on concatenated corpora, then split back."""
    all_texts = [x for corpus in corpora for x in corpus]
    vec = make_tfidf_vectorizer(spec)
    sparse = vec.fit_transform(all_texts)

    scfg = spec["svd"]
    svd = TruncatedSVD(
        n_components=int(scfg["n_components"]),
        random_state=int(scfg["random_state"]),
    )
    dense = svd.fit_transform(sparse).astype(np.float32)
    dense = Normalizer(copy=False).fit_transform(dense).astype(np.float32, copy=False)

    result, pos = [], 0
    for corpus in corpora:
        n = len(corpus)
        result.append(dense[pos:pos + n])
        pos += n
    return result


def load_n_chars(summary_path: Path) -> int:
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    val = obj.get("selected_n_chars")
    if not isinstance(val, int) or val <= 0:
        raise ValueError(f"No valid selected_n_chars in {summary_path}")
    return val


def output_base(out_dir: Path, exp: str, rid: str, lang: str, trunc: str) -> Path:
    return out_dir / f"{exp}__{rid}__{lang}__{trunc}"


def common_meta(
    *,
    exp: str, rid: str, lang: str, trunc: str, spec: dict, data_path: Path,
    models_path: Path, n_chars: int | None, max_tokens: int | None,
    X: np.ndarray, app_ids: list[str], extra: dict,
) -> dict:
    app_hash = hashlib.sha256("\n".join(map(str, app_ids)).encode("utf-8")).hexdigest()
    return {
        "project": "PAAC-VN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": exp,
        "representation": rid,
        "language": lang,
        "truncation_mode": trunc,
        "n_chars": n_chars if trunc == "char_budget" else None,
        "max_tokens_requested": max_tokens if trunc == "token_budget" else None,
        "hf_name": spec["hf_name"],
        "revision": spec["revision"],
        "max_len_native": spec["max_len"],
        "pooling": spec["pooling"],
        "needs_word_segmentation": spec["needs_word_segmentation"],
        "clustering_prefix": spec.get("clustering_prefix", ""),
        "embedding_dim": int(X.shape[1]),
        "embedding_dtype": "float32",
        "l2_normalized": True,
        "n_rows": int(X.shape[0]),
        "dataset_path": str(data_path.as_posix()),
        "dataset_sha256": sha256(data_path),
        "models_yaml_sha256": sha256(models_path),
        "app_ids_sha256": app_hash,
        "package_versions": package_versions(),
        **extra,
    }


ACTIVE_PRODUCTION_NAMESPACE: Path | None = None
ACTIVE_PRODUCTION_PHASE: str | None = None
ACTIVE_EXECUTION_AUTHORIZATION_SHA256: str | None = None


def activate_new_production_namespace(namespace: Path, phase: str) -> None:
    """Atomically claim a new namespace before enabling failure-receipt writes."""
    global ACTIVE_PRODUCTION_NAMESPACE, ACTIVE_PRODUCTION_PHASE
    try:
        namespace.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ProductionGuardError("OUTPUT_COLLISION", str(namespace)) from exc
    ACTIVE_PRODUCTION_NAMESPACE = namespace
    ACTIVE_PRODUCTION_PHASE = phase


def runtime_environment_diagnostics(namespace: Path, runtime: dict, torch_module=torch) -> dict:
    cuda = torch_module.cuda
    count = int(cuda.device_count()) if cuda.is_available() else 0
    names = [str(cuda.get_device_name(i)) for i in range(count)] if hasattr(cuda, "get_device_name") else []
    try:
        available_memory = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        available_memory = "UNAVAILABLE"
    try:
        free_output_bytes = int(shutil.disk_usage(namespace.parent).free)
    except OSError:
        free_output_bytes = "UNAVAILABLE"
    return {
        "python_version": sys.version,
        "packages": package_versions(),
        "cuda_available": bool(cuda.is_available()),
        "cuda_device_count": count,
        "cuda_device_names": names,
        "selected_device": runtime["device"],
        "parameter_dtype": runtime["dtype"],
        "compute_dtype": runtime["dtype"],
        "batch_size": runtime["batch_size"],
        "autocast": runtime["autocast"],
        "cuda_matmul_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "available_host_memory_bytes": available_memory,
        "free_output_bytes": free_output_bytes,
    }


def write_text_create_only(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ProductionGuardError("OUTPUT_COLLISION", str(path)) from exc


def validate_production_paths(args, contract: dict) -> Path:
    paths = contract["paths"]
    namespace = Path(paths["namespace"]).resolve()
    project_root = Path(paths["project_root"]).resolve()
    require_production(Path(sys.executable).resolve() == Path(paths["python"]).resolve(), "PYTHON_EXECUTABLE_MISMATCH")
    require_production(args.production_contract.resolve() == (project_root / "pipeline/e5_production_contract.json").resolve(), "CONTRACT_PATH_MISMATCH")
    require_production(args.models.resolve() == (project_root / "pipeline/models.yaml").resolve(), "MODELS_PATH_MISMATCH")
    require_production(args.run_id == contract["run_id"], "RUN_ID_MISMATCH")
    require_production(args.device == contract["runtime"]["device"], "DEVICE_CONTRACT_MISMATCH")
    require_production(args.dtype == contract["runtime"]["dtype"], "DTYPE_CONTRACT_MISMATCH")
    require_production(args.batch_size == contract["runtime"]["batch_size"], "BATCH_CONTRACT_MISMATCH")
    require_production(args.data_dir.resolve() == Path(paths["dataset_root"]).resolve(), "DATA_ROOT_MISMATCH")
    require_production(args.out_dir.resolve() == (namespace / "embeddings").resolve(), "OUTPUT_ROOT_MISMATCH")
    require_production(args.report_dir.resolve() == (namespace / "reports").resolve(), "REPORT_ROOT_MISMATCH")
    require_production(args.require_full_cohort, "FULL_COHORT_GUARD_REQUIRED")
    require_production(args.fail_if_exists, "FAIL_IF_EXISTS_REQUIRED")
    return namespace


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_and_validate_execution_authorization(path: Path, contract_path: Path, contract: dict) -> dict:
    """Validate the separate Owner Phase B authorization before runtime access."""
    expected_path = (
        Path(contract["paths"]["project_root"])
        / contract["phase_b_authorization"]["receipt_relative_path"]
    ).resolve()
    require_production(path.resolve() == expected_path, "EXECUTION_AUTHORIZATION_PATH_MISMATCH")
    require_production(path.is_file(), "EXECUTION_AUTHORIZATION_MISSING", str(path))
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionGuardError("EXECUTION_AUTHORIZATION_INVALID", str(path)) from exc
    require_production(receipt.get("task_id") == E5_PRODUCTION_TASK_ID, "EXECUTION_AUTHORIZATION_TASK_MISMATCH")
    require_production(receipt.get("plan_sha256") == E5_PRODUCTION_PLAN_SHA256, "EXECUTION_AUTHORIZATION_PLAN_MISMATCH")
    require_production(receipt.get("authorization_status") == E5_PHASE_B_AUTHORIZATION_STATUS, "EXECUTION_NOT_AUTHORIZED")
    require_production(receipt.get("run_id") == E5_PRODUCTION_RUN_ID, "EXECUTION_AUTHORIZATION_RUN_MISMATCH")
    require_production(set(receipt.get("authorized_phases", [])) == {"TOKEN_AUDIT", "TN2_NATIVE_INFERENCE_AND_T512_REUSE"}, "EXECUTION_PHASE_SET_MISMATCH")
    require_production(receipt.get("automatic_retry") is False, "AUTOMATIC_RETRY_FORBIDDEN")
    require_production(receipt.get("full_e5_rerun_authorized") is False, "FULL_E5_RERUN_FORBIDDEN")
    require_production(receipt.get("tn1_execution_authorized") is False, "TN1_EXECUTION_FORBIDDEN")
    require_production(receipt.get("asset_mode") in set(contract["phase_b_authorization"]["allowed_asset_modes"]), "ASSET_MODE_NOT_AUTHORIZED")
    for key, expected in contract["phase_b_authorization"]["required_fixed_bindings"].items():
        require_production(receipt.get(key) == expected, "EXECUTION_AUTHORIZATION_BINDING_MISMATCH", key)
    root = Path(contract["paths"]["project_root"])
    for key, relative in (
        ("phase_a_approval_receipt_sha256", "PAAC_MULTI_AGENT_STARTER/01_tasks/V4-G1A-E5-PROD-001/PLAN_V3_1_APPROVAL_RECEIPT.md"),
        ("research_lead_acceptance_sha256", "PAAC_MULTI_AGENT_STARTER/01_tasks/V4-G1A-E5-PROD-001/NGHIEM_THU_V3_1.md"),
    ):
        evidence_path = root / relative
        require_production(evidence_path.is_file(), "EXECUTION_AUTHORIZATION_EVIDENCE_MISSING", relative)
        require_production(receipt.get(key) == sha256(evidence_path), "EXECUTION_AUTHORIZATION_BINDING_MISMATCH", key)
    for key, expected in contract["paths"].items():
        require_production(receipt.get("paths", {}).get(key) == expected, "EXECUTION_AUTHORIZATION_PATH_BINDING_MISMATCH", key)
    model = receipt.get("model", {})
    require_production(model.get("checkpoint") == E5_CHECKPOINT, "EXECUTION_AUTHORIZATION_CHECKPOINT_MISMATCH")
    require_production(model.get("revision") == E5_REVISION, "EXECUTION_AUTHORIZATION_REVISION_MISMATCH")
    runtime = receipt.get("runtime", {})
    for key in ("device", "dtype", "batch_size", "autocast", "cuda_matmul_tf32", "cudnn_tf32"):
        require_production(runtime.get(key) == contract["runtime"][key], "EXECUTION_AUTHORIZATION_RUNTIME_MISMATCH", key)
    sources = receipt.get("source_sha256", {})
    expected_sources = contract["token_audit_lineage"]["fixed_source_bindings"]
    require_production(set(sources) == set(expected_sources) | {"pipeline/embed.py", "pipeline/e5_production_contract.json", "tests/test_e5_production_guards.py"}, "EXECUTION_AUTHORIZATION_SOURCE_SET_MISMATCH")
    for key, expected in expected_sources.items():
        require_production(sources.get(key) == expected, "EXECUTION_AUTHORIZATION_SOURCE_HASH_MISMATCH", key)
        current_path = root / key
        require_production(current_path.is_file(), "PROTECTED_SOURCE_MISSING", key)
        require_production(sha256(current_path) == expected, "PROTECTED_SOURCE_HASH_MISMATCH", key)
    require_production(sha256(Path(__file__).resolve()) == sources.get("pipeline/embed.py"), "EXECUTION_AUTHORIZATION_SOURCE_HASH_MISMATCH", "pipeline/embed.py")
    require_production(sha256(contract_path.resolve()) == sources.get("pipeline/e5_production_contract.json"), "EXECUTION_AUTHORIZATION_SOURCE_HASH_MISMATCH", "pipeline/e5_production_contract.json")
    require_production(_valid_sha256(sources.get("tests/test_e5_production_guards.py")), "EXECUTION_AUTHORIZATION_SOURCE_HASH_INVALID", "tests/test_e5_production_guards.py")
    return {"receipt": receipt, "sha256": sha256(path), "path": path.resolve().as_posix()}


def inventory_hf_assets(contract: dict) -> dict:
    cache_root = Path(contract["paths"]["model_cache"]).resolve()
    snapshot = (
        cache_root
        / "models--intfloat--multilingual-e5-large"
        / "snapshots"
        / E5_REVISION
    )
    require_production(snapshot.is_dir(), "PINNED_SNAPSHOT_MISSING", str(snapshot))
    files = []
    for path in sorted((p for p in snapshot.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        resolved = path.resolve()
        try:
            resolved.relative_to(cache_root)
        except ValueError as exc:
            raise ProductionGuardError("ASSET_PATH_ESCAPE", str(path)) from exc
        files.append({
            "path": path.relative_to(snapshot).as_posix(),
            "resolved_path": resolved.as_posix(),
            "size": resolved.stat().st_size,
            "sha256": sha256(resolved),
            "is_symlink": path.is_symlink(),
        })
    require_production(bool(files), "PINNED_SNAPSHOT_EMPTY")
    return {
        "checkpoint": E5_CHECKPOINT,
        "revision": E5_REVISION,
        "snapshot_path": snapshot.as_posix(),
        "files": files,
    }


def production_source_manifest(contract_path: Path, contract: dict, authorization: dict) -> str:
    root = Path(contract["paths"]["project_root"]).resolve()
    paths = {
        "pipeline/embed.py": Path(__file__).resolve(),
        "pipeline/e5_production_contract.json": contract_path.resolve(),
        "pipeline/models.yaml": root / "pipeline/models.yaml",
        "revision_v4_20261020/00_governance/PROTOCOL_LOCK.json": root / "revision_v4_20261020/00_governance/PROTOCOL_LOCK.json",
        "revision_v4_20261020/00_governance/GOV02_EXPERIMENT_MATRIX_V4_DRAFT.jsonl": root / "revision_v4_20261020/00_governance/GOV02_EXPERIMENT_MATRIX_V4_DRAFT.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require_production(not missing, "PROTECTED_SOURCE_MISSING", ",".join(missing))
    expected = authorization["receipt"]["source_sha256"]
    for relative, path in paths.items():
        require_production(sha256(path) == expected[relative], "PROTECTED_SOURCE_HASH_MISMATCH", relative)
    return "".join(f"{sha256(path)}  {relative}\n" for relative, path in paths.items())


def validate_audit_only_namespace(namespace: Path, contract: dict, authorization: dict) -> dict:
    """Validate the immutable token-audit-only namespace before model loading."""
    require_production(namespace.is_dir(), "TOKEN_AUDIT_NAMESPACE_MISSING", str(namespace))
    expected_files = {
        "token_audit/native_t512_equality.json",
        "environment.json",
        "source_manifest.sha256",
    }
    actual_files = {
        path.relative_to(namespace).as_posix()
        for path in namespace.rglob("*")
        if path.is_file()
    }
    require_production(
        actual_files == expected_files,
        "AUDIT_NAMESPACE_NOT_CLEAN",
        ",".join(sorted(actual_files ^ expected_files)),
    )
    audit_path = namespace / "token_audit/native_t512_equality.json"
    audit = load_and_validate_token_audit(audit_path, contract)
    require_production(audit.get("execution_authorization_sha256") == authorization["sha256"], "TOKEN_AUDIT_AUTHORIZATION_HASH_MISMATCH")
    current_inventory = inventory_hf_assets(contract)
    inventory_view = lambda payload: [
        (item["path"], int(item["size"]), item["sha256"])
        for item in payload["files"]
    ]
    require_production(
        inventory_view(current_inventory) == inventory_view(audit["asset_inventory"]),
        "TOKEN_AUDIT_ASSET_INVENTORY_DRIFT",
    )
    environment_path = namespace / "environment.json"
    source_manifest_path = namespace / "source_manifest.sha256"
    try:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionGuardError("TOKEN_AUDIT_ENVIRONMENT_INVALID", str(environment_path)) from exc
    runtime = contract["runtime"]
    for key, expected in (
        ("device", runtime["device"]),
        ("dtype", runtime["dtype"]),
        ("batch_size", runtime["batch_size"]),
        ("autocast", runtime["autocast"]),
    ):
        require_production(environment.get(key) == expected, "TOKEN_AUDIT_ENVIRONMENT_MISMATCH", key)
    require_production(environment.get("execution_authorization_sha256") == authorization["sha256"], "TOKEN_AUDIT_AUTHORIZATION_HASH_MISMATCH")
    source_manifest = source_manifest_path.read_text(encoding="utf-8")
    lineage = contract["token_audit_lineage"]
    expected_manifest_hash = lineage.get("source_manifest_sha256")
    if expected_manifest_hash is not None:
        require_production(
            sha256(source_manifest_path) == expected_manifest_hash,
            "TOKEN_AUDIT_SOURCE_MANIFEST_HASH_MISMATCH",
        )
    parsed_bindings: dict[str, str] = {}
    for raw_line in source_manifest.splitlines():
        parts = raw_line.split("  ", 1)
        require_production(len(parts) == 2, "TOKEN_AUDIT_SOURCE_MANIFEST_INVALID", raw_line)
        digest, recorded_path = parts
        require_production(
            len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            "TOKEN_AUDIT_SOURCE_MANIFEST_INVALID",
            raw_line,
        )
        normalized_path = recorded_path.replace("\\", "/")
        require_production(normalized_path not in parsed_bindings, "TOKEN_AUDIT_SOURCE_PATH_DUPLICATE", normalized_path)
        parsed_bindings[normalized_path] = digest
    required_bindings = authorization["receipt"]["source_sha256"]
    required_bindings = {key: value for key, value in required_bindings.items() if key != "tests/test_e5_production_guards.py"}
    require_production(len(parsed_bindings) == len(required_bindings), "TOKEN_AUDIT_SOURCE_SET_MISMATCH")
    for relative_path, expected_digest in required_bindings.items():
        require_production(relative_path in parsed_bindings, "TOKEN_AUDIT_SOURCE_LINEAGE_MISSING", relative_path)
        require_production(parsed_bindings[relative_path] == expected_digest, "TOKEN_AUDIT_SOURCE_HASH_MISMATCH", relative_path)
        current_path = Path(__file__).resolve() if relative_path == "pipeline/embed.py" else (
            Path(contract["paths"]["project_root"]) / relative_path
        )
        require_production(current_path.is_file(), "PROTECTED_SOURCE_MISSING", relative_path)
        require_production(sha256(current_path) == expected_digest, "PROTECTED_SOURCE_HASH_MISMATCH", relative_path)
    return {
        "receipt": audit,
        "receipt_sha256": sha256(audit_path),
        "environment_sha256": sha256(environment_path),
        "source_manifest_sha256": sha256(source_manifest_path),
    }


def copy_file_create_only(source: Path, target: Path) -> None:
    require_production(source.is_file(), "REUSE_SOURCE_MISSING", str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as src, target.open("xb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
    except FileExistsError as exc:
        raise ProductionGuardError("OUTPUT_COLLISION", str(target)) from exc


def materialize_t512_reuse(namespace: Path, contract: dict, audit_evidence: dict, authorization: dict) -> None:
    """Create separately named T512 artifacts from Native bytes after exact audit."""
    require_production(
        contract["outputs"].get("native_t512_reuse_authorized") is True,
        "T512_REUSE_NOT_AUTHORIZED",
    )
    units = contract["outputs"]["units"]
    native_by_language = {
        unit["language"]: unit for unit in units if unit["truncation_mode"] == "native"
    }
    t512_units = [unit for unit in units if unit["truncation_mode"] == "token_budget"]
    require_production(set(native_by_language) == {"en", "vi"}, "NATIVE_REUSE_SOURCE_SET_MISMATCH")
    require_production({unit["language"] for unit in t512_units} == {"en", "vi"}, "T512_REUSE_TARGET_SET_MISMATCH")
    target_bases = [namespace / "embeddings" / unit["stem"] for unit in t512_units]
    require_outputs_absent(target_bases)
    for unit in t512_units:
        source_unit = native_by_language[unit["language"]]
        source_base = namespace / "embeddings" / source_unit["stem"]
        target_base = namespace / "embeddings" / unit["stem"]
        source_paths = output_artifact_paths(source_base)
        require_production(all(path.is_file() for path in source_paths), "REUSE_SOURCE_INCOMPLETE", source_unit["stem"])
        source_metadata = json.loads(source_paths[1].read_text(encoding="utf-8"))
        copy_file_create_only(source_paths[0], target_base.with_suffix(".npy"))
        copy_file_create_only(source_paths[2], target_base.with_suffix(".rows.parquet"))
        reuse_metadata = dict(source_metadata)
        reuse_metadata.update({
            "truncation_mode": "token_budget",
            "n_chars": None,
            "max_tokens_requested": 512,
            "effective_max_tokens": 512,
            "reuse_status": "REUSED_AFTER_EXACT_TOKEN_EQUIVALENCE",
            "reused_from_stem": source_unit["stem"],
            "reused_source_npy_sha256": sha256(source_paths[0]),
            "reused_source_rows_sha256": sha256(source_paths[2]),
            "token_audit_receipt_sha256": audit_evidence["receipt_sha256"],
            "execution_authorization_sha256": authorization["sha256"],
        })
        write_json_create_only(target_base.with_suffix(".json"), reuse_metadata)


def validate_unit_metadata(metadata: dict, unit: dict, contract: dict) -> None:
    require_production(metadata.get("revision") == E5_REVISION, "METADATA_REVISION_MISMATCH", unit["stem"])
    expected_metadata = {
        "experiment": unit["experiment"],
        "representation": E5_REPRESENTATION_ID,
        "language": unit["language"],
        "truncation_mode": unit["truncation_mode"],
        "n_chars": unit["n_chars"],
        "max_tokens_requested": unit["max_tokens_requested"],
        "hf_name": E5_CHECKPOINT,
        "pooling": contract["model"]["pooling"],
        "clustering_prefix": contract["model"]["clustering_prefix"],
        "embedding_dim": unit["shape"][1],
        "embedding_dtype": "float32",
        "l2_normalized": True,
        "n_rows": unit["shape"][0],
        "dataset_sha256": contract["datasets"][unit["experiment"]]["sha256"],
        "app_ids_sha256": unit["ordered_id_sha256"],
        "runtime_ordered_id_sha256": contract["datasets"][unit["experiment"]]["runtime_ordered_id_sha256"],
        "frozen_ordered_id_sha256": contract["datasets"][unit["experiment"]]["frozen_ordered_id_sha256"],
        "ordered_id_binding_status": contract["datasets"][unit["experiment"]]["ordered_id_binding_status"],
        "execution_authorization_sha256": contract["_execution_authorization"]["sha256"],
    }
    for key, expected_value in expected_metadata.items():
        require_production(metadata.get(key) == expected_value, "METADATA_CONDITION_MISMATCH", f"{unit['stem']}:{key}")
    if unit["truncation_mode"] == "token_budget":
        native_unit = next(
            candidate for candidate in contract["outputs"]["units"]
            if candidate["language"] == unit["language"] and candidate["truncation_mode"] == "native"
        )
        require_production(
            metadata.get("reuse_status") == "REUSED_AFTER_EXACT_TOKEN_EQUIVALENCE",
            "T512_REUSE_PROVENANCE_MISSING",
            unit["stem"],
        )
        require_production(
            metadata.get("reused_from_stem") == native_unit["stem"],
            "T512_REUSE_SOURCE_MISMATCH",
            unit["stem"],
        )
        require_production(
            metadata.get("execution_authorization_sha256") == contract["_execution_authorization"]["sha256"],
            "T512_REUSE_APPROVAL_MISMATCH",
            unit["stem"],
        )


def validate_production_outputs(namespace: Path, contract: dict) -> dict:
    units = contract["outputs"]["units"]
    summaries = []
    audit_path = namespace / contract["token_audit_lineage"]["receipt_relative_path"]
    require_production(audit_path.is_file(), "TOKEN_AUDIT_RECEIPT_MISSING", str(audit_path))
    actual_audit_hash = sha256(audit_path)
    for unit in units:
        base = namespace / "embeddings" / unit["stem"]
        paths = output_artifact_paths(base)
        require_production(all(path.is_file() for path in paths), "OUTPUT_SET_INCOMPLETE", unit["stem"])
        vectors = np.load(paths[0], allow_pickle=False)
        rows = pd.read_parquet(paths[2])
        metadata = json.loads(paths[1].read_text(encoding="utf-8"))
        require_production(tuple(vectors.shape) == tuple(unit["shape"]), "OUTPUT_SHAPE_MISMATCH", unit["stem"])
        require_production(vectors.dtype == np.float32, "OUTPUT_DTYPE_MISMATCH", unit["stem"])
        require_production(bool(np.isfinite(vectors).all()), "NONFINITE_OUTPUT", unit["stem"])
        norms = np.linalg.norm(vectors, axis=1)
        require_production(bool(np.all(norms > 0)), "ZERO_OUTPUT_VECTOR", unit["stem"])
        norm_error = float(np.max(np.abs(norms - 1.0)))
        require_production(norm_error <= float(contract["validation"]["max_l2_norm_error"]), "L2_NORM_MISMATCH", unit["stem"])
        require_production(len(rows) == unit["shape"][0], "ROW_SIDECAR_COUNT_MISMATCH", unit["stem"])
        row_hash = ordered_ids_sha256(rows["appId"].astype(str).tolist())
        require_production(row_hash == unit["ordered_id_sha256"], "ROW_MAPPING_MISMATCH", unit["stem"])
        validate_unit_metadata(metadata, unit, contract)
        if unit["truncation_mode"] == "token_budget":
            native_unit = next(
                candidate for candidate in units
                if candidate["language"] == unit["language"] and candidate["truncation_mode"] == "native"
            )
            native_base = namespace / "embeddings" / native_unit["stem"]
            native_npy_hash = sha256(native_base.with_suffix(".npy"))
            native_rows_hash = sha256(native_base.with_suffix(".rows.parquet"))
            require_production(
                sha256(paths[0]) == native_npy_hash,
                "T512_REUSE_ARRAY_HASH_MISMATCH",
                unit["stem"],
            )
            require_production(
                sha256(paths[2]) == native_rows_hash,
                "T512_REUSE_ROWS_HASH_MISMATCH",
                unit["stem"],
            )
            require_production(
                metadata.get("reused_source_npy_sha256") == native_npy_hash,
                "T512_REUSE_SOURCE_HASH_MISMATCH",
                unit["stem"],
            )
            require_production(
                metadata.get("reused_source_rows_sha256") == native_rows_hash,
                "T512_REUSE_SOURCE_HASH_MISMATCH",
                unit["stem"],
            )
            require_production(
                metadata.get("token_audit_receipt_sha256") == actual_audit_hash,
                "T512_REUSE_AUDIT_HASH_MISMATCH",
                unit["stem"],
            )
        summaries.append({
            "unit": unit["name"],
            "stem": unit["stem"],
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "max_l2_norm_error": norm_error,
            "ordered_id_sha256": row_hash,
            "artifacts": {path.name: sha256(path) for path in paths},
        })
    dataset = contract["datasets"]["tn2"]
    return {
        "units": summaries,
        "unit_count": len(summaries),
        "token_audit_receipt_sha256": actual_audit_hash,
        "dataset_binding": {
            "dataset_sha256": dataset["sha256"],
            "row_count": dataset["row_count"],
            "runtime_ordered_id_sha256": dataset["runtime_ordered_id_sha256"],
            "frozen_ordered_id_sha256": dataset["frozen_ordered_id_sha256"],
            "ordered_id_binding_status": dataset["ordered_id_binding_status"],
        },
    }


def finalize_production_run(
    namespace: Path, contract: dict, authorization: dict, *, contract_path: Path = Path("pipeline/e5_production_contract.json")
) -> None:
    # Validate the complete targeted four-unit output set before performing any final
    # inventory work or emitting success artifacts.
    summary = validate_production_outputs(namespace, contract)
    asset_path = namespace / "asset_inventory.json"
    terminal_paths = [
        asset_path,
        namespace / "run_manifest.json",
        namespace / "run_manifest.sha256",
        namespace / "execution_receipt.json",
    ]
    collisions = [str(path) for path in terminal_paths if path.exists()]
    require_production(not collisions, "OUTPUT_COLLISION", ",".join(collisions))
    asset_inventory = inventory_hf_assets(contract)
    write_json_create_only(asset_path, asset_inventory)
    artifact_paths = sorted(
        (
            path for path in namespace.rglob("*")
            if path.is_file()
            and path.name not in {"run_manifest.json", "run_manifest.sha256", "execution_receipt.json", "failure_receipt.json"}
        ),
        key=lambda p: p.as_posix(),
    )
    current_source_paths = {
        "pipeline/embed.py": Path(__file__).resolve(),
        "pipeline/e5_production_contract.json": contract_path.resolve(),
        "pipeline/models.yaml": Path("pipeline/models.yaml").resolve(),
        "revision_v4_20261020/00_governance/PROTOCOL_LOCK.json": Path("revision_v4_20261020/00_governance/PROTOCOL_LOCK.json").resolve(),
        "revision_v4_20261020/00_governance/GOV02_EXPERIMENT_MATRIX_V4_DRAFT.jsonl": Path("revision_v4_20261020/00_governance/GOV02_EXPERIMENT_MATRIX_V4_DRAFT.jsonl").resolve(),
    }
    missing_sources = [name for name, path in current_source_paths.items() if not path.is_file()]
    require_production(not missing_sources, "PROTECTED_SOURCE_MISSING", ",".join(missing_sources))
    manifest = {
        "task_id": E5_PRODUCTION_TASK_ID,
        "run_id": E5_PRODUCTION_RUN_ID,
        "status": "TARGETED_TN2_RECOVERY_COMPLETE",
        "outputs": summary,
        "immutable_plan_sha256": E5_PRODUCTION_PLAN_SHA256,
        "execution_authorization_sha256": authorization["sha256"],
        "source_bindings": {
            name: sha256(path) for name, path in current_source_paths.items()
        },
        "artifacts": [
            {"path": path.relative_to(namespace).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        ],
    }
    manifest_path = namespace / "run_manifest.json"
    write_json_create_only(manifest_path, manifest)
    lines = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in manifest["artifacts"]
    ) + f"{sha256(manifest_path)}  run_manifest.json\n"
    write_text_create_only(namespace / "run_manifest.sha256", lines)
    write_json_create_only(
        namespace / "execution_receipt.json",
        {
            "task_id": E5_PRODUCTION_TASK_ID,
            "run_id": E5_PRODUCTION_RUN_ID,
            "status": "TARGETED_TN2_RECOVERY_COMPLETE",
            "unit_count": summary["unit_count"],
            "run_manifest_sha256": sha256(manifest_path),
            "experiment_scope": "TN2_FOUR_NAMED_UNITS_ONLY",
            "token_audit_receipt_sha256": summary["token_audit_receipt_sha256"],
            "dataset_binding": summary["dataset_binding"],
            "execution_authorization_sha256": authorization["sha256"],
        },
    )


def _main() -> None:
    global ACTIVE_PRODUCTION_NAMESPACE, ACTIVE_PRODUCTION_PHASE, ACTIVE_EXECUTION_AUTHORIZATION_SHA256
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["tn1", "tn2"], required=True)
    ap.add_argument("--repr", nargs="+", required=True, dest="reprs")
    ap.add_argument("--lang", nargs="+", choices=["vi", "en"], default=["vi", "en"])
    ap.add_argument("--trunc", choices=["char_budget", "token_budget", "native"], required=True)
    ap.add_argument("--n-chars", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--models", type=Path, default=Path("models.yaml"))
    ap.add_argument("--data-dir", type=Path, default=Path("../data_v1"))
    ap.add_argument("--out-dir", type=Path, default=Path("../embeddings"))
    ap.add_argument("--report-dir", type=Path, default=Path("../data_v1/report"))
    ap.add_argument("--production-contract", type=Path, default=None)
    ap.add_argument("--execution-authorization", type=Path, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--require-full-cohort", action="store_true")
    ap.add_argument("--fail-if-exists", action="store_true")
    ap.add_argument("--audit-native-t512-only", action="store_true")
    ap.add_argument("--reuse-native-for-t512", action="store_true")
    ap.add_argument("--finalize-production-run", action="store_true")
    ap.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("../data_v1/report/tokenizer_audit_summary.json"),
    )
    args = ap.parse_args()

    production_contract = None
    production_namespace = None
    token_audit_evidence = None
    dataset_validation_receipt = None
    execution_authorization = None
    if args.production_contract is not None:
        production_contract = load_production_contract(args.production_contract)
        require_production(args.execution_authorization is not None, "EXECUTION_AUTHORIZATION_REQUIRED")
        execution_authorization = load_and_validate_execution_authorization(
            args.execution_authorization, args.production_contract, production_contract
        )
        production_contract["_execution_authorization"] = execution_authorization
        ACTIVE_EXECUTION_AUTHORIZATION_SHA256 = execution_authorization["sha256"]
        production_namespace = validate_production_paths(args, production_contract)
        require_production(args.reprs == [E5_REPRESENTATION_ID], "E5_ONLY_PRODUCTION_SCOPE")
        require_production(args.exp in {"tn1", "tn2"}, "EXPERIMENT_SCOPE_MISMATCH")
        require_production(not (production_namespace / "failure_receipt.json").exists(), "FAILED_NAMESPACE_IMMUTABLE")
        require_production(not (production_namespace / "execution_receipt.json").exists(), "COMPLETED_NAMESPACE_IMMUTABLE")
        if args.audit_native_t512_only:
            require_production(production_contract.get("fresh_token_audit_required") is True, "FRESH_TOKEN_AUDIT_NOT_REQUIRED")
            require_production(not production_namespace.exists(), "OUTPUT_COLLISION", str(production_namespace))
            require_production("TOKEN_AUDIT" in execution_authorization["receipt"]["authorized_phases"], "TOKEN_AUDIT_NOT_AUTHORIZED")
            require_production(not args.reuse_native_for_t512, "AUDIT_REUSE_FLAG_FORBIDDEN")
            require_production(not args.finalize_production_run, "AUDIT_FINALIZATION_FORBIDDEN")
            require_production(args.exp == "tn2", "TN2_ONLY_SCOPE_REQUIRED")
            require_production(args.trunc == "native", "NATIVE_AUDIT_CONDITION_REQUIRED")
            require_production(args.lang == ["en", "vi"], "PRODUCTION_LANGUAGE_SET_MISMATCH")
        else:
            require_production("TN2_NATIVE_INFERENCE_AND_T512_REUSE" in execution_authorization["receipt"]["authorized_phases"], "TN2_INFERENCE_NOT_AUTHORIZED")
            require_production(args.reuse_native_for_t512, "T512_REUSE_FLAG_REQUIRED")
            require_production(args.finalize_production_run, "TARGETED_FINALIZATION_REQUIRED")
            require_production(args.exp == "tn2", "TN2_ONLY_SCOPE_REQUIRED")
            require_production(args.trunc == "native", "NATIVE_FORWARD_ONLY_REQUIRED")
            require_production(args.lang == ["en", "vi"], "PRODUCTION_LANGUAGE_SET_MISMATCH")
            token_audit_evidence = validate_audit_only_namespace(
                production_namespace,
                production_contract,
                execution_authorization,
            )
            all_output_bases = [
                production_namespace / "embeddings" / unit["stem"]
                for unit in production_contract["outputs"]["units"]
            ]
            require_outputs_absent(all_output_bases)
            preflight_paths = [
                production_namespace / "reports/truncation_rates.csv",
                production_namespace / "asset_inventory.json",
                production_namespace / "run_manifest.json",
                production_namespace / "run_manifest.sha256",
                production_namespace / "execution_receipt.json",
            ]
            collisions = [str(path) for path in preflight_paths if path.exists()]
            require_production(not collisions, "OUTPUT_COLLISION", ",".join(collisions))
            ACTIVE_PRODUCTION_NAMESPACE = production_namespace
            ACTIVE_PRODUCTION_PHASE = "TN2_NATIVE_INFERENCE_AND_T512_REUSE"

    cfg = load_models(args.models)
    reps = cfg["representations"]
    chosen = resolve_reps(args.reprs, args.exp, reps)

    if args.exp == "tn1" and args.trunc not in {"char_budget", "native"}:
        raise ValueError("TN1 only supports char_budget or native.")
    if args.exp == "tn2" and args.trunc not in {"token_budget", "native"}:
        raise ValueError("TN2 only supports token_budget or native.")

    n_chars = args.n_chars
    if args.trunc == "char_budget" and n_chars is None:
        n_chars = load_n_chars(args.audit_summary)
    if args.trunc == "token_budget" and args.max_tokens is None:
        raise ValueError("TN2 token_budget requires explicit --max-tokens 512.")
    if args.trunc == "token_budget" and args.max_tokens != 512:
        raise ValueError("PAAC-VN protocol freezes TN2 token_budget at exactly 512.")

    data_path = args.data_dir / (TN1_FILE if args.exp == "tn1" else TN2_FILE)
    if args.exp == "tn1":
        cols = ["appId", "genreId", TN1_FIELD]
    else:
        cols = ["appId", "genreId", TN2_FIELDS["en"], TN2_FIELDS["vi"]]
    df = pd.read_parquet(data_path, columns=cols)

    if production_contract is not None:
        expected = production_contract["datasets"][args.exp]
        dataset_validation_receipt = validate_production_dataset(
            df, data_path=data_path, expected=expected
        )

    # Pair-safe base guard. TN2 removes a pair if either side violates the guard.
    if args.exp == "tn1":
        base_texts = df[TN1_FIELD].fillna("").astype(str).tolist()
        keep = raw_guard_mask(base_texts)
    else:
        en_all = df[TN2_FIELDS["en"]].fillna("").astype(str).tolist()
        vi_all = df[TN2_FIELDS["vi"]].fillna("").astype(str).tolist()
        keep = raw_guard_mask(en_all) & raw_guard_mask(vi_all)

    if production_contract is not None:
        validate_full_cohort_mask(keep, int(production_contract["datasets"][args.exp]["row_count"]))

    if production_contract is not None and args.audit_native_t512_only:
        runtime = production_contract["runtime"]
        source_manifest_text = production_source_manifest(
            args.production_contract, production_contract, execution_authorization
        )
        runtime_diagnostics = configure_cuda_fp32(
            device_name=args.device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
        )
        spec = reps[E5_REPRESENTATION_ID]
        require_production(spec["hf_name"] == E5_CHECKPOINT, "CHECKPOINT_MISMATCH")
        require_production(spec["revision"] == E5_REVISION, "REVISION_MISMATCH")
        if execution_authorization["receipt"]["asset_mode"] == "PREPOPULATED_OFFLINE_CACHE":
            inventory_hf_assets(production_contract)
        tokenizer = AutoTokenizer.from_pretrained(
            E5_CHECKPOINT,
            revision=E5_REVISION,
            use_fast=bool(spec.get("tokenizer_use_fast", True)),
            trust_remote_code=False,
            cache_dir=production_contract["paths"]["model_cache"],
            local_files_only=execution_authorization["receipt"]["asset_mode"] == "PREPOPULATED_OFFLINE_CACHE",
        )
        expected_dataset = production_contract["datasets"]["tn2"]
        audit = {
            "task_id": E5_PRODUCTION_TASK_ID,
            "run_id": E5_PRODUCTION_RUN_ID,
            "checkpoint": E5_CHECKPOINT,
            "revision": E5_REVISION,
            "dataset": {
                "sha256": expected_dataset["sha256"],
                "ordered_id_sha256": expected_dataset["ordered_id_sha256"],
            },
            "tokenizer_batch_size": 8,
            "execution_authorization_sha256": execution_authorization["sha256"],
            "asset_inventory": inventory_hf_assets(production_contract),
            "languages": {},
        }
        for language in ("en", "vi"):
            texts = df[TN2_FIELDS[language]].fillna("").astype(str).tolist()
            native = prepare_neural_texts(texts, spec, "native", None)
            t512 = prepare_neural_texts(texts, spec, "token_budget", None)
            historical = ["query: " + str(text).strip() for text in texts]
            require_production(native == historical, "PREPROCESSING_EQUIVALENCE_MISMATCH", language)
            audit["languages"][language] = audit_native_t512_token_equality(
                tokenizer=tokenizer,
                prepared_native=native,
                prepared_t512=t512,
                batch_size=8,
            )
        activate_new_production_namespace(production_namespace, "TOKEN_AUDIT")
        write_text_create_only(
            production_namespace / "source_manifest.sha256",
            source_manifest_text,
        )
        write_json_create_only(production_namespace / "token_audit/native_t512_equality.json", audit)
        write_json_create_only(
            production_namespace / "environment.json",
            {
                **runtime_diagnostics,
                **runtime_environment_diagnostics(production_namespace, runtime),
                "runtime_contract": runtime,
                "asset_mode": execution_authorization["receipt"]["asset_mode"],
                "execution_authorization_sha256": execution_authorization["sha256"],
                "model_forward_status": "NOT_STARTED",
            },
        )
        print("TOKEN_AUDIT_PASS")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if production_contract is not None:
        validate_production_invocation(args, production_contract, n_chars)

    removed = np.flatnonzero(~keep)
    if len(removed):
        empty_rows = []
        for i in removed:
            empty_rows.append({
                "experiment": args.exp,
                "truncation_mode": args.trunc,
                "appId": str(df.iloc[i]["appId"]),
                "reason": f"pair/raw text < {MIN_LEN_AFTER_TRUNC} chars",
            })
        append_rows_csv(args.report_dir / "empty_after_trunc.csv", empty_rows)

    work = df.loc[keep].reset_index(drop=True)
    row_sidecar = work[["appId", "genreId"]].copy()
    row_sidecar.insert(0, "row_idx", np.arange(len(work), dtype=np.int64))

    trunc_rows = []

    # TF-IDF is special because TN2 must fit EN+VI jointly.
    if "tfidf_lsa" in chosen:
        spec = reps["tfidf_lsa"]
        if args.exp == "tn1":
            corpus, token_lens = prepare_tfidf_texts(
                work[TN1_FIELD].astype(str).tolist(), spec, args.trunc, n_chars, args.max_tokens
            )
            X = fit_tfidf_svd([corpus], spec)[0]
            extra = {
                "effective_max_tokens": None,
                "truncated_count": 0,
                "truncation_rate": 0.0,
            }
            base = output_base(args.out_dir, args.exp, "tfidf_lsa", "vi", args.trunc)
            meta = common_meta(
                exp=args.exp, rid="tfidf_lsa", lang="vi", trunc=args.trunc, spec=spec,
                data_path=data_path, models_path=args.models, n_chars=n_chars,
                max_tokens=args.max_tokens, X=X, app_ids=row_sidecar["appId"].astype(str).tolist(),
                extra=extra,
            )
            save_embedding(
                X,
                row_sidecar,
                base,
            )

            save_metadata(
                base,
                meta,
            )
            trunc_rows.append({
                "experiment": args.exp, "representation": "tfidf_lsa", "language": "vi",
                "truncation_mode": args.trunc, "n": len(X), "max_tokens": "",
                "truncated_count": 0, "truncation_rate": 0.0,
            })
        else:
            langs = list(dict.fromkeys(args.lang))
            if set(langs) != {"en", "vi"}:
                raise ValueError("TN2 TF-IDF must generate EN and VI together so vectorizer+SVD are fit jointly.")
            corpora, lens_by_lang = [], {}
            for lang in ["en", "vi"]:
                corpus, lens = prepare_tfidf_texts(
                    work[TN2_FIELDS[lang]].astype(str).tolist(),
                    spec, args.trunc, n_chars, args.max_tokens
                )
                corpora.append(corpus)
                lens_by_lang[lang] = lens
            X_en, X_vi = fit_tfidf_svd(corpora, spec)
            for lang, X in [("en", X_en), ("vi", X_vi)]:
                lens = lens_by_lang[lang]
                if args.trunc == "token_budget":
                    hit_count = int(np.sum(lens > args.max_tokens))
                    hit_rate = float(hit_count / len(lens))
                    eff = args.max_tokens
                else:
                    hit_count, hit_rate, eff = 0, 0.0, None
                extra = {
                    "effective_max_tokens": eff,
                    "truncated_count": hit_count,
                    "truncation_rate": hit_rate,
                    "tn2_joint_fit_en_vi": True,
                }
                base = output_base(args.out_dir, args.exp, "tfidf_lsa", lang, args.trunc)
                meta = common_meta(
                    exp=args.exp, rid="tfidf_lsa", lang=lang, trunc=args.trunc, spec=spec,
                    data_path=data_path, models_path=args.models, n_chars=n_chars,
                    max_tokens=args.max_tokens, X=X, app_ids=row_sidecar["appId"].astype(str).tolist(),
                    extra=extra,
                )
                save_embedding(
                    X,
                    row_sidecar,
                    base,
                )

                save_metadata(
                    base,
                    meta,
                )
                trunc_rows.append({
                    "experiment": args.exp, "representation": "tfidf_lsa", "language": lang,
                    "truncation_mode": args.trunc, "n": len(X), "max_tokens": eff or "",
                    "truncated_count": hit_count, "truncation_rate": hit_rate,
                })

    # Neural representations: one model in memory at a time. In production,
    # preflight every output for this invocation before the first model load or
    # write so a collision in a later language cannot leave earlier mutations.
    if production_contract is not None:
        invocation_bases = [
            output_base(args.out_dir, args.exp, rid, lang, args.trunc)
            for rid in chosen if rid != "tfidf_lsa"
            for lang in (["vi"] if args.exp == "tn1" else list(dict.fromkeys(args.lang)))
        ]
        require_outputs_absent(invocation_bases)
    for rid in chosen:
        if rid == "tfidf_lsa":
            continue
        spec = reps[rid]
        langs = ["vi"] if args.exp == "tn1" else list(dict.fromkeys(args.lang))
        for lang in langs:
            field = TN1_FIELD if args.exp == "tn1" else TN2_FIELDS[lang]
            texts = work[field].astype(str).tolist()
            print(f"[embed] {args.exp} {rid} {lang} {args.trunc} n={len(texts)}")
            t_total_start = time.perf_counter()
            X, extra = embed_neural(
                rid,
                spec,
                texts,
                args.trunc,
                n_chars,
                args.max_tokens,
                args.batch_size,
                production_contract,
            )
            base = output_base(args.out_dir, args.exp, rid, lang, args.trunc)

            save_seconds = save_embedding(
                X,
                row_sidecar,
                base,
                create_only=production_contract is not None,
            )

            total_seconds = (
                time.perf_counter() - t_total_start
            )

            extra["save_seconds"] = save_seconds
            extra["total_seconds"] = total_seconds
            if production_contract is not None:
                require_production(dataset_validation_receipt is not None, "DATASET_VALIDATION_RECEIPT_MISSING")
                extra.update({
                    "runtime_ordered_id_sha256": dataset_validation_receipt["runtime_ordered_id_sha256"],
                    "frozen_ordered_id_sha256": dataset_validation_receipt["frozen_ordered_id_sha256"],
                    "ordered_id_binding_status": dataset_validation_receipt["ordered_id_binding_status"],
                    "execution_authorization_sha256": execution_authorization["sha256"],
                })

            meta = common_meta(
                exp=args.exp,
                rid=rid,
                lang=lang,
                trunc=args.trunc,
                spec=spec,
                data_path=data_path,
                models_path=args.models,
                n_chars=n_chars,
                max_tokens=args.max_tokens,
                X=X,
                app_ids=row_sidecar[
                    "appId"
                ].astype(str).tolist(),
                extra=extra,
            )

            save_metadata(
                base,
                meta,
                create_only=production_contract is not None,
            )
            trunc_rows.append({
                "experiment": args.exp,
                "representation": rid,
                "language": lang,
                "truncation_mode": args.trunc,
                "n": len(X),
                "max_tokens": extra["effective_max_tokens"],
                "truncated_count": extra["truncated_count"],
                "truncation_rate": extra["truncation_rate"],
            })

    if production_contract is not None and args.reuse_native_for_t512:
        require_production(token_audit_evidence is not None, "TOKEN_AUDIT_EVIDENCE_MISSING")
        materialize_t512_reuse(production_namespace, production_contract, token_audit_evidence, execution_authorization)
    append_rows_csv(args.report_dir / "truncation_rates.csv", trunc_rows)
    if production_contract is not None and args.finalize_production_run:
        finalize_production_run(
            production_namespace,
            production_contract,
            execution_authorization,
            contract_path=args.production_contract,
        )
    print("DONE")


def main() -> None:
    try:
        _main()
    except Exception as exc:
        namespace = ACTIVE_PRODUCTION_NAMESPACE
        if namespace is not None:
            failure_path = namespace / "failure_receipt.json"
            success_path = namespace / "execution_receipt.json"
            if not failure_path.exists() and not success_path.exists():
                try:
                    write_json_create_only(
                        failure_path,
                        {
                            "task_id": E5_PRODUCTION_TASK_ID,
                            "run_id": E5_PRODUCTION_RUN_ID,
                            "status": "FAILED",
                            "phase": ACTIVE_PRODUCTION_PHASE or "UNKNOWN",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "execution_authorization_sha256": ACTIVE_EXECUTION_AUTHORIZATION_SHA256,
                            "source_manifest_sha256": (
                                sha256(namespace / "source_manifest.sha256")
                                if (namespace / "source_manifest.sha256").is_file() else None
                            ),
                            "model_forward_status": (
                                "NOT_STARTED" if ACTIVE_PRODUCTION_PHASE == "TOKEN_AUDIT" else "UNVERIFIED_AT_FAILURE"
                            ),
                            "success_receipt_emitted": False,
                        },
                    )
                except Exception:
                    pass
        raise


if __name__ == "__main__":
    main()
