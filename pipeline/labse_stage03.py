"""LaBSE Stage 03: pinned adapter, reference probe, tokenizer audit and inference.

Independent of the active E5 run. Never modifies frozen YAML or E5 outputs.
Examples: python -m pipeline.labse_stage03 preflight RUN_ID
          python -m pipeline.labse_stage03 probe RUN_ID --snapshot PATH
          python -m pipeline.labse_stage03 audit RUN_ID --snapshot PATH
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from pipeline import embed

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pipeline/labse_stage03_contract.json"
SMALL_ASSETS = ["modules.json", "sentence_bert_config.json", "config.json",
                "config_sentence_transformers.json", "1_Pooling/config.json", "2_Dense/config.json",
                "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.txt"]
WEIGHT_ASSETS = ["model.safetensors", "2_Dense/model.safetensors"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract():
    contract = read_json(CONTRACT)
    require(contract["model_id"] == "sentence-transformers/LaBSE", "MODEL_ID_CHANGED")
    require(contract["revision"] == "836121a0533e5664b21c7aacc5d22951f2b8b25b", "REVISION_CHANGED")
    require(contract["native_cap"] == 256 and contract["dimension"] == 768, "CAP_OR_DIM_CHANGED")
    require(contract["common_characters"] == 950 and not contract["word_segmentation"]
            and contract["prefix"] == "", "INPUT_RECIPE_CHANGED")
    return contract


def sources():
    names = ["pipeline/labse_stage03.py", "pipeline/labse_stage03_contract.json", "pipeline/embed.py"]
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sha256": {n: embed.sha256(ROOT / n) for n in names}}


def load_inputs(contract):
    frames = {}
    for exp, expected in contract["datasets"].items():
        path = ROOT / expected["path"]
        require(embed.sha256(path) == expected["sha256"], f"DATASET_HASH:{exp}")
        df = pd.read_parquet(path)
        require(len(df) == expected["n_rows"] and df.appId.is_unique, f"DATASET_ROWS:{exp}")
        require(embed.ordered_ids_sha256(df.appId) == expected["ordered_ids_sha256"], f"DATASET_ORDER:{exp}")
        frames[exp] = df
    return frames


def validate_configs(configs, contract):
    expected = [("", "Transformer"), ("1_Pooling", "Pooling"),
                ("2_Dense", "Dense"), ("3_Normalize", "Normalize")]
    graph = configs["modules.json"]
    require([(m["path"], m["type"].rsplit(".", 1)[-1]) for m in graph] == expected,
            "UNEXPECTED_MODULE_GRAPH")
    require([m["idx"] for m in graph] == [0, 1, 2, 3], "MODULE_ORDER")
    sentence = configs["sentence_bert_config.json"]
    require(sentence["max_seq_length"] == contract["native_cap"] and
            sentence.get("do_lower_case") is False, "EFFECTIVE_CAP_OR_CASE_MISMATCH")
    backbone = configs["config.json"]
    require(backbone["model_type"] == "bert" and backbone["hidden_size"] == 768 and
            backbone["max_position_embeddings"] >= contract["native_cap"], "BACKBONE_CONFIG")
    pooling = configs["1_Pooling/config.json"]
    require(pooling["word_embedding_dimension"] == 768 and
            pooling.get("pooling_mode_cls_token") is True, "CLS_POOLING_REQUIRED")
    require(not any(v for k, v in pooling.items() if k.startswith("pooling_mode_") and
                    k != "pooling_mode_cls_token"), "OTHER_POOLING_FORBIDDEN")
    dense = configs["2_Dense/config.json"]
    require(dense["in_features"] == 768 and dense["out_features"] == 768 and
            dense["bias"] is True and dense["activation_function"] == "torch.nn.modules.activation.Tanh",
            "DENSE_TANH_REQUIRED")
    return {"effective_native_cap": sentence["max_seq_length"], "dimension": 768,
            "graph": "Transformer/CLS/Dense-tanh/Normalize", "do_lower_case": False}


def resolve_snapshot(contract, snapshot, *, allow_download=False, weights=False, tokenizer_only=False):
    required = (["tokenizer.json", "sentence_bert_config.json"] if tokenizer_only else
                SMALL_ASSETS + (WEIGHT_ASSETS if weights else []))
    if snapshot is None:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            contract["model_id"], revision=contract["revision"],
            cache_dir=ROOT / "results/labse_stage03/cache",
            allow_patterns=required,
            local_files_only=not allow_download))
    require(snapshot.name == contract["revision"], "SNAPSHOT_REVISION_UNBOUND")
    require(all((snapshot / n).is_file() for n in required), "SNAPSHOT_INCOMPLETE")
    if tokenizer_only:
        sentence = read_json(snapshot / "sentence_bert_config.json")
        require(sentence["max_seq_length"] == contract["native_cap"] and sentence["do_lower_case"] is False,
                "EFFECTIVE_CAP_OR_CASE_MISMATCH")
        config_summary = {"effective_native_cap": sentence["max_seq_length"], "tokenizer_backend_only": True,
                          "full_module_graph_verified": False}
    else:
        configs = {name: read_json(snapshot / name) for name in SMALL_ASSETS if name.endswith(".json")}
        config_summary = validate_configs(configs, contract)
    assets = {name: embed.sha256(snapshot / name) for name in required}
    return snapshot, {"model_id": contract["model_id"], "revision": contract["revision"],
                      "config_summary": config_summary, "assets_sha256": assets}


def load_audit_tokenizer(snapshot):
    # tokenizer.json is self-contained for lengths: normalizer, WordPiece vocab,
    # special-token postprocessor and pretokenizer. Do not infer defaults from BERT.
    document = read_json(snapshot / "tokenizer.json")
    require(document["normalizer"]["lowercase"] is False and document["model"]["type"] == "WordPiece",
            "TOKENIZER_NORMALIZER_MISMATCH")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snapshot / "tokenizer.json"))
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    texts = ["An English description.", "Ứng dụng bản đồ và chỉ đường.", "日本語。", "word " * 600]
    wrapped = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    direct = [backend.encode(t, add_special_tokens=True).ids for t in texts]
    require(wrapped == direct, "TOKENIZER_BACKEND_WRAPPER_MISMATCH")
    return tokenizer


def prepared_texts(texts, condition, contract):
    require(condition in {"native", "char_budget", "token_budget"}, "UNKNOWN_INPUT_CONDITION")
    spec = {"needs_word_segmentation": False, "clustering_prefix": ""}
    return embed.prepare_neural_texts(list(texts), spec, condition,
                                     contract["common_characters"] if condition == "char_budget" else None)


def effective_cap(contract, condition, requested_tokens=512):
    # Backbone position capacity is not the SentenceTransformer's effective cap.
    return min(contract["native_cap"], requested_tokens) if condition == "token_budget" else contract["native_cap"]


def cls_dense_normalize(last_hidden_state, weight, bias):
    projected = F.linear(last_hidden_state[:, 0], weight, bias)
    return F.normalize(torch.tanh(projected).float(), p=2, dim=1)


class LaBSEAdapter:
    def __init__(self, tokenizer, backbone, dense_state, cap=256):
        require(set(dense_state) == {"linear.weight", "linear.bias"}, "DENSE_WEIGHT_KEYS")
        require(tuple(dense_state["linear.weight"].shape) == (768, 768) and
                tuple(dense_state["linear.bias"].shape) == (768,), "DENSE_WEIGHT_SHAPE")
        self.tokenizer, self.backbone, self.cap = tokenizer, backbone.eval(), cap
        device = next(backbone.parameters()).device
        self.weight = dense_state["linear.weight"].to(device=device, dtype=torch.float32)
        self.bias = dense_state["linear.bias"].to(device=device, dtype=torch.float32)
        self.device = device

    @classmethod
    def from_snapshot(cls, snapshot, device):
        tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True,
                                                  trust_remote_code=False)
        backbone = AutoModel.from_pretrained(snapshot, torch_dtype=torch.float32,
                                             use_safetensors=True, local_files_only=True,
                                             trust_remote_code=False).to(device)
        return cls(tokenizer, backbone, load_file(str(snapshot / "2_Dense/model.safetensors")))

    def encode(self, prepared, batch_size=8):
        lengths = embed.token_lengths(self.tokenizer, prepared)
        order = np.lexsort((np.arange(len(prepared)), lengths))
        result = np.empty((len(prepared), 768), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                tokens = self.tokenizer([prepared[i] for i in indices], padding=True, truncation=True,
                                        max_length=self.cap, add_special_tokens=True, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with torch.autocast(device_type=self.device.type, enabled=False):
                    output = self.backbone(**tokens, return_dict=True)
                    vectors = cls_dense_normalize(output.last_hidden_state, self.weight, self.bias)
                result[indices] = vectors.cpu().numpy()
        require(np.isfinite(result).all(), "NONFINITE_EMBEDDINGS")
        require(np.max(np.abs(np.linalg.norm(result, axis=1) - 1)) <= 1e-5, "INVALID_VECTOR_NORMS")
        return result, {"effective_cap": self.cap, "truncated_count": int((lengths > self.cap).sum()),
                        "prepared_text_sha256": embed.canonical_token_fingerprint({"texts": prepared}),
                        "batch_order_sha256": embed.canonical_token_fingerprint({"order": order.tolist()})}


def reference_probe(snapshot, contract):
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    reference = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True,
                                    model_kwargs={"torch_dtype": torch.float32}, trust_remote_code=False)
    require(reference.max_seq_length == contract["native_cap"], "REFERENCE_CAP_MISMATCH")
    # Share the loaded backbone to avoid duplicating ~1.9GB; tokenize and apply the
    # checkpoint's separate Dense head independently. This tests adapter composition.
    adapter = LaBSEAdapter(
        AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True),
        reference[0].auto_model, load_file(str(snapshot / "2_Dense/model.safetensors")))
    texts = ["An application for maps and navigation.", "Ứng dụng bản đồ và chỉ đường.",
             "Trò chơi giải đố trên điện thoại.", "Manage tasks and notes.",
             "日本語の説明とアプリ。", "word " * 600]
    prepared = prepared_texts(texts, "native", contract)
    actual, details = adapter.encode(prepared, batch_size=8)
    expected = reference.encode(prepared, batch_size=8, convert_to_numpy=True,
                                show_progress_bar=False, normalize_embeddings=False)
    difference = float(np.max(np.abs(actual - expected)))
    require(difference <= contract["reference_probe"]["max_abs_difference"], "REFERENCE_OUTPUT_MISMATCH")
    return {"status": "CHECKPOINT_REFERENCE_PASS", "device": "cpu", "dtype": "float32",
            "n_texts": len(texts), "max_abs_difference": difference,
            "max_norm_error": float(np.max(np.abs(np.linalg.norm(actual, axis=1) - 1))),
            "shared_backbone": True, "reference_library": importlib.metadata.version("sentence-transformers"),
            **details}, actual, expected


def tokenizer_audit(frame, tokenizer, contract):
    texts = frame.description_clean.tolist()
    summary = {}
    for condition in ("native", "char_budget"):
        prepared = prepared_texts(texts, condition, contract)
        lengths = embed.token_lengths(tokenizer, prepared)
        cap = effective_cap(contract, condition)
        summary[condition] = {"n_apps": len(frame), "cap": cap, "p95_tokens": float(np.percentile(lengths, 95)),
                              "median_tokens": float(np.median(lengths)), "token_truncated_count": int((lengths > cap).sum()),
                              "character_clipped_count": sum(a != str(b).strip() for a, b in zip(prepared, texts)),
                              "prepared_text_sha256": embed.canonical_token_fingerprint({"texts": prepared})}
    return {"ordered_ids_sha256": embed.ordered_ids_sha256(frame.appId), "conditions": summary,
            "cutoff_950_preserved": True, "is_optimal_cutoff_claim": False}


def configure_production():
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (None, ":4096:8"), "CUBLAS_CONFIG_CONFLICT")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE_NO_CPU_FALLBACK")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return embed.configure_cuda_fp32(device_name="cuda:0", dtype_name="float32", batch_size=8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "inspect", "audit", "probe", "encode"])
    parser.add_argument("run_id")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cell", choices=["tn1:vi:char_budget", "tn1:vi:native", "tn2:en:native", "tn2:vi:native"])
    parser.add_argument("--approved-by")
    parser.add_argument("--approval-reference")
    parser.add_argument("--reference-receipt", type=Path)
    args = parser.parse_args(argv)
    require(args.run_id and all(c.isalnum() or c in "_-" for c in args.run_id), "INVALID_RUN_ID")
    contract = load_contract()
    namespace = ROOT / "results/labse_stage03" / args.run_id
    require(not namespace.exists(), "OUTPUT_NAMESPACE_EXISTS")
    namespace.mkdir(parents=True, exist_ok=False)
    receipt = {"mode": args.mode, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "sources": sources(), "model_id": contract["model_id"], "revision": contract["revision"],
               "packages": embed.package_versions()}
    try:
        if args.mode == "preflight":
            frames = load_inputs(contract)
            receipt.update(status="INPUTS_VERIFIED_NOT_EXECUTED", n_apps={k: len(v) for k, v in frames.items()},
                           cuda_available=torch.cuda.is_available(), production_authorized=False)
        else:
            if args.mode == "encode":
                require(args.approved_by and args.approval_reference and args.cell and args.reference_receipt,
                        "OWNER_APPROVAL_CELL_AND_REFERENCE_REQUIRED")
                probe = read_json(args.reference_receipt)
                require(probe["status"] == "CHECKPOINT_REFERENCE_PASS" and probe["revision"] == contract["revision"]
                        and probe["sources"] == receipt["sources"], "REFERENCE_RECEIPT_MISMATCH")
                receipt["runtime"] = configure_production()
                receipt["approval"] = {"approved_by": args.approved_by, "reference": args.approval_reference,
                                       "reference_receipt_sha256": embed.sha256(args.reference_receipt)}
            snapshot, receipt["checkpoint"] = resolve_snapshot(
                contract, args.snapshot, allow_download=args.allow_download, weights=args.mode in {"probe", "encode"},
                tokenizer_only=args.mode == "audit")
            if args.mode == "inspect":
                receipt["status"] = "CHECKPOINT_CONFIG_VERIFIED"
            elif args.mode == "audit":
                frames = load_inputs(contract)
                tokenizer = load_audit_tokenizer(snapshot)
                receipt["audit"] = tokenizer_audit(frames["tn1"], tokenizer, contract)
                receipt["status"] = "TOKENIZER_AUDIT_COMPLETE"
            elif args.mode == "probe":
                result, actual, expected = reference_probe(snapshot, contract)
                receipt.update(result)
                for name, array in (("adapter.npy", actual), ("reference.npy", expected)):
                    with (namespace / name).open("xb") as handle:
                        np.save(handle, array)
            elif args.mode == "encode":
                require(probe["checkpoint"] == receipt["checkpoint"], "REFERENCE_ASSETS_MISMATCH")
                for name, digest in receipt["sources"]["sha256"].items():
                    committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
                    require(hashlib.sha256(committed).hexdigest() == digest, "UNCOMMITTED_PRODUCTION_SOURCE")
                frames = load_inputs(contract)
                exp, lang, condition = args.cell.split(":")
                frame = frames[exp]
                field = "description_clean" if lang == "vi" else "description_original_clean"
                adapter = LaBSEAdapter.from_snapshot(snapshot, "cuda:0")
                vectors, details = adapter.encode(prepared_texts(frame[field], condition, contract), batch_size=8)
                base = namespace / f"{exp}__labse__{lang}__{condition}"
                rows = frame[["appId", "genreId"]].reset_index(drop=True)
                rows.insert(0, "row_idx", np.arange(len(rows)))
                receipt.update(status="EMBEDDINGS_CREATED_PENDING_LOCAL_VALIDATION", cell=args.cell,
                               ordered_ids_sha256=embed.ordered_ids_sha256(rows.appId), **details)
                embed.save_embedding(vectors, rows, base, create_only=True)
                embed.save_metadata(base, receipt, create_only=True)
                embed.write_json_create_only(base.with_suffix(".app_ids.json"), rows.appId.tolist())
                receipt["artifacts"] = {p.name: embed.sha256(p) for p in namespace.glob(base.name + ".*")}
    except Exception as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        embed.write_json_create_only(namespace / "execution_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(namespace / "execution_receipt.json")}))


if __name__ == "__main__":
    main()
