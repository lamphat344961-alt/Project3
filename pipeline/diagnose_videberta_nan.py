#!/usr/bin/env python
"""
PAAC-VN — targeted ViDeBERTa NaN diagnosis.

Purpose:
- Find bad rows from the two existing ViDeBERTa embeddings.
- Reproduce only the original batches containing those rows.
- Compare:
    A) original inference recipe: torch_dtype="auto", original batch
    B) torch_dtype="auto", singleton row
    C) forced float32, original batch
- No dataset modification.
- No embedding artifact overwrite.
- Does NOT rerun full embeddings.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

# Import exact PAAC-VN preprocessing/pooling code.
from embed import cut_chars_at_word_boundary, pool_outputs
from pyvi import ViTokenizer


RID = "videberta_base"
EXP = "tn1"
LANG = "vi"
BATCH_SIZE = 32
N_CHARS = 950


def bad_rows(path: Path) -> list[int]:
    X = np.load(path, mmap_mode="r")
    out = []
    for start in range(0, len(X), 2048):
        stop = min(start + 2048, len(X))
        block = np.asarray(X[start:stop])
        mask = ~np.isfinite(block).all(axis=1)
        out.extend((np.flatnonzero(mask) + start).tolist())
    return out


def read_selected_rows(parquet_path: Path, wanted: set[int], columns: list[str]) -> dict[int, dict]:
    """Read only parquet row groups that contain requested absolute row indices."""
    pf = pq.ParquetFile(parquet_path)
    result: dict[int, dict] = {}
    offset = 0

    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        rg_start, rg_stop = offset, offset + n
        local_wanted = sorted(i - rg_start for i in wanted if rg_start <= i < rg_stop)

        if local_wanted:
            table = pf.read_row_group(rg, columns=columns)
            for local in local_wanted:
                row = {}
                for col in columns:
                    row[col] = table[col][local].as_py()
                result[rg_start + local] = row

        offset = rg_stop

    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(f"Could not read requested dataset rows: {missing}")
    return result


def prepare_one(text: str, trunc: str, spec: dict) -> str:
    text = str(text).strip()
    if trunc == "char_budget":
        text = cut_chars_at_word_boundary(text, N_CHARS)

    if spec["needs_word_segmentation"]:
        text = ViTokenizer.tokenize(text)

    return spec.get("clustering_prefix", "") + text


def run_forward(model, tokenizer, prepared: list[str], spec: dict, device: torch.device):
    enc = tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=int(spec["max_len"]),
        return_tensors="pt",
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = model(**enc, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = pool_outputs(outputs, enc["attention_mask"], spec["pooling"])
        normed = F.normalize(pooled.float(), p=2, dim=1)

    hidden_ok = torch.isfinite(hidden).all(dim=(1, 2)).detach().cpu().numpy()
    pooled_ok = torch.isfinite(pooled).all(dim=1).detach().cpu().numpy()
    normed_ok = torch.isfinite(normed).all(dim=1).detach().cpu().numpy()

    return hidden_ok, pooled_ok, normed_ok


def load_model_and_tokenizer(spec: dict, dtype_mode: str):
    tokenizer = AutoTokenizer.from_pretrained(
        spec["hf_name"],
        revision=spec["revision"],
        use_fast=bool(spec.get("tokenizer_use_fast", True)),
        trust_remote_code=False,
    )

    dtype = "auto" if dtype_mode == "auto" else torch.float32
    model = AutoModel.from_pretrained(
        spec["hf_name"],
        revision=spec["revision"],
        trust_remote_code=False,
        torch_dtype=dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/audit/videberta_nan_diagnosis.csv"),
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    models_path = repo / "pipeline" / "models.yaml"
    with models_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    spec = cfg["representations"][RID]

    emb_dir = repo / "embeddings" / "tn1"
    char_path = emb_dir / "tn1__videberta_base__vi__char_budget.npy"
    native_path = emb_dir / "tn1__videberta_base__vi__native.npy"

    bad_by_mode = {
        "char_budget": bad_rows(char_path),
        "native": bad_rows(native_path),
    }

    print("Bad rows found:")
    for mode, rows in bad_by_mode.items():
        print(f"  {mode}: {rows}")

    union_bad = sorted(set(bad_by_mode["char_budget"]) | set(bad_by_mode["native"]))

    # Exact original batches containing any bad row.
    batch_rows = set()
    for idx in union_bad:
        start = (idx // BATCH_SIZE) * BATCH_SIZE
        stop = min(start + BATCH_SIZE, 23883)
        batch_rows.update(range(start, stop))

    wanted = set(union_bad) | batch_rows
    data_path = repo / "data_v1" / "benchmark_v1.parquet"
    data = read_selected_rows(
        data_path,
        wanted,
        columns=["appId", "genreId", "description_clean"],
    )

    # Sidecar cross-check for the bad rows only.
    rows_path = emb_dir / "tn1__videberta_base__vi__char_budget.rows.parquet"
    side = read_selected_rows(
        rows_path,
        set(union_bad),
        columns=["row_idx", "appId", "genreId"],
    )
    for idx in union_bad:
        if int(side[idx]["row_idx"]) != idx:
            raise RuntimeError(f"row_idx mismatch at {idx}")
        if str(side[idx]["appId"]) != str(data[idx]["appId"]):
            raise RuntimeError(f"appId mismatch at {idx}")

    records = []
    for mode in ["char_budget", "native"]:
        for idx in bad_by_mode[mode]:
            prepared = prepare_one(data[idx]["description_clean"], mode, spec)
            records.append({
                "truncation": mode,
                "row_idx": idx,
                "appId": str(data[idx]["appId"]),
                "genreId": str(data[idx]["genreId"]),
                "clean_chars": len(str(data[idx]["description_clean"])),
                "prepared_chars": len(prepared),
            })

    # Token lengths can be checked once with the canonical tokenizer.
    tok = AutoTokenizer.from_pretrained(
        spec["hf_name"],
        revision=spec["revision"],
        use_fast=bool(spec.get("tokenizer_use_fast", True)),
        trust_remote_code=False,
    )
    for rec in records:
        idx = rec["row_idx"]
        prepared = prepare_one(data[idx]["description_clean"], rec["truncation"], spec)
        ids = tok(
            prepared,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        rec["token_len_before_trunc"] = len(ids)
        rec["would_truncate_at_512"] = len(ids) > int(spec["max_len"])

    del tok
    gc.collect()

    # A) original auto dtype on exact original batches.
    print("\n[A] auto dtype + original batch")
    model, tokenizer, device = load_model_and_tokenizer(spec, "auto")
    print(f"device={device}, model_dtype={next(model.parameters()).dtype}")

    auto_batch_results = {}
    for mode in ["char_budget", "native"]:
        starts = sorted({(i // BATCH_SIZE) * BATCH_SIZE for i in bad_by_mode[mode]})
        for start in starts:
            stop = min(start + BATCH_SIZE, 23883)
            idxs = list(range(start, stop))
            prepared = [prepare_one(data[i]["description_clean"], mode, spec) for i in idxs]
            h_ok, p_ok, n_ok = run_forward(model, tokenizer, prepared, spec, device)
            for pos, idx in enumerate(idxs):
                if idx in bad_by_mode[mode]:
                    auto_batch_results[(mode, idx)] = (
                        bool(h_ok[pos]), bool(p_ok[pos]), bool(n_ok[pos])
                    )
                    print(
                        f"  {mode} row={idx}: "
                        f"hidden={h_ok[pos]} pooled={p_ok[pos]} normalized={n_ok[pos]}"
                    )

    # B) same auto dtype, singleton input.
    print("\n[B] auto dtype + singleton")
    auto_single_results = {}
    for mode in ["char_budget", "native"]:
        for idx in bad_by_mode[mode]:
            prepared = [prepare_one(data[idx]["description_clean"], mode, spec)]
            h_ok, p_ok, n_ok = run_forward(model, tokenizer, prepared, spec, device)
            auto_single_results[(mode, idx)] = (
                bool(h_ok[0]), bool(p_ok[0]), bool(n_ok[0])
            )
            print(
                f"  {mode} row={idx}: "
                f"hidden={h_ok[0]} pooled={p_ok[0]} normalized={n_ok[0]}"
            )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # C) float32 on exact original batches.
    print("\n[C] float32 + original batch")
    model, tokenizer, device = load_model_and_tokenizer(spec, "float32")
    print(f"device={device}, model_dtype={next(model.parameters()).dtype}")

    fp32_batch_results = {}
    for mode in ["char_budget", "native"]:
        starts = sorted({(i // BATCH_SIZE) * BATCH_SIZE for i in bad_by_mode[mode]})
        for start in starts:
            stop = min(start + BATCH_SIZE, 23883)
            idxs = list(range(start, stop))
            prepared = [prepare_one(data[i]["description_clean"], mode, spec) for i in idxs]
            h_ok, p_ok, n_ok = run_forward(model, tokenizer, prepared, spec, device)
            for pos, idx in enumerate(idxs):
                if idx in bad_by_mode[mode]:
                    fp32_batch_results[(mode, idx)] = (
                        bool(h_ok[pos]), bool(p_ok[pos]), bool(n_ok[pos])
                    )
                    print(
                        f"  {mode} row={idx}: "
                        f"hidden={h_ok[pos]} pooled={p_ok[pos]} normalized={n_ok[pos]}"
                    )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Merge diagnostic results.
    for rec in records:
        key = (rec["truncation"], rec["row_idx"])
        a = auto_batch_results[key]
        b = auto_single_results[key]
        c = fp32_batch_results[key]

        rec.update({
            "auto_batch_hidden_finite": a[0],
            "auto_batch_pooled_finite": a[1],
            "auto_batch_normalized_finite": a[2],
            "auto_single_hidden_finite": b[0],
            "auto_single_pooled_finite": b[1],
            "auto_single_normalized_finite": b[2],
            "fp32_batch_hidden_finite": c[0],
            "fp32_batch_pooled_finite": c[1],
            "fp32_batch_normalized_finite": c[2],
        })

    df = pd.DataFrame(records).sort_values(["truncation", "row_idx"])
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    print(f"\nReport: {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
