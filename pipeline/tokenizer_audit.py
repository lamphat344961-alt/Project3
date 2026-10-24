#!/usr/bin/env python
"""
PAAC-VN tokenizer audit for TN1 common-content.

Protocol:
- sample >= 2,000 Vietnamese benchmark descriptions deterministically (SEED=42)
- test character budgets {450, 700, 950, 1200, 1500} by default
- cut the same cleaned text to N raw characters at the nearest previous
  whitespace boundary
- THEN apply model preprocessing (PyVi if required; clustering prefix if any)
- count tokenizer output INCLUDING special tokens, with NO truncation
- choose the largest tested N for which P95 token length <= max_len for every
  audit_limiter model (the 256-token PhoBERT family)
- write data_v1/report/tokenizer_audit.csv and tokenizer_audit_summary.json

This script does not modify either dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, logging as hf_logging

SEED = 42
DEFAULT_LEVELS = [450, 700, 950, 1200, 1500]
DEFAULT_SAMPLE = 2000
TEXT_FIELD = "description_clean"

hf_logging.set_verbosity_error()


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
    if len(reps) != 8:
        raise ValueError(f"Expected 8 representations, found {len(reps)}")
    for rid, spec in reps.items():
        for key in (
            "hf_name", "revision", "max_len", "pooling",
            "needs_word_segmentation", "query_prefix", "doc_prefix",
            "clustering_prefix", "dim", "license", "year", "params",
            "training_objective", "model_family",
        ):
            if key not in spec or spec[key] is None:
                raise ValueError(f"{rid}: required models.yaml field {key!r} is missing/null")
    return cfg


def cut_chars_at_word_boundary(text: str, n_chars: int) -> str:
    """Cut to at most n_chars, avoiding a partial final whitespace-delimited word."""
    text = str(text)
    if len(text) <= n_chars:
        return text.strip()

    prefix = text[:n_chars]
    # If the next character is whitespace, the first n chars already end on a word.
    if text[n_chars].isspace():
        return prefix.rstrip()

    # Otherwise retreat to the last whitespace boundary.
    matches = list(re.finditer(r"\s+", prefix))
    if not matches:
        return prefix.rstrip()
    cut_at = matches[-1].start()
    # Defensive fallback; benchmark descriptions are >=50 chars, so this should not fire.
    if cut_at < 20:
        return prefix.rstrip()
    return prefix[:cut_at].rstrip()


def preprocess_for_model(texts: list[str], spec: dict, segmented_texts: list[str] | None) -> list[str]:
    if spec["needs_word_segmentation"]:
        if segmented_texts is None:
            segmented_texts = [ViTokenizer.tokenize(t) for t in texts]
        base = segmented_texts
    else:
        base = texts
    prefix = spec.get("clustering_prefix", "")
    return [prefix + t for t in base]


def tokenizer_lengths(tokenizer, texts: list[str], batch_size: int = 256) -> np.ndarray:
    lens: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lens.extend(len(ids) for ids in encoded["input_ids"])
    return np.asarray(lens, dtype=np.int32)


def tfidf_token_lengths(texts: list[str], spec: dict) -> np.ndarray:
    vcfg = spec["vectorizer"]
    vec = TfidfVectorizer(
        analyzer=vcfg["analyzer"],
        ngram_range=tuple(vcfg["ngram_range"]),
        min_df=vcfg["min_df"],
        max_df=vcfg["max_df"],
        sublinear_tf=vcfg["sublinear_tf"],
        norm=vcfg["norm"],
        lowercase=vcfg["lowercase"],
        token_pattern=vcfg["token_pattern"],
    )
    # Count input word tokens before n-gram construction.
    preprocessor = vec.build_preprocessor()
    tokenizer = vec.build_tokenizer()
    return np.asarray([len(tokenizer(preprocessor(t))) for t in texts], dtype=np.int32)


def summarize(lengths: np.ndarray) -> dict:
    return {
        "n": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data_v1/benchmark_v1.parquet"))
    ap.add_argument("--models", type=Path, default=Path("models.yaml"))
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    ap.add_argument("--text-field", default=TEXT_FIELD)
    ap.add_argument("--out-dir", type=Path, default=Path("../data_v1/report"))
    args = ap.parse_args()

    if args.sample < 2000:
        raise ValueError("Protocol requires --sample >= 2000.")
    levels = sorted(set(args.levels))
    if not levels or any(x <= 0 for x in levels):
        raise ValueError("--levels must contain positive character budgets.")

    cfg = load_models(args.models)
    reps = cfg["representations"]

    df = pd.read_parquet(args.data, columns=[c for c in ["appId", "genreId", args.text_field] if c])
    if args.text_field not in df.columns:
        raise KeyError(f"{args.text_field!r} not found in {args.data}")
    eligible = df[df[args.text_field].notna()].copy()
    eligible[args.text_field] = eligible[args.text_field].astype(str)
    eligible = eligible[eligible[args.text_field].str.len() >= 20]
    if len(eligible) < args.sample:
        raise ValueError(f"Only {len(eligible)} valid rows; requested sample={args.sample}")

    sample = eligible.sample(n=args.sample, random_state=SEED, replace=False)
    raw_texts = sample[args.text_field].tolist()

    # Load each HF tokenizer exactly at the frozen revision.
    tokenizers = {}
    for rid, spec in reps.items():
        if spec["backend"] != "transformers":
            continue
        print(f"[load tokenizer] {rid}: {spec['hf_name']} @ {spec['revision'][:8]}")
        tokenizers[rid] = AutoTokenizer.from_pretrained(
            spec["hf_name"],
            revision=spec["revision"],
            use_fast=bool(spec.get("tokenizer_use_fast", True)),
            trust_remote_code=False,
        )

    rows: list[dict] = []
    needs_any_seg = any(s.get("needs_word_segmentation") for s in reps.values())

    for n_chars in levels:
        print(f"\n[audit] N_CHARS={n_chars}")
        cut = [cut_chars_at_word_boundary(t, n_chars) for t in raw_texts]
        segmented = [ViTokenizer.tokenize(t) for t in cut] if needs_any_seg else None

        for rid, spec in reps.items():
            if rid == "tfidf_lsa":
                lengths = tfidf_token_lengths(cut, spec)
            else:
                prepared = preprocess_for_model(cut, spec, segmented)
                lengths = tokenizer_lengths(tokenizers[rid], prepared)

            s = summarize(lengths)
            max_len = int(spec["max_len"])
            over = None if max_len < 0 else float(np.mean(lengths > max_len))
            row = {
                "n_chars": n_chars,
                "representation": rid,
                "display_name": spec.get("display_name", rid),
                "max_len": max_len,
                "audit_limiter": bool(spec.get("audit_limiter", False)),
                **s,
                "over_native_max_rate": over,
            }
            rows.append(row)
            over_text = "NA" if over is None else f"{100*over:.2f}%"
            print(f"  {rid:28s} p95={s['p95']:7.1f} max={s['max']:5d} over={over_text}")

    result = pd.DataFrame(rows)

    limiter_ids = [rid for rid, s in reps.items() if s.get("audit_limiter") is True]
    if not limiter_ids:
        raise RuntimeError("No audit_limiter models are defined in models.yaml.")

    eligible_levels: list[int] = []
    for level in levels:
        part = result[(result["n_chars"] == level) & (result["representation"].isin(limiter_ids))]
        if len(part) != len(limiter_ids):
            raise RuntimeError(f"Missing limiter rows at level={level}")
        ok = True
        for _, r in part.iterrows():
            if float(r["p95"]) > int(r["max_len"]):
                ok = False
                break
        if ok:
            eligible_levels.append(level)

    if not eligible_levels:
        raise RuntimeError(
            "No tested N_CHARS satisfies P95 <= native max_len for every audit-limiter model. "
            "Do not guess N; inspect tokenizer_audit.csv."
        )
    selected = max(eligible_levels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "tokenizer_audit.csv"
    json_path = args.out_dir / "tokenizer_audit_summary.json"
    result.to_csv(csv_path, index=False)

    selected_rows = result[
        (result["n_chars"] == selected) & (result["representation"].isin(limiter_ids))
    ][["representation", "p95", "max_len", "over_native_max_rate"]].to_dict("records")

    summary = {
        "protocol": "PAAC-VN TN1 common-content tokenizer audit",
        "seed": SEED,
        "sample_size": args.sample,
        "levels_tested": levels,
        "text_field": args.text_field,
        "counts_include_special_tokens": True,
        "word_segmentation_before_tokenization": True,
        "prefix_applied_after_char_cut": True,
        "limiter_models": limiter_ids,
        "criterion": "largest tested N_CHARS such that P95 token length <= max_len for every audit_limiter model",
        "selected_n_chars": selected,
        "selected_limiter_stats": selected_rows,
        "benchmark_sha256": sha256(args.data),
        "models_yaml_sha256": sha256(args.models),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"SELECTED N_CHARS = {selected}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
