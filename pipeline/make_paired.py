# -*- coding: utf-8 -*-
"""
make_paired.py — Build TN2 paired English-source / Vietnamese-MT dataset.

Input:
    data_v1/benchmark_v1.parquet

Output:
    data_v1/paired_en.parquet
    data_v1/report/paired_*.csv / paired_summary.json / paired_run_meta.json

Protocol:
    1) Start only from benchmark_v1 (TN2 is a subset of TN1 benchmark representatives).
    2) Keep is_translated == True.
    3) Require both VI description_clean and source description_original_clean >= 50 chars.
    4) Confirm source language with two independent detectors: fastText lid.176 + py3langid.
       Primary TN2 keeps only consensus English (both == 'en'). Disagreements are reported, not silently kept.
    5) Near-deduplicate again on EN source text using the same MinHash-LSH settings as TN1.
    6) Representative rule: maxInstalls DESC, fallback installs_num, appId ASC.
    7) paired_en is therefore a strict subset of benchmark_v1 and retains paired EN/VI text for the same app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------ CONFIG ---------------------------------
MIN_DESC_LEN = 50
NEAR_DUP_JACCARD = 0.8
NUM_PERM = 128
SHINGLE = 3
SEED = 42
WORD_RE = re.compile(r"\w+")

REQUIRED_COLUMNS = {
    "appId",
    "genreId",
    "description_clean",
    "description_original_clean",
    "is_translated",
    "maxInstalls",
    "installs_num",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_len(s) -> int:
    return len(s) if isinstance(s, str) else 0


def shingles(text: str) -> set[str]:
    toks = WORD_RE.findall(text.lower())
    if len(toks) < SHINGLE:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def near_dup_groups(texts: list[str]) -> np.ndarray:
    """Deterministic connected components from MinHash-LSH candidate matches.

    Empty-shingle texts are singletons and are never inserted into the LSH.
    Input order must already be deterministic (we sort by appId before calling).
    """
    from datasketch import MinHash, MinHashLSH

    n = len(texts)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    lsh = MinHashLSH(threshold=NEAR_DUP_JACCARD, num_perm=NUM_PERM)
    mhs: list[MinHash | None] = [None] * n

    for i, t in enumerate(texts):
        sh = shingles(t)
        if not sh:
            continue
        m = MinHash(num_perm=NUM_PERM, seed=SEED)
        m.update_batch([x.encode("utf-8") for x in sorted(sh)])
        mhs[i] = m
        lsh.insert(str(i), m)
        if (i + 1) % 5000 == 0:
            print(f"   EN MinHash {i + 1}/{n}")

    for i, m in enumerate(mhs):
        if m is None:
            continue
        # Sort query hits so union order is deterministic.
        for j in sorted(map(int, lsh.query(m))):
            if j <= i or mhs[j] is None:
                continue
            # Same criterion as TN1 implementation: MinHash-estimated Jaccard.
            if m.jaccard(mhs[j]) >= NEAR_DUP_JACCARD:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    return np.array([find(i) for i in range(n)], dtype=np.int64)


def load_language_detectors(fasttext_model_path: Path):
    if not fasttext_model_path.exists():
        raise FileNotFoundError(
            f"Missing fastText language-id model: {fasttext_model_path}\n"
            "Place lid.176.ftz (or lid.176.bin) there or pass --fasttext-model PATH."
        )

    try:
        import fasttext
    except ImportError as e:
        raise RuntimeError("Missing dependency: pip install fasttext") from e

    try:
        import py3langid as py3langid
    except ImportError as e:
        raise RuntimeError("Missing dependency: pip install py3langid") from e

    ft = fasttext.load_model(str(fasttext_model_path))
    return ft, py3langid


def detect_languages(df: pd.DataFrame, fasttext_model_path: Path) -> pd.DataFrame:
    """Run both detectors on cleaned source listing text.

    No confidence threshold is used: protocol requires detector agreement, not score tuning.
    Scores are retained only for audit.
    """
    ft, py3langid = load_language_detectors(fasttext_model_path)

    ft_lang, ft_score, py_lang, py_score = [], [], [], []
    texts = df["description_original_clean"].tolist()
    n = len(texts)

    for i, text in enumerate(texts):
        one_line = re.sub(r"\s+", " ", text).strip()

        labels, probs = ft.predict(one_line, k=1)
        lab = labels[0].replace("__label__", "")
        ft_lang.append(lab)
        ft_score.append(float(probs[0]))

        plab, pscore = py3langid.classify(one_line)
        py_lang.append(str(plab))
        py_score.append(float(pscore))

        if (i + 1) % 2000 == 0:
            print(f"   language ID {i + 1}/{n}")

    out = df.copy()
    out["fasttext_lang"] = ft_lang
    out["fasttext_score"] = ft_score
    out["py3langid_lang"] = py_lang
    out["py3langid_score"] = py_score
    out["source_lang_agree"] = out["fasttext_lang"] == out["py3langid_lang"]
    out["source_lang_en_consensus"] = (
        (out["fasttext_lang"] == "en") & (out["py3langid_lang"] == "en")
    )
    return out


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"benchmark_v1 missing required columns: {sorted(missing)}")
    if df["appId"].isna().any() or df["appId"].duplicated().any():
        raise ValueError("benchmark_v1 appId must be non-null and unique")
    if df["genreId"].nunique(dropna=True) != 49:
        raise ValueError(f"Expected 49 genreId in benchmark_v1, got {df['genreId'].nunique(dropna=True)}")

    # is_translated is expected to be bool/null. Do not parse strings implicitly.
    bad = df["is_translated"].dropna().map(lambda x: not isinstance(x, (bool, np.bool_)))
    if bad.any():
        vals = df.loc[bad[bad].index, "is_translated"].head(10).tolist()
        raise TypeError(f"is_translated schema drift; expected bool/null, examples={vals}")


def build_paired(
    benchmark_path: Path,
    out_dir: Path,
    fasttext_model_path: Path,
) -> pd.DataFrame:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] load benchmark: {benchmark_path}")
    df = pd.read_parquet(benchmark_path)
    validate_input(df)
    df = df.sort_values("appId", kind="stable").reset_index(drop=True)
    n0 = len(df)
    print(f"    {n0:,} benchmark apps")

    # TN2 requires actual Google-translated VI listing, not provenance label aliases.
    translated = df[df["is_translated"] == True].copy()  # noqa: E712
    n_translated = len(translated)
    print(f"[2] is_translated=True: {n_translated:,}")

    translated["vi_len"] = translated["description_clean"].map(text_len)
    translated["source_len"] = translated["description_original_clean"].map(text_len)
    valid_text = translated[
        (translated["vi_len"] >= MIN_DESC_LEN)
        & (translated["source_len"] >= MIN_DESC_LEN)
    ].copy()
    n_valid_text = len(valid_text)
    print(f"[3] both EN-source candidate and VI text >= {MIN_DESC_LEN}: {n_valid_text:,}")

    print("[4] source language ID: fastText + py3langid")
    lang_df = detect_languages(valid_text, fasttext_model_path)

    # Full predictions for reproducibility/audit.
    pred_cols = [
        "appId", "genreId", "fasttext_lang", "fasttext_score",
        "py3langid_lang", "py3langid_score", "source_lang_agree",
        "source_lang_en_consensus", "source_len", "vi_len",
    ]
    lang_df[pred_cols].to_csv(report_dir / "source_lang_predictions.csv", index=False)

    disagreements = lang_df[~lang_df["source_lang_agree"]]
    disagreements[pred_cols + ["description_original_clean"]].to_csv(
        report_dir / "source_lang_disagreements.csv", index=False
    )

    lang_pair_counts = (
        lang_df.groupby(["fasttext_lang", "py3langid_lang"], dropna=False)
        .size()
        .rename("count")
        .sort_values(ascending=False)
        .reset_index()
    )
    lang_pair_counts.to_csv(report_dir / "source_lang_pair_counts.csv", index=False)

    en = lang_df[lang_df["source_lang_en_consensus"]].copy()
    n_en = len(en)
    print(f"    consensus English: {n_en:,}")
    print(f"    detector disagreements: {len(disagreements):,}")

    if n_en == 0:
        raise RuntimeError("No consensus-English source descriptions found")

    # EN-side near-dedup: input order is deterministic.
    print("[5] near-dedup English source text")
    en = en.sort_values("appId", kind="stable").reset_index(drop=True)
    roots = near_dup_groups(en["description_original_clean"].tolist())
    en["_en_root"] = roots
    en["en_dup_group_size"] = en.groupby("_en_root")["appId"].transform("size")

    dup_roots = sorted(en.loc[en["en_dup_group_size"] > 1, "_en_root"].unique())
    root_to_gid = {r: f"edg{k:05d}" for k, r in enumerate(dup_roots)}
    en["en_dup_group_id"] = en["_en_root"].map(root_to_gid)

    # Deterministic representative: maxInstalls, fallback installs_num, appId ASC.
    en["_rep_installs"] = en["maxInstalls"].where(en["maxInstalls"].notna(), en["installs_num"])
    order = en.sort_values(
        ["_rep_installs", "appId"], ascending=[False, True], na_position="last", kind="stable"
    )
    rep_idx = order.drop_duplicates("_en_root", keep="first").index
    en["_keep_en_rep"] = False
    en.loc[rep_idx, "_keep_en_rep"] = True

    paired = en[en["_keep_en_rep"]].copy().sort_values("appId", kind="stable").reset_index(drop=True)
    n_paired = len(paired)
    print(f"[6] paired_en after EN dedup: {n_paired:,}")

    # Reports for EN duplicate groups.
    dup = en[en["en_dup_group_size"] > 1]
    if len(dup):
        dup_report = (
            dup.groupby("en_dup_group_id")
            .agg(
                size=("appId", "size"),
                n_genres=("genreId", "nunique"),
                representative=("_keep_en_rep", "sum"),
                example_appId=("appId", "first"),
            )
            .sort_values("size", ascending=False)
        )
    else:
        dup_report = pd.DataFrame(columns=["size", "n_genres", "representative", "example_appId"])
    dup_report.to_csv(report_dir / "paired_en_dup_groups.csv")

    # Funnel.
    funnel = pd.DataFrame(
        {
            "step": [
                "0. benchmark_v1",
                "1. is_translated = true",
                f"2. both texts >= {MIN_DESC_LEN} chars",
                "3. source language consensus = en",
                "4. EN near-deduplicated paired set",
            ],
            "n_apps": [n0, n_translated, n_valid_text, n_en, n_paired],
        }
    )
    funnel["removed"] = (-funnel["n_apps"].diff()).fillna(0).astype(int)
    funnel.to_csv(report_dir / "paired_funnel.csv", index=False)

    genre_counts = paired["genreId"].value_counts().sort_index()
    genre_counts.rename("count").to_csv(report_dir / "paired_genre_distribution.csv")

    # Hard validation: TN2 still needs all 49 reference labels.
    if paired["genreId"].nunique() != 49:
        missing_genres = sorted(set(df["genreId"].dropna().unique()) - set(paired["genreId"].dropna().unique()))
        raise RuntimeError(
            f"LEVEL C: paired_en has {paired['genreId'].nunique()} genres, expected 49. "
            f"Missing={missing_genres}"
        )
    if paired["appId"].duplicated().any():
        raise AssertionError("paired_en appId is not unique")
    if not paired["is_translated"].eq(True).all():  # noqa: E712
        raise AssertionError("paired_en contains is_translated != True")
    if (paired["description_clean"].map(text_len) < MIN_DESC_LEN).any():
        raise AssertionError("paired_en contains short VI description")
    if (paired["description_original_clean"].map(text_len) < MIN_DESC_LEN).any():
        raise AssertionError("paired_en contains short EN source description")
    if not ((paired["fasttext_lang"] == "en") & (paired["py3langid_lang"] == "en")).all():
        raise AssertionError("paired_en contains non-consensus-English source")

    # Keep canonical model inputs + necessary metadata. Raw source is retained for audit.
    keep_cols = [
        "appId", "genreId", "genre",
        "description_clean",                    # VI machine-translated branch
        "description_original",                 # raw-ish source listing retained from MongoDB export
        "description_original_clean",           # EN source branch model input
        "is_translated",
        "fasttext_lang", "fasttext_score", "py3langid_lang", "py3langid_score",
        "source_len", "vi_len",
        "maxInstalls", "installs_num",
        "en_dup_group_id", "en_dup_group_size",
        "dup_group_id", "dup_group_size",
    ]
    keep_cols = [c for c in keep_cols if c in paired.columns]

    out_path = out_dir / "paired_en.parquet"
    paired[keep_cols].to_parquet(out_path, index=False)

    summary = {
        "config": {
            "MIN_DESC_LEN": MIN_DESC_LEN,
            "NEAR_DUP_JACCARD": NEAR_DUP_JACCARD,
            "NUM_PERM": NUM_PERM,
            "SHINGLE": SHINGLE,
            "SEED": SEED,
            "language_policy": "fastText=en AND py3langid=en",
            "dedup_scope": "consensus-English candidates already in benchmark_v1",
            "representative_rule": "maxInstalls DESC, fallback installs_num, appId ASC",
        },
        "funnel": funnel.set_index("step")["n_apps"].to_dict(),
        "detector_agreement": {
            "n_checked": int(len(lang_df)),
            "n_agree": int(lang_df["source_lang_agree"].sum()),
            "n_disagree": int((~lang_df["source_lang_agree"]).sum()),
            "agreement_rate": round(float(lang_df["source_lang_agree"].mean()), 6),
            "n_consensus_en": int(n_en),
        },
        "en_dup_groups": int(len(dup_report)),
        "apps_in_en_dup_groups": int(len(dup)),
        "paired": {
            "n_apps": int(n_paired),
            "n_labels": int(paired["genreId"].nunique()),
            "min_genre_n": int(genre_counts.min()),
            "max_genre_n": int(genre_counts.max()),
        },
    }
    (report_dir / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Reproducibility metadata.
    import datasketch

    try:
        import fasttext
        fasttext_version = getattr(fasttext, "__version__", "unknown")
    except Exception:
        fasttext_version = "unknown"
    try:
        import py3langid
        py3langid_version = getattr(py3langid, "__version__", "unknown")
    except Exception:
        py3langid_version = "unknown"

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "datasketch": getattr(datasketch, "__version__", "unknown"),
        "fasttext": fasttext_version,
        "py3langid": py3langid_version,
        "input": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "rows": int(n0),
        },
        "fasttext_model": {
            "path": str(fasttext_model_path),
            "sha256": sha256_file(fasttext_model_path),
        },
        "output": {
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "rows": int(n_paired),
        },
    }
    (report_dir / "paired_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path.resolve()}")
    return paired


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, default=Path("data_v1/benchmark_v1.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("data_v1"))
    p.add_argument("--fasttext-model", type=Path, default=Path("models/lid.176.ftz"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_paired(args.benchmark, args.out_dir, args.fasttext_model)
