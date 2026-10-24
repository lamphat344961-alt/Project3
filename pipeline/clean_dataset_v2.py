# -*- coding: utf-8 -*-
"""
clean_dataset_v2.py — rebuild PAAC-VN data_v1 from raw MongoDB.

Source of truth:
  MongoDB app_store_research.google_play_apps

Pipeline:
  1. Load raw documents (read-only)
  2. Validate required schema
  3. Deterministic ordering by appId
  4. Clean text (raw columns are preserved)
  5. Derive Vietnamese-language flags and provenance metadata
  6. Build benchmark eligibility mask
  7. Near-deduplicate ONLY eligible benchmark candidates
  8. Select one representative per duplicate component
  9. Export parquet + deterministic reports + run metadata

Important semantics:
  - is_translated == False: Vietnamese listing is original (not Google-translated)
  - is_translated == True : Vietnamese description is translated from another language
  - description_original is source-listing text; it is NOT assumed to be English
"""

from __future__ import annotations

import hashlib
import html
import json
import platform
import re
import shutil
import sys
import unicodedata as ud
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
MONGO_URI = "mongodb://localhost:27017"
DB, COLL = "app_store_research", "google_play_apps"
OUT = Path("data_v1")
ARCHIVE_ROOT = Path("archive")

VI_MIN = 0.10
MIXED_MIN = 0.05
MIN_DESC_LEN = 50
SUMMARY_LIMIT = 80
DESC_LIMIT = 4000
NEAR_DUP_JACCARD = 0.8
NUM_PERM = 128
SHINGLE = 3
SEED = 42
EXPECTED_GENRE_COUNT = 49

REQUIRED_SOURCE_FIELDS = {
    "appId", "genreId", "description", "is_translated", "description_original"
}

FIELDS = [
    "appId", "title", "summary", "description", "genre", "genreId",
    "developerId", "installs", "minInstalls", "maxInstalls", "ratings", "score",
    "contentRating", "price", "IAPRange", "platform", "free", "offersIAP",
    "adSupported", "currency", "reviews", "released", "updated", "developer",
    "developerLegalAddress",
    # recrawl_translation.mjs
    "is_translated", "description_original", "description_translated",
    "description_changed", "recrawl_error", "recrawled_at",
]
TEXT = ["title", "summary", "description"]

# ============================== REGEX ==============================
VI_STRICT = set("ăơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịĩọỏốồổỗộớờởỡợụủũứừửữựỳỵỷỹ")
INVISIBLE_RE = re.compile("[\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060\ufeff\u00ad]")
CTRL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TAG_RE = re.compile(r"</?[a-zA-Z][^<>]{0,200}>")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
HSPACE_RE = re.compile(r"[^\S\n]+")
WORD_RE = re.compile(r"\w+")

# ============================== IO HELPERS ==============================
def archive_existing_output() -> Path | None:
    """Move current data_v1 aside before a rebuild. Never delete it."""
    if not OUT.exists() or not any(OUT.iterdir()):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_ROOT / f"data_v1_pre_reset_{stamp}"
    shutil.move(str(OUT), str(dest))
    print(f"[archive] moved previous {OUT} -> {dest}")
    return dest


def load_mongo() -> pd.DataFrame:
    from pymongo import MongoClient

    client = MongoClient(MONGO_URI)
    col = client[DB][COLL]
    proj = {"_id": 0, **{f: 1 for f in FIELDS}}
    docs = list(col.find({}, proj))
    return pd.DataFrame(docs)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# ============================== TEXT ==============================
def clean_text(s, multiline: bool) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = html.unescape(str(s))
    s = ud.normalize("NFC", s)
    s = INVISIBLE_RE.sub("", s)
    s = CTRL_RE.sub(" ", s)
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if multiline:
        lines = (HSPACE_RE.sub(" ", ln).strip() for ln in s.split("\n"))
        return "\n".join(ln for ln in lines if ln)
    return re.sub(r"\s+", " ", s).strip()


def vi_density(s: str) -> float:
    s = s.lower()
    letters = sum(c.isalpha() for c in s)
    return sum(c in VI_STRICT for c in s) / letters if letters else 0.0


def parse_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    d = re.sub(r"\D", "", str(x))
    return float(d) if d else np.nan


def shingles(text: str) -> set[str]:
    toks = WORD_RE.findall(text.lower())
    if len(toks) < SHINGLE:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}

# ============================== VALIDATION ==============================
def validate_source_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(df.columns))
    if missing:
        raise RuntimeError(f"MongoDB projection missing required fields: {missing}")

    if df["appId"].isna().any():
        raise RuntimeError("appId contains null values")
    if df["appId"].astype(str).duplicated().any():
        dups = df.loc[df["appId"].astype(str).duplicated(False), "appId"].head(10).tolist()
        raise RuntimeError(f"Duplicate appId in raw collection; examples: {dups}")

    # Schema guard only: current MongoDB schema is bool/null.
    bad = []
    for idx, v in df["is_translated"].items():
        if pd.isna(v):
            continue
        if not isinstance(v, (bool, np.bool_)):
            bad.append((idx, repr(v), type(v).__name__))
            if len(bad) >= 10:
                break
    if bad:
        raise TypeError(
            "is_translated must be bool/null; schema drift detected. "
            f"Examples: {bad}"
        )


def add_missing_optional_fields(df: pd.DataFrame) -> pd.DataFrame:
    optional_missing = [f for f in FIELDS if f not in df.columns and f not in REQUIRED_SOURCE_FIELDS]
    if optional_missing:
        print(f"[warning] optional fields missing and filled with null: {optional_missing}")
    for f in optional_missing:
        df[f] = np.nan
    return df

# ============================== DEDUP ==============================
def near_dup_groups(texts: list[str]) -> np.ndarray:
    """Union-find roots for texts. Empty-shingle texts stay singleton and never enter LSH."""
    from datasketch import MinHash, MinHashLSH

    n = len(texts)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lsh = MinHashLSH(threshold=NEAR_DUP_JACCARD, num_perm=NUM_PERM)
    mhs: list[object | None] = [None] * n

    for i, t in enumerate(texts):
        sh = shingles(t)
        if not sh:
            continue  # singleton; do NOT insert an empty MinHash
        m = MinHash(num_perm=NUM_PERM, seed=SEED)
        m.update_batch([x.encode("utf8") for x in sorted(sh)])
        mhs[i] = m
        lsh.insert(str(i), m)
        if (i + 1) % 5000 == 0:
            print(f"   MinHash {i + 1}/{n}")

    for i, m in enumerate(mhs):
        if m is None:
            continue
        for j in map(int, lsh.query(m)):
            if j <= i or mhs[j] is None:
                continue
            if m.jaccard(mhs[j]) >= NEAR_DUP_JACCARD:
                ri, rj = find(i), find(j)
                if ri != rj:
                    # deterministic union: lower root wins
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj

    return np.array([find(i) for i in range(n)])


def assign_stable_dup_ids(df: pd.DataFrame, eligible_index: pd.Index, roots: np.ndarray) -> None:
    """Assign stable duplicate metadata only to eligible benchmark candidates."""
    df["dup_group_size"] = 1
    df["dup_group_id"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["dup_genre_conflict"] = False

    work = df.loc[eligible_index, ["appId", "genreId"]].copy()
    work["_root_local"] = roots

    # A component's stable key is the lexicographically smallest appId in it.
    comp_key = work.groupby("_root_local")["appId"].transform(lambda s: min(map(str, s)))
    work["_comp_key"] = comp_key
    size_map = work.groupby("_comp_key").size()
    conflict_map = work.groupby("_comp_key")["genreId"].nunique().gt(1)

    duplicate_keys = sorted(size_map[size_map > 1].index.tolist())
    gid_map = {key: f"dg{k:05d}" for k, key in enumerate(duplicate_keys)}

    df.loc[eligible_index, "dup_group_size"] = work["_comp_key"].map(size_map).astype(int).values
    df.loc[eligible_index, "dup_group_id"] = work["_comp_key"].map(gid_map).values
    df.loc[eligible_index, "dup_genre_conflict"] = work["_comp_key"].map(conflict_map).fillna(False).values
    df.loc[eligible_index, "_dup_component_key"] = work["_comp_key"].values

# ============================== PIPELINE ==============================
def run(df: pd.DataFrame) -> pd.DataFrame:
    validate_source_schema(df)
    df = add_missing_optional_fields(df.copy())

    # Deterministic global row order.
    df["appId"] = df["appId"].astype(str)
    df = df.sort_values("appId", kind="stable").reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rep = OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)

    for c in TEXT + ["genre", "genreId", "developerId"]:
        df[c] = df[c].astype(object)

    n0 = len(df)
    print(f"[1] loaded {n0:,} apps")

    # ---- 2. Clean text; preserve raw source columns unchanged.
    print("[2] cleaning text")
    df["title_clean"] = df["title"].map(lambda s: clean_text(s, False))
    df["summary_clean"] = df["summary"].map(lambda s: clean_text(s, False))
    df["description_clean"] = df["description"].map(lambda s: clean_text(s, True))
    df["description_original_clean"] = df["description_original"].map(lambda s: clean_text(s, True))

    for c in TEXT:
        df[f"{c}_len"] = df[f"{c}_clean"].str.len()
        df[f"{c}_words"] = df[f"{c}_clean"].map(lambda s: len(WORD_RE.findall(s)))
    df["description_original_len"] = df["description_original_clean"].str.len()

    # ---- 3. Language flags for displayed Vietnamese description.
    print("[3] language flags")
    df["has_vi"] = df["description_clean"].map(lambda s: any(c in VI_STRICT for c in s.lower()))
    df["vi_density"] = df["description_clean"].map(vi_density)
    df["lang_flag"] = np.select(
        [df["vi_density"] >= VI_MIN, df["vi_density"] >= MIXED_MIN],
        ["vi", "mixed"],
        default="other",
    )

    # ---- 4. Quality / provenance metadata.
    print("[4] quality and provenance flags")
    df["short_desc"] = df["description_len"] < MIN_DESC_LEN

    raw_sum_len = df["summary"].map(lambda s: len(html.unescape(str(s))) if isinstance(s, str) else 0)
    raw_desc_len = df["description"].map(lambda s: len(str(s)) if isinstance(s, str) else 0)
    df["likely_translated"] = (raw_sum_len > SUMMARY_LIMIT) | (raw_desc_len > DESC_LIMIT)

    recrawl_ok = df["is_translated"].notna()
    if "recrawl_error" in df.columns:
        recrawl_ok &= df["recrawl_error"].isna()

    # Semantics confirmed from raw MongoDB schema:
    # False = original Vietnamese listing; True = Google-translated from another language.
    df["text_source"] = "unknown"
    df.loc[recrawl_ok & (df["is_translated"] == False), "text_source"] = "human"  # noqa: E712
    df.loc[recrawl_ok & (df["is_translated"] == True), "text_source"] = "machine"  # noqa: E712

    # Representative criterion: maxInstalls first; fallback only if absent/null.
    df["installs_num"] = (
        df["maxInstalls"].map(parse_int)
        .fillna(df["minInstalls"].map(parse_int))
        .fillna(df["installs"].map(parse_int))
    )
    df["ratings_num"] = df["ratings"].map(parse_int)

    # ---- 5. Benchmark eligibility BEFORE dedup.
    eligible = df["has_vi"] & (df["lang_flag"] == "vi") & ~df["short_desc"]
    df["_eligible"] = eligible
    eligible_idx = df.index[eligible]
    print(f"[5] eligible before dedup: {len(eligible_idx):,}")

    # ---- 6. Near-dedup ONLY eligible candidates.
    print("[6] near-dedup eligible benchmark candidates")
    eligible_texts = df.loc[eligible_idx, "description_clean"].tolist()
    roots = near_dup_groups(eligible_texts)
    assign_stable_dup_ids(df, eligible_idx, roots)

    # Select representative per component using installs DESC, appId ASC.
    work = df.loc[eligible_idx, ["appId", "installs_num", "_dup_component_key"]].copy()
    work["_rep_key"] = work["_dup_component_key"].fillna(work["appId"])
    order = work.sort_values(
        ["installs_num", "appId"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    )
    reps = order.drop_duplicates("_rep_key", keep="first").index
    df["in_benchmark"] = False
    df.loc[reps, "in_benchmark"] = True

    # ---- 7. Funnel.
    steps = [
        ("0. collected", pd.Series(True, index=df.index)),
        ("1. has Vietnamese-specific char", df["has_vi"]),
        ("2. lang_flag = vi", df["has_vi"] & (df["lang_flag"] == "vi")),
        (f"3. description >= {MIN_DESC_LEN} chars", eligible),
        ("4. near-deduplicated benchmark", df["in_benchmark"]),
    ]
    funnel = pd.DataFrame({"step": [s for s, _ in steps], "n_apps": [int(m.sum()) for _, m in steps]})
    funnel["removed"] = (-funnel["n_apps"].diff()).fillna(0).astype(int)
    funnel.to_csv(rep / "funnel.csv", index=False)

    gfun = pd.DataFrame({s: df[m].groupby("genreId").size() for s, m in steps}).fillna(0).astype(int)
    gfun["kept_pct"] = (100 * gfun.iloc[:, -1] / gfun.iloc[:, 0].replace(0, np.nan)).round(1)
    gfun.sort_values(gfun.columns[0], ascending=False).to_csv(rep / "genre_funnel.csv")

    # ---- 8. Benchmark stats.
    bm = df[df["in_benchmark"]].copy()
    counts = bm["genreId"].value_counts()
    if counts.size != EXPECTED_GENRE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_GENRE_COUNT} genreId values in benchmark, got {counts.size}")

    p = counts / counts.sum()
    imbalance = {
        "n_apps": int(len(bm)),
        "n_labels": int(counts.size),
        "max_label": counts.index[0],
        "max": int(counts.iloc[0]),
        "min_label": counts.index[-1],
        "min": int(counts.iloc[-1]),
        "imbalance_ratio": round(float(counts.iloc[0] / counts.iloc[-1]), 2),
        "normalized_entropy": round(float(-(p * np.log(p)).sum() / np.log(len(p))), 4),
        "cv": round(float(counts.std() / counts.mean()), 3),
    }
    counts.rename("count").to_csv(rep / "genre_distribution_benchmark.csv")

    ct = pd.crosstab(bm["genreId"], bm["text_source"], normalize="index").round(3)
    if "machine" in ct.columns:
        ct = ct.sort_values("machine", ascending=False)
    ct.to_csv(rep / "text_source_by_genre.csv")

    lens = bm[[f"{c}_{k}" for c in TEXT for k in ("len", "words")]].describe(
        percentiles=[.25, .5, .75, .95]
    ).T.round(1)
    lens.to_csv(rep / "length_stats_benchmark.csv")

    # ---- 9. Language density diagnostics.
    bins = [0, .005, .01, .02, .03, .04, .05, .06, .08, .10, .15, 1.0]
    hv = df[df["has_vi"]]
    pd.cut(hv["vi_density"], bins, include_lowest=True).value_counts().sort_index().rename("count").to_csv(
        rep / "density_hist.csv"
    )
    with (rep / "density_samples.txt").open("w", encoding="utf-8") as f:
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = hv[(hv["vi_density"] > lo) & (hv["vi_density"] <= hi)]
            f.write(f"\n===== ({lo}, {hi}]  n={len(s)} =====\n")
            if len(s):
                for _, r in s.sample(min(6, len(s)), random_state=SEED).iterrows():
                    preview = r.description_clean[:200].replace(chr(10), " ")
                    f.write(f"{r.vi_density:.3f} | {r.genreId} | {preview}\n")

    # ---- 10. Duplicate reports (eligible candidates only).
    dup = df.loc[eligible_idx]
    dup = dup[dup["dup_group_size"] > 1]
    if len(dup):
        grp = dup.groupby("dup_group_id", dropna=True).agg(
            size=("appId", "size"),
            developers=("developerId", lambda x: ", ".join(map(str, x.value_counts().index[:3]))),
            genres=("genreId", lambda x: ", ".join(map(str, x.value_counts().index[:3]))),
            conflict=("dup_genre_conflict", "first"),
            example_title=("title_clean", "first"),
        ).sort_values(["size", "example_title"], ascending=[False, True])
    else:
        grp = pd.DataFrame(columns=["size", "developers", "genres", "conflict", "example_title"])
    grp.to_csv(rep / "dup_groups.csv")
    grp[grp.get("conflict", pd.Series(dtype=bool)) == True].to_csv(rep / "dup_groups_genre_conflict.csv")  # noqa: E712

    summary = {
        "config": {
            "VI_MIN": VI_MIN,
            "MIXED_MIN": MIXED_MIN,
            "MIN_DESC_LEN": MIN_DESC_LEN,
            "NEAR_DUP_JACCARD": NEAR_DUP_JACCARD,
            "NUM_PERM": NUM_PERM,
            "SHINGLE": SHINGLE,
            "SEED": SEED,
            "dedup_scope": "eligible_benchmark_candidates_only",
            "representative_rule": "maxInstalls DESC, fallback minInstalls/installs, appId ASC",
        },
        "funnel": funnel.set_index("step")["n_apps"].to_dict(),
        "lang_flag": df["lang_flag"].value_counts().to_dict(),
        "text_source_all": df["text_source"].value_counts().to_dict(),
        "text_source_benchmark": bm["text_source"].value_counts().to_dict(),
        "description_changed_since_crawl": int(df["description_changed"].fillna(False).astype(bool).sum()),
        "heuristic_vs_flag": pd.crosstab(df["likely_translated"], df["text_source"]).to_dict(),
        "dup_groups": int(len(grp)),
        "apps_in_dup_groups": int(grp["size"].sum()) if len(grp) else 0,
        "dup_groups_genre_conflict": int(grp["conflict"].sum()) if len(grp) else 0,
        "benchmark_imbalance": imbalance,
    }
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 11. Export data.
    drop_internal = ["_eligible", "_dup_component_key"]
    full_out = df.drop(columns=[c for c in drop_internal if c in df.columns])
    full_path = OUT / "dataset_v1_full.parquet"
    full_out.to_parquet(full_path, index=False)

    benchmark_keep = [
        "appId", "genreId", "genre",
        "title_clean", "summary_clean", "description_clean",
        "description_original", "description_original_clean", "is_translated",
        "developerId", "dup_group_id", "dup_group_size", "text_source",
        "description_changed", "vi_density", "lang_flag",
        "installs_num", "maxInstalls", "ratings_num", "score",
        "contentRating", "price", "free", "offersIAP", "adSupported",
        "released", "updated", "developerLegalAddress",
    ]
    benchmark_keep = [c for c in benchmark_keep if c in full_out.columns]
    benchmark_path = OUT / "benchmark_v1.parquet"
    full_out.loc[full_out["in_benchmark"], benchmark_keep].reset_index(drop=True).to_parquet(
        benchmark_path, index=False
    )

    # ---- 12. Hard postconditions.
    out_bm = pd.read_parquet(benchmark_path)
    assert out_bm["appId"].is_unique, "benchmark appId must be unique"
    assert len(out_bm) == int(df["in_benchmark"].sum())
    assert (out_bm["vi_density"] >= VI_MIN).all()
    assert set(out_bm["text_source"].dropna().unique()) <= {"human", "machine", "unknown"}
    assert out_bm["genreId"].nunique() == EXPECTED_GENRE_COUNT

    # ---- 13. Run metadata after outputs exist.
    output_files = sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "run_meta.json"])
    try:
        import pymongo
        pymongo_version = pymongo.version
    except Exception:
        pymongo_version = None
    try:
        import datasketch
        datasketch_version = getattr(datasketch, "__version__", None)
    except Exception:
        datasketch_version = None
    try:
        import pyarrow
        pyarrow_version = pyarrow.__version__
    except Exception:
        pyarrow_version = None

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pymongo": pymongo_version,
            "datasketch": datasketch_version,
            "pyarrow": pyarrow_version,
        },
        "mongo": {"db": DB, "collection": COLL, "uri": MONGO_URI},
        "config": summary["config"],
        "row_counts": {
            "raw": int(n0),
            "eligible_before_dedup": int(eligible.sum()),
            "benchmark": int(len(out_bm)),
        },
        "outputs": {
            str(p.relative_to(OUT)): {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in output_files
        },
    }
    (rep / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved rebuild to {OUT.resolve()}")
    return full_out


if __name__ == "__main__":
    archive_existing_output()
    run(load_mongo())
