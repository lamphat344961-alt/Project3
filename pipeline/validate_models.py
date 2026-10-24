#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path
import yaml

REQUIRED_FIELDS = (
    "hf_name",
    "revision",
    "max_len",
    "pooling",
    "needs_word_segmentation",
    "query_prefix",
    "doc_prefix",
    "dim",
    "license",
    "year",
    "params",
    "training_objective",
    "model_family",
)

EXPECTED_REPRS = {
    "tfidf_lsa",
    "phobert_base_v2",
    "videberta_base",
    "sup_simcse_phobert",
    "bkai_vietnamese_biencoder",
    "multilingual_e5_large",
    "bge_m3",
    "qwen3_embedding_0_6b",
}

SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(path: Path) -> None:
    cfg = load_yaml(path)
    reps = cfg.get("representations")
    if not isinstance(reps, dict):
        raise ValueError("models.yaml must contain mapping 'representations'.")

    if set(reps) != EXPECTED_REPRS:
        missing = sorted(EXPECTED_REPRS - set(reps))
        extra = sorted(set(reps) - EXPECTED_REPRS)
        raise ValueError(f"Representation set mismatch. missing={missing}, extra={extra}")

    errors = []
    for rid, spec in reps.items():
        for key in REQUIRED_FIELDS:
            if key not in spec:
                errors.append(f"{rid}: missing required field {key!r}")
            elif spec[key] is None:
                errors.append(f"{rid}: required field {key!r} is null")

        if spec.get("backend") == "transformers":
            if not SHA40.fullmatch(str(spec.get("revision", ""))):
                errors.append(f"{rid}: revision must be a frozen 40-char git SHA")
            if not isinstance(spec.get("max_len"), int) or spec["max_len"] <= 0:
                errors.append(f"{rid}: max_len must be a positive int")
            if not isinstance(spec.get("dim"), int) or spec["dim"] <= 0:
                errors.append(f"{rid}: dim must be a positive int")
        elif rid == "tfidf_lsa":
            if spec.get("max_len") != -1:
                errors.append("tfidf_lsa: max_len must be -1 sentinel (unbounded native input)")
        else:
            errors.append(f"{rid}: unsupported backend={spec.get('backend')!r}")

        if not isinstance(spec.get("needs_word_segmentation"), bool):
            errors.append(f"{rid}: needs_word_segmentation must be bool")

        if "clustering_prefix" not in spec or spec["clustering_prefix"] is None:
            errors.append(f"{rid}: clustering_prefix must be present and non-null")

    # Protocol-specific hard checks: fail loudly if the frozen recipe drifts.
    hard = {
        "phobert_base_v2": (256, 768, "mean_attention_mask", True),
        "videberta_base": (512, 768, "mean_attention_mask", True),
        "sup_simcse_phobert": (256, 768, "pooler_output", True),
        "bkai_vietnamese_biencoder": (256, 768, "mean_attention_mask", True),
        "multilingual_e5_large": (512, 1024, "mean_attention_mask", False),
        "bge_m3": (8192, 1024, "cls", False),
        "qwen3_embedding_0_6b": (32768, 1024, "last_token", False),
    }
    for rid, (max_len, dim, pooling, seg) in hard.items():
        spec = reps[rid]
        got = (spec["max_len"], spec["dim"], spec["pooling"], spec["needs_word_segmentation"])
        want = (max_len, dim, pooling, seg)
        if got != want:
            errors.append(f"{rid}: frozen tuple drifted: got={got}, expected={want}")

    if reps["multilingual_e5_large"]["clustering_prefix"] != "query: ":
        errors.append("multilingual_e5_large: clustering_prefix must be 'query: ' for this clustering task")
    if reps["bge_m3"].get("dense_only") is not True:
        errors.append("bge_m3: dense_only must be true")
    if reps["qwen3_embedding_0_6b"]["clustering_prefix"] != "":
        errors.append("qwen3_embedding_0_6b: documents must not receive a query instruction")

    if errors:
        raise ValueError("models.yaml validation failed:\n- " + "\n- ".join(errors))

    print(f"OK: {path} -- 8 representations locked, all mandatory fields non-null.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("models.yaml"))
    args = ap.parse_args()
    validate(args.models)


if __name__ == "__main__":
    main()
