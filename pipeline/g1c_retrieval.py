#!/usr/bin/env python3
"""Exact, tie-aware, blockwise cross-lingual retrieval for PAAC-VN G1C.

This module contains no model or embedding inference.  It validates paired
embedding inputs, normalizes a float64 copy, and evaluates both retrieval
directions against the complete target catalogue.  Ties are defined by exact
equality of the computed float64 cosine scores.

Real retrieval is fail-closed until a separate execution receipt authorizes a
specific set of input hashes and output namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_ID = "V4-G1C-RET-001"
REAL_AUTHORIZATION_STATUS = "AUTHORIZED_FOR_REAL_RETRIEVAL"
PER_QUERY_FILENAME = "per_query.jsonl"
AGGREGATE_FILENAME = "aggregate.json"
MANIFEST_FILENAME = "provenance_manifest.json"


class G1CRetrievalError(RuntimeError):
    """Fail-closed error with a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}:{detail}")


@dataclass(frozen=True)
class QueryRank:
    query_index: int
    query_app_id: str
    positive_target_app_id: str
    direction: str
    positive_score: float
    rank_min: int
    rank_max: int
    rank_average: float
    tie_group_size: int
    expected_recall_at_1: float
    expected_recall_at_10: float
    expected_reciprocal_rank: float
    target_count: int


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(app_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(app_ids).encode("utf-8")).hexdigest()


def _error(code: str, detail: str = "") -> G1CRetrievalError:
    return G1CRetrievalError(code, detail)


def _validate_ids(ids: Sequence[Any], label: str) -> list[str]:
    values = [str(value) for value in ids]
    if not values or any(not value for value in values):
        raise _error("INVALID_APP_IDS", f"{label}: empty catalogue or empty ID")
    if len(set(values)) != len(values):
        raise _error("DUPLICATE_APP_IDS", label)
    return values


def _validate_row_indices(indices: Sequence[Any] | None, n: int, label: str) -> None:
    if indices is None:
        return
    actual = np.asarray(indices)
    expected = np.arange(n, dtype=np.int64)
    if actual.ndim != 1 or actual.shape[0] != n or not np.array_equal(actual, expected):
        raise _error("ROW_INDEX_MISMATCH", label)


def _validate_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise _error("INVALID_EMBEDDING_SHAPE", f"{label}:{array.shape}")
    converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise _error("NONFINITE_VECTOR", label)
    squared_norms = np.sum(converted * converted, axis=1, dtype=np.float64)
    if np.any(squared_norms <= 0.0):
        raise _error("ZERO_VECTOR", label)
    return converted


def validate_and_normalize_inputs(
    en_vectors: np.ndarray,
    vi_vectors: np.ndarray,
    en_ids: Sequence[Any],
    vi_ids: Sequence[Any],
    *,
    en_row_indices: Sequence[Any] | None = None,
    vi_row_indices: Sequence[Any] | None = None,
    expected_count: int | None = None,
    expected_ordered_ids_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate paired inputs and return normalized float64 copies.

    Input arrays are never mutated.  Paired IDs must contain the same unique
    values in the same physical order; no sorting or silent realignment occurs.
    """

    en = _validate_matrix(en_vectors, "en")
    vi = _validate_matrix(vi_vectors, "vi")
    if en.shape != vi.shape:
        raise _error("EMBEDDING_SHAPE_MISMATCH", f"en={en.shape},vi={vi.shape}")

    en_id_values = _validate_ids(en_ids, "en")
    vi_id_values = _validate_ids(vi_ids, "vi")
    if len(en_id_values) != en.shape[0] or len(vi_id_values) != vi.shape[0]:
        raise _error("ID_VECTOR_LENGTH_MISMATCH")
    if set(en_id_values) != set(vi_id_values):
        raise _error("PAIRED_ID_SET_MISMATCH")
    if en_id_values != vi_id_values:
        raise _error("PAIRED_ID_ORDER_MISMATCH")
    if expected_count is not None and len(en_id_values) != expected_count:
        raise _error(
            "CATALOGUE_SIZE_MISMATCH",
            f"expected={expected_count},actual={len(en_id_values)}",
        )

    _validate_row_indices(en_row_indices, len(en_id_values), "en")
    _validate_row_indices(vi_row_indices, len(vi_id_values), "vi")

    id_digest = ordered_ids_sha256(en_id_values)
    if expected_ordered_ids_sha256 is not None and id_digest != expected_ordered_ids_sha256:
        raise _error(
            "ORDERED_ID_HASH_MISMATCH",
            f"expected={expected_ordered_ids_sha256},actual={id_digest}",
        )

    en_norms = np.sqrt(np.sum(en * en, axis=1, dtype=np.float64))
    vi_norms = np.sqrt(np.sum(vi * vi, axis=1, dtype=np.float64))
    en_unit = en / en_norms[:, None]
    vi_unit = vi / vi_norms[:, None]
    return en_unit, vi_unit, en_id_values


def tie_aware_metrics(better_count: int, tie_count: int) -> dict[str, float | int]:
    """Return rank evidence and expected metrics for one tied positive."""

    if better_count < 0 or tie_count <= 0:
        raise _error(
            "INVALID_RANK_COUNTS",
            f"better_count={better_count},tie_count={tie_count}",
        )
    rank_min = better_count + 1
    rank_max = better_count + tie_count
    rank_average = (rank_min + rank_max) / 2.0

    def expected_recall(k: int) -> float:
        return min(tie_count, max(0, k - better_count)) / tie_count

    expected_rr = math.fsum(
        1.0 / (better_count + offset) for offset in range(1, tie_count + 1)
    ) / tie_count
    return {
        "rank_min": rank_min,
        "rank_max": rank_max,
        "rank_average": rank_average,
        "tie_group_size": tie_count,
        "expected_recall_at_1": expected_recall(1),
        "expected_recall_at_10": expected_recall(10),
        "expected_reciprocal_rank": expected_rr,
    }


def _directional_ranks(
    query: np.ndarray,
    target: np.ndarray,
    app_ids: Sequence[str],
    direction: str,
    block_size: int,
) -> list[QueryRank]:
    if block_size <= 0:
        raise _error("INVALID_BLOCK_SIZE", str(block_size))
    n = query.shape[0]
    positive_scores = np.einsum(
        "ij,ij->i", query, target, dtype=np.float64, optimize=False
    )
    better = np.zeros(n, dtype=np.int64)
    ties = np.zeros(n, dtype=np.int64)

    for query_start in range(0, n, block_size):
        query_end = min(query_start + block_size, n)
        query_block = query[query_start:query_end]
        positive_block = positive_scores[query_start:query_end, None]
        block_better = np.zeros(query_end - query_start, dtype=np.int64)
        block_ties = np.zeros(query_end - query_start, dtype=np.int64)
        for target_start in range(0, n, block_size):
            target_end = min(target_start + block_size, n)
            scores = np.einsum(
                "id,jd->ij",
                query_block,
                target[target_start:target_end],
                dtype=np.float64,
                optimize=False,
            )
            block_better += np.count_nonzero(scores > positive_block, axis=1)
            block_ties += np.count_nonzero(scores == positive_block, axis=1)
        better[query_start:query_end] = block_better
        ties[query_start:query_end] = block_ties

    if np.any(ties <= 0):
        raise _error("POSITIVE_NOT_IN_TIE_GROUP")

    rows: list[QueryRank] = []
    for index, app_id in enumerate(app_ids):
        metrics = tie_aware_metrics(int(better[index]), int(ties[index]))
        rows.append(
            QueryRank(
                query_index=index,
                query_app_id=app_id,
                positive_target_app_id=app_id,
                direction=direction,
                positive_score=float(positive_scores[index]),
                rank_min=int(metrics["rank_min"]),
                rank_max=int(metrics["rank_max"]),
                rank_average=float(metrics["rank_average"]),
                tie_group_size=int(metrics["tie_group_size"]),
                expected_recall_at_1=float(metrics["expected_recall_at_1"]),
                expected_recall_at_10=float(metrics["expected_recall_at_10"]),
                expected_reciprocal_rank=float(metrics["expected_reciprocal_rank"]),
                target_count=n,
            )
        )
    return rows


def _aggregate(rows: Sequence[QueryRank]) -> dict[str, Any]:
    if not rows:
        raise _error("EMPTY_QUERY_RESULTS")
    return {
        "direction": rows[0].direction,
        "query_count": len(rows),
        "target_count": rows[0].target_count,
        "recall_at_1_primary": math.fsum(
            row.expected_recall_at_1 for row in rows
        )
        / len(rows),
        "recall_at_10_supplementary": math.fsum(
            row.expected_recall_at_10 for row in rows
        )
        / len(rows),
        "mrr_supplementary": math.fsum(
            row.expected_reciprocal_rank for row in rows
        )
        / len(rows),
    }


def evaluate_bidirectional(
    en_vectors: np.ndarray,
    vi_vectors: np.ndarray,
    en_ids: Sequence[Any],
    vi_ids: Sequence[Any],
    *,
    block_size: int,
    en_row_indices: Sequence[Any] | None = None,
    vi_row_indices: Sequence[Any] | None = None,
    expected_count: int | None = None,
    expected_ordered_ids_sha256: str | None = None,
) -> dict[str, Any]:
    en_unit, vi_unit, app_ids = validate_and_normalize_inputs(
        en_vectors,
        vi_vectors,
        en_ids,
        vi_ids,
        en_row_indices=en_row_indices,
        vi_row_indices=vi_row_indices,
        expected_count=expected_count,
        expected_ordered_ids_sha256=expected_ordered_ids_sha256,
    )
    en_to_vi = _directional_ranks(en_unit, vi_unit, app_ids, "EN_TO_VI", block_size)
    vi_to_en = _directional_ranks(vi_unit, en_unit, app_ids, "VI_TO_EN", block_size)
    return {
        "app_ids_sha256": ordered_ids_sha256(app_ids),
        "catalogue_size": len(app_ids),
        "score_dtype": "float64",
        "normalization": "float64_l2_copy",
        "tie_rule": "exact_float64_score_equality",
        "per_query": [asdict(row) for row in (*en_to_vi, *vi_to_en)],
        "aggregate": {
            "EN_TO_VI": _aggregate(en_to_vi),
            "VI_TO_EN": _aggregate(vi_to_en),
        },
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def persist_results_create_only(
    result: Mapping[str, Any],
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    """Atomically publish deterministic outputs into a new directory."""

    output_dir = Path(output_dir)
    required_provenance = {
        "authorization_receipt_sha256",
        "contract_sha256",
        "code_sha256",
        "input_sha256",
        "block_size",
        "cell_id",
    }
    missing = sorted(required_provenance - set(provenance))
    if missing:
        raise _error("PROVENANCE_FIELD_MISSING", ",".join(missing))
    if not isinstance(provenance.get("input_sha256"), Mapping):
        raise _error("PROVENANCE_INPUT_HASHES_INVALID")
    if output_dir.exists():
        raise _error("OUTPUT_COLLISION", str(output_dir))
    parent = output_dir.parent
    if not parent.is_dir():
        raise _error("OUTPUT_PARENT_MISSING", str(parent))

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        per_query_path = staging / PER_QUERY_FILENAME
        aggregate_path = staging / AGGREGATE_FILENAME
        manifest_path = staging / MANIFEST_FILENAME
        per_query_path.write_bytes(_jsonl_bytes(result["per_query"]))
        aggregate_path.write_bytes(_canonical_json_bytes(result["aggregate"]))
        output_hashes = {
            PER_QUERY_FILENAME: sha256_file(per_query_path),
            AGGREGATE_FILENAME: sha256_file(aggregate_path),
        }
        manifest = {
            "task_id": TASK_ID,
            "status": "SUCCESS",
            "catalogue_size": result["catalogue_size"],
            "app_ids_sha256": result["app_ids_sha256"],
            "score_dtype": result["score_dtype"],
            "normalization": result["normalization"],
            "tie_rule": result["tie_rule"],
            "directions": ["EN_TO_VI", "VI_TO_EN"],
            "output_sha256": output_hashes,
            "provenance": dict(provenance),
        }
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        if output_dir.exists():
            raise _error("OUTPUT_COLLISION", str(output_dir))
        staging.rename(output_dir)
        return {
            **output_hashes,
            MANIFEST_FILENAME: sha256_file(output_dir / MANIFEST_FILENAME),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("INVALID_JSON", f"{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise _error("INVALID_JSON_OBJECT", str(path))
    return value


def _load_rows(path: Path) -> tuple[list[int], list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise _error("PYARROW_REQUIRED_FOR_REAL_SIDECAR") from exc
    try:
        table = pq.read_table(path, columns=["row_idx", "appId"])
    except Exception as exc:
        raise _error("INVALID_ROW_SIDECAR", f"{path}:{exc}") from exc
    return (
        [int(value) for value in table.column("row_idx").to_pylist()],
        [str(value) for value in table.column("appId").to_pylist()],
    )


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise _error("ARTIFACT_HASH_MISMATCH", f"{label}:expected={expected},actual={actual}")


def _validate_real_authorization(
    contract_path: Path,
    receipt_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _load_json(contract_path)
    receipt = _load_json(receipt_path)
    if contract.get("real_retrieval_execution_authorized") is not False:
        raise _error("CONTRACT_AUTHORIZATION_STATE_INVALID")
    if receipt.get("task_id") != TASK_ID:
        raise _error("AUTHORIZATION_TASK_MISMATCH")
    if receipt.get("authorization_status") != REAL_AUTHORIZATION_STATUS:
        raise _error("REAL_RETRIEVAL_NOT_AUTHORIZED")
    if receipt.get("contract_sha256") != sha256_file(contract_path):
        raise _error("AUTHORIZATION_CONTRACT_HASH_MISMATCH")
    if receipt.get("ordered_app_ids_sha256") != contract.get("input_contract", {}).get(
        "ordered_app_ids_sha256"
    ):
        raise _error("AUTHORIZATION_ORDERED_ID_HASH_MISMATCH")
    if receipt.get("output_namespace") != str(output_dir):
        raise _error("AUTHORIZATION_OUTPUT_NAMESPACE_MISMATCH")
    return contract, receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-npy", type=Path, required=True)
    parser.add_argument("--vi-npy", type=Path, required=True)
    parser.add_argument("--en-rows", type=Path, required=True)
    parser.add_argument("--vi-rows", type=Path, required=True)
    parser.add_argument("--en-meta", type=Path, required=True)
    parser.add_argument("--vi-meta", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        contract, receipt = _validate_real_authorization(
            args.contract, args.authorization_receipt, args.output_dir
        )
        if args.output_dir.exists():
            raise _error("OUTPUT_COLLISION", str(args.output_dir))
        bindings = receipt.get("inputs")
        if not isinstance(bindings, dict):
            raise _error("AUTHORIZATION_INPUT_BINDINGS_MISSING")
        paths = {
            "en_npy": args.en_npy,
            "vi_npy": args.vi_npy,
            "en_rows": args.en_rows,
            "vi_rows": args.vi_rows,
            "en_meta": args.en_meta,
            "vi_meta": args.vi_meta,
        }
        for label, path in paths.items():
            expected = bindings.get(label, {}).get("sha256")
            if not isinstance(expected, str):
                raise _error("AUTHORIZATION_INPUT_HASH_MISSING", label)
            _require_hash(path, expected, label)

        en_indices, en_ids = _load_rows(args.en_rows)
        vi_indices, vi_ids = _load_rows(args.vi_rows)
        en_vectors = np.load(args.en_npy, mmap_mode="r", allow_pickle=False)
        vi_vectors = np.load(args.vi_npy, mmap_mode="r", allow_pickle=False)
        result = evaluate_bidirectional(
            en_vectors,
            vi_vectors,
            en_ids,
            vi_ids,
            block_size=args.block_size,
            en_row_indices=en_indices,
            vi_row_indices=vi_indices,
            expected_count=int(contract["input_contract"]["catalogue_size"]),
            expected_ordered_ids_sha256=receipt["ordered_app_ids_sha256"],
        )
        provenance = {
            "authorization_receipt_sha256": sha256_file(args.authorization_receipt),
            "contract_sha256": sha256_file(args.contract),
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "input_sha256": {label: sha256_file(path) for label, path in paths.items()},
            "block_size": args.block_size,
            "cell_id": receipt.get("cell_id"),
        }
        persist_results_create_only(result, args.output_dir, provenance)
        return 0
    except G1CRetrievalError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

