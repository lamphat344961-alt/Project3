"""Synthetic CPU tests for the PAAC-VN G1C retrieval implementation."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.g1c_retrieval import (
    AGGREGATE_FILENAME,
    MANIFEST_FILENAME,
    PER_QUERY_FILENAME,
    G1CRetrievalError,
    evaluate_bidirectional,
    ordered_ids_sha256,
    persist_results_create_only,
    tie_aware_metrics,
    validate_and_normalize_inputs,
)


def dense_oracle(query: np.ndarray, target: np.ndarray) -> list[dict[str, float | int]]:
    """Independent dense reference used only by tests."""

    query64 = np.asarray(query, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    query64 = query64 / np.sqrt(np.sum(query64 * query64, axis=1))[:, None]
    target64 = target64 / np.sqrt(np.sum(target64 * target64, axis=1))[:, None]
    similarities = query64 @ target64.T
    expected: list[dict[str, float | int]] = []
    for index in range(len(query64)):
        positive = similarities[index, index]
        better = int(np.count_nonzero(similarities[index] > positive))
        tied = int(np.count_nonzero(similarities[index] == positive))
        rank_min = better + 1
        rank_max = better + tied
        expected.append(
            {
                "rank_min": rank_min,
                "rank_max": rank_max,
                "rank_average": (rank_min + rank_max) / 2.0,
                "tie_group_size": tied,
                "expected_recall_at_1": min(tied, max(0, 1 - better)) / tied,
                "expected_recall_at_10": min(tied, max(0, 10 - better)) / tied,
                "expected_reciprocal_rank": math.fsum(
                    1.0 / rank for rank in range(rank_min, rank_max + 1)
                )
                / tied,
            }
        )
    return expected


class TieMetricTests(unittest.TestCase):
    def test_unique_positive(self) -> None:
        vectors = np.eye(4, dtype=np.float32)
        ids = [f"app-{index}" for index in range(4)]
        result = evaluate_bidirectional(vectors, vectors, ids, ids, block_size=2)
        for row in result["per_query"]:
            self.assertEqual(row["rank_min"], 1)
            self.assertEqual(row["rank_max"], 1)
            self.assertEqual(row["tie_group_size"], 1)
            self.assertEqual(row["expected_recall_at_1"], 1.0)
            self.assertEqual(row["expected_reciprocal_rank"], 1.0)

    def test_top_one_two_way_tie(self) -> None:
        metrics = tie_aware_metrics(0, 2)
        self.assertEqual(metrics["expected_recall_at_1"], 0.5)
        self.assertEqual(metrics["expected_reciprocal_rank"], 0.75)
        self.assertEqual(metrics["rank_average"], 1.5)

    def test_top_ten_boundary_tie(self) -> None:
        metrics = tie_aware_metrics(9, 3)
        self.assertEqual(metrics["rank_min"], 10)
        self.assertEqual(metrics["rank_max"], 12)
        self.assertEqual(metrics["expected_recall_at_10"], 1.0 / 3.0)
        expected_rr = (1.0 / 10.0 + 1.0 / 11.0 + 1.0 / 12.0) / 3.0
        self.assertAlmostEqual(metrics["expected_reciprocal_rank"], expected_rr)

    def test_three_way_top_tie_exact_oracle(self) -> None:
        metrics = tie_aware_metrics(0, 3)
        self.assertEqual(metrics["expected_recall_at_1"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["expected_reciprocal_rank"], 11.0 / 18.0)
        self.assertNotEqual(
            metrics["expected_reciprocal_rank"], 1.0 / metrics["rank_average"]
        )

    def test_three_way_top_tie_through_retrieval(self) -> None:
        vectors = np.ones((3, 2), dtype=np.float32)
        ids = ["a", "b", "c"]
        result = evaluate_bidirectional(vectors, vectors, ids, ids, block_size=1)
        for row in result["per_query"]:
            self.assertEqual(row["tie_group_size"], 3)
            self.assertEqual(row["expected_recall_at_1"], 1.0 / 3.0)
            self.assertAlmostEqual(row["expected_reciprocal_rank"], 11.0 / 18.0)


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = np.eye(3, dtype=np.float32)
        self.ids = ["a", "b", "c"]

    def assert_code(self, expected: str, callable_obj) -> None:
        with self.assertRaises(G1CRetrievalError) as caught:
            callable_obj()
        self.assertEqual(caught.exception.code, expected)

    def test_duplicate_ids_rejected(self) -> None:
        self.assert_code(
            "DUPLICATE_APP_IDS",
            lambda: validate_and_normalize_inputs(
                self.vectors, self.vectors, ["a", "a", "c"], self.ids
            ),
        )

    def test_id_set_mismatch_rejected(self) -> None:
        self.assert_code(
            "PAIRED_ID_SET_MISMATCH",
            lambda: validate_and_normalize_inputs(
                self.vectors, self.vectors, self.ids, ["a", "b", "x"]
            ),
        )

    def test_id_order_mismatch_rejected(self) -> None:
        self.assert_code(
            "PAIRED_ID_ORDER_MISMATCH",
            lambda: validate_and_normalize_inputs(
                self.vectors, self.vectors, self.ids, ["b", "a", "c"]
            ),
        )

    def test_row_alignment_rejected(self) -> None:
        self.assert_code(
            "ROW_INDEX_MISMATCH",
            lambda: validate_and_normalize_inputs(
                self.vectors,
                self.vectors,
                self.ids,
                self.ids,
                en_row_indices=[0, 2, 1],
                vi_row_indices=[0, 1, 2],
            ),
        )

    def test_shape_mismatch_rejected(self) -> None:
        self.assert_code(
            "EMBEDDING_SHAPE_MISMATCH",
            lambda: validate_and_normalize_inputs(
                self.vectors, np.ones((3, 2)), self.ids, self.ids
            ),
        )

    def test_nan_and_inf_rejected(self) -> None:
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                vectors = self.vectors.copy()
                vectors[0, 0] = invalid
                self.assert_code(
                    "NONFINITE_VECTOR",
                    lambda vectors=vectors: validate_and_normalize_inputs(
                        vectors, self.vectors, self.ids, self.ids
                    ),
                )

    def test_zero_vector_rejected(self) -> None:
        vectors = self.vectors.copy()
        vectors[1] = 0
        self.assert_code(
            "ZERO_VECTOR",
            lambda: validate_and_normalize_inputs(
                vectors, self.vectors, self.ids, self.ids
            ),
        )

    def test_ordered_id_hash_guard(self) -> None:
        self.assertEqual(
            ordered_ids_sha256(self.ids),
            __import__("hashlib").sha256(b"a\nb\nc").hexdigest(),
        )
        self.assert_code(
            "ORDERED_ID_HASH_MISMATCH",
            lambda: validate_and_normalize_inputs(
                self.vectors,
                self.vectors,
                self.ids,
                self.ids,
                expected_ordered_ids_sha256="0" * 64,
            ),
        )

    def test_input_arrays_are_not_mutated(self) -> None:
        en = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        vi = np.array([[4.0, 0.0], [0.0, 5.0]], dtype=np.float32)
        en_before = en.copy()
        vi_before = vi.copy()
        validate_and_normalize_inputs(en, vi, ["a", "b"], ["a", "b"])
        np.testing.assert_array_equal(en, en_before)
        np.testing.assert_array_equal(vi, vi_before)


class BlockwiseAndPersistenceTests(unittest.TestCase):
    def test_blockwise_matches_independent_dense_oracle(self) -> None:
        rng = np.random.default_rng(20261023)
        en = rng.normal(size=(9, 5)).astype(np.float32)
        vi = rng.normal(size=(9, 5)).astype(np.float32)
        ids = [f"id-{index:02d}" for index in range(9)]
        en_oracle = dense_oracle(en, vi)
        vi_oracle = dense_oracle(vi, en)

        for block_size in (1, 2, 4, 20):
            with self.subTest(block_size=block_size):
                result = evaluate_bidirectional(
                    en, vi, ids, ids, block_size=block_size
                )
                self.assertEqual(
                    [row["direction"] for row in result["per_query"][:9]],
                    ["EN_TO_VI"] * 9,
                )
                self.assertEqual(
                    [row["direction"] for row in result["per_query"][9:]],
                    ["VI_TO_EN"] * 9,
                )
                for actual, expected in zip(
                    result["per_query"][:9], en_oracle, strict=True
                ):
                    for key, value in expected.items():
                        self.assertAlmostEqual(actual[key], value)
                for actual, expected in zip(
                    result["per_query"][9:], vi_oracle, strict=True
                ):
                    for key, value in expected.items():
                        self.assertAlmostEqual(actual[key], value)

    def test_bidirectional_metrics_are_independent(self) -> None:
        en = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        vi = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        ids = ["a", "b", "c"]
        result = evaluate_bidirectional(en, vi, ids, ids, block_size=2)
        self.assertNotEqual(
            result["aggregate"]["EN_TO_VI"]["mrr_supplementary"],
            result["aggregate"]["VI_TO_EN"]["mrr_supplementary"],
        )

    def test_invalid_block_size_rejected(self) -> None:
        vectors = np.eye(2)
        with self.assertRaises(G1CRetrievalError) as caught:
            evaluate_bidirectional(vectors, vectors, ["a", "b"], ["a", "b"], block_size=0)
        self.assertEqual(caught.exception.code, "INVALID_BLOCK_SIZE")

    def test_persistence_is_deterministic_and_create_only(self) -> None:
        vectors = np.eye(3)
        ids = ["a", "b", "c"]
        result = evaluate_bidirectional(vectors, vectors, ids, ids, block_size=2)
        provenance = {
            "contract_sha256": "1" * 64,
            "authorization_receipt_sha256": "2" * 64,
            "code_sha256": "5" * 64,
            "input_sha256": {"en_npy": "3" * 64, "vi_npy": "4" * 64},
            "block_size": 2,
            "cell_id": "synthetic",
        }
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = parent / "first"
            second = parent / "second"
            first_hashes = persist_results_create_only(result, first, provenance)
            second_hashes = persist_results_create_only(result, second, provenance)
            self.assertEqual(first_hashes, second_hashes)
            for filename in (
                PER_QUERY_FILENAME,
                AGGREGATE_FILENAME,
                MANIFEST_FILENAME,
            ):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            with self.assertRaises(G1CRetrievalError) as caught:
                persist_results_create_only(result, first, provenance)
            self.assertEqual(caught.exception.code, "OUTPUT_COLLISION")
            manifest = json.loads((first / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "SUCCESS")
            self.assertEqual(manifest["provenance"]["code_sha256"], "5" * 64)

    def test_missing_provenance_fails_before_output(self) -> None:
        vectors = np.eye(2)
        ids = ["a", "b"]
        result = evaluate_bidirectional(vectors, vectors, ids, ids, block_size=1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaises(G1CRetrievalError) as caught:
                persist_results_create_only(result, output, {})
            self.assertEqual(caught.exception.code, "PROVENANCE_FIELD_MISSING")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

