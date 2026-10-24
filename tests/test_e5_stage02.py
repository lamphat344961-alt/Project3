"""Stage 02 fixtures: no checkpoint, network, CUDA inference or real experiment."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pipeline import e5_stage02 as stage


class Tokenizer:
    def __call__(self, texts, **kwargs):
        # Two tokens per input, variable audit length to force reordering.
        if not kwargs.get("return_tensors"):
            return {"input_ids": [list(range(len(text) + 2)) for text in texts]}
        ids = torch.tensor([[int(text.split(":")[0]), 0] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.tensor([[1, 0]] * len(texts))}


class Model:
    def __call__(self, input_ids, attention_mask, return_dict):
        hidden = torch.zeros((len(input_ids), 2, 1024), dtype=torch.float32)
        for i, row in enumerate(input_ids):
            hidden[i, 0, int(row[0])] = 2
            hidden[i, 1, 99] = 100  # Must be excluded by attention mask.
        return SimpleNamespace(last_hidden_state=hidden)


class Stage02Tests(unittest.TestCase):
    def test_six_cells_and_no_unapproved_reuse(self):
        contract = stage.load_contract()
        self.assertEqual(len(contract["cells"]), 6)
        self.assertFalse(contract["reuse_native_t512"])
        self.assertFalse(contract["full_run_authorized"])

    def test_reuse_and_missing_cell_are_rejected(self):
        for change in (lambda c: c.update(reuse_native_t512=True), lambda c: c["cells"].pop()):
            contract = stage.load_contract()
            change(contract)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "contract.json"
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    stage.load_contract(path)

    def test_sorting_restores_rows_and_masked_pooling(self):
        texts = ["3:longest input", "1:x", "2:medium"]
        vectors, metadata = stage.encode_prepared(texts, Tokenizer(), Model(), torch.device("cpu"), 2)
        self.assertEqual(np.argmax(vectors, axis=1).tolist(), [3, 1, 2])
        np.testing.assert_array_equal(vectors[:, 99], 0)
        self.assertEqual(np.linalg.norm(vectors, axis=1).tolist(), [1, 1, 1])
        self.assertIn("batch_order_sha256", metadata)

    def test_batch_size_does_not_change_fixture_rows(self):
        texts = ["3:longest input", "1:x", "2:medium"]
        a, _ = stage.encode_prepared(texts, Tokenizer(), Model(), torch.device("cpu"), 1)
        b, _ = stage.encode_prepared(texts, Tokenizer(), Model(), torch.device("cpu"), 2)
        np.testing.assert_array_equal(a, b)

    def test_nonfinite_zero_norm_and_wrong_shape_rejected(self):
        valid = np.eye(2, 1024, dtype=np.float32)
        invalid = [valid[:, :10], np.zeros_like(valid), valid.astype(np.float64)]
        nonfinite = valid.copy()
        nonfinite[0, 1] = np.nan
        invalid.append(nonfinite)
        for vectors in invalid:
            with self.assertRaises(RuntimeError):
                stage.validate_vectors(vectors, 2)

    def test_repeat_tolerance_fails_for_changed_unit_vectors(self):
        first = np.eye(2, 1024, dtype=np.float32)
        self.assertTrue(stage.compare_repeats(first, first.copy())["passed"])
        second = first.copy()
        second[0] = first[1]
        self.assertFalse(stage.compare_repeats(first, second)["passed"])

    def test_cpu_only_runtime_fails_before_model_loading(self):
        runtime = stage.load_contract()["runtime"]
        with patch.dict(stage.os.environ, {}, clear=True), patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "NO_CPU_FALLBACK"):
                stage.configure_runtime(runtime)

    def test_conflicting_cublas_setting_rejected(self):
        with patch.dict(stage.os.environ, {"CUBLAS_WORKSPACE_CONFIG": "bad"}):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_CONFIG_CONFLICT"):
                stage.configure_runtime(stage.load_contract()["runtime"])

    def test_namespace_cannot_escape_or_target_historical_folder(self):
        for run_id in ("../embeddings", "a/b", "a.b", "", "C:\\embeddings"):
            with self.assertRaises(RuntimeError):
                stage.safe_namespace(run_id)
        self.assertEqual(stage.safe_namespace("smoke_001").parent, stage.ROOT / "results/e5_stage02")

    def test_create_only_save_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            stage.embed.write_json_create_only(path, {"original": True})
            original = path.read_bytes()
            with self.assertRaises(stage.embed.ProductionGuardError):
                stage.embed.write_json_create_only(path, {"original": False})
            self.assertEqual(path.read_bytes(), original)

    def test_existing_namespace_rejected_before_input_or_model_access(self):
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            with patch.object(stage, "safe_namespace", return_value=namespace), patch.object(stage, "load_inputs") as loader:
                with self.assertRaisesRegex(RuntimeError, "OUTPUT_NAMESPACE_EXISTS"):
                    stage.main(["smoke", "exists"])
                loader.assert_not_called()

    def test_saved_row_permutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "vectors"
            vectors = np.eye(2, 1024, dtype=np.float32)
            rows = pd.DataFrame({"row_idx": [0, 1], "appId": ["app-b", "app-a"], "genreId": ["x", "y"]})
            stage.embed.save_embedding(vectors, rows, base, create_only=True)
            stage.validate_saved_output(base, vectors, ["app-b", "app-a"])
            with self.assertRaisesRegex(RuntimeError, "SAVED_ROW_ORDER_MISMATCH"):
                stage.validate_saved_output(base, vectors, ["app-a", "app-b"])

    def test_input_hash_or_id_drift_stops_before_model_load(self):
        contract = stage.load_contract()
        contract["datasets"]["tn1"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "DATASET_HASH:tn1"):
            stage.load_inputs(contract)


if __name__ == "__main__":
    unittest.main()
