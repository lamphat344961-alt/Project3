"""Offline LaBSE fixtures. No pretrained weights, network, GPU or E5 execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pipeline import labse_stage03 as stage


def configs():
    return {
        "modules.json": [{"idx": i, "path": path, "type": "sentence_transformers.models." + kind}
                         for i, (path, kind) in enumerate([
                             ("", "Transformer"), ("1_Pooling", "Pooling"),
                             ("2_Dense", "Dense"), ("3_Normalize", "Normalize")])],
        "sentence_bert_config.json": {"max_seq_length": 256, "do_lower_case": False},
        "config.json": {"model_type": "bert", "hidden_size": 768, "max_position_embeddings": 512},
        "1_Pooling/config.json": {"word_embedding_dimension": 768, "pooling_mode_cls_token": True,
                                  "pooling_mode_mean_tokens": False},
        "2_Dense/config.json": {"in_features": 768, "out_features": 768, "bias": True,
                                "activation_function": "torch.nn.modules.activation.Tanh"},
    }


class Tokenizer:
    def __call__(self, texts, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([[int(t[0]), 0] for t in texts]),
                    "attention_mask": torch.tensor([[1, 1]] * len(texts))}
        return {"input_ids": [[1] * (len(text) + 2) for text in texts]}


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fixture_parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, return_dict):
        hidden = torch.zeros((len(input_ids), 2, 768))
        for i, row in enumerate(input_ids):
            hidden[i, 0, int(row[0])] = 2
            hidden[i, 1, 99] = 100
        return SimpleNamespace(last_hidden_state=hidden)


class LaBSETests(unittest.TestCase):
    def test_native_cap_comes_from_sentence_config_not_backbone(self):
        summary = stage.validate_configs(configs(), stage.load_contract())
        self.assertEqual(summary["effective_native_cap"], 256)
        self.assertEqual(stage.effective_cap(stage.load_contract(), "token_budget", 512), 256)

    def test_wrong_graph_mean_pooling_cap_and_dense_are_rejected(self):
        bad_configs = []
        bad = configs(); bad["modules.json"].pop(2); bad_configs.append(bad)
        bad = configs(); bad["1_Pooling/config.json"]["pooling_mode_mean_tokens"] = True; bad_configs.append(bad)
        bad = configs(); bad["sentence_bert_config.json"]["max_seq_length"] = 512; bad_configs.append(bad)
        bad = configs(); bad["2_Dense/config.json"]["activation_function"] = "identity"; bad_configs.append(bad)
        for config in bad_configs:
            with self.assertRaises(RuntimeError):
                stage.validate_configs(config, stage.load_contract())

    def test_head_matches_installed_sentence_transformers_modules(self):
        from sentence_transformers.models import Pooling, Dense, Normalize
        torch.manual_seed(42)
        hidden = torch.randn(3, 5, 768)
        weight, bias = torch.randn(768, 768) * 0.01, torch.randn(768) * 0.01
        features = {"token_embeddings": hidden.clone(), "attention_mask": torch.ones(3, 5, dtype=torch.int64)}
        pooling = Pooling(768, pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
        dense = Dense(768, 768, init_weight=weight.clone(), init_bias=bias.clone())
        reference = Normalize()(dense(pooling(features)))["sentence_embedding"]
        actual = stage.cls_dense_normalize(hidden, weight, bias)
        np.testing.assert_allclose(actual.detach().numpy(), reference.detach().numpy(), atol=1e-7, rtol=1e-6)

    def test_dense_head_is_not_raw_cls_or_mean(self):
        hidden = torch.zeros(1, 2, 768)
        hidden[0, 0, 0] = 1
        hidden[0, 1, 99] = 10
        weight = torch.eye(768)
        weight[[0, 1]] = weight[[1, 0]]
        output = stage.cls_dense_normalize(hidden, weight, torch.zeros(768))
        self.assertEqual(int(output.argmax()), 1)
        self.assertEqual(float(output[0, 99]), 0)

    def test_adapter_restores_original_row_order(self):
        adapter = stage.LaBSEAdapter(Tokenizer(), Backbone(), {
            "linear.weight": torch.eye(768), "linear.bias": torch.zeros(768)})
        actual, metadata = adapter.encode(["3:longest input", "1:x", "2:medium"], batch_size=2)
        self.assertEqual(actual.argmax(axis=1).tolist(), [3, 1, 2])
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-7)
        self.assertEqual(metadata["effective_cap"], 256)

    def test_missing_dense_weights_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(RuntimeError, "DENSE_WEIGHT_KEYS"):
            stage.LaBSEAdapter(Tokenizer(), Backbone(), {})

    def test_no_prefix_no_segmentation_and_950_preserved(self):
        contract = stage.load_contract()
        text = "Ứng dụng bản đồ. " * 100
        native = stage.prepared_texts([text], "native", contract)[0]
        clipped = stage.prepared_texts([text], "char_budget", contract)[0]
        self.assertEqual(native, text.strip())
        self.assertLessEqual(len(clipped), 950)
        self.assertFalse(native.startswith("query:"))

    def test_tokenizer_audit_counts_character_and_token_cuts_separately(self):
        frame = pd.DataFrame({"appId": ["a", "b"], "description_clean": ["x" * 1200, "short"]})
        result = stage.tokenizer_audit(frame, Tokenizer(), stage.load_contract())
        native, clipped = result["conditions"]["native"], result["conditions"]["char_budget"]
        self.assertEqual(native["character_clipped_count"], 0)
        self.assertEqual(clipped["character_clipped_count"], 1)
        self.assertEqual(clipped["token_truncated_count"], 1)
        self.assertTrue(result["cutoff_950_preserved"])

    def test_production_has_no_cpu_fallback(self):
        with patch.dict(stage.os.environ, {}, clear=True), patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "NO_CPU_FALLBACK"):
                stage.configure_production()

    def test_encode_requires_owner_and_reference_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(stage, "ROOT", Path(temp)), \
             patch.object(stage, "sources", return_value={"fixture": True}), \
             patch.object(stage, "resolve_snapshot") as loader:
            with self.assertRaisesRegex(RuntimeError, "OWNER_APPROVAL_CELL_AND_REFERENCE_REQUIRED"):
                stage.main(["encode", "fixture"])
            loader.assert_not_called()
            receipt = json.loads((Path(temp) / "results/labse_stage03/fixture/execution_receipt.json").read_text())
            self.assertEqual(receipt["status"], "FAILED")

    def test_unknown_revision_folder_rejected_before_config_use(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "SNAPSHOT_REVISION_UNBOUND"):
                stage.resolve_snapshot(stage.load_contract(), Path(temp))

    def test_partial_tokenizer_assets_never_certify_full_graph(self):
        contract = stage.load_contract()
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp) / contract["revision"]
            snapshot.mkdir()
            (snapshot / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 256, "do_lower_case": False}))
            (snapshot / "tokenizer.json").write_text("{}")
            _, receipt = stage.resolve_snapshot(contract, snapshot, tokenizer_only=True)
            self.assertFalse(receipt["config_summary"]["full_module_graph_verified"])
            with self.assertRaisesRegex(RuntimeError, "SNAPSHOT_INCOMPLETE"):
                stage.resolve_snapshot(contract, snapshot)

    def test_tokenizer_wrapper_matches_saved_backend(self):
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
        tokenizer = Tokenizer(models.WordPiece({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "word": 3}, unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            tokenizer.save(str(snapshot / "tokenizer.json"))
            wrapped = stage.load_audit_tokenizer(snapshot)
            self.assertEqual(wrapped("word", add_special_tokens=True)["input_ids"], [1, 3, 2])


if __name__ == "__main__":
    unittest.main()
