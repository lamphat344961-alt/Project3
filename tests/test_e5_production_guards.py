from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

# The focused guard suite is hermetic: scientific third-party modules that are
# irrelevant to these pure guard functions are replaced before importing the
# production module. No tokenizer/model/network API is loaded or called.
fake_pyvi = types.ModuleType("pyvi")
fake_pyvi.ViTokenizer = type("ViTokenizer", (), {"tokenize": staticmethod(lambda text: text)})
sys.modules.setdefault("pyvi", fake_pyvi)

fake_sklearn = types.ModuleType("sklearn")
fake_decomposition = types.ModuleType("sklearn.decomposition")
fake_feature = types.ModuleType("sklearn.feature_extraction")
fake_feature_text = types.ModuleType("sklearn.feature_extraction.text")
fake_preprocessing = types.ModuleType("sklearn.preprocessing")
fake_decomposition.TruncatedSVD = type("TruncatedSVD", (), {})
fake_feature_text.TfidfVectorizer = type("TfidfVectorizer", (), {})
fake_preprocessing.Normalizer = type("Normalizer", (), {})
sys.modules.setdefault("sklearn", fake_sklearn)
sys.modules.setdefault("sklearn.decomposition", fake_decomposition)
sys.modules.setdefault("sklearn.feature_extraction", fake_feature)
sys.modules.setdefault("sklearn.feature_extraction.text", fake_feature_text)
sys.modules.setdefault("sklearn.preprocessing", fake_preprocessing)

class _ForbiddenFactory:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise AssertionError("REAL_MODEL_OR_TOKENIZER_FACTORY_CALLED")

fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoModel = _ForbiddenFactory
fake_transformers.AutoTokenizer = _ForbiddenFactory
fake_transformers.logging = SimpleNamespace(set_verbosity_error=lambda: None)
sys.modules.setdefault("transformers", fake_transformers)

from pipeline import embed


class _MockTokenizer:
    def __init__(self, mutate_t512: bool = False):
        self.calls = 0
        self.mutate_t512 = mutate_t512

    def __call__(self, texts, **kwargs):
        self.calls += 1
        ids = [[101, len(text), 102] for text in texts]
        if self.mutate_t512 and self.calls > 1 and ids:
            ids[0] = [101, 999, 102]
        return {
            "input_ids": ids,
            "attention_mask": [[1] * len(row) for row in ids],
            "token_type_ids": [[0] * len(row) for row in ids],
        }


class _MockCuda:
    def __init__(self, available=True, count=1):
        self._available = available
        self._count = count

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count


class _MockTorch:
    __version__ = "mock"
    def __init__(self, available=True, count=1):
        self.cuda = _MockCuda(available, count)
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        )
        self.precision = None

    def set_float32_matmul_precision(self, value):
        self.precision = value

    def get_float32_matmul_precision(self):
        return self.precision


class ProductionContractTests(unittest.TestCase):
    @staticmethod
    def _authorization(digest="a" * 64):
        return {"sha256": digest, "receipt": {"source_sha256": {}}}

    @staticmethod
    def _bind_authorization(contract, digest="a" * 64):
        contract["_execution_authorization"] = ProductionContractTests._authorization(digest)
        return contract

    def _authorization_fixture(self, temp: str):
        root = Path(temp) / "project"
        task_root = root / "PAAC_MULTI_AGENT_STARTER/01_tasks/V4-G1A-E5-PROD-001"
        task_root.mkdir(parents=True)
        approval = task_root / "PLAN_V3_1_APPROVAL_RECEIPT.md"
        acceptance = task_root / "NGHIEM_THU_V3_1.md"
        approval.write_text("phase-a", encoding="utf-8")
        acceptance.write_text("accepted", encoding="utf-8")
        contract = json.loads(Path("pipeline/e5_production_contract.json").read_text(encoding="utf-8"))
        contract["paths"] = {
            "project_root": root.as_posix(),
            "python": (root / "venv/python").as_posix(),
            "dataset_root": (root / "data_v1").as_posix(),
            "output_root": (root / "output").as_posix(),
            "model_cache": (root / "cache").as_posix(),
            "namespace": (root / "output" / embed.E5_PRODUCTION_RUN_ID).as_posix(),
        }
        contract_path = root / "pipeline/e5_production_contract.json"
        contract_path.parent.mkdir(parents=True)
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        for relative in contract["token_audit_lineage"]["fixed_source_bindings"]:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(relative).read_bytes())
        receipt = {
            "task_id": embed.E5_PRODUCTION_TASK_ID,
            "plan_sha256": embed.E5_PRODUCTION_PLAN_SHA256,
            "authorization_status": embed.E5_PHASE_B_AUTHORIZATION_STATUS,
            "run_id": embed.E5_PRODUCTION_RUN_ID,
            "authorized_phases": ["TOKEN_AUDIT", "TN2_NATIVE_INFERENCE_AND_T512_REUSE"],
            "automatic_retry": False,
            "full_e5_rerun_authorized": False,
            "tn1_execution_authorized": False,
            "asset_mode": "PREPOPULATED_OFFLINE_CACHE",
            "phase_a_approval_receipt_sha256": embed.sha256(approval),
            "research_lead_acceptance_sha256": embed.sha256(acceptance),
            **contract["phase_b_authorization"]["required_fixed_bindings"],
            "paths": contract["paths"],
            "model": {"checkpoint": embed.E5_CHECKPOINT, "revision": embed.E5_REVISION},
            "runtime": {key: contract["runtime"][key] for key in ("device", "dtype", "batch_size", "autocast", "cuda_matmul_tf32", "cudnn_tf32")},
            "source_sha256": {
                **contract["token_audit_lineage"]["fixed_source_bindings"],
                "pipeline/embed.py": embed.sha256(Path(embed.__file__)),
                "pipeline/e5_production_contract.json": embed.sha256(contract_path),
                "tests/test_e5_production_guards.py": "e" * 64,
            },
        }
        authorization_path = task_root / "KAGGLE_EXECUTION_AUTHORIZATION_V3_1.json"
        authorization_path.write_text(json.dumps(receipt), encoding="utf-8")
        return contract, contract_path, authorization_path, receipt

    def test_actual_contract_is_valid_and_keeps_four_separate_tn2_units(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        self.assertEqual(4, len(contract["outputs"]["units"]))
        self.assertTrue(contract["outputs"]["native_t512_reuse_authorized"])
        self.assertEqual({"tn2"}, {unit["experiment"] for unit in contract["outputs"]["units"]})
        self.assertEqual("TARGETED_TN2_RECOVERY_COMPLETE", contract["completion_status"])
        self.assertEqual("PROVENANCE_INCOMPLETE", contract["historical_tn1"]["status"])
        self.assertFalse(contract["validation"]["batch_8_vs_4_required"])
        self.assertEqual(embed.E5_CANONICAL_ORDERED_ID_SHA256, contract["datasets"]["tn2"]["ordered_id_sha256"])
        self.assertEqual("CANONICAL_MATCH", contract["datasets"]["tn2"]["ordered_id_binding_status"])
        self.assertEqual("/kaggle/working/PAAC-VN", contract["paths"]["project_root"])

    def test_stale_ordered_id_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            contract = json.loads(Path("pipeline/e5_production_contract.json").read_text(encoding="utf-8"))
            contract["datasets"]["tn2"]["runtime_ordered_id_sha256"] = "d52e4c1f9815e922c07a8d9813a5163223494e3a5eac2c41d4a70da1f3dd567f"
            path = Path(temp) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "CANONICAL_ORDERED_ID_BINDING_MISMATCH"):
                embed.load_production_contract(path)

    def test_phase_b_authorization_binds_plan_paths_sources_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            contract, contract_path, path, receipt = self._authorization_fixture(temp)
            evidence = embed.load_and_validate_execution_authorization(path, contract_path, contract)
            self.assertEqual(receipt, evidence["receipt"])
            self.assertEqual(embed.sha256(path), evidence["sha256"])

    def test_phase_b_authorization_rejects_wrong_asset_mode_before_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            contract, contract_path, path, receipt = self._authorization_fixture(temp)
            receipt["asset_mode"] = "AUTO"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "ASSET_MODE_NOT_AUTHORIZED"):
                embed.load_and_validate_execution_authorization(path, contract_path, contract)

    def test_phase_b_authorization_rejects_stale_code_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            contract, contract_path, path, receipt = self._authorization_fixture(temp)
            receipt["source_sha256"]["pipeline/embed.py"] = "0" * 64
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "EXECUTION_AUTHORIZATION_SOURCE_HASH_MISMATCH"):
                embed.load_and_validate_execution_authorization(path, contract_path, contract)

    def test_phase_b_authorization_rejects_unauthorized_phase_set(self):
        with tempfile.TemporaryDirectory() as temp:
            contract, contract_path, path, receipt = self._authorization_fixture(temp)
            receipt["authorized_phases"] = ["TOKEN_AUDIT"]
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "EXECUTION_PHASE_SET_MISMATCH"):
                embed.load_and_validate_execution_authorization(path, contract_path, contract)

    def test_phase_b_authorization_rejects_wrong_receipt_path(self):
        with tempfile.TemporaryDirectory() as temp:
            contract, contract_path, path, _ = self._authorization_fixture(temp)
            wrong = path.parent / "other.json"
            wrong.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(embed.ProductionGuardError, "EXECUTION_AUTHORIZATION_PATH_MISMATCH"):
                embed.load_and_validate_execution_authorization(wrong, contract_path, contract)

    def test_missing_phase_b_authorization_fails_before_factories(self):
        argv = [
            "embed.py", "--exp", "tn2", "--repr", embed.E5_REPRESENTATION_ID,
            "--lang", "en", "vi", "--trunc", "native", "--batch-size", "8",
            "--production-contract", "pipeline/e5_production_contract.json",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            embed.AutoTokenizer, "from_pretrained"
        ) as tokenizer_factory, mock.patch.object(embed.AutoModel, "from_pretrained") as model_factory:
            with self.assertRaisesRegex(embed.ProductionGuardError, "EXECUTION_AUTHORIZATION_REQUIRED"):
                embed._main()
        tokenizer_factory.assert_not_called()
        model_factory.assert_not_called()

    def test_token_audit_default_batch_is_exactly_8(self):
        tokenizer = _MockTokenizer()
        texts = [f"query: {index}" for index in range(17)]
        result = embed.audit_native_t512_token_equality(
            tokenizer=tokenizer, prepared_native=texts, prepared_t512=texts
        )
        self.assertEqual(8, result["tokenizer_batch_size"])
        self.assertEqual(6, tokenizer.calls)

    def test_cuda_fp32_tf32_off_batch_8(self):
        backend = _MockTorch()
        receipt = embed.configure_cuda_fp32(
            device_name="cuda:0", dtype_name="float32", batch_size=8, torch_module=backend
        )
        self.assertFalse(receipt["cuda_matmul_allow_tf32"])
        self.assertFalse(receipt["cudnn_allow_tf32"])
        self.assertEqual("highest", receipt["float32_matmul_precision"])

    def test_cuda_unavailable_fails_closed(self):
        with self.assertRaisesRegex(embed.ProductionGuardError, "CUDA_UNAVAILABLE"):
            embed.configure_cuda_fp32(
                device_name="cuda:0", dtype_name="float32", batch_size=8,
                torch_module=_MockTorch(available=False),
            )

    def test_cpu_fallback_is_rejected(self):
        with self.assertRaisesRegex(embed.ProductionGuardError, "CUDA_DEVICE_REQUIRED"):
            embed.configure_cuda_fp32(
                device_name="cpu", dtype_name="float32", batch_size=8,
                torch_module=_MockTorch(),
            )

    def test_non_fp32_is_rejected(self):
        with self.assertRaisesRegex(embed.ProductionGuardError, "FP32_REQUIRED"):
            embed.configure_cuda_fp32(
                device_name="cuda:0", dtype_name="float16", batch_size=8,
                torch_module=_MockTorch(),
            )

    def test_batch_other_than_8_is_rejected_without_reduction(self):
        with self.assertRaisesRegex(embed.ProductionGuardError, "BATCH_SIZE_MUST_BE_8"):
            embed.configure_cuda_fp32(
                device_name="cuda:0", dtype_name="float32", batch_size=4,
                torch_module=_MockTorch(),
            )

    def test_full_cohort_filtering_fails_closed(self):
        with self.assertRaisesRegex(embed.ProductionGuardError, "FULL_COHORT_FILTERING_FORBIDDEN"):
            embed.validate_full_cohort_mask(np.array([True, False, True]), 3)

    def test_dataset_hash_count_unique_and_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.bin"
            path.write_bytes(b"fixture")
            expected_hash = hashlib.sha256(b"fixture").hexdigest()
            frame = pd.DataFrame({"appId": ["a", "b"], "genreId": [1, 2]})
            receipt = embed.validate_production_dataset(
                frame,
                data_path=path,
                expected={
                    "sha256": expected_hash,
                    "row_count": 2,
                    "runtime_ordered_id_sha256": hashlib.sha256("a\nb".encode("utf-8")).hexdigest(),
                    "frozen_ordered_id_sha256": hashlib.sha256("a\nb".encode("utf-8")).hexdigest(),
                    "ordered_id_binding_status": "CANONICAL_MATCH",
                },
            )
            self.assertEqual(2, receipt["row_count"])
            self.assertEqual(receipt["runtime_ordered_id_sha256"], receipt["frozen_ordered_id_sha256"])

    def test_dataset_order_mismatch_is_specific(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.bin"
            path.write_bytes(b"fixture")
            frame = pd.DataFrame({"appId": ["b", "a"]})
            with self.assertRaisesRegex(embed.ProductionGuardError, "DATASET_ORDER_MISMATCH"):
                embed.validate_production_dataset(
                    frame,
                    data_path=path,
                    expected={
                        "sha256": hashlib.sha256(b"fixture").hexdigest(),
                        "row_count": 2,
                        "runtime_ordered_id_sha256": hashlib.sha256("a\nb".encode("utf-8")).hexdigest(),
                        "frozen_ordered_id_sha256": "f" * 64,
                        "ordered_id_binding_status": "ORDERED_ID_BINDING_DISCREPANCY_OWNER_ACCEPTED_FOR_THIS_TN2_RECOVERY",
                    },
                )

    def test_ordered_id_discrepancy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.bin"
            path.write_bytes(b"fixture")
            frame = pd.DataFrame({"appId": ["a", "b"]})
            with self.assertRaisesRegex(embed.ProductionGuardError, "ORDERED_ID_BINDING_DISCREPANCY"):
                embed.validate_production_dataset(
                    frame,
                    data_path=path,
                    expected={
                        "sha256": hashlib.sha256(b"fixture").hexdigest(),
                        "row_count": 2,
                        "runtime_ordered_id_sha256": hashlib.sha256(b"a\nb").hexdigest(),
                        "frozen_ordered_id_sha256": "f" * 64,
                        "ordered_id_binding_status": "CANONICAL_MATCH",
                    },
                )

    def test_native_t512_exact_token_contract(self):
        result = embed.audit_native_t512_token_equality(
            tokenizer=_MockTokenizer(),
            prepared_native=["query: a", "query: b"],
            prepared_t512=["query: a", "query: b"],
            batch_size=2,
        )
        self.assertTrue(result["exact_equal"])
        self.assertEqual(result["native_sha256"], result["t512_sha256"])
        self.assertEqual(2, result["tokenizer_batch_size"])
        self.assertTrue(result["input_ids_exact_equal"])
        self.assertTrue(result["attention_mask_exact_equal"])
        self.assertTrue(result["token_type_ids_present"])
        self.assertTrue(result["token_type_ids_exact_equal"])

    def test_native_t512_token_mismatch_fails_closed(self):
        with self.assertRaisesRegex(embed.ProductionGuardError, "NATIVE_T512_TOKEN_CONTRACT_MISMATCH"):
            embed.audit_native_t512_token_equality(
                tokenizer=_MockTokenizer(mutate_t512=True),
                prepared_native=["query: a"],
                prepared_t512=["query: a"],
                batch_size=1,
            )

    def test_token_audit_receipt_requires_bound_complete_evidence(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "TOKEN_AUDIT_TASK_MISMATCH"):
                embed.load_and_validate_token_audit(path, contract)
            digest = "a" * 64
            receipt = {
                "task_id": embed.E5_PRODUCTION_TASK_ID,
                "run_id": embed.E5_PRODUCTION_RUN_ID,
                "checkpoint": embed.E5_CHECKPOINT,
                "revision": embed.E5_REVISION,
                "dataset": {
                    "sha256": contract["datasets"]["tn2"]["sha256"],
                    "ordered_id_sha256": contract["datasets"]["tn2"]["audit_ordered_id_sha256"],
                },
                "asset_inventory": {
                    "checkpoint": embed.E5_CHECKPOINT,
                    "revision": embed.E5_REVISION,
                    "files": [{"path": "tokenizer.json", "size": 1, "sha256": digest}],
                },
                "languages": {
                    language: {
                        "row_count": 13657,
                        "max_length": 512,
                        "tokenizer_batch_size": 8,
                        "prepared_text_sha256": digest,
                        "native_sha256": digest,
                        "t512_sha256": digest,
                        "native_input_ids_sha256": digest,
                        "t512_input_ids_sha256": digest,
                        "input_ids_exact_equal": True,
                        "native_attention_mask_sha256": digest,
                        "t512_attention_mask_sha256": digest,
                        "attention_mask_exact_equal": True,
                        "token_type_ids_present": False,
                        "native_token_type_ids_sha256": None,
                        "t512_token_type_ids_sha256": None,
                        "token_type_ids_exact_equal": True,
                        "exact_equal": True,
                        "mismatch_count": 0,
                    }
                    for language in ("en", "vi")
                },
            }
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(receipt, embed.load_and_validate_token_audit(path, contract))

    def test_production_invocation_binds_exact_condition(self):
        contract = self._bind_authorization(embed.load_production_contract(Path("pipeline/e5_production_contract.json")))
        args = mock.Mock(exp="tn2", trunc="native", lang=["en", "vi"], max_tokens=None)
        units = embed.validate_production_invocation(args, contract, None)
        self.assertEqual(["TN2_E5_EN_NATIVE", "TN2_E5_VI_NATIVE"], [unit["name"] for unit in units])
        with self.assertRaisesRegex(embed.ProductionGuardError, "PRODUCTION_N_CHARS_MISMATCH"):
            embed.validate_production_invocation(args, contract, 950)

    def test_production_invocation_rejects_partial_or_duplicate_languages(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        for languages in (["en"], ["en", "en", "vi"], ["vi", "en"]):
            args = mock.Mock(exp="tn2", trunc="native", lang=languages, max_tokens=None)
            with self.assertRaisesRegex(embed.ProductionGuardError, "PRODUCTION_LANGUAGE_SET_MISMATCH"):
                embed.validate_production_invocation(args, contract, None)

    def test_output_metadata_is_bound_to_exact_unit_condition(self):
        contract = self._bind_authorization(embed.load_production_contract(Path("pipeline/e5_production_contract.json")))
        unit = contract["outputs"]["units"][0]
        metadata = {
            "experiment": "tn2",
            "representation": embed.E5_REPRESENTATION_ID,
            "language": "en",
            "truncation_mode": "native",
            "n_chars": None,
            "max_tokens_requested": None,
            "hf_name": embed.E5_CHECKPOINT,
            "revision": embed.E5_REVISION,
            "pooling": contract["model"]["pooling"],
            "clustering_prefix": contract["model"]["clustering_prefix"],
            "embedding_dim": 1024,
            "embedding_dtype": "float32",
            "l2_normalized": True,
            "n_rows": 13657,
            "dataset_sha256": contract["datasets"]["tn2"]["sha256"],
            "app_ids_sha256": unit["ordered_id_sha256"],
            "runtime_ordered_id_sha256": contract["datasets"]["tn2"]["runtime_ordered_id_sha256"],
            "frozen_ordered_id_sha256": contract["datasets"]["tn2"]["frozen_ordered_id_sha256"],
            "ordered_id_binding_status": contract["datasets"]["tn2"]["ordered_id_binding_status"],
            "execution_authorization_sha256": "a" * 64,
        }
        embed.validate_unit_metadata(metadata, unit, contract)
        metadata["n_chars"] = 949
        with self.assertRaisesRegex(embed.ProductionGuardError, "METADATA_CONDITION_MISMATCH"):
            embed.validate_unit_metadata(metadata, unit, contract)

    def test_prepared_text_mismatch_fails_before_tokenizer(self):
        tokenizer = mock.Mock()
        with self.assertRaisesRegex(embed.ProductionGuardError, "NATIVE_T512_PREPARED_TEXT_MISMATCH"):
            embed.audit_native_t512_token_equality(
                tokenizer=tokenizer,
                prepared_native=["query: a"],
                prepared_t512=["query: a "],
            )
        tokenizer.assert_not_called()

    def test_historical_e5_preprocessing_contract_is_preserved(self):
        spec = {"needs_word_segmentation": False, "clustering_prefix": "query: "}
        raw = ["  alpha  ", "beta"]
        self.assertEqual(
            ["query: alpha", "query: beta"],
            embed.prepare_neural_texts(raw, spec, "native", None),
        )

    def test_output_collision_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "unit"
            base.with_suffix(".npy").write_bytes(b"existing")
            with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                embed.require_outputs_absent([base])

    def test_multi_output_collision_is_preflighted_as_one_set(self):
        with tempfile.TemporaryDirectory() as temp:
            en_base = Path(temp) / "en"
            vi_base = Path(temp) / "vi"
            vi_base.with_suffix(".npy").write_bytes(b"existing")
            with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                embed.require_outputs_absent([en_base, vi_base])
            self.assertFalse(any(path.exists() for path in embed.output_artifact_paths(en_base)))

    def test_create_only_json_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            embed.write_json_create_only(path, {"status": "FIRST"})
            with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                embed.write_json_create_only(path, {"status": "SECOND"})
            self.assertEqual("FIRST", json.loads(path.read_text(encoding="utf-8"))["status"])

    def test_atomic_namespace_collision_never_activates_or_mutates_existing_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp) / "existing"
            namespace.mkdir()
            marker = namespace / "marker.bin"
            marker.write_bytes(b"immutable")
            old_namespace = embed.ACTIVE_PRODUCTION_NAMESPACE
            old_phase = embed.ACTIVE_PRODUCTION_PHASE
            embed.ACTIVE_PRODUCTION_NAMESPACE = None
            embed.ACTIVE_PRODUCTION_PHASE = None
            try:
                with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                    embed.activate_new_production_namespace(namespace, "TOKEN_AUDIT")
                self.assertIsNone(embed.ACTIVE_PRODUCTION_NAMESPACE)
                self.assertIsNone(embed.ACTIVE_PRODUCTION_PHASE)
                self.assertEqual(b"immutable", marker.read_bytes())
                self.assertFalse((namespace / "failure_receipt.json").exists())
            finally:
                embed.ACTIVE_PRODUCTION_NAMESPACE = old_namespace
                embed.ACTIVE_PRODUCTION_PHASE = old_phase

    def test_incomplete_outputs_cannot_emit_success_receipt(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            audit_path = namespace / "token_audit/native_t512_equality.json"
            audit_path.parent.mkdir(parents=True)
            audit_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_SET_INCOMPLETE"):
                embed.finalize_production_run(namespace, contract, self._authorization())
            self.assertFalse((namespace / "execution_receipt.json").exists())

    def test_finalization_collision_causes_zero_mutation(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            collision = namespace / "run_manifest.json"
            collision.write_text("existing", encoding="utf-8")
            with mock.patch.object(embed, "validate_production_outputs", return_value={"units": [], "unit_count": 4, "token_audit_receipt_sha256": "a" * 64, "dataset_binding": {}}), mock.patch.object(
                embed, "inventory_hf_assets", return_value={"files": []}
            ) as inventory:
                with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                    embed.finalize_production_run(namespace, contract, self._authorization())
            inventory.assert_not_called()
            self.assertFalse((namespace / "asset_inventory.json").exists())
            self.assertFalse((namespace / "execution_receipt.json").exists())

    def test_preexisting_asset_inventory_is_not_trusted(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            (namespace / "asset_inventory.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(embed, "validate_production_outputs", return_value={"units": [], "unit_count": 4, "token_audit_receipt_sha256": "a" * 64, "dataset_binding": {}}), mock.patch.object(
                embed, "inventory_hf_assets"
            ) as inventory:
                with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                    embed.finalize_production_run(namespace, contract, self._authorization())
            inventory.assert_not_called()
            self.assertFalse((namespace / "execution_receipt.json").exists())

    def test_success_manifest_binds_asset_inventory_bytes(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            with mock.patch.object(embed, "validate_production_outputs", return_value={"units": [], "unit_count": 4, "token_audit_receipt_sha256": "a" * 64, "dataset_binding": {"ordered_id_binding_status": "CANONICAL_MATCH"}}), mock.patch.object(
                embed, "inventory_hf_assets", return_value={"checkpoint": embed.E5_CHECKPOINT, "files": []}
            ):
                embed.finalize_production_run(namespace, contract, self._authorization())
            manifest = json.loads((namespace / "run_manifest.json").read_text(encoding="utf-8"))
            inventory_entry = next(
                item for item in manifest["artifacts"] if item["path"] == "asset_inventory.json"
            )
            self.assertEqual(embed.sha256(namespace / "asset_inventory.json"), inventory_entry["sha256"])
            receipt = json.loads((namespace / "execution_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(embed.sha256(namespace / "run_manifest.json"), receipt["run_manifest_sha256"])
            self.assertEqual("TARGETED_TN2_RECOVERY_COMPLETE", receipt["status"])
            self.assertEqual("TN2_FOUR_NAMED_UNITS_ONLY", receipt["experiment_scope"])

    def test_audit_only_namespace_requires_exact_three_evidence_files(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        contract = json.loads(json.dumps(contract))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            namespace = Path(temp) / "namespace"
            contract["paths"]["project_root"] = root.as_posix()
            contract["paths"]["model_cache"] = (root / "cache").as_posix()
            snapshot_file = root / "cache/models--intfloat--multilingual-e5-large/snapshots" / embed.E5_REVISION / "tokenizer.json"
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file.write_text("fixture", encoding="utf-8")
            source_paths = {
                "pipeline/embed.py": Path(embed.__file__).resolve(),
                "pipeline/e5_production_contract.json": root / "pipeline/e5_production_contract.json",
                "pipeline/models.yaml": root / "pipeline/models.yaml",
                "revision_v4_20261020/00_governance/PROTOCOL_LOCK.json": root / "revision_v4_20261020/00_governance/PROTOCOL_LOCK.json",
                "revision_v4_20261020/00_governance/GOV02_EXPERIMENT_MATRIX_V4_DRAFT.jsonl": root / "revision_v4_20261020/00_governance/GOV02_EXPERIMENT_MATRIX_V4_DRAFT.jsonl",
            }
            for relative, source in source_paths.items():
                if relative == "pipeline/embed.py":
                    continue
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(relative, encoding="utf-8")
            bindings = {relative: embed.sha256(source) for relative, source in source_paths.items()}
            authorization = {"sha256": "c" * 64, "receipt": {"source_sha256": {**bindings, "tests/test_e5_production_guards.py": "d" * 64}}}
            (namespace / "token_audit").mkdir(parents=True)
            digest = "a" * 64
            audit = {
                "task_id": embed.E5_PRODUCTION_TASK_ID,
                "run_id": embed.E5_PRODUCTION_RUN_ID,
                "checkpoint": embed.E5_CHECKPOINT,
                "revision": embed.E5_REVISION,
                "dataset": {
                    "sha256": contract["datasets"]["tn2"]["sha256"],
                    "ordered_id_sha256": contract["datasets"]["tn2"]["audit_ordered_id_sha256"],
                },
                "asset_inventory": embed.inventory_hf_assets(contract),
                "execution_authorization_sha256": authorization["sha256"],
                "languages": {
                    language: {
                        "row_count": 13657, "max_length": 512, "tokenizer_batch_size": 8,
                        "prepared_text_sha256": digest, "native_sha256": digest,
                        "t512_sha256": digest,
                        "native_input_ids_sha256": digest, "t512_input_ids_sha256": digest, "input_ids_exact_equal": True,
                        "native_attention_mask_sha256": digest, "t512_attention_mask_sha256": digest, "attention_mask_exact_equal": True,
                        "token_type_ids_present": False, "native_token_type_ids_sha256": None,
                        "t512_token_type_ids_sha256": None, "token_type_ids_exact_equal": True,
                        "exact_equal": True, "mismatch_count": 0,
                    }
                    for language in ("en", "vi")
                },
            }
            (namespace / "token_audit/native_t512_equality.json").write_text(json.dumps(audit), encoding="utf-8")
            (namespace / "environment.json").write_text(json.dumps({
                "device": "cuda:0", "dtype": "float32", "batch_size": 8, "autocast": False,
                "execution_authorization_sha256": authorization["sha256"],
            }), encoding="utf-8")
            valid_manifest = "".join(f"{value}  {key}\n" for key, value in bindings.items())
            (namespace / "source_manifest.sha256").write_text(valid_manifest, encoding="utf-8")
            evidence = embed.validate_audit_only_namespace(namespace, contract, authorization)
            self.assertEqual(embed.sha256(namespace / "token_audit/native_t512_equality.json"), evidence["receipt_sha256"])
            (namespace / "source_manifest.sha256").write_text(
                "prose containing " + ("b" * 64) + " but no binding\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(embed.ProductionGuardError, "TOKEN_AUDIT_SOURCE_MANIFEST_INVALID"):
                embed.validate_audit_only_namespace(namespace, contract, authorization)
            (namespace / "source_manifest.sha256").write_text(
                ("b" * 64) + "  /evil/unapproved-root/pipeline/embed.py\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(embed.ProductionGuardError, "TOKEN_AUDIT_SOURCE_SET_MISMATCH"):
                embed.validate_audit_only_namespace(namespace, contract, authorization)
            (namespace / "source_manifest.sha256").write_text(valid_manifest, encoding="utf-8")
            (namespace / "unexpected.txt").write_text("collision", encoding="utf-8")
            with self.assertRaisesRegex(embed.ProductionGuardError, "AUDIT_NAMESPACE_NOT_CLEAN"):
                embed.validate_audit_only_namespace(namespace, contract, authorization)

    def test_t512_reuse_is_create_only_and_byte_identical(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            for unit in contract["outputs"]["units"]:
                if unit["truncation_mode"] != "native":
                    continue
                base = namespace / "embeddings" / unit["stem"]
                base.parent.mkdir(parents=True, exist_ok=True)
                base.with_suffix(".npy").write_bytes((unit["language"] + "-vectors").encode())
                base.with_suffix(".rows.parquet").write_bytes((unit["language"] + "-rows").encode())
                base.with_suffix(".json").write_text(json.dumps({
                    "experiment": "tn2", "representation": embed.E5_REPRESENTATION_ID,
                    "language": unit["language"], "truncation_mode": "native",
                    "n_chars": None, "max_tokens_requested": None,
                }), encoding="utf-8")
            evidence = {"receipt_sha256": "a" * 64}
            embed.materialize_t512_reuse(namespace, contract, evidence, self._authorization())
            for unit in contract["outputs"]["units"]:
                if unit["truncation_mode"] != "token_budget":
                    continue
                native = next(candidate for candidate in contract["outputs"]["units"] if candidate["language"] == unit["language"] and candidate["truncation_mode"] == "native")
                target_base = namespace / "embeddings" / unit["stem"]
                source_base = namespace / "embeddings" / native["stem"]
                self.assertEqual(target_base.with_suffix(".npy").read_bytes(), source_base.with_suffix(".npy").read_bytes())
                self.assertEqual(target_base.with_suffix(".rows.parquet").read_bytes(), source_base.with_suffix(".rows.parquet").read_bytes())
                metadata = json.loads(target_base.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual("REUSED_AFTER_EXACT_TOKEN_EQUIVALENCE", metadata["reuse_status"])
            with self.assertRaisesRegex(embed.ProductionGuardError, "OUTPUT_COLLISION"):
                embed.materialize_t512_reuse(namespace, contract, evidence, self._authorization())

    def test_native_invocation_excludes_t512_forward_units(self):
        contract = embed.load_production_contract(Path("pipeline/e5_production_contract.json"))
        units = embed.expected_units_for_invocation(contract, exp="tn2", trunc="native")
        self.assertEqual(2, len(units))
        self.assertTrue(all(unit["truncation_mode"] == "native" for unit in units))
        self.assertFalse(any(unit["experiment"] == "tn1" for unit in contract["outputs"]["units"]))

    def test_t512_metadata_must_bind_actual_audit_receipt_hash(self):
        contract = self._bind_authorization(embed.load_production_contract(Path("pipeline/e5_production_contract.json")))
        contract = json.loads(json.dumps(contract))
        row_hash = hashlib.sha256(b"a").hexdigest()
        contract["datasets"]["tn2"]["runtime_ordered_id_sha256"] = row_hash
        contract["datasets"]["tn2"]["ordered_id_sha256"] = row_hash
        contract["datasets"]["tn2"]["row_count"] = 1
        for unit in contract["outputs"]["units"]:
            unit["shape"] = [1, 2]
            unit["ordered_id_sha256"] = row_hash
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp)
            audit_path = namespace / "token_audit/native_t512_equality.json"
            audit_path.parent.mkdir(parents=True)
            audit_path.write_text("{}", encoding="utf-8")
            for unit in contract["outputs"]["units"]:
                base = namespace / "embeddings" / unit["stem"]
                base.parent.mkdir(parents=True, exist_ok=True)
                payload = (unit["language"] + "-bytes").encode()
                base.with_suffix(".npy").write_bytes(payload)
                base.with_suffix(".rows.parquet").write_bytes(payload)
                metadata = {
                    "experiment": "tn2", "representation": embed.E5_REPRESENTATION_ID,
                    "language": unit["language"], "truncation_mode": unit["truncation_mode"],
                    "n_chars": unit["n_chars"], "max_tokens_requested": unit["max_tokens_requested"],
                    "hf_name": embed.E5_CHECKPOINT, "revision": embed.E5_REVISION,
                    "pooling": contract["model"]["pooling"],
                    "clustering_prefix": contract["model"]["clustering_prefix"],
                    "embedding_dim": 2, "embedding_dtype": "float32", "l2_normalized": True,
                    "n_rows": 1, "dataset_sha256": contract["datasets"]["tn2"]["sha256"],
                    "app_ids_sha256": row_hash, "runtime_ordered_id_sha256": row_hash,
                    "frozen_ordered_id_sha256": contract["datasets"]["tn2"]["frozen_ordered_id_sha256"],
                    "ordered_id_binding_status": contract["datasets"]["tn2"]["ordered_id_binding_status"],
                    "execution_authorization_sha256": "a" * 64,
                }
                if unit["truncation_mode"] == "token_budget":
                    native = next(candidate for candidate in contract["outputs"]["units"] if candidate["language"] == unit["language"] and candidate["truncation_mode"] == "native")
                    native_base = namespace / "embeddings" / native["stem"]
                    metadata.update({
                        "reuse_status": "REUSED_AFTER_EXACT_TOKEN_EQUIVALENCE",
                        "reused_from_stem": native["stem"],
                        "reused_source_npy_sha256": embed.sha256(native_base.with_suffix(".npy")),
                        "reused_source_rows_sha256": embed.sha256(native_base.with_suffix(".rows.parquet")),
                        "token_audit_receipt_sha256": "f" * 64,
                        "execution_authorization_sha256": "a" * 64,
                    })
                base.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            vectors = np.array([[1.0, 0.0]], dtype=np.float32)
            rows = pd.DataFrame({"appId": ["a"]})
            with mock.patch.object(embed.np, "load", return_value=vectors), mock.patch.object(
                embed.pd, "read_parquet", return_value=rows
            ):
                with self.assertRaisesRegex(embed.ProductionGuardError, "T512_REUSE_AUDIT_HASH_MISMATCH"):
                    embed.validate_production_outputs(namespace, contract)

    def test_no_real_model_or_tokenizer_factory_is_called_by_unit_tests(self):
        with mock.patch.object(embed.AutoTokenizer, "from_pretrained") as tok, mock.patch.object(
            embed.AutoModel, "from_pretrained"
        ) as model:
            embed.configure_cuda_fp32(
                device_name="cuda:0", dtype_name="float32", batch_size=8,
                torch_module=_MockTorch(),
            )
            tok.assert_not_called()
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()

