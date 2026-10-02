"""Contract checks for externally supplied iteration profiles; no accelerator needed."""

import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from profiled_performance_model import ProfiledPerformanceModel
from task import PromptTask, TokenTask


class ProfiledPerformanceModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile_path = Path(self.temp.name) / "profile.csv"
        self.metadata_path = Path(self.temp.name) / "profile.json"
        self.metadata = {
            "schema_version": 1,
            "provenance": "measured",
            "description": "Unit-test fixture, not an actual hardware measurement.",
            "latency_unit": "ms",
            "source": "Temporary unit-test fixture",
            "environment": {"fixture": True},
        }
        self.rows = [self.row("prefill", 1, 16, 0, 4)]
        self.instance = SimpleNamespace(
            processors=[SimpleNamespace(name="test-npu")],
            model=SimpleNamespace(
                name="test-model",
                parallelism=SimpleNamespace(tensor_parallelism=1, pipeline_parallelism=1),
            ),
        )

    @staticmethod
    def row(phase, batch_size, batch_tokens, context_tokens, latency_ms):
        return dict(model="test-model", hardware="test-npu", tensor_parallel=1,
                    phase=phase, batch_size=batch_size, batch_tokens=batch_tokens,
                    context_tokens=context_tokens, latency_ms=latency_ms)

    def make_model(self, **kwargs):
        with self.profile_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ProfiledPerformanceModel.COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        return ProfiledPerformanceModel(self.profile_path, self.metadata_path, **kwargs)

    @staticmethod
    def prompt(length=16, node_id=0):
        return PromptTask(node_id=node_id, prompt_size=length,
                          request=SimpleNamespace(batch_size=1, prompt_size=length, processed_tokens=0))

    @staticmethod
    def decode(context, node_id=1):
        return TokenTask(node_id=node_id, token_size=8,
                         request=SimpleNamespace(batch_size=1, prompt_size=16, processed_tokens=context))

    def test_duplicates_use_median_and_convert_milliseconds_to_seconds(self):
        self.rows = [self.row("prefill", 1, 16, 0, latency) for latency in (2, 100, 6)]
        model = self.make_model()
        self.assertAlmostEqual(model.get_iteration_duration([self.prompt()], self.instance), 0.006)
        metadata = model.get_metadata()
        self.assertEqual(metadata["profile_row_count"], 3)
        self.assertEqual(metadata["profile_shape_count"], 1)
        self.assertEqual(metadata["exact_lookup_count"], 1)
        self.assertEqual(metadata["nearest_lookup_count"], 0)

    def test_decode_context_is_requeried_after_each_iteration(self):
        self.rows = [self.row("decode", 1, 1, 16, 2), self.row("decode", 1, 1, 17, 3)]
        model = self.make_model()
        task = self.decode(16)
        self.assertAlmostEqual(model.get_iteration_duration([task], self.instance), 0.002)
        task.request.processed_tokens += 1
        self.assertAlmostEqual(model.get_iteration_duration([task], self.instance), 0.003)
        self.assertTrue(model.context_sensitive)
        self.assertFalse(model.supports_iteration_coalescing)

    def test_mixed_shape_counts_only_decode_context(self):
        self.rows = [self.row("mixed", 3, 18, 40, 7)]
        model = self.make_model()
        prompt = self.prompt(16)
        prompt.request.processed_tokens = 999  # Never included in decode context.
        batch = [prompt, self.decode(16, 1), self.decode(24, 2)]
        self.assertAlmostEqual(model.get_iteration_duration(batch, self.instance), 0.007)

    def test_default_exact_refuses_an_unmeasured_shape(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "No exact profile"):
            model.get_iteration_duration([self.prompt(17)], self.instance)
        self.assertEqual(model.get_metadata()["nearest_lookup_count"], 0)

    def test_nearest_is_explicit_warns_once_and_counts_each_approximation(self):
        self.rows = [self.row("decode", 1, 1, 12, 8), self.row("decode", 1, 1, 8, 2)]
        model = self.make_model(lookup_mode="nearest")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Equal distances resolve deterministically to the smaller shape.
            self.assertAlmostEqual(model.get_iteration_duration([self.decode(10)], self.instance), 0.002)
            self.assertAlmostEqual(model.get_iteration_duration([self.decode(11)], self.instance), 0.008)
            self.assertAlmostEqual(model.get_iteration_duration([self.decode(12)], self.instance), 0.008)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertEqual(model.get_metadata()["nearest_lookup_count"], 2)
        self.assertEqual(model.get_metadata()["exact_lookup_count"], 1)

    def test_nearest_never_crosses_model_hardware_tp_or_phase(self):
        model = self.make_model(lookup_mode="nearest")
        for field in ("model", "hardware", "tp", "phase"):
            with self.subTest(field=field):
                instance = SimpleNamespace(
                    processors=[SimpleNamespace(name="other" if field == "hardware" else "test-npu")
                                for _ in range(2 if field == "tp" else 1)],
                    model=SimpleNamespace(
                        name="other" if field == "model" else "test-model",
                        parallelism=SimpleNamespace(tensor_parallelism=2 if field == "tp" else 1,
                                                    pipeline_parallelism=1),
                    ),
                )
                task = self.decode(16) if field == "phase" else self.prompt()
                with self.assertRaisesRegex(ValueError, "No profile for"):
                    model.get_iteration_duration([task], instance)

    def test_synthetic_profiles_require_explicit_permission(self):
        self.metadata["provenance"] = "synthetic"
        del self.metadata["source"]
        del self.metadata["environment"]
        with self.assertRaisesRegex(ValueError, "Synthetic profile refused"):
            self.make_model()
        model = self.make_model(allow_synthetic=True)
        self.assertEqual(model.get_metadata()["provenance"], "synthetic")

    def test_measured_metadata_requires_only_documented_source_and_environment(self):
        self.make_model()  # Minimal documented measured metadata is accepted.
        for key, bad in (("source", ""), ("source", None), ("environment", {}),
                         ("environment", "unknown"), ("description", " "),
                         ("latency_unit", "s"), ("schema_version", 2),
                         ("schema_version", True), ("provenance", "unknown")):
            with self.subTest(key=key, bad=bad):
                previous = self.metadata[key]
                self.metadata[key] = bad
                with self.assertRaises(ValueError):
                    self.make_model()
                self.metadata[key] = previous

    def test_invalid_csv_dimensions_and_latencies_are_rejected(self):
        invalid = (
            ("model", ""), ("hardware", " "), ("tensor_parallel", 0),
            ("tensor_parallel", "1.0"), ("batch_size", -1), ("batch_tokens", "1.5"),
            ("context_tokens", -1), ("phase", "attention"),
            ("latency_ms", 0), ("latency_ms", -1), ("latency_ms", "NaN"),
            ("latency_ms", "inf"), ("latency_ms", "-inf"), ("latency_ms", "bad"),
        )
        for key, bad in invalid:
            with self.subTest(key=key, bad=bad):
                self.rows = [self.row("prefill", 1, 16, 0, 4)]
                self.rows[0][key] = bad
                with self.assertRaises(ValueError):
                    self.make_model()

    def test_inconsistent_phase_shapes_and_empty_profiles_are_rejected(self):
        invalid_rows = (
            [], [self.row("prefill", 1, 16, 1, 4)],
            [self.row("decode", 2, 3, 32, 4)],
            [self.row("mixed", 1, 16, 16, 4)],
            [self.row("prefill", 3, 2, 0, 4)],
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows):
                self.rows = rows
                with self.assertRaises(ValueError):
                    self.make_model()

    def test_missing_extra_duplicate_columns_and_short_rows_are_rejected(self):
        self.make_model()
        header = ",".join(ProfiledPerformanceModel.COLUMNS)
        contents = (
            "model,latency_ms\ntest-model,4\n",
            header + ",extra\ntest-model,test-npu,1,prefill,1,16,0,4,x\n",
            header.replace("hardware", "model") + "\n",
            header + "\ntest-model,test-npu,1,prefill,1,16,0\n",
            header + "\ntest-model,test-npu,1,prefill,1,16,0,4,extra\n",
        )
        for content in contents:
            with self.subTest(content=content):
                self.profile_path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    ProfiledPerformanceModel(self.profile_path, self.metadata_path)

    def test_unsupported_runtime_configurations_fail_clearly(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "empty"):
            model.get_iteration_duration([], self.instance)
        with self.assertRaisesRegex(ValueError, "Unsupported iteration task"):
            model.get_iteration_duration([object()], self.instance)
        self.instance.model.parallelism.pipeline_parallelism = 2
        with self.assertRaisesRegex(ValueError, "pipeline_parallelism=1"):
            model.get_iteration_duration([self.prompt()], self.instance)
        self.instance.model.parallelism.pipeline_parallelism = 1
        self.instance.model.parallelism.tensor_parallelism = 2
        with self.assertRaisesRegex(ValueError, "processor count"):
            model.get_iteration_duration([self.prompt()], self.instance)
        self.instance.model.parallelism.tensor_parallelism = 1
        packed_request = self.prompt()
        packed_request.request.batch_size = 2
        with self.assertRaisesRegex(ValueError, "request.batch_size=1"):
            model.get_iteration_duration([packed_request], self.instance)
        self.instance.processors.append(SimpleNamespace(name="other-npu"))
        with self.assertRaisesRegex(ValueError, "one hardware name"):
            model.get_iteration_duration([self.prompt()], self.instance)
        with self.assertRaisesRegex(NotImplementedError, "iteration scheduling"):
            model.get_duration(self.prompt(), [self.prompt()], self.instance)

    def test_metadata_snapshot_is_serializable_and_does_not_leak_mutability(self):
        model = self.make_model()
        snapshot = model.get_metadata()
        json.dumps(snapshot)
        self.assertEqual(len(snapshot["profile_sha256"]), 64)
        self.assertEqual(len(snapshot["metadata_sha256"]), 64)
        snapshot["profile_metadata"]["environment"]["fixture"] = False
        self.assertTrue(model.get_metadata()["profile_metadata"]["environment"]["fixture"])

    def test_invalid_lookup_and_permission_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "lookup_mode"):
            self.make_model(lookup_mode="interpolate")
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.make_model(allow_synthetic="false")


if __name__ == "__main__":
    unittest.main()
