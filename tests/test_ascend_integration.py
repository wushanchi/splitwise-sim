"""Subprocess tests for synthetic NPU paths and the original GPU baseline."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_python(*args):
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=60,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HYDRA_FULL_ERROR": "1"})


class AscendIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="splitwise-ascend-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name)
        run = run_python("scripts/run_ascend_examples.py", "--output-dir", str(cls.output / "demos"))
        if run.returncode:
            raise AssertionError(run.stdout + run.stderr)

    def test_aggregate_completes_on_two_instances_and_accepts_time_zero(self):
        output = self.output / "demos" / "aggregate"
        requests = read_csv(output / "detailed" / "0.csv")
        nodes = read_csv(output / "request_nodes.csv")
        self.assertEqual(len(requests), 8)
        self.assertEqual({row["node_type"] for row in nodes}, {"PROMPT", "TOKEN"})
        prompts = [row for row in nodes if row["node_type"] == "PROMPT"]
        self.assertEqual(len({row["runner"] for row in prompts}), 2)
        first = next(row for row in prompts if float(row["request_id"]) == 0)
        self.assertEqual(float(first["start_timestamp"]), 0.0)
        self.assertFalse((output / "oom.csv").exists())

    def test_pd_transfers_follow_dag_and_share_a_serial_link(self):
        output = self.output / "demos" / "pd"
        requests = read_csv(output / "detailed" / "0.csv")
        nodes = read_csv(output / "request_nodes.csv")
        self.assertEqual(len(requests), 8)
        by_request = {}
        for row in nodes:
            by_request.setdefault(row["request_id"], {})[row["node_type"]] = row
        for request in by_request.values():
            self.assertEqual(set(request), {"PROMPT", "KVCacheTransfer", "TOKEN"})
            p, f, d = (request[name] for name in ("PROMPT", "KVCacheTransfer", "TOKEN"))
            self.assertNotEqual(p["runner"], d["runner"])
            self.assertLessEqual(float(p["completion_timestamp"]), float(f["start_timestamp"]) + 1e-12)
            self.assertLessEqual(float(f["completion_timestamp"]), float(d["start_timestamp"]) + 1e-12)
            self.assertGreater(float(f["completion_timestamp"]), float(f["start_timestamp"]))
        flows = sorted((row for row in nodes if row["node_type"] == "KVCacheTransfer"),
                       key=lambda row: float(row["start_timestamp"]))
        self.assertEqual(len(flows), 8)
        self.assertEqual(len({row["runner"] for row in flows}), 1)
        for previous, following in zip(flows, flows[1:]):
            self.assertLessEqual(float(previous["completion_timestamp"]), float(following["start_timestamp"]) + 1e-12)

    def test_default_entry_refuses_synthetic_profile(self):
        output = self.output / "refused"
        run = run_python("run.py", "--config-name", "config_ascend", f"output_dir={output.as_posix()}")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("synthetic", (run.stdout + run.stderr).lower())
        self.assertFalse((output / "summary.csv").exists())

    def test_exact_profile_updates_decode_context_each_iteration(self):
        fixture = self.output / "exact_context"
        fixture.mkdir()
        trace = fixture / "trace.csv"
        trace.write_text("request_id,request_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size\n"
                         "0,2,0,0,1,2,4\n", encoding="utf-8")
        profile = fixture / "profile.csv"
        profile.write_text(
            "model,hardware,tensor_parallel,phase,batch_size,batch_tokens,context_tokens,latency_ms\n"
            "synthetic-llm,ascend-generic,1,prefill,1,2,0,10\n"
            "synthetic-llm,ascend-generic,1,decode,1,1,2,1\n"
            "synthetic-llm,ascend-generic,1,decode,1,1,3,2\n"
            "synthetic-llm,ascend-generic,1,decode,1,1,4,4\n", encoding="utf-8")
        metadata_path = fixture / "profile.metadata.json"
        metadata_path.write_text(json.dumps({"schema_version": 1, "provenance": "synthetic",
                                             "description": "Exact context-change integration fixture",
                                             "latency_unit": "ms"}), encoding="utf-8")
        output = fixture / "output"
        run = run_python("run.py", "--config-name", "config_ascend", "cluster.servers.0.count=1",
                         f"trace.path={trace.as_posix()}", f"performance_model.profile_path={profile.as_posix()}",
                         f"performance_model.metadata_path={metadata_path.as_posix()}",
                         "performance_model.allow_synthetic=True", "performance_model.lookup_mode=exact",
                         f"output_dir={output.as_posix()}")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        requests = read_csv(output / "detailed" / "0.csv")
        self.assertEqual(len(requests), 1)
        # Prefill creates token 1, then contexts 2/3/4 cost 1/2/4 ms.
        # Reusing the first decode latency would incorrectly produce 13 ms.
        self.assertAlmostEqual(float(requests[0]["response_times"]), 0.017, places=12)
        metadata = json.loads((output / "simulation_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["performance_model"]["nearest_lookup_count"], 0)
        self.assertTrue(metadata["all_requests_completed"])

    def test_outputs_preserve_synthetic_provenance_and_approximation_counts(self):
        for case in ("aggregate", "pd"):
            with self.subTest(case=case):
                path = self.output / "demos" / case / "simulation_metadata.json"
                metadata = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["execution"], "cpu_discrete_event_simulation")
                self.assertEqual(metadata["run_status"], "completed")
                self.assertEqual(metadata["num_input_requests"], 8)
                self.assertEqual(metadata["num_completed_requests"], 8)
                self.assertTrue(metadata["all_requests_completed"])
                self.assertFalse(metadata["power_model"]["enabled"])
                profile = metadata["performance_model"]
                self.assertEqual(profile["provenance"], "synthetic")
                self.assertEqual(profile["lookup_mode"], "nearest")
                self.assertFalse(profile["supports_iteration_coalescing"])
                self.assertGreater(profile["nearest_lookup_count"], 0)
                self.assertEqual(len(profile["profile_sha256"]), 64)
        pd_metadata = json.loads((self.output / "demos" / "pd" / "simulation_metadata.json").read_text(encoding="utf-8"))
        link = pd_metadata["schedulers"]["0"]
        self.assertEqual(link["link_count"], 1)
        self.assertEqual(link["contention_scope"], "directed_instance_pair")
        self.assertFalse(link["shared_nic_or_switch_contention"])

    def test_original_gpu_baseline_still_completes(self):
        # Replace launcher only in a disposable copy; preserve GPU data/configs.
        config_dir = self.output / "gpu_configs"
        shutil.copytree(REPO / "configs", config_dir)
        config = OmegaConf.load(config_dir / "config.yaml")
        for item in config.defaults:
            if not isinstance(item, str) and "override hydra/launcher" in item:
                item["override hydra/launcher"] = "basic"
        del config.hydra.launcher
        OmegaConf.save(config, config_dir / "config.yaml")
        output = self.output / "gpu_regression"
        run = run_python("run.py", "--config-path", config_dir.as_posix(),
                         "cluster=half_half", "cluster.servers.0.count=0", "cluster.servers.1.count=1",
                         "start_state=baseline", "applications.0.scheduler=token_jsq", "performance_model=db",
                         "trace.filename=test_trace", "debug=True", f"output_dir={output.as_posix()}")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(len(read_csv(output / "detailed" / "0.csv")), 3)
        nodes = read_csv(output / "request_nodes.csv")
        self.assertEqual(len(nodes), 6)
        self.assertEqual({row["node_type"] for row in nodes}, {"PROMPT", "TOKEN"})
        self.assertEqual({row["runner"] for row in nodes}, {"h100-80gb_0"})
        self.assertFalse((output / "oom.csv").exists())


if __name__ == "__main__":
    unittest.main()
