"""Resource and context contracts shared by the Ascend configurations."""
import unittest
import math
from types import SimpleNamespace
from unittest.mock import patch

import performance_model
from instance import ORCAInstance
from model import LLMArchitecture
from power_model import DisabledPowerModel
from processor import NPU, ProcessorType
from request import GenerativeLLMRequest
from simulator import Simulator, TraceSimulator
from utils import get_statistics


class KVCapacityTests(unittest.TestCase):
    def architecture(self, **kwargs):
        return LLMArchitecture(name="example", num_layers=8,
                               hidden_size=512, num_heads=8, **kwargs)

    def test_default_preserves_mha(self):
        self.assertEqual(self.architecture().get_kv_cache_bytes_per_token(2),
                         2 * 512 * 8 * 2)

    def test_gqa_and_independent_kv_dtype(self):
        architecture = self.architecture(num_kv_heads=2, head_dim=64, kv_dtype_size=1)
        self.assertEqual(architecture.get_kv_cache_bytes_per_token(2), 2048)

    def test_opaque_cache_layout(self):
        architecture = self.architecture(kv_cache_bytes_per_token=1024)
        request = GenerativeLLMRequest(request_id=1, application_id=0,
                                       request_type=2, batch_size=2,
                                       prompt_size=128, token_size=8)
        model = SimpleNamespace(architecture=architecture,
                                size=SimpleNamespace(dtype_size=2))
        self.assertEqual(request.estimate_kv_cache_size(128, model), 262144)

    def test_invalid_cache_configuration(self):
        for kwargs in ({"num_kv_heads": 3}, {"head_dim": 0},
                       {"kv_dtype_size": float("nan")},
                       {"kv_cache_bytes_per_token": -1},
                       {"kv_cache_bytes_per_token": 4, "num_kv_heads": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.architecture(**kwargs)


class ExecutionContractTests(unittest.TestCase):
    def test_no_completions_produce_missing_latency_statistics(self):
        statistics = get_statistics([])
        self.assertIn("p99", statistics)
        self.assertTrue(all(math.isnan(value) for value in statistics.values()))

    def test_npu_is_a_resource_without_runtime_dependency(self):
        npu = NPU(name="ascend-generic", memory_size=64 * 1024 ** 3)
        self.assertEqual(npu.processor_type, ProcessorType.NPU)
        self.assertEqual(npu.memory_free, 64 * 1024 ** 3)

    def test_disabled_power_does_not_assume_dgx_h100(self):
        model = DisabledPowerModel()
        self.assertEqual(model.get_server_idle_power(SimpleNamespace(name="ascend-generic")), 0)
        self.assertFalse(model.get_metadata()["enabled"])

    def test_context_dependent_model_advances_one_iteration(self):
        instance = SimpleNamespace(
            batch=[SimpleNamespace(token_size=15, generated_tokens=2)],
            prompt_tasks_in_batch=[],
        )
        with patch.object(performance_model, "performance_model",
                          SimpleNamespace(supports_iteration_coalescing=False)):
            self.assertEqual(ORCAInstance.get_num_contiguous_iterations(instance), 1)
        with patch.object(performance_model, "performance_model", SimpleNamespace()):
            self.assertEqual(ORCAInstance.get_num_contiguous_iterations(instance), 13)

    def test_zero_time_trace_is_dispatched_after_initialization(self):
        calls = []
        ready = {"value": False}
        def arrive(request):
            self.assertTrue(ready["value"])
            calls.append(request)
        router = SimpleNamespace(run=lambda: None, request_arrival=arrive)
        stub = SimpleNamespace(run=lambda: None)
        trace = SimpleNamespace(requests=[SimpleNamespace(arrival_timestamp=0)])
        with patch("simulator.utils.file_logger"):
            sim = TraceSimulator(trace, stub, {}, router, stub, end_time=10)
            self.assertEqual(calls, [])
            ready["value"] = True
            with patch.object(sim, "save_results"):
                sim.run()
        self.assertEqual(len(calls), 1)

    def test_arrival_at_single_iteration_boundary_does_not_extend_decode(self):
        request = GenerativeLLMRequest(request_id=1, application_id=0,
                                       request_type=2, prompt_size=128, token_size=2)
        token = request.nodes[1]
        token.generating_tokens = 1
        token.processing_tokens = 1
        completed_steps = []
        with patch("simulator.utils.file_logger"):
            sim = Simulator(end_time=10)
            event = sim.schedule(1, lambda: completed_steps.append(token.generating_tokens))
            instance = SimpleNamespace(
                pause_next_iteration=False, prompt_tasks_in_batch=[],
                num_contiguous_iterations=1, iteration_duration=1,
                completion_events={"iteration": event}, batch=[token],
            )
            # A same-time arrival runs just before the existing completion.
            sim.time = 1
            ORCAInstance.pause_iteration(instance)
            sim.run()
        self.assertEqual(completed_steps, [1])
        self.assertEqual(sim.time, 1)
        self.assertIs(instance.completion_events["iteration"], event)


if __name__ == "__main__":
    unittest.main()
