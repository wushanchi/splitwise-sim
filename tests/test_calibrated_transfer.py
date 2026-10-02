import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import simulator
from calibrated_interconnect import CalibratedLink
from flow import FlowType
from model import LLMArchitecture
from node import NodeState
from request import GenerativeLLMRequest
from scheduler import CalibratedKVScheduler


class MemoryInstance:
    """Capacity/accounting endpoint; use real Flow and Task lifecycle methods."""

    def __init__(self, instance_id, tag, max_memory=10000):
        self.instance_id = instance_id
        self.tag = tag
        self.name = "test-device"
        self.memory = 0
        self.max_memory = max_memory
        self.sched_memory = 0
        self.sched_pending_tokens = 0
        self.notifications = []
        self.allocations = {}
        self.model = SimpleNamespace(
            architecture=LLMArchitecture(name="test", hidden_size=2,
                                         num_layers=1, num_heads=1),
            size=SimpleNamespace(dtype_size=2, total_size=0))

    def alloc_memory(self, request, amount):
        self.memory += amount
        self.allocations[request] = self.allocations.get(request, 0) + amount

    def free_memory(self, request, amount):
        self.memory -= amount
        self.sched_memory -= amount
        self.allocations[request] -= amount

    def notify_flow_completion(self, flow):
        self.notifications.append(flow)


class CalibratedTransferTests(unittest.TestCase):
    def setUp(self):
        previous_simulator = simulator.sim
        self.addCleanup(setattr, simulator, "sim", previous_simulator)
        logger = patch("utils.file_logger", return_value=Mock())
        logger.start()
        self.addCleanup(logger.stop)
        previous_directory = os.getcwd()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        os.chdir(directory.name)
        self.addCleanup(os.chdir, previous_directory)
        self.sim = simulator.Simulator(end_time=20)
        self.src = MemoryInstance(0, "prompt")
        self.dest = MemoryInstance(1, "token")
        self.finished = []

    def request(self, request_id=0, prompt_size=2, token_size=4):
        return GenerativeLLMRequest(request_id=request_id, request_type=2,
                                    application_id=0, prompt_size=prompt_size,
                                    token_size=token_size)

    def link(self, src=None, dest=None, **overrides):
        params = dict(name="test-link", src=src or self.src, dest=dest or self.dest,
                      transfer_bandwidth_gib_s=1, transfer_latency_ms=100,
                      retry_delay=0.25)
        params.update(overrides)
        return CalibratedLink(**params)

    def flow(self, request_id, size=1024**3, memory=10, src=None, dest=None):
        src, dest = src or self.src, dest or self.dest
        request = self.request(request_id)
        request.memory = memory
        src.alloc_memory(request, memory)
        flow = request.create_flow(FlowType.KVCacheTransfer, size=size, src=src, dest=dest)
        flow.notify = True
        flow.executor = SimpleNamespace(
            finish_flow=lambda completed, link: self.finished.append(completed))
        return flow

    def test_same_pair_serializes_and_preserves_memory_lifecycle(self):
        link = self.link()
        first, second = self.flow(1), self.flow(2)
        self.assertEqual(first.node_id, second.node_id)  # IDs are per Request.
        link.flow_arrival(first)
        link.flow_arrival(second)
        self.assertEqual(first.state, NodeState.RUNNING)
        self.assertEqual(second.state, NodeState.QUEUED)
        self.assertEqual(self.src.memory, 20)
        self.assertEqual(self.dest.memory, 10)
        self.assertEqual(link.bandwidth_free, 0)
        self.sim.run()
        self.assertAlmostEqual(first.metrics.completion_timestamp, 1.1)
        self.assertAlmostEqual(second.metrics.start_timestamp, 1.1)
        self.assertAlmostEqual(second.metrics.completion_timestamp, 2.2)
        self.assertEqual(self.src.memory, 0)
        self.assertEqual(self.dest.memory, 20)
        self.assertEqual([id(flow) for flow in self.finished], [id(first), id(second)])
        self.assertEqual(len(self.src.notifications), 2)
        self.assertEqual(link.bandwidth_used, 0)
        link.flow_completion(first)  # A stale callback must have no effects.
        self.assertEqual(len(self.finished), 2)
        self.assertEqual(self.src.memory, 0)

    def test_independent_pairs_transfer_in_parallel(self):
        other_dest = MemoryInstance(2, "token")
        first = self.flow(1)
        second = self.flow(2, dest=other_dest)
        self.link().flow_arrival(first)
        self.link(dest=other_dest).flow_arrival(second)
        self.sim.run()
        self.assertEqual(first.metrics.start_timestamp, 0)
        self.assertEqual(second.metrics.start_timestamp, 0)
        self.assertAlmostEqual(first.metrics.completion_timestamp, 1.1)
        self.assertAlmostEqual(second.metrics.completion_timestamp, 1.1)

    def test_memory_retry_is_fifo_and_starts_each_flow_once(self):
        self.dest.max_memory = 20
        self.dest.memory = 15
        first = self.flow(1, size=1024**3 / 10, memory=10)
        second = self.flow(2, size=1024**3 / 10, memory=5)
        link = self.link(transfer_latency_ms=0)
        link.flow_arrival(first)
        retry_event = link._retry_event
        link.flow_arrival(second)
        self.assertIs(link._retry_event, retry_event)
        self.assertEqual(first.state, NodeState.QUEUED)
        self.assertEqual(second.state, NodeState.QUEUED)  # Cannot bypass queue head.
        self.sim.schedule(0.4, lambda: setattr(self.dest, "memory", 0))
        self.sim.run()
        self.assertAlmostEqual(first.metrics.start_timestamp, 0.5)
        self.assertAlmostEqual(first.metrics.completion_timestamp, 0.6)
        self.assertAlmostEqual(second.metrics.start_timestamp, 0.6)
        self.assertAlmostEqual(second.metrics.completion_timestamp, 0.7)
        self.assertEqual(self.dest.memory, 15)
        self.assertEqual(self.src.memory, 0)
        self.assertEqual(len(self.finished), 2)
        self.assertIsNone(link._retry_event)
        self.assertEqual(len(self.sim.events), 0)

    def test_duplicate_arrival_and_wrong_endpoint_are_rejected(self):
        link = self.link()
        first = self.flow(1)
        link.flow_arrival(first)
        with self.assertRaisesRegex(ValueError, "only once"):
            link.flow_arrival(first)
        wrong = self.flow(2, dest=MemoryInstance(2, "token"))
        with self.assertRaisesRegex(ValueError, "endpoints"):
            link.flow_arrival(wrong)

    def test_invalid_transfer_parameters_fail_fast(self):
        for bandwidth in (0, -1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(bandwidth=bandwidth), self.assertRaises(ValueError):
                self.link(transfer_bandwidth_gib_s=bandwidth)
        for latency in (-1, float("nan"), float("inf"), True):
            with self.subTest(latency=latency), self.assertRaises(ValueError):
                self.link(transfer_latency_ms=latency)
        for retry_delay in (0, -1, float("inf")):
            with self.subTest(retry_delay=retry_delay), self.assertRaises(ValueError):
                self.link(retry_delay=retry_delay)
        with self.assertRaises(ValueError):
            self.link().flow_arrival(self.flow(1, size=-1))

    def scheduler(self, **overrides):
        params = dict(application=SimpleNamespace(application_id=0), router=Mock(),
                      overheads={}, executor_overheads=SimpleNamespace(
                          submit_task=0, submit_flow=0, finish_request=0),
                      prompt_processors=[], token_processors=[],
                      transfer_bandwidth_gib_s=1, transfer_latency_ms=5)
        params.update(overrides)
        scheduler = CalibratedKVScheduler(**params)
        scheduler.add_instance(self.src)
        scheduler.add_instance(self.dest)
        return scheduler

    def test_scheduler_reuses_pair_and_accounts_for_completed_tasks(self):
        scheduler = self.scheduler()
        request = self.request(1)
        scheduler.schedule(request)
        second = self.request(2)
        scheduler.schedule(second)
        self.assertIs(request.flow_node.link, second.flow_node.link)
        self.assertEqual(len(scheduler.links), 1)
        self.assertEqual(self.src.sched_pending_tokens, 4)
        self.assertEqual(self.dest.sched_pending_tokens, 2)
        self.assertEqual(scheduler.get_metadata()["overlap"], False)
        self.assertEqual(scheduler.get_metadata()["transfer_bandwidth_gib_s"], 1)
        prompt = request.root_node
        token = next(request.successors(request.flow_node))
        self.assertIs(prompt.instance, self.src)
        self.assertIs(token.instance, self.dest)
        prompt.arrive()
        prompt.run()
        prompt.processing_tokens = prompt.prompt_size
        prompt.generating_tokens = 1
        prompt.complete_iteration()
        prompt.complete()
        self.assertEqual(self.src.sched_pending_tokens, 2)

        def finish_flow(flow, link):
            token.arrive()
            token.run()
            token.processing_tokens = token.token_size
            token.generating_tokens = token.token_size
            token.complete_iteration()
            token.complete()

        request.flow_node.executor = SimpleNamespace(finish_flow=finish_flow)
        request.flow_node.link.flow_arrival(request.flow_node)
        self.sim.run()
        self.assertEqual(self.dest.sched_pending_tokens, 1)
        self.assertEqual(self.src.memory, 0)
        self.assertEqual(self.dest.memory, 0)
        # Reservation for the second, not-yet-running request remains intact.
        self.assertEqual(self.src.sched_memory, second.root_node.max_memory(self.src))
        second_token = next(second.successors(second.flow_node))
        self.assertEqual(self.dest.sched_memory,
                         second.root_node.max_memory(self.dest) +
                         second_token.max_memory(self.dest))

    def test_scheduler_uses_tags_and_updated_load(self):
        scheduler = self.scheduler()
        other_src, other_dest = MemoryInstance(2, "prompt"), MemoryInstance(3, "token")
        scheduler.add_instance(other_src)
        scheduler.add_instance(other_dest)
        first, second = self.request(1), self.request(2)
        scheduler.schedule(first)
        scheduler.schedule(second)
        self.assertIs(first.root_node.instance, self.src)
        self.assertIs(second.root_node.instance, other_src)
        self.assertIs(second.flow_node.dest, other_dest)
        self.assertEqual(len(scheduler.links), 2)
        with self.assertRaisesRegex(ValueError, "tags"):
            scheduler.add_instance(MemoryInstance(4, None))
        with self.assertRaises(ValueError):
            self.scheduler(transfer_bandwidth_gib_s=0)


if __name__ == "__main__":
    unittest.main()
