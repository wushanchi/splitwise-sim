"""A serial KV transfer model calibrated in GiB/s and milliseconds.

Contention is scoped to one directed instance pair. Different links do not share
NIC or switch capacity, and this model does not simulate HCCL or transfer overlap.
"""

import math

from node import NodeState
from simulator import cancel_event, schedule_event


def _finite_number(value, name, *, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return number


class CalibratedLink:
    """FIFO transfers sharing the full bandwidth of a directed instance pair.

    Only the queue head can start. If receiver memory is unavailable, one retry
    event polls at ``retry_delay`` seconds; smaller followers cannot bypass it.
    Flow.run/complete retain the existing destination allocation/source release
    lifecycle. This class intentionally does not inherit the legacy Link logic.
    """

    @staticmethod
    def validate_parameters(transfer_bandwidth_gib_s, transfer_latency_ms):
        bandwidth = _finite_number(transfer_bandwidth_gib_s,
                                   "transfer_bandwidth_gib_s", positive=True)
        latency = _finite_number(transfer_latency_ms, "transfer_latency_ms")
        if not math.isfinite(bandwidth * 1024**3):
            raise ValueError("transfer_bandwidth_gib_s overflows bytes/second")
        return bandwidth, latency

    def __init__(self, *, name, src, dest, transfer_bandwidth_gib_s,
                 transfer_latency_ms, retry_delay=1.0):
        bandwidth, latency = self.validate_parameters(
            transfer_bandwidth_gib_s, transfer_latency_ms)
        self.retry_delay = _finite_number(retry_delay, "retry_delay", positive=True)
        if src is dest:
            raise ValueError("CalibratedLink requires distinct source and destination")
        self.name = name
        self.src = src
        self.dest = dest
        self.transfer_bandwidth_gib_s = bandwidth
        self.transfer_latency_ms = latency
        self.bandwidth = bandwidth * 1024**3
        self.max_flows = 1
        self.pending_queue = []
        self.executing_queue = []
        self.completed_queue = []
        self._active_flow = None
        self._retry_event = None
        self._finishing = False
        self._draining = False

    @property
    def bandwidth_used(self):
        return self.bandwidth if self._active_flow is not None else 0.0

    @property
    def bandwidth_free(self):
        return self.bandwidth - self.bandwidth_used

    def get_duration(self, flow):
        size = _finite_number(flow.size, "flow.size")
        duration = self.transfer_latency_ms / 1000 + size / self.bandwidth
        if not math.isfinite(duration):
            raise ValueError("KV transfer duration must be finite")
        return duration

    def flow_arrival(self, flow):
        if flow.src is not self.src or flow.dest is not self.dest:
            raise ValueError("Flow endpoints do not match this directed link")
        if flow.state != NodeState.NONE:
            raise ValueError("A flow can arrive at its link only once")
        self.get_duration(flow)
        _finite_number(flow.request.memory, "flow.request.memory")
        flow.link = self
        flow.arrive()
        self.pending_queue.append(flow)
        self._try_start_next()

    def _cancel_retry(self):
        if self._retry_event is not None:
            cancel_event(self._retry_event)
            self._retry_event = None

    def _retry(self):
        self._retry_event = None
        self._try_start_next()

    def _try_start_next(self):
        # Completion callbacks may synchronously submit more flows. Keep those
        # arrivals queued until the current callback and FIFO drain are finished.
        if self._finishing or self._draining:
            return
        self._draining = True
        try:
            while self._active_flow is None and self.pending_queue:
                flow = self.pending_queue[0]
                if self.dest.memory + flow.request.memory > self.dest.max_memory:
                    if self._retry_event is None:
                        self._retry_event = schedule_event(self.retry_delay, self._retry)
                    return
                self._cancel_retry()
                self.pending_queue.pop(0)
                self._active_flow = flow
                self.executing_queue.append(flow)
                flow.duration = self.get_duration(flow)
                flow.run()
                schedule_event(flow.duration, lambda flow=flow: self.flow_completion(flow))
                # A zero-byte, zero-latency flow may finish synchronously. The
                # loop then drains its successor without recursive start calls.
        finally:
            self._draining = False

    def flow_completion(self, flow):
        # Identity matters: Node ids are only unique within each Request.
        if self._active_flow is not flow or flow.state != NodeState.RUNNING:
            return
        self._finishing = True
        try:
            flow.complete()
            self._active_flow = None
            self.executing_queue.clear()
            self.completed_queue.append(flow)
            flow.executor.finish_flow(flow, self)
            if flow.notify:
                flow.src.notify_flow_completion(flow)
        finally:
            self._finishing = False
        self._try_start_next()
