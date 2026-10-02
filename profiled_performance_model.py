"""Iteration latency lookup with explicit profile provenance and shape semantics.

This module does not infer NPU performance from GPU data. Profiles are measured
externally or explicitly labelled synthetic; all CSV latencies are milliseconds.
"""

import copy
import csv
import hashlib
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

from hydra.utils import get_original_cwd

from performance_model import PerformanceModel
from task import PromptTask, TokenTask


class ProfiledPerformanceModel(PerformanceModel):
    """Look up one iteration using phase, batch work, and decode context.

    ``exact`` is the default. ``nearest`` explicitly permits approximation only
    within the same model, hardware, TP and phase. Its distance is the sum of
    abs(candidate - query) / max(query, 1) over the three shape dimensions.
    Ties are resolved by ascending (batch_size, batch_tokens, context_tokens).

    Context changes after each decode iteration, so callers must not coalesce
    several iterations using one predicted latency.
    """

    context_sensitive = True
    supports_iteration_coalescing = False
    COLUMNS = (
        "model", "hardware", "tensor_parallel", "phase", "batch_size",
        "batch_tokens", "context_tokens", "latency_ms",
    )
    PHASES = frozenset(("prefill", "decode", "mixed"))

    def __init__(self, profile_path, metadata_path, allow_synthetic=False,
                 lookup_mode="exact"):
        if lookup_mode not in ("exact", "nearest"):
            raise ValueError("lookup_mode must be 'exact' or 'nearest'")
        if not isinstance(allow_synthetic, bool):
            raise ValueError("allow_synthetic must be a boolean")
        self.profile_path = self._resolve_path(profile_path)
        self.metadata_path = self._resolve_path(metadata_path)
        self.lookup_mode = lookup_mode
        self.allow_synthetic = allow_synthetic
        self.exact_lookup_count = 0
        self.nearest_lookup_count = 0
        self._nearest_warning_emitted = False
        self._metadata = self._load_metadata()
        if self._metadata["provenance"] == "synthetic" and not allow_synthetic:
            raise ValueError(
                "Synthetic profile refused: set allow_synthetic=true explicitly "
                "for a demonstration; synthetic latency is not hardware evidence."
            )
        self._profiles, self.profile_row_count = self._load_profiles()
        self.profile_sha256 = self._sha256(self.profile_path)
        self.metadata_sha256 = self._sha256(self.metadata_path)
        # Register the global model only after all validation has succeeded.
        super().__init__()

    @staticmethod
    def _resolve_path(path):
        result = Path(path).expanduser()
        if not result.is_absolute():
            try:
                base = Path(get_original_cwd())
            except ValueError:  # Ordinary Python/unittest, outside Hydra.
                base = Path.cwd()
            result = base / result
        return result.resolve()

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _load_metadata(self):
        with self.metadata_path.open(encoding="utf-8-sig") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("Profile metadata must be a JSON object")
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise ValueError("Profile metadata schema_version must be 1")
        if metadata.get("provenance") not in ("measured", "synthetic"):
            raise ValueError("Profile metadata provenance must be measured or synthetic")
        if not isinstance(metadata.get("description"), str) or not metadata["description"].strip():
            raise ValueError("Profile metadata requires a nonempty description")
        if metadata.get("latency_unit") != "ms":
            raise ValueError("Profile metadata latency_unit must be 'ms'")
        if metadata["provenance"] == "measured":
            if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
                raise ValueError("Measured profile metadata requires a nonempty source")
            if not isinstance(metadata.get("environment"), dict) or not metadata["environment"]:
                raise ValueError("Measured profile metadata requires a nonempty environment object")
        return metadata

    @staticmethod
    def _integer(value, field, minimum, where):
        # CSV integer fields deliberately reject fractional/exponent notation.
        if isinstance(value, bool):
            raise ValueError(f"{where}: {field} must be an integer >= {minimum}")
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{where}: {field} must be an integer >= {minimum}") from exc
        if not isinstance(value, str) and parsed != value:
            raise ValueError(f"{where}: {field} must be an integer >= {minimum}")
        if parsed < minimum:
            raise ValueError(f"{where}: {field} must be an integer >= {minimum}")
        return parsed

    @classmethod
    def _validate_shape(cls, phase, batch_size, batch_tokens, context_tokens, where):
        if phase not in cls.PHASES:
            raise ValueError(f"{where}: unknown phase {phase!r}")
        if batch_tokens < batch_size:
            raise ValueError(f"{where}: batch_tokens must be >= batch_size")
        if phase == "prefill" and context_tokens != 0:
            raise ValueError(f"{where}: prefill context_tokens must be 0")
        if phase == "decode" and batch_tokens != batch_size:
            raise ValueError(f"{where}: decode batch_tokens must equal batch_size")
        if phase == "mixed" and batch_size < 2:
            raise ValueError(f"{where}: mixed phase requires at least two tasks")

    def _load_profiles(self):
        samples = defaultdict(list)
        row_count = 0
        with self.profile_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if (reader.fieldnames is None or len(reader.fieldnames) != len(self.COLUMNS)
                    or set(reader.fieldnames) != set(self.COLUMNS)):
                raise ValueError(f"Profile CSV columns must be exactly {','.join(self.COLUMNS)}")
            for row in reader:
                where = f"Profile CSV line {reader.line_num}"
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{where}: incorrect number of fields")
                model, hardware = row["model"].strip(), row["hardware"].strip()
                if not model or not hardware:
                    raise ValueError(f"{where}: model and hardware must be nonempty")
                tp = self._integer(row["tensor_parallel"], "tensor_parallel", 1, where)
                batch_size = self._integer(row["batch_size"], "batch_size", 1, where)
                batch_tokens = self._integer(row["batch_tokens"], "batch_tokens", 1, where)
                context_tokens = self._integer(row["context_tokens"], "context_tokens", 0, where)
                phase = row["phase"].strip()
                self._validate_shape(phase, batch_size, batch_tokens, context_tokens, where)
                try:
                    latency_ms = float(row["latency_ms"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{where}: latency_ms must be positive and finite") from exc
                if not math.isfinite(latency_ms) or latency_ms <= 0:
                    raise ValueError(f"{where}: latency_ms must be positive and finite")
                key = (model, hardware, tp, phase)
                shape = (batch_size, batch_tokens, context_tokens)
                samples[(key, shape)].append(latency_ms)
                row_count += 1
        if not row_count:
            raise ValueError("Profile CSV must contain at least one data row")
        profiles = defaultdict(dict)
        for (key, shape), latencies in samples.items():
            latency_s = statistics.median(latencies) / 1000.0
            if not math.isfinite(latency_s) or latency_s <= 0:
                raise ValueError("Profile latency cannot be represented as positive finite seconds")
            profiles[key][shape] = latency_s
        return dict(profiles), row_count

    def _query(self, key, shape):
        candidates = self._profiles.get(key)
        if candidates is None:
            raise ValueError(
                f"No profile for model/hardware/TP/phase {key!r}; "
                "nearest never crosses these categories."
            )
        if shape in candidates:
            self.exact_lookup_count += 1
            return candidates[shape]
        if self.lookup_mode == "exact":
            raise ValueError(
                f"No exact profile for {key!r}, shape {shape!r} "
                "(batch_size, batch_tokens, context_tokens). "
                "Collect this shape or explicitly select lookup_mode=nearest."
            )
        def distance(candidate):
            return sum(abs(actual - requested) / max(requested, 1)
                       for actual, requested in zip(candidate, shape))
        chosen = min(candidates, key=lambda candidate: (distance(candidate), candidate))
        self.nearest_lookup_count += 1
        if not self._nearest_warning_emitted:
            warnings.warn(
                f"Approximate nearest profile lookup enabled: {key!r}, "
                f"requested {shape!r}, using {chosen!r}. These estimates are not "
                "measured at the requested shape; usage is recorded in metadata.",
                RuntimeWarning, stacklevel=3,
            )
            self._nearest_warning_emitted = True
        return candidates[chosen]

    def get_iteration_duration(self, batch, instance, *args, **kwargs):
        if not batch:
            raise ValueError("Cannot profile an empty iteration batch")
        processors = instance.processors
        if not processors or any(proc.name != processors[0].name for proc in processors):
            raise ValueError("Profiled iterations require processors with one hardware name")
        tp = self._integer(instance.model.parallelism.tensor_parallelism,
                           "tensor_parallel", 1, "Iteration")
        # No PP feature exists in this schema: reject instead of silently reusing TP data.
        pp = instance.model.parallelism.pipeline_parallelism
        if pp != 1:
            raise ValueError("ProfiledPerformanceModel schema v1 supports pipeline_parallelism=1 only")
        if len(processors) != tp:
            raise ValueError("Profiled iterations require processor count equal to tensor_parallelism")
        prompts = 0
        decodes = 0
        batch_tokens = 0
        context_tokens = 0
        for task in batch:
            if not isinstance(task, (PromptTask, TokenTask)):
                raise ValueError(f"Unsupported iteration task type {type(task).__name__}")
            request_batch = self._integer(task.request.batch_size, "request.batch_size", 1, "Iteration")
            if request_batch != 1:
                raise ValueError("ProfiledPerformanceModel supports request.batch_size=1 only")
            if isinstance(task, PromptTask):
                prompts += 1
                batch_tokens += self._integer(task.request.prompt_size, "prompt_size", 1, "Iteration")
            elif isinstance(task, TokenTask):
                decodes += 1
                batch_tokens += 1
                context_tokens += self._integer(task.request.processed_tokens,
                                                "processed_tokens", 0, "Iteration")
        phase = "mixed" if prompts and decodes else "prefill" if prompts else "decode"
        shape = (len(batch), batch_tokens, context_tokens)
        key = (instance.model.name, processors[0].name, tp, phase)
        return self._query(key, shape)

    def get_duration(self, task, batch, instance, *args, **kwargs):
        raise NotImplementedError(
            "ProfiledPerformanceModel requires iteration scheduling; whole-task "
            "duration cannot hold a changing decode context fixed."
        )

    def get_metadata(self):
        """Return a JSON-serializable snapshot, including approximation counts."""
        return {
            "model_class": f"{type(self).__module__}.{type(self).__name__}",
            "profile_path": str(self.profile_path),
            "metadata_path": str(self.metadata_path),
            "profile_sha256": self.profile_sha256,
            "metadata_sha256": self.metadata_sha256,
            "profile_metadata": copy.deepcopy(self._metadata),
            "provenance": self._metadata["provenance"],
            "lookup_mode": self.lookup_mode,
            "allow_synthetic": self.allow_synthetic,
            "context_sensitive": self.context_sensitive,
            "supports_iteration_coalescing": self.supports_iteration_coalescing,
            "profile_row_count": self.profile_row_count,
            "profile_shape_count": sum(len(shapes) for shapes in self._profiles.values()),
            "exact_lookup_count": self.exact_lookup_count,
            "nearest_lookup_count": self.nearest_lookup_count,
            "nearest_distance": "sum(abs(candidate-query)/max(query,1))",
        }
