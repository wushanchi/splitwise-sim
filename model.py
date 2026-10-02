import logging
import math

from dataclasses import dataclass


@dataclass(kw_only=True)
class ModelArchitecture():
    name: str
    num_layers: int


@dataclass(kw_only=True)
class LLMArchitecture(ModelArchitecture):
    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    head_dim: int | None = None
    kv_dtype_size: float | None = None
    kv_cache_bytes_per_token: float | None = None

    def __post_init__(self):
        for name in ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or
                                      not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("kv_dtype_size", "kv_cache_bytes_per_token"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.num_kv_heads is not None:
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError("num_kv_heads must divide num_heads")
        if self.head_dim is None and self.hidden_size % self.num_heads:
            raise ValueError("Set head_dim when hidden_size is not divisible by num_heads")
        if self.kv_cache_bytes_per_token is not None and any(
            value is not None for value in
            (self.num_kv_heads, self.head_dim, self.kv_dtype_size)
        ):
            raise ValueError("Use either KV head/dtype fields or kv_cache_bytes_per_token")

    def get_kv_cache_bytes_per_token(self, model_dtype_size):
        """Whole-model bytes for one sequence token, before TP/PP distribution.

        Unspecified fields retain the original MHA capacity formula. The opaque
        override can represent a measured/compressed cache layout such as MLA;
        it does not simulate that layout's kernels, buffers, or communication.
        """
        if self.kv_cache_bytes_per_token is not None:
            return self.kv_cache_bytes_per_token
        heads = self.num_kv_heads if self.num_kv_heads is not None else self.num_heads
        dimension = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_heads
        dtype_size = self.kv_dtype_size if self.kv_dtype_size is not None else model_dtype_size
        if isinstance(dtype_size, bool) or not math.isfinite(dtype_size) or dtype_size <= 0:
            raise ValueError("KV dtype size must be finite and positive")
        return 2 * heads * dimension * self.num_layers * dtype_size


@dataclass(kw_only=True)
class ModelParallelism():
    """
    Captures the different parallelisms of a Model.
    """
    pipeline_parallelism: int
    tensor_parallelism: int

    @property
    def num_processors(self):
        """
        The number of accelerators required is the product of the parallelisms.
        """
        return self.pipeline_parallelism * self.tensor_parallelism


@dataclass(kw_only=True)
class ModelSize():
    """
    Captures the various sizes of a Model.
    """
    weights: int
    dtype_size: int

    @property
    def total_size(self):
        return self.weights


@dataclass(kw_only=True)
class Model():
    name: str
    architecture: ModelArchitecture
    parallelism: ModelParallelism
    size: ModelSize

    @property
    def size_per_processor(self):
        return self.size.total_size / self.parallelism.num_processors


@dataclass(kw_only=True)
class GenerativeLLM(Model):
    """
    Generative Large Language Model.
    NOTE: We currently don't capture embeddings, variable context lengths, etc.
    """
    context_size: int = 0
