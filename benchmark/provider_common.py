"""Shared provider contracts for RSNN benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from benchmark.benchmark_persistent_snn import RSNNResult


ResetPolicy = Literal["stateless", "before_phase", "before_sample"]
ExecutionClass = Literal["device_native", "device_adapted", "host_wrapper"]
TimingMode = Literal["isolated", "queued"]


@dataclass
class PreparedMetadata:
    """Describe buffers and execution semantics produced during preparation."""

    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    input_layout: str
    operator_layout: str
    index_dtype: str
    value_dtype: str
    compute_dtype: str
    workspace_bytes: int
    persistent_bytes: int
    padding_ratio: float
    execution_class: ExecutionClass
    timing_scope: str
    fusion_level: str = "none"
    state_semantics: str = "independent_inference"
    uses_host_control: bool = False
    uses_internal_sync: bool = False
    uses_d2h: bool = False
    uses_h2d: bool = False
    allocates_during_run: bool = False
    graph_capturable: bool = False
    layout_transform_mode: str = "none"
    layout_transform_in_timing: bool = False
    spike_representation: str = "dense"
    input_contiguous: bool = True
    input_address_mod: int = 0
    run_allocated_bytes: int = 0
    run_reserved_delta: int = 0
    native_available: bool = False
    native_status: str = "not_provided"
    selected_kernels: str = ""
    selector_model: str = ""

    @property
    def logical_n(self) -> int:
        """Return the logical neuron count."""

        return self.logical_shape[-1]

    @property
    def logical_batch(self) -> int:
        """Return the logical batch size."""

        return self.logical_shape[-2]

    @property
    def physical_n(self) -> int:
        """Return the physical neuron count."""

        return self.physical_shape[-1]

    @property
    def physical_batch(self) -> int:
        """Return the physical batch size."""

        return self.physical_shape[-2]

    @property
    def total_prepared_bytes(self) -> int:
        """Return all live tensor bytes after prepare.

        ``workspace_bytes`` is the reusable scratch subset and must not be
        added to this value. ``persistent_bytes`` remains as a compatibility
        field for existing CSV readers.
        """

        return self.persistent_bytes


@dataclass
class BenchmarkRunner:
    """Execute one prepared benchmark provider with explicit reset semantics."""

    run_fn: Callable[[], RSNNResult]
    reset_fn: Callable[[], None]
    metadata: PreparedMetadata
    reset_policy: ResetPolicy = "stateless"

    def prepare(self) -> None:
        """Complete the common interface after eager adapter preparation."""

    def reset(self) -> None:
        """Restore the prepared state."""

        self.reset_fn()

    def run(self) -> RSNNResult:
        """Execute one benchmark sample."""

        return self.run_fn()

    def __call__(self) -> RSNNResult:
        """Preserve callable compatibility with timing helpers."""

        return self.run()


@dataclass
class PreparedOperator:
    """Expose genuinely distinct native and adapted operator entry points."""

    native_run: Callable[[], None] | None
    adapted_run: Callable[[], None]
    transform_run: Callable[[], None] | None
    native_output: torch.Tensor | None
    adapted_output: torch.Tensor
    metadata: PreparedMetadata
    release_fn: Callable[[], None]

    def release(self) -> None:
        """Release native handles owned by the prepared operator."""

        self.release_fn()


@dataclass(frozen=True)
class AuditResult:
    """Summarize one profiler audit of a prepared device provider."""

    provider: str
    audited: bool
    has_cuda_malloc: bool
    has_cuda_free: bool
    has_d2h: bool
    has_h2d: bool
    has_internal_sync: bool
    unexpected_alloc_ops: tuple[str, ...]
    observed_ops: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit manifest entry."""

        return {
            "provider": self.provider,
            "audited": self.audited,
            "cuda_malloc_in_run": self.has_cuda_malloc,
            "cuda_free_in_run": self.has_cuda_free,
            "d2h_in_run": self.has_d2h,
            "h2d_in_run": self.has_h2d,
            "internal_sync_in_run": self.has_internal_sync,
            "unexpected_alloc_ops": list(self.unexpected_alloc_ops),
            "observed_ops": list(self.observed_ops),
        }


def _reset(
    reset_fn: Callable[[], None],
    reset_policy: ResetPolicy,
    *,
    before_sample: bool,
) -> None:
    if not before_sample or reset_policy == "before_sample":
        reset_fn()


def run_warmup(
    run_fn: Callable[[], object],
    reset_fn: Callable[[], None],
    reset_policy: ResetPolicy,
    warmup: int,
) -> None:
    """Execute one explicitly reset warmup phase."""

    _reset(reset_fn, reset_policy, before_sample=False)
    for _ in range(warmup):
        _reset(reset_fn, reset_policy, before_sample=True)
        run_fn()
    torch.cuda.synchronize()


def time_cuda_callable(
    run_fn: Callable[[], object],
    *,
    reset_fn: Callable[[], None] = lambda: None,
    reset_policy: ResetPolicy = "stateless",
    warmup: int,
    repeat: int,
    timing_mode: TimingMode = "queued",
) -> list[float]:
    """Measure isolated latency or queued steady-state GPU execution."""

    run_warmup(run_fn, reset_fn, reset_policy, warmup)
    _reset(reset_fn, reset_policy, before_sample=False)
    samples = []
    if timing_mode == "isolated":
        for _ in range(repeat):
            _reset(reset_fn, reset_policy, before_sample=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends, strict=True):
        _reset(reset_fn, reset_policy, before_sample=True)
        start.record()
        run_fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def inspect_tensor(
    tensor: torch.Tensor,
    *,
    required_alignment: int = 256,
) -> dict[str, object]:
    """Return physical tensor layout and address information."""

    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
        "address_alignment": tensor.data_ptr() % required_alignment,
        "dtype": str(tensor.dtype),
    }


def tensor_bytes(tensors: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> int:
    """Return storage bytes represented by a tensor collection."""

    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
