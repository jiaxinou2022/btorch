"""Torch process setup — device resolution."""
from __future__ import annotations

import pytest

from connectome_dataset.benchmarks.backend_utils import resolve_torch_device


@pytest.fixture(scope="session")
def torch_device(bench_cfg) -> str:
    return resolve_torch_device(bench_cfg.device or "cuda")
